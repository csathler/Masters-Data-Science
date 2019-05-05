#####################################################################################
# Author:  Carlos Sathler
# Email:   cssathler@gmail.com
# Class:   Big Data Application and Analytics
# Date:    11/14/2016 
# Program: load_data.py
# Purpose: This program reads the data for the project and creates csv files 
#          that will be used for exploratory data analysis and predictive analitics
#####################################################################################

import sys
import pandas as pd
import numpy as np
import string
import time
import json 
import commands as cmd
import util_load_data_call as call_check 
import util_download_data  as down_data 

#------------------------------------------------------------------------------------
# Display timestamp and message 
# 
def timestamp(msg):
    x = string.split(time.ctime())
    now = x[1] + " " + x[2] + " " + x[4] + " " + x[3]
    print now + "\t> " + msg

#------------------------------------------------------------------------------------
# Create empty directory to store discussion tree nodes and comments 
# If one already exists, make sure to remove all files from previous run
# 
def get_temp_out_dir():
    (ret, pwd) = cmd.getstatusoutput('pwd')
    # cleans directory; assumes permission is not an issue
    (ret, dummy) = cmd.getstatusoutput('rm -r temp_data')
    # recreate directory; assumes permission is not an issue 
    (ret, dummy) = cmd.getstatusoutput('mkdir temp_data') 
    # return full path to temp output directory
    return pwd + "/" + "temp_data"

#------------------------------------------------------------------------------------
# Check data was properly load
# This function used during testing 
#
def verify_load(partition):

    # check if counts match
    # this only works for test runs
    # for regular run, getting "Argument list too long", so ignore

    # check count of discussion tree node files first
    timestamp("Verifying counts of tree node file")
    shellcmd = "cat disc_tree_nodes_" + partition + ".csv | wc -l" 
    (ret, count1) = cmd.getstatusoutput(shellcmd)
    (ret, count2) = cmd.getstatusoutput('cat temp_data/*nodes.csv | wc -l')
    if (int(count1) != int(count2)):
        print "Something went wrong concatenating nodes files!"

    # check count of comments files
    timestamp("Verifying counts of comments file")
    shellcmd = "cat raw_comments_" + partition + ".csv | wc -l" 
    (ret, count1) = cmd.getstatusoutput(shellcmd)
    (ret, count2) = cmd.getstatusoutput('cat temp_data/*comms.csv | wc -l')
    if (int(count1) != int(count2)):
        print "Something went wrong concatenating comments files!"

#------------------------------------------------------------------------------------
# Create filtered files
# Only include discussions where at least 10 users replied to OP
# Only include discussions where OP engaged at least once after posting challenge 
# and when she engaged, it wasn't to award a delta
#
def create_filtered_files(partition):

    timestamp("Filtering out less relevant '" + partition + "' discussions...")
    ops_source   = "ops_" + partition + ".csv"
    nodes_source = "disc_tree_nodes_" + partition + ".csv"

    df_ops   = pd.read_csv(ops_source)
    df_nodes = pd.read_csv(nodes_source)

    # project root_id and author of each node
    x = df_nodes[['root_id','author']]

    # select distinct authors per discussion tree
    x = x.drop_duplicates() 
     
    # count distinct authors per discussion tree
    x = x.groupby('root_id').agg( {'author':np.size} )

    # get list of root_ids that had at least 10 users reply to original post
    # note one of the authors is the OP, hence count need s to be at least 11
    well_challenged_trees = x[ x['author'] >= 11 ].index

    # get list of root_ids that have OP reply at least once not to award a delta
    # these are trees in which a delta did not occur but not for lack of engagement

    # first remove nodes where there was a delta award
    x = df_nodes[ df_nodes['delta_award_tf'] != True ]

    # next remove nodes that are not by the op
    x = x[ x['author'] == x['op_author'] ]

    # count how many time op_author posted to the tree
    x = x.groupby('root_id').agg( {'op_author':np.size} )

    # good trees are the ones there was no delta not for lack of engagement 
    op_engaged_trees = x[ x['op_author'] > 1 ].index

    # create dataframe for filtered tree root_ids
    filtered_root_ids = set(well_challenged_trees) & set(op_engaged_trees)
    df_filtered_root_ids = pd.DataFrame(pd.Series(list(filtered_root_ids)))
    df_filtered_root_ids.columns = [['root_id']]

    # remove rows from ops dataframe
    df_ops_filtered = pd.merge(df_ops, df_filtered_root_ids, how='inner', 
                      left_on='id', right_on='root_id')
    df_ops_filtered.drop('root_id', axis=1, inplace=True)

    # create filtered dataset 
    df_nodes_filtered = pd.merge(df_nodes, df_filtered_root_ids, on='root_id')

    ops_target   = "ops_" + partition + "_filtered.csv"
    nodes_target = "disc_tree_nodes_" + partition + "_filtered.csv"

    df_ops_filtered.to_csv(ops_target, index=False, encoding='utf-8')
    df_nodes_filtered.to_csv(nodes_target, index=False, encoding='utf-8')

#------------------------------------------------------------------------------------
# Flag delta nodes 
# 
def flag_deltas(partition):

    timestamp("Flagging deltas...")   
    ops_fname = "ops_" + partition + ".csv"
    nodes_fname = "disc_tree_nodes_" + partition + ".csv"
    df_ops   = pd.read_csv(ops_fname)
    df_nodes = pd.read_csv(nodes_fname)

    # create a copy of df_nodes just with needed columns 
    #df_deltabot_nodes = df_nodes[ df_nodes.author=='DeltaBot' ][['parent_id','id']].copy()
    df_deltabot_nodes = df_nodes[ df_nodes.author=='DeltaBot' ][['parent_id','id']]
    df_deltabot_nodes.columns = ['op_awarding_node_id','deltabot_node_id']
    
    # left join with nodes' parent records 
    # to find the nodes that the deltabot was confirming 
    df_op_awarding_nodes = pd.merge(df_nodes, df_deltabot_nodes, how='left',\
                                    left_on='id', right_on='op_awarding_node_id')

    # fill NaN with False, will use that later
    df_op_awarding_nodes = df_op_awarding_nodes.fillna(False)

    # row indexer 1: author of post is author of original post 
    # row indexer 2: no deltabot child
    op_author_post = df_op_awarding_nodes.author == df_op_awarding_nodes.op_author        
    deltabot_child = df_op_awarding_nodes.deltabot_node_id                                

    ids = df_op_awarding_nodes[ op_author_post & deltabot_child ]['id'].tolist()
    parent_ids = df_op_awarding_nodes[ op_author_post & deltabot_child ]['parent_id'].tolist()

    timestamp("Updating disc_tree_node dataframe; flagging 'delta_award_tf' and 'deltanode_tf'")
    df_nodes = df_nodes.set_index('id')
    df_nodes.loc[ ids,        'delta_award_tf' ] = True
    df_nodes.loc[ parent_ids, 'deltanode_tf'   ] = True 
    df_nodes = df_nodes.reset_index()

    timestamp("Updating ops dataframe; flagging 'deltatree_tf'")
    df_ops   = df_ops.set_index('id').copy()
    root_ids = df_nodes[ df_nodes.deltanode_tf==True ].root_id.drop_duplicates().tolist()
    df_ops.loc[ root_ids, 'deltatree_tf'   ] = True 
    df_ops = df_ops.reset_index()

    timestamp("Saving disc_tree_node data, with flagged delta nodes")
    df_nodes.to_csv(nodes_fname, index=False, encoding='utf-8')

    timestamp("Saving ops data, with flagged delta op nodes")
    df_ops.to_csv(ops_fname, index=False, encoding='utf-8')
    
#------------------------------------------------------------------------------------
# Remove repeating header rows from data files 
# The header rows exist mixed with the data because the files were created
# from a concatanation of temporary files created for each discussion tree 
#
def clean_up_files(partition):

    timestamp("Cleaning up nodes file")
    nodes_file = "disc_tree_nodes_" + partition + ".csv"
    df_nodes = pd.read_csv(nodes_file)
    df_nodes = df_nodes[ (df_nodes['id'] <> 'id') & (df_nodes['parent_id'] <> 'parent_id')]
    df_nodes.to_csv(nodes_file, index=False, encoding='utf-8')

    timestamp("Cleaning up comments file")
    comms_file = "raw_comments_" + partition + ".csv"
    df_comms = pd.read_csv(comms_file)
    df_comms = df_comms[ (df_comms['id'] <> 'id') & (df_comms['parent_id'] <> 'parent_id')]
    df_comms.to_csv(comms_file, index=False, encoding='utf-8')

#------------------------------------------------------------------------------------
# Concatenate temp data file into final data files 
#
def concat_temp_files(partition):

    timestamp("Concatenating temp nodes files")
    target_file = "disc_tree_nodes_" + partition + ".csv"

    # delete file from previous run, if any
    shellcmd = "rm " + target_file
    (rc, files) = cmd.getstatusoutput(shellcmd)
    
    (rc, sfiles) = cmd.getstatusoutput('ls temp_data | grep nodes')
    lfiles = sfiles.split()

    for file in lfiles:
        full_file = "temp_data/" + file
        shellcmd = "cat " + full_file + " >> " + target_file 
        cmd.getstatusoutput(shellcmd)

    timestamp("Created nodes file: '" + target_file + "'") 

    timestamp("Concatenating temp comments files")
    target_file = "raw_comments_" + partition + ".csv"

    # delete file from previous run, if any
    shellcmd = "rm " + target_file
    (rc, files) = cmd.getstatusoutput(shellcmd)

    (rc, sfiles) = cmd.getstatusoutput('ls temp_data | grep comms')
    lfiles = sfiles.split()

    for file in lfiles:
        full_file = "temp_data/" + file
        shellcmd = "cat " + full_file + " >> " + target_file 
        cmd.getstatusoutput(shellcmd)

    timestamp("Created comments file: '" + target_file + "'") 

#------------------------------------------------------------------------------------
# Add original post to original post dataframe
# 
def insert_op(df, cols, idx, node_id, op_author, title, url, treesize, deltatree_tf): 
    treesize = treesize + 1   # account for root node
    rec_op = dict([('id',             node_id),
                     ('op_author',    op_author),
                     ('title',        title),
                     ('url',          url),
                     ('treesize',     treesize),
                     ('deltatree_tf', deltatree_tf)])
    df_op = pd.DataFrame(data=rec_op, index=[idx], columns=cols)
    return pd.concat( [df, df_op] )

#------------------------------------------------------------------------------------
# Add discussion tree node to disc tree node dataframe 
# 
def insert_disc_tree_node(df, cols, idx, node_id, parent_id, author, op_author, 
    posttimeraw, degree, level, height, deltanode_tf, delta_award_tf, root_id):
    rec_disc_tree = dict([('id',        node_id),
                     ('parent_id',      parent_id),
                     ('author',         author),
                     ('op_author',      op_author),
                     ('posttimeraw',    posttimeraw),
                     ('degree',         degree),
                     ('level',          level),
                     ('height',         height),
                     ('deltanode_tf',   deltanode_tf),
                     ('delta_award_tf', delta_award_tf),
                     ('root_id',        root_id)])
    df_disc_tree = pd.DataFrame(data=rec_disc_tree, index=[idx], columns=cols)
    return pd.concat( [df, df_disc_tree] )

#------------------------------------------------------------------------------------
# Add comment row to comment dataframe 
# 
def insert_comment(df, cols, idx, node_id, parent_id, author, op_author, comm, 
    comm_html):
    rec_comm = dict([('id',           node_id),
                     ('parent_id',    parent_id),
                     ('author',       author),
                     ('op_author',    op_author),
                     ('comm',         comm),
                     ('comm_html',    comm_html)])
    df_comm = pd.DataFrame(data=rec_comm, index=[idx], columns=cols)
    return pd.concat( [df, df_comm] )

#------------------------------------------------------------------------------------
# Save discussion tree information to csv files in temp directory
# these files will be concatenated at the end of the program run 
#
def save_tree_data(op_id, temp_out_path, df_dt_nodes, df_dt_comms):
    comms_file = temp_out_path + "/" + op_id + "_dc_comms.csv"
    nodes_file = temp_out_path + "/" + op_id + "_dc_nodes.csv" 
    df_dt_nodes.to_csv(nodes_file, index=False, encoding='utf-8') 
    df_dt_comms.to_csv(comms_file, index=False, encoding='utf-8') 

#------------------------------------------------------------------------------------
# Traverse and update discussion three
# Fields: level, degree, height
# 
def traverse_and_update(node_id, level):
    global df_dt_nodes
    level = level + 1
    where_child    = df_dt_nodes['parent_id'] == node_id 
    where_not_root = df_dt_nodes['id'] != node_id 
    children = df_dt_nodes[ where_child & where_not_root ]['id'].tolist()
    degree   = len(children)
    max_height = 0 
    if degree > 0:
        for child in children: 
            height = traverse_and_update(child, level) + 1
            if height > max_height:
                max_height = height
    df_dt_nodes.loc[ df_dt_nodes['id'] == node_id, 'level']  = level - 1
    df_dt_nodes.loc[ df_dt_nodes['id'] == node_id, 'degree'] = degree
    df_dt_nodes.loc[ df_dt_nodes['id'] == node_id, 'height'] = max_height 
    return max_height

#------------------------------------------------------------------------------------
# Load data
#
def load_data(partition, source_file, rec_count):
    global df_dt_nodes

    timestamp("Begin load of '" + partition + "' data")

    # need to create temp files due to RAM limitations 
    timestamp("Preparing directory for temporary data files")
    temp_out_path = get_temp_out_dir() 

    timestamp("Begin reading file '" + source_file + "'")
    print "You need plenty of RAM for this step..."
    #----------------------------------------------------------------------
    # Begin citation 
    # Book: Python for Data Analysis, page 18
    # Author: Wes McKinney
    #
    records = [json.loads(dtree) for dtree in open(source_file)]
    #
    # End citation 
    #---------------------------------------------------------------------- 
    timestamp("End reading " + source_file)
 
    # Define columns for dataframes used to store
    # (1) original post information
    # (2) discussion tree information 
    # (3) comments information
    cols_op        = ['id','op_author','title','url','treesize','deltatree_tf']
    cols_disc_tree = ['id','parent_id','author','op_author','posttimeraw','degree',
                      'level','height','deltanode_tf','delta_award_tf','root_id']
    cols_comms     = ['id','parent_id','author','op_author','comm','comm_html']

    # initialize dataframe to store original posts
    df_all_ops = pd.DataFrame()

    # initialize discussion tree count (original post - "op") 
    op_count = 1

    # traverse the list of JSON records 
    for discussion in records: 

        # initialize count of nodes and comments for this discussion tree
        node_count = 1
        comm_count = 1

        op_id = discussion['id']            
        op_author = discussion['author']     

        # op_replies is list of dictionary objects
        op_replies = discussion['comments']

        # initializes flag that indicates if tree has delta  
        tree_has_delta = False               
    
        # initialize/reset dataframes to store info about current discussion tree
        df_dt_nodes = pd.DataFrame()
        df_dt_comms = pd.DataFrame()
   
        msg = "Data: " + partition + ", Tree: " + str(op_count) + ", Comms: " + str(len(op_replies))
        timestamp(msg)

        for comm in op_replies: 

            if not comm.has_key('author'):
                # bad record: skip
                continue 
    
            delta_award_tf = False 
            tree_has_delta = False 
            comment   = comm['body']
            author    = comm['author'] 
            parent_id = comm['parent_id']
            if ( string.find( parent_id, '_' ) != -1 ):
                parent_id = string.split( parent_id, '_')[1]

            # add comment to discussion tree comments dataset
            df_dt_comms = insert_comment(df_dt_comms, cols_comms, comm_count,
                                        comm['id'], parent_id, author, op_author, 
                                        comment, comm['body_html'])
            comm_count += 1 
    
            # add node to discussion tree dataset
            df_dt_nodes = insert_disc_tree_node(df_dt_nodes, cols_disc_tree, 
                                        node_count, comm['id'], parent_id, 
                                        author, op_author, comm['created_utc'], 0, 0, 0, 
                                        False, delta_award_tf, op_id)
            node_count += 1

        #endfor (comm)

        # if op only entered title, make that the op comment        
        if discussion['selftext'] == "":
            discussion['selftext'] = discussion['title']
            discussion['selftext_html'] = discussion['title']

        # add original post text to comments dataframe 
        df_dt_comms = insert_comment(df_dt_comms, cols_comms, comm_count,
                                     op_id, op_id, op_author, op_author, 
                                     discussion['selftext'], 
                                     discussion['selftext_html'])

        # add root node for the present discussion tree (the original post)
        df_dt_nodes = insert_disc_tree_node(df_dt_nodes, cols_disc_tree, 
                                     node_count, op_id, op_id, 
                                     discussion['author'], op_author, 
                                     discussion['created_utc'], 0, 0, 0, 
                                     False, delta_award_tf, op_id)

        # add op to original post dataframe 
        df_all_ops = insert_op(df_all_ops, cols_op, op_count, op_id, op_author, 
                                     discussion['title'], discussion['url'], 
                                     len(op_replies), tree_has_delta)
        op_count += 1 

        # traverse df_dt_nodes to update degree, level, height
        traverse_and_update(op_id, 0)  

        # dump dataframes to disk to keep memory use low 
        save_tree_data(op_id, temp_out_path, df_dt_nodes, df_dt_comms)

        if (op_count > rec_count):
            break 

    #endfor (discussion)

    op_fname = "ops_" + partition + ".csv" 
    timestamp("Saving original post data to file '" + op_fname + "'")
    df_all_ops.to_csv(op_fname, index=False, encoding='utf-8') 

    # concatenate temporary files
    concat_temp_files(partition)

    # if test run, check if files have proper counts
    if rec_count <= 500: 
        verify_load(partition) 

    # remove header rows created due to concatenation of temp files 
    clean_up_files(partition)

    # remove header rows created due to concatenation of temp files 
    flag_deltas(partition)

    # create files with filtered rows (only relevant discussions)
    create_filtered_files(partition)

#------------------------------------------------------------------------------------
# MAIN 
#
if __name__ == '__main__':

    # ensures program was properly called
    (data_to_load, dt_count) = call_check.get_good_parms_or_exit(sys.argv)

    # If needed, download data from original research team site
    fname = "cmv.tar.bz2"
    url = "https://chenhaot.com/data/cmv/"
    if down_data.needs_download(fname): 
        (train_data, test_data) = down_data.download_files(url, fname)
    else:
        (train_data, test_data) = down_data.get_input_file_names()

    if data_to_load == 'ALL':
        load_data('train', train_data, dt_count) 
        load_data('test',  test_data,  dt_count) 
    elif data_to_load == 'TRAIN':
        load_data('train', train_data, dt_count) 
    elif data_to_load == 'TEST':
        load_data('test',  test_data,  dt_count) 
