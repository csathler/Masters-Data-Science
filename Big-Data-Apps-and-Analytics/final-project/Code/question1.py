#####################################################################################
# Author:  Carlos Sathler
# Email:   cssathler@gmail.com
# Class:   Big Data Application and Analytics
# Date:    11/14/2016 
# Program: question1.py 
# Purpose: This program addresses the question: Was the data properly loaded?
#          It creates plots that are compared to the plots produced by the original
#          paper we are studying on the project.
#####################################################################################

import sys
import numpy as np
import pandas as pd
import time

#------------------------------------------------------------------------------------
# BEGIN CITATION - Stackoverflow
# Was getting error "RuntimeError: Python is not installed as a framework. on mac"
#http://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python
# Author: "Iron Pillow"
#
import matplotlib as mlp
#
if sys.platform == 'darwin':
	# in mac os only
        mlp.use('TkAgg')
#
import matplotlib.pyplot as plt
#
# END CITATION - Stackoverflow
#------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------
# Set plot tickers and label sizes
#
def set_tickers(pname, xlabel, ylabel):
    l = pname.set_xlabel(xlabel)
    l.set_fontsize('small')
    l = pname.set_ylabel(ylabel)
    l.set_fontsize('small')
    xlabels = pname.get_xticklabels()
    for label in xlabels: label.set_fontsize(6)
    ylabels = pname.get_yticklabels()
    for label in ylabels: label.set_fontsize(8)

#------------------------------------------------------------------------------------
#
# Gets list of labels and return list with blanks in the mix
def trim_labels(labels_in):
    labels_out = list()
    for item in labels_in:
        if (int(item) % 2) == 1:
            labels_out.append(item)
        else:
            labels_out.append("")
    return labels_out

#------------------------------------------------------------------------------------
# Read data needed for the plots
# Create and save plots
#
def analysis_q1():

    # populate dataframes from csv file
    df_nodes_fname = "disc_tree_nodes_train.csv"
    df_ops_fname = "ops_train.csv"
    df_nodes = pd.read_csv(df_nodes_fname)
    df_ops   = pd.read_csv(df_ops_fname)

    # remove header rows from nodes and comms; these were created during the load
    df_nodes = df_nodes[ (df_nodes['id'] <> 'id') & (df_nodes['parent_id'] <> 'parent_id')]

    # create column for year and month of the post
    df_nodes['YearMonth'] = df_nodes['posttimeraw'].\
                        apply(lambda x: time.strftime('%Y%m',time.gmtime(float(x))))

    fig = plt.figure()
    #fig.suptitle("Monthly Activity Metrics ([1, p616])", fontsize=14)
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, 
                        wspace=0.25, hspace=0.45)

    #------------------------------------------------
    # Will reproduce plots from the research paper
    #
    #------------------------------------------------
    # Plot 1: Number of discussions started per month
    #
    # plot1
    f_plot1 = fig.add_subplot(221)
    # create dataframe with original posts only
    df_op_nodes = df_nodes[ (df_nodes['id'] == df_nodes['parent_id']) ]
    plot1 = df_op_nodes.groupby('YearMonth')['id'].count()
    labels = plot1.index.tolist()
    x = np.arange(1, len(labels)+1).tolist()
    y = plot1.tolist()
    f_plot1.set_title("(a) Discussions Started per Month", fontsize=10)
    set_tickers(f_plot1, "Month", "Number of Posts") 
    plt.xticks(x, trim_labels(labels), rotation=45)
    plt.plot(x, y)
    
    #
    #---------------------------------------------------
    # Plot 2: Average number of replies to original post 
    #
    # plot2
    f_plot2 = fig.add_subplot(222)
    # Join op with with op_nodes dataframes to get treesize
    # Number of replies is treesize minus one  
    df_op_replies = pd.merge(df_ops[['id','treesize']], df_op_nodes, on='id')
    df_op_replies['replies'] = df_op_replies['treesize'].apply(lambda x: x-1)
    plot2 = df_op_replies.groupby('YearMonth').agg( {'replies' : np.mean} ) 
    labels = plot2.index.tolist()
    x = np.arange(1, len(labels)+1).tolist()
    y = plot2['replies'].tolist()
    f_plot2.set_title("(b) Average Replies per Discussion", fontsize=10)
    f_plot2.set_ylim(30,80)
    set_tickers(f_plot2, "Month", "Average Number of Replies") 
    plt.xticks(x, trim_labels(labels), rotation=45)
    plt.plot(x, y)
    
    #
    #-----------------------------------------------------
    # Plot 3: Average number challengers per original post 
    #
    # plot3
    f_plot3 = fig.add_subplot(223)
    # Count unique authors in reply nodes for each discussion 
    df_op_nodes  = df_nodes[ (df_nodes['id'] == df_nodes['parent_id']) ]
    df_rep_nodes = df_nodes[ (df_nodes['id'] <> df_nodes['parent_id']) ]
    df_rep_nodes = df_nodes[ (df_nodes['author'] <> df_nodes['op_author']) ]
    df_rep_nodes = df_nodes[['YearMonth','op_author','author']]
    df_rep_nodes = df_rep_nodes.drop_duplicates()
    plot3 = df_rep_nodes.groupby( ['YearMonth','op_author'] ).count()
    plot3 = plot3.groupby(level=['YearMonth']).mean()
    labels = plot3.index.tolist()
    x = np.arange(1, len(labels)+1).tolist()
    y = plot3['author'].tolist()
    f_plot3.set_title("(c) Average Challengers per Post", fontsize=10)
    f_plot3.set_ylim(10,30)
    set_tickers(f_plot3, "Month", "Average Number of Challengers") 
    plt.xticks(x, trim_labels(labels), rotation=45)
    plt.plot(x, y)
    
    #
    #-----------------------------------------------------
    # Plot 4: Average delta percentage 
    #
    # plot4
    f_plot4 = fig.add_subplot(224)
    
    # create column for year and month of the post
    df_nodes['YearMonth'] = df_nodes['posttimeraw'].\
                        apply(lambda x: time.strftime('%Y%m',time.gmtime(float(x))))
   
    # get count of discussions per month 
    df_discussions = df_nodes[df_nodes.id == df_nodes.parent_id]
    disc_per_month = pd.DataFrame(df_discussions.groupby('YearMonth')['id'].count())
    disc_per_month.columns = [['discussions']]

    # get count of deltas per month
    df_deltas = df_nodes[(df_nodes.deltanode_tf)][['YearMonth','root_id']]
    df_deltas = df_deltas.drop_duplicates()   # one delta per tree only
    deltas_per_month = pd.DataFrame(df_deltas.groupby('YearMonth')['root_id'].count())
    deltas_per_month.columns = [['deltas']]
        
    plot4 = pd.merge(disc_per_month, deltas_per_month, how='left',
                  left_index=True, right_index=True)

    plot4 = plot4.fillna(0)

    plot4['delta_percent'] = plot4.deltas / plot4.discussions * 100

    labels = plot4.index.tolist()
    x = np.arange(1, len(labels)+1).tolist()
    y = plot4['delta_percent'].tolist()
    f_plot4.set_title("(d) Delta Percentage", fontsize=10)
    f_plot4.set_ylim(15,40)
    set_tickers(f_plot4, "Month", "Delta Percentage") 
    plt.xticks(x, trim_labels(labels), rotation=45)
    plt.plot(x, y)

    # Saving first set of plots (same plots as paper)
    plt.savefig('question1.png')
    
    
    #------------------------------------------------
    # Won't be using the additonal plots below...
    #------------------------------------------------
    #
    #fig = plt.figure()
    #fig.suptitle("Question 1 additional plots", fontsize=14)
    # Plot 1.1: Number of posts per month
    #
    # plot5
    #plot5 = fig.add_subplot(121)
    #plot11 = df_nodes.groupby('YearMonth')['id'].count()
    #labels = plot11.index.tolist()
    #x = np.arange(1, len(labels)+1).tolist()
    #y = plot11.tolist()
    #plt.xticks(x, trim_labels(labels), rotation=45)
    #plt.plot(x, y)
    
    #---------------------------------------------------
    # Plot 2.1: Average number of root replies per month 
    #
    # plot6
    #plot5 = fig.add_subplot(122)
    # Degree captures number of child nodes
    #df_op_nodes = df_nodes[ (df_nodes['id'] == df_nodes['parent_id']) ].copy()
    #df_op_nodes['root_replies'] = df_op_nodes['degree'].apply(lambda x: int(x))
    #plot21 = df_op_nodes.groupby('YearMonth').agg( {'root_replies' : np.mean} ) 
    #labels = plot21.index.tolist()
    #x = np.arange(1, len(labels)+1).tolist()
    #y = plot21['root_replies'].tolist()
    #plt.xticks(x, trim_labels(labels), rotation=45)
    #plt.plot(x, y)

    #plt.savefig('question1_fig2.png')

    
#####################################################################################
# MAIN
#
#
if __name__ == '__main__':

    analysis_q1()
