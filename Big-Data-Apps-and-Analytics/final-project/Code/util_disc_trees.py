#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Author: Carlos Sathler
# cssathler@gmail.com
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

#####################################################################################
# Discussion tree utility functions
#
import sys
import pandas as pd
import numpy as np
import string
import time
import json 
import commands as cmd

#------------------------------------------------------------------------------------
# Check if node is delta node or has a child which is a delta node 
#
def is_delta(df_nodes, id):
    delta_tf = df_nodes[ df_nodes['id'] == id ]['deltanode_tf'].values[0]
    if (delta_tf == 'True'):
        # delta node, returns one
        return 1
    else:
        # look for delta node among children
        children = df_nodes[ df_nodes['parent_id'] == id ]['id'].tolist()
        for child in children:
            #print child
            has_delta_child = is_delta(df_nodes, child)
            if has_delta_child == 1:
                # child is delta node, returns one
                return 1 

    # not delta node; no delta nodes among children
    return 0 

#------------------------------------------------------------------------------------
# Print discussion tree node, with attributes
# 
def print_node_all(df_dt_nodes, id, level):
    tab = '    '
    s = df_dt_nodes[ df_dt_nodes['id'] == id ]
    idx = s.index[0]
    level  = s.get_value(idx, 'level')
    author = s.get_value(idx, 'author')
    degree = s.get_value(idx, 'degree')
    height = s.get_value(idx, 'height')
    delta  = s.get_value(idx, 'deltanode_tf') 
    node_info = tab*int(level) + str(id) + " level: " + str(level) + " degree: " + str(degree) + " height: " + str(height) + " (" + author + ")"
    if (delta):
        node_info = node_info + '  >>> Delta!'
    print node_info

#------------------------------------------------------------------------------------
# Print discussion 
# 
def traverse_and_print(df_dt_nodes, id, level):
    level = level + 1
    where_child    = df_dt_nodes['parent_id'] == id 
    where_not_root = df_dt_nodes['id'] != id 
    children = df_dt_nodes[ where_child & where_not_root ]['id'].tolist()
    degree   = len(children)
    if degree > 0:
        for child in children: 
            print_node_all(df_dt_nodes, child, level)
            traverse_and_print(df_dt_nodes, child, level)

#------------------------------------------------------------------------------------
# Debugging: print discussion tree
# 
def print_tree(df, id):
    print_node_all(df, id, 0)
    traverse_and_print(df , id, -1)

#------------------------------------------------------------------------------------
# Testing... 
# 
if __name__ == '__main__':
    infile = 'disc_tree_nodes_train.csv'
    df = pd.read_csv(infile)
    #print '\nTree with no delta ---------------------------'
    #print_tree(df, '16rve8')
    print '\nTree with delta ---------------------------'
    print_tree(df, '2rosbp')
    print_tree(df, '2ro9ux')
