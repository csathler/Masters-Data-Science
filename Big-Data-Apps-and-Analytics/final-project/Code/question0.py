#####################################################################################
# Author:  Carlos Sathler
# Email:   cssathler@gmail.com
# Class:   Big Data Application and Analytics
# Date:    11/14/2016 
# Program: question0.py 
# Purpose: This program prints basic statitics about the project data 
#####################################################################################

import sys
import pandas as pd
import numpy as np
import string

#------------------------------------------------------------------------------------
# prints dataset statistics 
# 
def print_dataset_stats(type):

    # populate dataframes from csv file
    df_nodes_fname = "disc_tree_nodes_" + type + ".csv"
    df_ops_fname = "ops_" + type + ".csv"
    df_nodes = pd.read_csv(df_nodes_fname)
    df_ops   = pd.read_csv(df_ops_fname)

    # Number of discussion trees:
    tree_count = str(df_ops['op_author'].size)
    node_count = str(df_nodes['id'].size)
    op_count   = str(df_ops['op_author'].drop_duplicates().size)
    user_count = str(df_nodes['author'].drop_duplicates().size)

    if type == 'train':
        print "Training Set Statistics:"
    else: 
        print "Test Set Statistics:"

    print "\tDiscussion trees:   " + tree_count
    print "\tNumber of nodes : " + node_count
    print "\tOP count        :   " + op_count 
    print "\tUnique users    :   " + user_count 


#####################################################################################
#   MAIN
#
#
if __name__ == '__main__':

    print "\n\n"
    print_dataset_stats('train')
    print "\n\n"
    print_dataset_stats('test')
    print "\n\n"
