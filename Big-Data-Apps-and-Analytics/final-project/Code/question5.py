####################################################################################
# Author:  Carlos Sathler
# Email:   cssathler@gmail.com
# Class:   Big Data Application and Analytics
# Date:    11/14/2016 
# Program: load_data.py
# Purpose: This program addresses the question: Is there a relationship between the
#          shape of the discussion tree and persuasion? 
#####################################################################################

import sys
import numpy as np
import pandas as pd
import matplotlib as mlp
import time
from scipy import stats

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

import matplotlib.pyplot as plt

#------------------------------------------------------------------------------------
# Set plot tickers and label sizes 
#
def set_tickers(pname, xlabel, plot_tag):
    plot_tag = plot_tag + " " + xlabel 
    l = pname.set_xlabel(plot_tag)
    l.set_fontsize('smaller')
    l = pname.set_ylabel('Percentage of Deltas Received')
    l.set_fontsize('smaller')
    xlabels = pname.get_xticklabels()
    for label in xlabels: label.set_fontsize(8)
    ylabels = pname.get_yticklabels()
    for label in ylabels: label.set_fontsize(8)

#------------------------------------------------------------------------------------
# Read data needed for the plots 
#
def analysis_q5():

    df_nodes_fname = "disc_tree_nodes_train_filtered.csv"
    df_nodes = pd.read_csv(df_nodes_fname)

    # remove posts by DeltaBot (post 'author' is index)
    df_nodes = df_nodes[ df_nodes['author'] != 'DeltaBot' ] 

    fig = plt.figure()
    #fig.suptitle("Delta Posts Percentage as a Function of Node Level and Degree", fontsize=14)

    #------
    # plot1 
    # percentage of delta posts per node level 

    df_posts   = df_nodes[ df_nodes.level != 0 ].groupby('level').count() 
    df_posts   = pd.DataFrame(df_posts['id']) 
    df_posts.columns = [['posts']]

    # number of delta posts per day
    df_delta_posts = df_nodes[ df_nodes.deltanode_tf == True  ].groupby('level').count() 
    df_delta_posts = pd.DataFrame(df_delta_posts['id'])
    df_delta_posts.columns = [['deltas']]

    # outer join to capture levels that have 0 deltas
    df_level_counts = pd.merge(df_posts,df_delta_posts,how='left',left_index=True,right_index=True)
    df_level_counts = df_level_counts.fillna(0) 
    
    all_posts   = np.array(df_level_counts.posts.tolist())
    delta_posts = np.array(df_level_counts.deltas.tolist())

    # percentage of delta posts per day
    delta_percent = list( delta_posts / all_posts.astype(float) * 100)

    x_cats   = df_posts.index.tolist()

    plot1 = fig.add_subplot(121)
    plot1.set_title("Delta Posts % per Node Level", fontsize=10) 
    plt.xlim([1,10])
    set_tickers(plot1, "Node Level", "(a)")
    plot1 = plt.bar(x_cats, delta_percent, facecolor = 'black')

    #------
    # plot2 
    # percentage of delta posts per degree 

    df_posts   = df_nodes[ df_nodes.level != 0 ].groupby('degree').count() 
    df_posts   = pd.DataFrame(df_posts['id']) 
    df_posts.columns = [['posts']]

    # number of delta posts per day
    df_delta_posts = df_nodes[ df_nodes.deltanode_tf == True  ].groupby('degree').count() 
    df_delta_posts = pd.DataFrame(df_delta_posts['id'])
    df_delta_posts.columns = [['deltas']]

    # outer join to capture levels that have 0 deltas
    df_degree_counts = pd.merge(df_posts,df_delta_posts,how='left',left_index=True,right_index=True)
    df_degree_counts = df_degree_counts.fillna(0) 

    # remove degrees values that have less than 10 posts
    df_degree_counts = df_degree_counts[ df_degree_counts.posts > 9 ]
    
    all_posts   = np.array(df_degree_counts.posts.tolist())
    delta_posts = np.array(df_degree_counts.deltas.tolist())

    # percentage of delta posts per day
    delta_percent = list( delta_posts / all_posts.astype(float) * 100)

    x_cats   = df_degree_counts.index.tolist()

    plot2 = fig.add_subplot(122)
    plot2.set_title("Delta Posts % per Node Degree", fontsize=10) 
    plt.xlim([0,24])
    set_tickers(plot2, "Degree of Node", "(b)")
    plot2 = plt.bar(x_cats, delta_percent, facecolor='black')

    plt.savefig('question5.png')


#####################################################################################
# Will address QUESION 5 of analysis:
#
# We choose node properties of level and degree as representation of interaction 
# dynamics and try to establish a connection between these properties and deltas
# received by users
#
if __name__ == '__main__':

   analysis_q5()
