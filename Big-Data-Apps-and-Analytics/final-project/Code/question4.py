#####################################################################################
# Author:  Carlos Sathler
# Email:   cssathler@gmail.com
# Class:   Big Data Application and Analytics
# Date:    11/14/2016 
# Program: question4.py 
# Purpose: This program addresses the question: Does it matter when a post was sub-
#          mitted?  Are persuasive posts more likely to occur at a certain time of
#          day or certain day of the week? 
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
def set_hour_tickers(pname, ylabel, plot_tag):
    plot_tag = plot_tag + " " + "Hour of Day"
    l = pname.set_xlabel(plot_tag)
    l.set_fontsize('smaller')
    l = pname.set_ylabel(ylabel)
    l.set_fontsize('smaller')
    xlabels = pname.get_xticklabels()
    for label in xlabels: label.set_fontsize(8)
    ylabels = pname.get_yticklabels()
    for label in ylabels: label.set_fontsize(8)

#------------------------------------------------------------------------------------
# Set plot tickers and label sizes 
#
def set_week_day_tickers(pname, ylabel, plot_tag):
    plot_tag = plot_tag + " " + "Day of Week"
    l = pname.set_xlabel(plot_tag)
    l.set_fontsize('smaller')
    l = pname.set_ylabel(ylabel)
    l.set_fontsize('smaller')
    pname.set_xticklabels(['         Sun', '         Mon', '         Tue', \
                           '         Wed', '         Thu', '         Fri', \
                           '         Sat'])
    xlabels = pname.get_xticklabels()
    for label in xlabels: label.set_fontsize(8)
    ylabels = pname.get_yticklabels()
    for label in ylabels: label.set_fontsize(8)

#------------------------------------------------------------------------------------
# Read data needed for the plots 
#
def analysis_q4():

    df_nodes_fname = "disc_tree_nodes_train_filtered.csv"
    df_nodes = pd.read_csv(df_nodes_fname)

    # remove posts by DeltaBot (post 'author' is index)
    df_nodes = df_nodes[ df_nodes['author'] != 'DeltaBot' ] 

    #df_nodes['WeekDay'] = df_nodes['posttimeraw'].apply(lambda x: \
    #                        time.strftime('%a',time.gmtime(float(x))))

    df_nodes['WeekDayNo'] = df_nodes['posttimeraw'].apply(lambda x: \
                            int(time.strftime('%w',time.gmtime(float(x)))))

    df_nodes['H24'] = df_nodes['posttimeraw'].apply(lambda x: \
                            int(time.strftime('%-H',time.gmtime(float(x)))))

    #df_nodes['Min'] = df_nodes['posttimeraw'].apply(lambda x: \
    #                        int(time.strftime('%-M',time.gmtime(float(x)))))

    #df_nodes['Sec'] = df_nodes['posttimeraw'].apply(lambda x: \
    #                        int(time.strftime('%-S',time.gmtime(float(x)))))

    fig = plt.figure()
    #fig.suptitle("Delta Posts as a Function of Time", fontsize=14)

    #------
    # plot1 
    # percentage of delta posts per day 

    df_posts    = df_nodes.groupby('WeekDayNo').count() 
    all_posts   = np.array(df_posts.id.tolist())

    # number of delta posts per day
    df_delta_posts = df_nodes[ df_nodes.deltanode_tf == True  ].groupby('WeekDayNo').count() 
    delta_posts = np.array(df_delta_posts.id.tolist())

    # percentage of delta posts per day
    delta_percent = list( delta_posts / all_posts.astype(float) * 100)

    x_cats = df_posts.index.tolist()

    plot1 = fig.add_subplot(221)
    plot1.set_title("Delta Posts % per Week Day", fontsize=10) 
    avg = np.mean(delta_percent)
    std = np.std(delta_percent)
    plt.axhline(avg, linestyle='dashed', linewidth=1, color='r')
    plot1.text(4.6, avg+0.01, "Average: " + str(round(avg,2)) + \
                              "\nStd: " + str(round(std,2)), color='r', fontdict=dict(size=8))
    set_week_day_tickers(plot1, "% of Delta Posts", "(a)")
    plot1 = plt.bar(x_cats, delta_percent, facecolor='black')

    #------
    # plot2 
    # normalize percentage of delta posts per day 

    # normalize the data
    delta_percent_norm = (delta_percent - np.mean(delta_percent)) / np.std(delta_percent)

    plot2 = fig.add_subplot(223)
    plot2.set_title("Normalized Delta Posts % per Week Day", fontsize=10) 
    plt.ylim([-1.5,1.5])
    set_week_day_tickers(plot2, "Normalize % of Delta Posts", "(c)")
    plot2 = plt.bar(x_cats, delta_percent_norm, facecolor='black')

    #------
    # plot3 
    # percentage of delta posts per hour of day 

    df_posts    = df_nodes.groupby('H24').count() 
    all_posts   = np.array(df_posts.id.tolist())

    # number of delta posts per hours of day  
    df_delta_posts = df_nodes[ df_nodes.deltanode_tf == True  ].groupby('H24').count() 
    delta_posts = np.array(df_delta_posts.id.tolist())

    # percentage of delta posts per day
    delta_percent = list( delta_posts / all_posts.astype(float) * 100)

    x_cats   = df_posts.index.tolist()

    plot3 = fig.add_subplot(222)

    plot3.set_title("Delta Posts % per Hour", fontsize=10) 
    avg = np.mean(delta_percent)
    std = np.std(delta_percent)
    #plt.axhline(avg, linestyle='dashed', linewidth=1, color='r')
    #plot3.text(7.5, avg+0.01, "Average: " + str(round(avg,2)) + \
    #                          "\nStd: " + str(round(std,2)), color='b', fontdict=dict(size=8))
    plot3.text(7.5, avg+0.01, "Std: " + str(round(std,2)), color='r', fontdict=dict(size=8))
    plt.xlim([0,24])
    set_hour_tickers(plot3, "% of Delta Posts", "(b)")
    plot3 = plt.bar(x_cats, delta_percent, facecolor='black')

    #------
    # plot4 
    # normalize percentage of delta posts per hour of day

    # normalize the data
    delta_percent_norm = (delta_percent - np.mean(delta_percent)) / np.std(delta_percent)

    plot4 = fig.add_subplot(224)
    plot4.set_title("Normalized Delta Posts % per Hour", fontsize=10) 
    plt.xlim([0,24])
    set_hour_tickers(plot4, "Normalize % of Delta Posts", "(d)")
    plot4 = plt.bar(x_cats, delta_percent_norm, facecolor='black')

    fig.subplots_adjust(left=None, bottom=None, right=None,
                        top=None, wspace=0.35, hspace=0.35)

    plt.savefig('question4.png')


#####################################################################################
#  MAIN 
#
#
if __name__ == '__main__':

    analysis_q4()

