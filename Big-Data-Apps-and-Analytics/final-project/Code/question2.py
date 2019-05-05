#####################################################################################
# Author:  Carlos Sathler
# Email:   cssathler@gmail.com
# Class:   Big Data Application and Analytics
# Date:    11/14/2016 
# Program: question2.py 
# Purpose: This program addresses the question: Are there highly persuasive users?
#          Are certain users more skilled at getting deltas, i.e., should we focus 
#          on studying certain users who are "good" at persuasion and compare them 
#          with others who are "bad" at persuasion?
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
import seaborn

#------------------------------------------------------------------------------------
# Set plot tickers and label sizes 
#
def set_tickers(pname, plot_tag):
    plot_tag = plot_tag + " " + "Percent Deltas/Posts"
    l = pname.set_xlabel(plot_tag)
    l.set_fontsize('small')
    l = pname.set_ylabel('Frequency')
    l.set_fontsize('small')
    xlabels = pname.get_xticklabels()
    for label in xlabels: label.set_fontsize(8)
    ylabels = pname.get_yticklabels()
    for label in ylabels: label.set_fontsize(8)

#------------------------------------------------------------------------------------
# Get certain df stats, configure text to display them, and return text 
#
def get_stats_string(df):
    total_users  = df.count()[0]
    (avg_posts, avg_deltas, avg_pct) = df.describe()['mean':'mean'].values[0]
    txt = "Total users: "      + str(total_users)        + "\n" + \
          "Avg. posts/user: "  + str(round(avg_posts,2)) + "\n" + \
          "Avg. deltas/user: " + str(round(avg_deltas,2))
    return txt

#------------------------------------------------------------------------------------
# Read data needed for the plots 
#
def analysis_q2():

    df_nodes_fname = "disc_tree_nodes_train_filtered.csv"
    df_nodes = pd.read_csv(df_nodes_fname)

    # remove "unqualified users"

    # remove user '[deleted]'
    df_nodes = df_nodes[ df_nodes['author'] != '[deleted]' ] 

    # remove posts by DeltaBot (post 'author' is index)
    df_nodes = df_nodes[ df_nodes['author'] != 'DeltaBot' ] 

    # get the count of posts per user
    post_per_author = df_nodes.groupby('author').agg({'id':np.size})

    # get the count of posts per user
    delta_per_author = df_nodes[ df_nodes['deltanode_tf'] == True ] \
                       .groupby('author').agg({'id':np.size})

    # merge 
    df_q2 = post_per_author.merge(delta_per_author, how='left', on=None, 
            left_on=None, right_on=None, left_index=True, right_index=True, 
            sort=False, suffixes=('_x', '_y'), copy=True, indicator=False)

    # give columns meaningful names
    df_q2.columns = [['total_posts','deltas']]   

    # nan for delta count implies deltas = 0
    df_q2 = df_q2.fillna(0)

    # calculate percentage of deltas received by user
    df_q2['delta_percent'] = df_q2['deltas'] / df_q2['total_posts'] * 100

    # remove users who posted less than 4 times total
    df_q2 = df_q2[ df_q2['total_posts'] >= 5 ]

    #----------------
    # create figure 1

    x = df_q2['delta_percent'].tolist() 

    fig = plt.figure()
    #fig.suptitle("Authors' Percentage of Delta Awards over His/Her Total Posts", fontsize=14)

    #------
    # plot1
    plot1 = fig.add_subplot(221)        

    txt = get_stats_string(df_q2)
    plot1.text(18.5, 850, txt, color='black', fontdict=dict(size=8))

    plot1.hist(x, bins=30)
    plot1.set_title('Entire % range, bins=30', fontsize=10)
    set_tickers(plot1, "(a)")

    #------
    # plot2
    plot2 = fig.add_subplot(222)        
    plot2.set_title('Range between 0 and 1%, bins=5', fontsize=10)
    avg_delta = df_nodes[ df_nodes.deltanode_tf==True ].count()[0] / \
                float(df_nodes.count()[0]) * 100
    avg_delta = round(avg_delta,3) 
    plot2.axvline(x=avg_delta, linestyle='dashed', linewidth=1, color='r')
    plot2.text(avg_delta+0.02,12500,'Average awards\n' + str(avg_delta) + "% of posts",\
               color='r', fontdict=dict(size=8))
    plot2.hist(x, bins=5, range=(0,1))  
    set_tickers(plot2, "(b)")

    #------
    # plot3
    plot3 = fig.add_subplot(223)        
    plt.xlim([1,10])
    plot3.set_title('Range between 1 and 10%, bins=9', fontsize=10)

    y = df_q2[ (df_q2.delta_percent >= 1) & (df_q2.delta_percent <= 10) ] 
    txt = get_stats_string(y)
    plot3.text(5.7, 400, txt, color='black', fontdict=dict(size=8))

    plot3.hist(x, bins=9, range=(1,10)) 
    set_tickers(plot3, "(c)")

    #------
    # plot4
    plot4 = fig.add_subplot(224)        
    plt.xlim([10,30])
    plot4.set_title('Range between 10 and 30%, bins=8', fontsize=10)

    # displays some stats specific to this group
    y = df_q2[ df_q2.delta_percent >= 10 ] 
    txt = get_stats_string(y)
    plot4.text(21, 95, txt, color='black', fontdict=dict(size=8))

    plot4.hist(x, bins=8, range=(10,30))
    set_tickers(plot4, "(d)")

    fig.subplots_adjust(left=None, bottom=None, right=None, 
                        top=None, wspace=0.35, hspace=0.35) 

    plt.savefig('question2_fig1.png')

    #----------------
    # create figure 2
   
    # find correlation between number of deltas and number of posts 
    x = df_q2.total_posts.tolist()
    y = df_q2.deltas.tolist() 
    ss = stats.spearmanr(y, x)
    ss_corr   = round(ss[0],4)
    ss_pvalue = round(ss[1],4)
    lmplot_txt1 = "Spearman Correlation: " + str(ss_corr)
    lmplot_txt2 = "p-value: " + str(ss_pvalue)

    p = seaborn.lmplot(y='deltas', x='total_posts', data=df_q2)
    p.fig.text(0.20, 0.80, lmplot_txt1, color='black', fontsize=10)
    p.fig.text(0.38, 0.76, lmplot_txt2, color='black', fontsize=10)
    #p.fig.text(0.48, 0.68, lmplot_txt1, color='black', fontsize=10)
    #p.fig.text(0.66, 0.64, lmplot_txt2, color='black', fontsize=10)

    plt.xlabel("Number of Posts")
    plt.ylabel("Deltas Received")
    plt.title("Deltas Received x Number of Posts by Users")
    #plt.text(25, 1550, "oi", color='r', fontdict=dict(size=12))
    plt.xlim([0,3500])
    plt.ylim([0,25])

    plt.savefig('question2_fig2.png')


#####################################################################################
#  MAIN 
#
#
if __name__ == '__main__':

    analysis_q2()

