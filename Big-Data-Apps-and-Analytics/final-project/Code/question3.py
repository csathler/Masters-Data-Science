#####################################################################################
# Author:  Carlos Sathler
# Email:   cssathler@gmail.com
# Class:   Big Data Application and Analytics
# Date:    11/14/2016 
# Program: question3.py 
# Purpose: This program addresses the question: Are there highly malleable users?
#          Do we need to exclude certain users from the analysis because they are
#          too easy to persuade?
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
    plot_tag = plot_tag + " " + "% Times Persuaded"
    l = pname.set_xlabel(plot_tag)
    l.set_fontsize('smaller')
    l = pname.set_ylabel('Frequency')
    l.set_fontsize('smaller')
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
    txt = "Total OP's: "      + str(total_users)        + "\n" + \
          "Avg. posts/OP: "  + str(round(avg_posts,2)) + "\n" + \
          "Avg. deltas/OP: " + str(round(avg_deltas,2))
    return txt

#------------------------------------------------------------------------------------
#needed Rea_ data  for the plots 
#
def analysis_q3():

    df_ops_fname = "ops_train_filtered.csv"
    df_ops = pd.read_csv(df_ops_fname)

    df_nodes_fname = "disc_tree_nodes_train_filtered.csv"
    df_nodes = pd.read_csv(df_nodes_fname)

    # remove "unqualified users"

    # remove user '[deleted]'
    df_nodes = df_nodes[ df_nodes['author'] != '[deleted]' ] 

    # remove posts by DeltaBot (post 'author' is index)
    df_nodes = df_nodes[ df_nodes['author'] != 'DeltaBot' ] 

    # get the count of posts per op author 
    post_per_author = df_ops.groupby('op_author').agg({'id':np.size})

    # get the count of delta awards per op author 
    delta_per_author = df_ops[ df_ops['deltatree_tf'] == True ] \
                       .groupby('op_author').agg({'id':np.size})

    # merge 
    df_q3 = post_per_author.merge(delta_per_author, how='left', on=None, 
            left_on=None, right_on=None, left_index=True, right_index=True, 
            sort=False, suffixes=('_x', '_y'), copy=True, indicator=False)

    # give columns meaningful names
    df_q3.columns = [['total_ops','deltas_awarded']]   

    # nan for delta count implies deltas = 0
    df_q3 = df_q3.fillna(0)

    # calculate percentage of deltas awarded by user
    df_q3['delta_percent'] = df_q3['deltas_awarded'] / df_q3['total_ops'] * 100

    # remove users who posted less than 4 times total
    df_q3 = df_q3[ df_q3['total_ops'] >= 5 ]

    #----------------
    # create figure 1

    x = df_q3['delta_percent'].tolist() 

    fig = plt.figure()
    #fig.suptitle('Percentage of Times Author Persuaded on Discussions He/She Started', fontsize=14)

    #------
    # plot1
    plot1 = fig.add_subplot(121)        

    avg_delta = df_ops[ df_ops.deltatree_tf==True ].count()[0] / \
                float(df_ops.count()[0]) * 100
    avg_delta = round(avg_delta,3)
    plot1.axvline(x=avg_delta, linestyle='dashed', linewidth=1, color='r')
    plot1.text(avg_delta+2, 35,'Average deltas\n' + str(avg_delta) + "%",\
               color='r', fontdict=dict(size=8))

    txt = get_stats_string(df_q3)
    plot1.text(56, 44.3, txt, color='black', fontdict=dict(size=8))

    plot1.set_title('Entire % range, bins=20', fontsize=10)
    set_tickers(plot1, "(a)")
    plot1.hist(x, bins=20, range=(0,100))

    #------
    # plot2
    plot2 = fig.add_subplot(122)        

    txt = get_stats_string(df_q3[ df_q3.delta_percent >= 30 ])
    plot2.text(69, 26.5, txt, color='black', fontdict=dict(size=8))

    plot2.set_title('Range: 30-100%, bins=14', fontsize=10)
    plt.xlim([30,100])
    set_tickers(plot2, "(b)")
    plot2.hist(x, bins=14, range=(30,100))  

    fig.subplots_adjust(left=None, bottom=None, right=None,
                        top=None, wspace=0.3, hspace=0.3) 

    plt.savefig('question3_fig1.png')

    #----------------
    # create figure 2
   
    # find correlation between number of deltas awarded and discussions started
    x = df_q3.total_ops.tolist()
    y = df_q3.deltas_awarded.tolist()
    ss = stats.spearmanr(y, x)
    ss_corr     = round(ss[0],4)
    ss_pvalue   = round(ss[1],4)
    lmplot_txt1 = "Spearman Correlation: " + str(ss_corr) 
    lmplot_txt2 = "p-value: " + str(ss_pvalue)

    p = seaborn.lmplot(y='deltas_awarded', x='total_ops', data=df_q3)
    p.fig.text(0.23, 0.82, lmplot_txt1, color='black', fontsize=10)
    p.fig.text(0.41, 0.78, lmplot_txt2, color='black', fontsize=10)

    plt.xlabel("Number of Discussions Started")
    plt.ylabel("Deltas Awarded")
    plt.title("Deltas Awarded x Number of Discussions Started")
    plt.xlim([0,50])
    plt.ylim([0,16])
    plt.savefig('question3_fig2.png')


#####################################################################################
#  MAIN 
#
#
if __name__ == '__main__':

    analysis_q3()

