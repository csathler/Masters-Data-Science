#####################################################################################
# Author:  Carlos Sathler
# Email:   cssathler@gmail.com
# Class:   Big Data Application and Analytics
# Date:    11/15/2016 
# Program: predict.py 
# Purpose: Program does predictive analytics on the project data 
#####################################################################################

import sys, os
import string
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
# Display timestamp and message
#
def timestamp(msg):
    x = string.split(time.ctime())
    now = x[1] + " " + x[2] + " " + x[4] + " " + x[3]
    print now + "\t> " + msg

#------------------------------------------------------------------------------------
#  Print prediction model metrics based on confusion matrix 
#
def print_metrics(cm, msg): 

    TP = float(cm[1][1])
    FP = float(cm[0][1])
    TN = float(cm[0][0])
    FN = float(cm[1][0])
    
    print " "
    print msg 
    print " "
    print "TP - True  Positives: " + str(int(TP))
    print "FP - False Positives: " + str(int(FP))
    print "TN - True  Negatives: " + str(int(TN))
    print "FN - False Negatives: " + str(int(FN))

    print " "
    print "Accuracy (TP+TN)/(TP+FP+FN+TN)......: " + \
           str(round((TP+TN)/(TP+FP+FN+TN)*100,4)) + "%"

    if (TP+FP) > 0:
        print "Positive Predictive Value TP/(TP+FP): " + \
               str(round(TP/(TP+FP)*100,4)) + "%" 
    else:
        print "Positive Predictive Value TP/(TP+FP): 0.0000%" 

    
    if (TP+FN) > 0:
        print "Sensitivity TP/(TP+FN)..............: " + \
               str(round(TP/(TP+FN)*100,4)) + "%"
    else:
        print "Sensitivity TP/(TP+FN)..............: 0.0000%"


#------------------------------------------------------------------------------------
#  Separate features and target 
#
def get_X_and_y(df):
    
    # target is in the first column in the dataframe
    df_X = df.iloc[:, 1:]  
    df_y = df.iloc[:, 0:1]  
   
    return (df_X, df_y) 

#------------------------------------------------------------------------------------
#  Get count of positive words in comments vs. negative words 
#
def get_word_count(comm, good_word_list, bad_word_list):

    good_word_count = 0 
    bad_word_count  = 0 
    comm_type = str(type(comm))

    if 'str' in comm_type:

        comm = comm.upper()

        for good_word in good_word_list:
            if ( string.find( comm, good_word ) != -1 ):
                 good_word_count += 1

        for bad_word in bad_word_list:
            if ( string.find( comm, bad_word ) != -1 ):
                 bad_word_count += 1

    return (good_word_count - bad_word_count)

#------------------------------------------------------------------------------------
#  Return number of hops from a given node (id) to the closest OP post 
#
def get_hops_to_op(df, id):        

     try:
         parent_id = df[id:id].parent_id.values[0]
         # x[0] will be author; x[1] will be op_author
         x = list(df.loc[parent_id:parent_id][['author','op_author']].values[0])
         if (x[0] == x[1]):
             return 1
         else:
             return 1 + get_hops_to_op(df, parent_id)
     except KeyError:
         return 99999999999.9

#------------------------------------------------------------------------------------
#  Prepross
#
def preprocess(df_train, df_teste):
    #------------------------------------------------------------------------------------
    # BEGIN CITATION - Book Python Machine Learning 
    # Author: Sebastian Raschka 
    #
    # create dummy features for week day number
    # Used some command ideas, loosely, from pp 104, 105, 108 
    wkdayno_map = {0:'Sun',1:'Mon',2:'Tue',3:'Wed',4:'Thu',5:'Fri',6:'Sat'}
    df_train.WeekDayNo = df_train.WeekDayNo.map(wkdayno_map) 
    df_teste.WeekDayNo = df_teste.WeekDayNo.map(wkdayno_map) 
    df_train           = pd.get_dummies(df_train[df_train.columns])  
    df_teste           = pd.get_dummies(df_teste[df_teste.columns])  
    #
    # will use columns later to restore column names for dataframes
    idx_columns = df_teste.columns
    #
    # impute values if NaN present
    # df_teste.isnull().sum()
    # code from book, page 102, with a few modifications...
    from sklearn.preprocessing import Imputer
    imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imr = imr.fit(df_teste)   # only test partition had NaN values for workcount feature
    df_teste = pd.DataFrame(imr.transform(df_teste))
    df_teste.columns = idx_columns
    #
    # standardize degree, level and WordCount
    # ideas from book, page 111; here we convert to array and back to df
    from sklearn.preprocessing import StandardScaler
    stdsc = StandardScaler()
    df_train = pd.DataFrame(stdsc.fit_transform(df_train.values))
    df_teste = pd.DataFrame(stdsc.transform(df_teste.values))
    df_train.columns = idx_columns
    df_teste.columns = idx_columns
    #
    # END CITATION - Book Python Machine Learning 
    #------------------------------------------------------------------------------------

    # drop H24 column (3)
    df_train = df_train.iloc[:, [0,1,3,4,5,6,7,8,9,10,11]]
    df_teste = df_teste.iloc[:, [0,1,3,4,5,6,7,8,9,10,11]]

    return (df_train, df_teste)


#------------------------------------------------------------------------------------
#  Read data and select/prepare features
#  Features:
#       day of week (number)
#       time of day (H24)
#       level
#	degree 
#       number of words in post
#       hops to nearest OP post
#
def get_clean_data(partition):

    # look for file from previous run
    fname = "predict_" + partition + ".csv" 
    if os.path.exists(fname):
        # return found file        
        df_clean = pd.read_csv(fname)         
        return df_clean
 
    timestamp('Reading nodes file...') 
    df_nodes_fname = "disc_tree_nodes_" + partition + "_filtered.csv"
    df_nodes = pd.read_csv(df_nodes_fname, \
               dtype={'degree':np.float64, 'level':np.float64})

    timestamp('Reading comments file...') 
    df_comms_fname = "raw_comments_" + partition + ".csv"
    df_comms = pd.read_csv(df_comms_fname)

    timestamp('Creating column for day of the week number')
    df_nodes['WeekDayNo'] = df_nodes['posttimeraw'].apply(lambda x: \
                            int(time.strftime('%w',time.gmtime(float(x)))))

    timestamp('Creating column for hour of the day')
    df_nodes['H24'] = df_nodes['posttimeraw'].apply(lambda x: \
                            int(time.strftime('%-H',time.gmtime(float(x)))))

    timestamp('Converting delta from boolean to float') 
    df_nodes['delta'] = df_nodes['deltanode_tf'].apply(lambda x: float(x)) 

    timestamp('Getting word count of comments for each post') 
    df_nodes['WordCount'] = pd.merge(df_nodes, df_comms, on='id').comm.\
                            apply(lambda x: len(str(x).split()))

    df_nodes_copy = df_nodes.copy()
    df_nodes_copy = df_nodes_copy.set_index('id')
    timestamp('Calculating post closeness to closest op post - hops_to_op')
    df_nodes['hops_to_op'] = df_nodes.id.apply(lambda x:\
                             get_hops_to_op(df_nodes_copy, x))

    timestamp('Removing users who could never receive deltas: OP, DeltaBot')
    df_nodes = df_nodes[ df_nodes['author'] != 'DeltaBot' ]
    df_nodes = df_nodes[ df_nodes['author'] != df_nodes['op_author'] ]

    timestamp('Creating df with desired features only')
    df_clean = df_nodes[['delta','degree','level','WeekDayNo',\
                         'H24','WordCount','hops_to_op']].copy()

    # save file, in case we need to run prediction again...
    timestamp('Saving prediction data file')
    df_clean.to_csv(fname, index=False, encoding='utf-8')
    return df_clean


#####################################################################################
#  MAIN 
#
if __name__ == '__main__':

    #del sys.modules['predict']                                              

    timestamp('Getting training data...')
    df_train   = get_clean_data('train') 
    timestamp('Getting test data...')
    df_teste   = get_clean_data('test') 
   
    timestamp('Separate features and target in both train and test partitions')
    (df_X_train, df_y_train_f) = get_X_and_y(df_train) 
    (df_X_teste, df_y_teste_f) = get_X_and_y(df_teste) 

    timestamp('Preprosses train and test partitions')
    (df_X_train_f, df_X_teste_f) = preprocess(df_X_train, df_X_teste)
 
    timestamp('Create np arrays from dataframes')
    X_train = df_X_train_f.values
    X_teste = df_X_teste_f.values
    y_train = np.ravel(df_y_train_f.values)
    y_teste = np.ravel(df_y_teste_f.values)

    from sklearn.linear_model import LogisticRegression         
    from sklearn.neighbors    import KNeighborsClassifier
    from sklearn.tree         import DecisionTreeClassifier
    from sklearn.ensemble     import RandomForestClassifier
    #from sklearn.ensemble     import AdaBoostClassifier 
    from sklearn.metrics      import confusion_matrix
    #from sklearn.metrics     import accuracy_score
    #acc = accuracy_score(y_teste, y_pred)

    clf_lrg = LogisticRegression(random_state=0)
    clf_knb = KNeighborsClassifier()
    clf_dtr = DecisionTreeClassifier(random_state=0)
    #clf_abc = AdaBoostClassifier(n_estimators=100, random_state=0)
    clf_rfo = RandomForestClassifier(n_estimators=100, random_state=0)

    timestamp('Fitting - Logistic Regression')
    clf_lrg = clf_lrg.fit(X_train, y_train) 
    timestamp('Fitting - KNN')
    clf_knb = clf_knb.fit(X_train, y_train) 
    timestamp('Fitting - Decision Tree Classifier')
    clf_dtr = clf_dtr.fit(X_train, y_train) 
    #timestamp('Fitting - AdaBoost Classifier')
    #clf_abc = clf_abc.fit(X_train, y_train) 
    timestamp('Fitting - Random Forest')
    clf_rfo = clf_rfo.fit(X_train, y_train) 

    timestamp('Predicting for all classifiers...')
    y_pred_lrg = clf_lrg.predict(X_teste)
    y_pred_knb = clf_knb.predict(X_teste)
    y_pred_dtr = clf_dtr.predict(X_teste)
    #y_pred_abc = clf_abc.predict(X_teste)
    y_pred_rfo = clf_rfo.predict(X_teste)

    timestamp('Calculating confusion matrix for all classifiers...')
    cm_lrg = confusion_matrix(y_teste, y_pred_lrg)
    cm_knb = confusion_matrix(y_teste, y_pred_knb)
    cm_dtr = confusion_matrix(y_teste, y_pred_dtr)
    #cm_abc = confusion_matrix(y_teste, y_pred_abc)
    cm_rfo = confusion_matrix(y_teste, y_pred_rfo)

    print_metrics(cm_lrg, "Metrics - Logistic Regression")
    print_metrics(cm_knb, "Metrics - KNN Classifier")
    print_metrics(cm_dtr, "Metrics - Decision Tree Classifier")
    #print_metrics(cm_abc, "Metrics - AdaBoost Classifier")
    print_metrics(cm_rfo, "Metrics - Random Forest")

    print "\n"
