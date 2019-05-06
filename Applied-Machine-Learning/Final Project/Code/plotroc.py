import sys
import numpy as np
import pandas as pd
import matplotlib as mlp
if sys.platform == 'darwin':
     # in mac os only
     mlp.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import tree
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 

#------------------------------------------------------------------------------------------------------ 
# loads dataset
# in:  none 
# out: dataset as dataframe 
# 
def load_data(fname):
    # define column names for data import
    df_col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 
                    'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 
                    'hours_per_week', 'native_country', 'income']
    df_col_dtype = {'age': np.float64, 'fnlwgt': np.float64, 'education_num': np.float64, 
                    'capital_gain': np.float64, 'capital_loss': np.float64, 'hours_per_week': np.float64} 
    # read data
    df = pd.read_csv(fname, index_col=None, header=None, na_values=' ?', 
                     names=df_col_names, dtype=df_col_dtype)

    return df 

#------------------------------------------------------------------------------------------------------ 
# gets dictionary and swaps keys with values
# in : my_dict_in
# out: my_dict_out
# 
def invert_dict(my_dict_in):
    my_dict_out = {}
    for x, y in my_dict_in.iteritems(): my_dict_out[y] = float(x)
    return my_dict_out

#------------------------------------------------------------------------------------------------------ 
# Common pre processing I 
# Common preprocessing to prepare 3 of 5 dataframe versions 
#
# in:  dataset as dataframe 
# out: dataset as dataframe with the following characteristics: 
#      1 - All columns in dataset set to float, including ordinal and categorical features 
#      2 - remove rows with nan's 
#      3 - remove redundant features
# 
def common_preprocess_I(df):

    # found 2399 rows with NaN in the following columns: workclass, occupation, native_country 
    # these are categorical features, could use mode to impute values, but will dropna
    df.dropna(inplace=True)
    
    # drop education since we have education_num
    df.drop('education', axis=1, inplace=True)

    df_source = df.copy()

    col_map = {}

    # create dictionary objects for each column containing nominal or ordinal values
    workclass_dict      = pd.Series(df.workclass.unique()).to_dict()      
    marital_status_dict = pd.Series(df.marital_status.unique()).to_dict() 
    occupation_dict     = pd.Series(df.occupation.unique()).to_dict()     
    relationship_dict   = pd.Series(df.relationship.unique()).to_dict()   
    race_dict           = pd.Series(df.race.unique()).to_dict()           
    sex_dict            = pd.Series(df.sex.unique()).to_dict()           
    native_country_dict = pd.Series(df.native_country.unique()).to_dict()
    income_dict         = pd.Series(df.income.unique()).to_dict()         
    
    # create inverted dictionary objects to update dataframe columns
    workclass_dict_inv      = invert_dict( workclass_dict      )
    marital_status_dict_inv = invert_dict( marital_status_dict )
    occupation_dict_inv     = invert_dict( occupation_dict     ) 
    relationship_dict_inv   = invert_dict( relationship_dict   ) 
    race_dict_inv           = invert_dict( race_dict           ) 
    sex_dict_inv            = invert_dict( sex_dict            )
    native_country_dict_inv = invert_dict( native_country_dict ) 
    income_dict_inv         = invert_dict( income_dict         )

    col_map['workclass']      = [ workclass_dict, workclass_dict_inv ]
    col_map['marital_status'] = [ marital_status_dict, marital_status_dict_inv ] 
    col_map['occupation']     = [ occupation_dict, occupation_dict_inv ]    
    col_map['relationship']   = [ relationship_dict, relationship_dict_inv ] 
    col_map['race']           = [ race_dict, race_dict_inv ] 
    col_map['sex']            = [ sex_dict, sex_dict_inv ] 
    col_map['native_country'] = [ native_country_dict, native_country_dict_inv ] 
    col_map['income']         = [ income_dict, income_dict_inv ] 
    
    # convert each nominal column value to a number using inverted dictionaries 
    df.workclass.replace(      workclass_dict_inv      , inplace=True )
    df.marital_status.replace( marital_status_dict_inv , inplace=True )
    df.occupation.replace(     occupation_dict_inv     , inplace=True )
    df.relationship.replace(   relationship_dict_inv   , inplace=True )
    df.race.replace(           race_dict_inv           , inplace=True )
    df.sex.replace(            sex_dict_inv            , inplace=True )
    df.native_country.replace( native_country_dict_inv , inplace=True )
    df.income.replace(         income_dict_inv         , inplace=True )

    # switch flag to beginning of each row
    df = pd.concat([df.iloc[:,13:14], df.iloc[:,0:13]], axis=1)
  
    return df

#------------------------------------------------------------------------------------------------------ 
# Common pre processing II
# Common preprocessing to prepare 2 of 5 dataframe versions 
#
# in:  dataset as dataframe 
# out: dataset as dataframe with the following characteristics: 
#      1 - All columns in dataset set to float, except categorical features 
#      2 - remove rows with nan's 
#      3 - remove redundant features
#      4 - convert all categorical features to dummies
# 
def common_preprocess_II(df):

    # found 2399 rows with NaN in the following columns: workclass, occupation, native_country 
    # these are categorical features, could use mode to impute values, but will dropna
    df.dropna(inplace=True)
    
    # drop education since we have education_num
    df.drop('education', axis=1, inplace=True)

    # convert income flag 
    col_map = {}
    income_dict       = pd.Series(df.income.unique()).to_dict()
    income_dict_inv   = invert_dict( income_dict )
    col_map['income'] = [ income_dict, income_dict_inv ]
    df.income.replace(income_dict_inv, inplace=True )

    # switch flag to beginning of each row
    df = pd.concat([df.iloc[:,13:14],df.iloc[:,0:13]], axis=1)

    return df

#------------------------------------------------------------------------------------------------------ 
# Get percentile using global percentile list 
# Used by discretize function 
# in: value from dataframe continuous feature
# out: return percentile corresponding to value from dataframe column using global percentiles list 
#
def getpercentile(x):
    # return percentile for x 
    if (x <= percentiles[0]): return 1.0
    elif (x <= percentiles[1]): return 2.0
    elif (x <= percentiles[2]): return 3.0
    elif (x <= percentiles[3]): return 4.0
    elif (x <= percentiles[4]): return 5.0
    elif (x <= percentiles[5]): return 6.0
    elif (x <= percentiles[6]): return 7.0
    elif (x <= percentiles[7]): return 8.0
    elif (x <= percentiles[8]): return 9.0
    else: return 10.0

#------------------------------------------------------------------------------------------------------ 
# Discretize
# Replace continous feature with value from 1 to 10 value corresponding to its percentile 
# in: dataframe and name of the column to be discretize
#
def discretize(df, col_name):

    # create list with percentiles
    global percentiles
    percentiles = []
    x = []
    for i in range(1,11,1): x.append(float(i)/10)
    percentiles = df[col_name].quantile(x).tolist()

    # discretize column
    df[col_name] = df[col_name].apply(lambda x: getpercentile(x)) 

    return df

#------------------------------------------------------------------------------------------------------ 
# Pre-Processing #1 
# Create data set #1 
# in:  dataframe processed by general pre-processing routine 
# out: dataframe modified as follows: 
#      Continuous features  - unchanged 
#      Ordinal featuers     - unchanged
#      Categorical features - unchanged
# (note: the only ordinal feature is education_num)
#
def pre_processing_I(dfin):

    # perform general pre processing steps
    df = common_preprocess_I( dfin.copy() )

    # No changes to dataframe 
    return df

#------------------------------------------------------------------------------------------------------ 
# Pre-Processing #2 
# Create data set #2 
# in:  dataframe processed by general pre-processing routine 
# out: dataframe modified as follows: 
#      Continuous features  - discretized
#      Ordinal featuers     - unchanged
#      Categorical features - unchanged
# (note: the only ordinal feature is education_num)
#
def pre_processing_II(dfin):

    # perform general pre processing steps
    df = common_preprocess_I( dfin.copy() )

    # Discretize all continous columns 
    df = discretize( df, 'age'            )
    df = discretize( df, 'fnlwgt'         )
    df = discretize( df, 'capital_gain'   )
    df = discretize( df, 'capital_loss'   )
    df = discretize( df, 'hours_per_week' )

    return df

#------------------------------------------------------------------------------------------------------ 
# Pre-Processing #3 
# Create data set #3 
# in:  dataframe processed by general pre-processing routine 
# out: dataframe modified as follows: 
#      Continuous features  - standardize 
#      Ordinal featuers     - standardize
#      Categorical features - unchanged
# (note: the only ordinal feature is education_num)
#
def pre_processing_III(dfin):

    # perform general pre processing steps
    df = common_preprocess_I( dfin.copy() )

    # standirdize all non-categorical features (i.e., continuous and ordinal) 
    df.age            = scale(df.age)
    df.fnlwgt         = scale(df.fnlwgt)
    df.education_num  = scale(df.education_num)
    df.capital_gain   = scale(df.capital_gain)
    df.capital_loss   = scale(df.capital_loss)
    df.hours_per_week = scale(df.hours_per_week)

    return df

#------------------------------------------------------------------------------------------------------ 
# Pre-Processing #4 
# Create data set #4 
# in:  dataframe processed by general pre-processing routine 
# out: dataframe modified as follows: 
#      Continuous features  - discretize 
#      Ordinal featuers     - unchanged
#      Categorical features - create dummy variables 
# (note: the only ordinal feature is education_num)
#
def pre_processing_IV(dfin):

    # perform general pre processing steps
    df = common_preprocess_II( dfin.copy() )

    # Discretize all continous columns 
    df = discretize( df, 'age'            )
    df = discretize( df, 'fnlwgt'         )
    df = discretize( df, 'capital_gain'   )
    df = discretize( df, 'capital_loss'   )
    df = discretize( df, 'hours_per_week' )

    # create dummy variables for categorical features
    df = pd.get_dummies(df)  

    return df

#------------------------------------------------------------------------------------------------------ 
# Pre-Processing #5 
# Create data set #5 
# in:  dataframe processed by general pre-processing routine 
# out: dataframe modified as follows: 
#      Continuous features  - standardize 
#      Ordinal featuers     - standardize
#      Categorical features - create dummy variables 
#
def pre_processing_V(dfin):

    # perform general pre processing steps
    df = common_preprocess_II( dfin.copy() )

    # standirdize all non-categorical features (i.e., continuous and ordinal) 
    df.age            = scale(df.age)
    df.fnlwgt         = scale(df.fnlwgt)
    df.education_num  = scale(df.education_num)
    df.capital_gain   = scale(df.capital_gain)
    df.capital_loss   = scale(df.capital_loss)
    df.hours_per_week = scale(df.hours_per_week)

    # create dummy variables for categorical features
    df = pd.get_dummies(df)  

    return df

#------------------------------------------------------------------------------------------------------
# plot naive bayes roc
#
def plot_roc_naive_bayes(df_list, df_TEST_list):
    
    best_idx = 1 
    # retrive best datasets
    #
    # train
    #
    X_train = df_list[best_idx].copy()
    y_train = X_train.income
    X_train.drop('income', axis=1, inplace=True)
    # test
    #
    X_TEST = df_TEST_list[best_idx].copy()
    y_TEST = np.array(X_TEST.income)
    X_TEST.drop('income', axis=1, inplace=True)

    # remove column which doesn't exist in TEST dataset
    col_to_del = 'native_country_ Holand-Netherlands'
    if col_to_del in X_train.columns.tolist():
        X_train.drop(col_to_del, axis=1, inplace=True)

    clf = GaussianNB() 

    # BEGIN CITATION
    # code from sklearn man pages
    # 04/30/2017
    #
    # URL: http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py 
    #
    probas = clf.fit( X_train.values, y_train.values ).predict_proba(X_TEST)
    fpr, tpr, thresholds = roc_curve(y_TEST, probas[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, color='blue', label='Naive Bayes (area = %0.4f)' % (roc_auc))
    # 
    # END CITATION

#------------------------------------------------------------------------------------------------------ 
# plot roc knn algorithm 
#
def plot_roc_knn(df, df_TEST):

    best_idx = 4
    best_idx_parm = 21 

    # retrive best datasets
    #
    # train
    #
    X_train = df_list[best_idx].copy()
    y_train = X_train.income
    X_train.drop('income', axis=1, inplace=True)
    # test
    #
    X_TEST = df_TEST_list[best_idx].copy()
    y_TEST = np.array(X_TEST.income)
    X_TEST.drop('income', axis=1, inplace=True)

    # remove column which doesn't exist in TEST dataset
    col_to_del = 'native_country_ Holand-Netherlands'
    if col_to_del in X_train.columns.tolist():
        X_train.drop(col_to_del, axis=1, inplace=True)
  
    clf = KNeighborsClassifier(n_neighbors=best_idx_parm)

    # BEGIN CITATION
    # code from sklearn man pages
    # 04/30/2017
    #
    # URL: http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py 
    #
    probas = clf.fit( X_train.values, y_train.values ).predict_proba(X_TEST)
    fpr, tpr, thresholds = roc_curve(y_TEST, probas[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, color='limegreen', label='KNN (area = %0.4f)' % (roc_auc))
    # 
    # END CITATION

#------------------------------------------------------------------------------------------------------ 
# plot roc logistic regression 
#
def plot_roc_logit(df, df_TEST):

    best_idx = 4
    best_idx_parm_l = 'l1' 
    best_idx_parm_C = 1.0

    # retrive best datasets
    #
    # train
    #
    X_train = df_list[best_idx].copy()
    y_train = X_train.income
    X_train.drop('income', axis=1, inplace=True)
    # test
    #
    X_TEST = df_TEST_list[best_idx].copy()
    y_TEST = np.array(X_TEST.income)
    X_TEST.drop('income', axis=1, inplace=True)

    # remove column which doesn't exist in TEST dataset
    col_to_del = 'native_country_ Holand-Netherlands'
    if col_to_del in X_train.columns.tolist():
        X_train.drop(col_to_del, axis=1, inplace=True)
  
    clf = LogisticRegression(penalty=best_idx_parm_l, C=best_idx_parm_C)

    # BEGIN CITATION
    # code from sklearn man pages
    # 04/30/2017
    #
    # URL: http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py 
    #
    probas = clf.fit( X_train.values, y_train.values ).predict_proba(X_TEST)
    fpr, tpr, thresholds = roc_curve(y_TEST, probas[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, color='red', label='Logit (area = %0.4f)' % (roc_auc))
    # 
    # END CITATION

#------------------------------------------------------------------------------------------------------ 
# plot roc decision tree 
#
def plot_roc_decision_tree(df, df_TEST):
    
    best_idx = 4 
    best_idx_parm = 15 
   
    # retrive best datasets
    #
    # train
    #
    X_train = df_list[best_idx].copy()
    y_train = X_train.income
    X_train.drop('income', axis=1, inplace=True)
    # test
    #
    X_TEST = df_TEST_list[best_idx].copy()
    y_TEST = np.array(X_TEST.income)
    X_TEST.drop('income', axis=1, inplace=True)

    # remove column which doesn't exist in TEST dataset
    col_to_del = 'native_country_ Holand-Netherlands'
    if col_to_del in X_train.columns.tolist():
        X_train.drop(col_to_del, axis=1, inplace=True)
  
    clf = tree.DecisionTreeClassifier(max_depth=best_idx_parm)

    # BEGIN CITATION
    # code from sklearn man pages
    # 04/30/2017
    #
    # URL: http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py 
    #
    probas = clf.fit( X_train.values, y_train.values ).predict_proba(X_TEST)
    fpr, tpr, thresholds = roc_curve(y_TEST, probas[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, color='cyan', label='Decision Tree (area = %0.4f)' % (roc_auc))
    # 
    # END CITATION

#------------------------------------------------------------------------------------------------------ 
# plot roc SVM 
#
def plot_roc_svm(df, df_TEST):
    
    best_idx = 4
    best_idx_kernel = 'rbf' 
    best_idx_C = 1

    # retrive best datasets
    #
    # train
    #
    X_train = df_list[best_idx].copy()
    y_train = X_train.income
    X_train.drop('income', axis=1, inplace=True)
    # test
    #
    X_TEST = df_TEST_list[best_idx].copy()
    y_TEST = np.array(X_TEST.income)
    X_TEST.drop('income', axis=1, inplace=True)

    # remove column which doesn't exist in TEST dataset
    col_to_del = 'native_country_ Holand-Netherlands'
    if col_to_del in X_train.columns.tolist():
        X_train.drop(col_to_del, axis=1, inplace=True)

    clf = SVC(kernel=best_idx_kernel, C=best_idx_C)

    # BEGIN CITATION
    # code from sklearn man pages
    # 04/30/2017
    #
    # URL: http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py 
    #
    probas = clf.fit( X_train.values, y_train.values ).decision_function(X_TEST)
    fpr, tpr, thresholds = roc_curve(y_TEST, probas)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, color='magenta', label='SVM (area = %0.4f)' % (roc_auc))
    # 
    # END CITATION

#------------------------------------------------------------------------------------------------------ 
# plot roc AdaBoost 
#
def plot_roc_adaboost(df, df_TEST):
    
    best_idx = 2
    best_idx_depth = 6
    best_idx_n_est = 10

    # retrive best datasets
    #
    # train
    #
    X_train = df_list[best_idx].copy()
    y_train = X_train.income
    X_train.drop('income', axis=1, inplace=True)
    # test
    #
    X_TEST = df_TEST_list[best_idx].copy()
    y_TEST = np.array(X_TEST.income)
    X_TEST.drop('income', axis=1, inplace=True)

    # remove column which doesn't exist in TEST dataset
    col_to_del = 'native_country_ Holand-Netherlands'
    if col_to_del in X_train.columns.tolist():
        X_train.drop(col_to_del, axis=1, inplace=True)

    clf = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=best_idx_depth), 
                                     n_estimators=best_idx_n_est)

    # BEGIN CITATION
    # code from sklearn man pages
    # 04/30/2017
    #
    # URL: http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py 
    #
    probas = clf.fit( X_train.values, y_train.values ).predict_proba(X_TEST)
    fpr, tpr, thresholds = roc_curve(y_TEST, probas[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, color='goldenrod', label='Adaboost (area = %0.4f)' % (roc_auc))
    # 
    # END CITATION

#------------------------------------------------------------------------------------------------------ 
# plot roc Random Forest 
#
def plot_roc_random_forest(df, df_TEST):
    
    best_idx = 0
    best_idx_depth = 12 
    best_idx_n_est = 50

    # retrive best datasets
    #
    # train
    #
    X_train = df_list[best_idx].copy()
    y_train = X_train.income
    X_train.drop('income', axis=1, inplace=True)
    # test
    #
    X_TEST = df_TEST_list[best_idx].copy()
    y_TEST = np.array(X_TEST.income)
    X_TEST.drop('income', axis=1, inplace=True)

    # remove column which doesn't exist in TEST dataset
    col_to_del = 'native_country_ Holand-Netherlands'
    if col_to_del in X_train.columns.tolist():
        X_train.drop(col_to_del, axis=1, inplace=True)

    clf = RandomForestClassifier(n_estimators=best_idx_n_est, max_depth=best_idx_depth)

    # BEGIN CITATION
    # code from sklearn man pages
    # 04/30/2017
    #
    # URL: http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py 
    #
    probas = clf.fit( X_train.values, y_train.values ).predict_proba(X_TEST)
    fpr, tpr, thresholds = roc_curve(y_TEST, probas[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, color='yellow', label='Random Forest (area = %0.4f)' % (roc_auc))
    # 
    # END CITATION

#####################################################################################################
#
if __name__ == '__main__':
    
    # load data
    df = load_data('adult.data')
    df_TEST = load_data('adult.test')

    # obtain 5 versions of dataset as follows:
    '''
                	Continouse Feature	Ordinal Features	Categorical Features
    Pre-processing 1	Leave unchanged		Leave unchanged		Leave unchanged
    Pre-processing 2	Discretize		Leave unchanged		Leave unchanged
    Pre-processing 3	Standardize		Standardize		Leave unchanged
    Pre-processing 4	Discretize		Leave unchanged		Create dummy variables
    Pre-processing 5	Standardize		Standardize		Create dummy variables
    '''

    df_list = list()
    # populate list with training dataset versions     
    df_list.append( pre_processing_I( df.copy() ) )
    df_list.append( pre_processing_II( df.copy() ) )
    df_list.append( pre_processing_III( df.copy() ) )
    df_list.append( pre_processing_IV( df.copy() ) )
    df_list.append( pre_processing_V( df.copy() ) )

    df_TEST_list = list()
    # populate list with training dataset versions     
    df_TEST_list.append( pre_processing_I( df_TEST.copy() ) )
    df_TEST_list.append( pre_processing_II( df_TEST.copy() ) )
    df_TEST_list.append( pre_processing_III( df_TEST.copy() ) )
    df_TEST_list.append( pre_processing_IV( df_TEST.copy() ) )
    df_TEST_list.append( pre_processing_V( df_TEST.copy() ) )

    plot_roc_naive_bayes( df_list[:], df_TEST_list[:] )
    plot_roc_knn( df_list[:], df_TEST_list[:] )
    plot_roc_logit( df_list[:], df_TEST_list[:] )
    plot_roc_decision_tree( df_list[:], df_TEST_list[:] )
    plot_roc_svm( df_list[:], df_TEST_list[:] )
    plot_roc_adaboost( df_list[:], df_TEST_list[:] )
    plot_roc_random_forest( df_list[:], df_TEST_list[:] )

    # BEGIN CITATION
    # code from sklearn man pages
    # 04/30/2017
    #
    # URL: http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py 
    #
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', label='Luck')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    #
    # END CITATION

    plt.savefig('ROC.png')

