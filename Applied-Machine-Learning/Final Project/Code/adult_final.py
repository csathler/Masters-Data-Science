import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import tree
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 

#------------------------------------------------------------------------------------------------------ 
# print metrics 
# print metrics results for algo, along with confustion matrix
# in: string identifying dataset, accuracy, precision, recall, f1 score, roc, confusion matrix 
# out: n/a
#
def print_metrics(dataset, ac, pr, rc, f1, roc, TN, FP, FN, TP):

    msg = dataset + " DATASET RESULTS"
    print('\n           ' + msg)
    print("           Accuracy........: %.3f" % ac)
    print("           Precision Score.: %.3f" % pr)
    print("           Recall Score....: %.3f" % rc)
    print("           F1 Score........: %.3f" % f1)
    #print("           ROC AUC.........: %.3f" % roc)
    print("\n           Confusion Matrix:")

    print("           +----------------------------+")
    print("           |         Prediction         |")
    print("           +----------------------------+")
    print("           |      0      |       1      |")
    print("       +---|----------------------------|")
    print("       | 0 |    %5d    |     %5d    |" % (TN, FP))
    print(" Truth +   |----------------------------|")
    print("       | 1 |    %5d    |     %5d    |" % (FN, TP))
    print("       +---+----------------------------+")

#------------------------------------------------------------------------------------------------------ 
# print algo header 
# in:  
# out: 
# 
def print_algo_header(algo, best_dataset, best_parm_descr):
    header1 = 'RESULTS FOR ' + algo + ' CLASSIFER\n'
    header2 = '-' * len(header1)
    header3 = '\nBest dataset.........: ' + str(best_dataset) + '\n'
    header4 = 'Best hyper-parameters: ' + best_parm_descr + '\n'
    print '\n' + header1 + header2 + header3 + header4

#------------------------------------------------------------------------------------------------------ 
# get metrics 
# in:  2 arrays: actual Y values, predicted Y values 
# out: tuple with the following metrics:
#      accuracy, precision, recall, f1 score, roc auc, confusion matrix data TN, FP, FN, TP  
# 
def get_metrics(y_actual, y_pred):
    ac  = accuracy_score(y_actual, y_pred)
    pr  = precision_score(y_actual, y_pred)
    rc  = recall_score(y_actual, y_pred)
    f1  = f1_score(y_actual, y_pred)
    # incorrect calculation
    #roc = roc_auc_score(y_actual, y_pred)
    roc = -1 
    cm  = confusion_matrix(y_actual, y_pred)
    return (ac, pr, rc, f1, roc, cm[0][0], cm[0][1], cm[1][0], cm[1][1])

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
# Learn naive bayes 
# in:  list of 5 dataframes for naive bayes tuninng and selection
# out: 
#      tuple with 6 elements:
#           best dataset version
#           best metric score obtained for algo
#           string with description of best hyper-parameters for algo
#           2 arrays with Y values (actual, predicted) for training dataset
#           2 arrays with Y values (actual, predicted) for test dataset
#
def learn_naive_bayes(df_list, df_TEST_list):
    
    best_idx = 0
    best_idx_score = 0  

    for idx in range(5):

        df = df_list[idx]
 
        # partition dataset
        # will tune using stratified kfold and get final score on test set
        X_train, X_test, y_train, y_test = train_test_split( 
               df.iloc[:,1:].values, df.iloc[:,0:1].values, test_size=0.2, random_state=0)
    
        # create nvaive bayes classifier object
        clf = GaussianNB() 
    
        # will train model using kfold validation and compute average score
        scores = []
        kf = KFold(n_splits=10)
    
        # split "Generate indices to split data into training and test set" test = validation
        for train_idx, valid_idx in kf.split(X_train, y_train):
    
            # load train and validation partition for this iteration of score computing
            X_train_part = X_train[np.ravel(train_idx)] 
            y_train_part = y_train[np.ravel(train_idx)] 
            X_valid_part = X_train[np.ravel(valid_idx)] 
            y_valid_part = y_train[np.ravel(valid_idx)] 
   
            # learn/fit model for this fold
            clf = clf.fit( X_train_part, np.ravel(y_train_part))
            
            # calculate f1 score 
            y_pred = clf.predict( X_valid_part ) 
            scores.append( f1_score(y_valid_part, y_pred) )
         
        # track best score and corresponding hyperparameter value
        if best_idx_score < np.mean(scores):
            best_idx_score = np.mean(scores)
            best_idx = idx 

    # obtain results on full training dataset and then on test dataset

    # retrive best training dataset
    X_train = df_list[best_idx].copy()
    y_train = df.income
    X_train.drop('income', axis=1, inplace=True)

    # removing column which doesn't exist in TEST dataset
    col_to_del = 'native_country_ Holand-Netherlands'
    if col_to_del in X_train.columns.tolist():
        X_train.drop(col_to_del, axis=1, inplace=True)

    # repeat learning on full train partition using the best parameters
    clf = GaussianNB() 
    clf = clf.fit( X_train.values, y_train.values )
    y_pred = clf.predict(X_train)

    # select test dataset with went through proper preprocessing
    X_TEST = df_TEST_list[best_idx].copy()
    y_TEST = np.array(X_TEST.income)
    X_TEST.drop('income', axis=1, inplace=True)
    y_pred_TEST = clf.predict(X_TEST)

    return (best_idx, best_idx_score, 'No hyper-parameter tunning', np.array(y_train), y_pred, y_TEST, y_pred_TEST) 

#------------------------------------------------------------------------------------------------------ 
# Learn knn algorithm 
# in:  list of 5 dataframes for decision tree tuninng and selection
# out: 
#      tuple with 6 elements:
#           best dataset version
#           best metric score obtained for algo
#           string with description of best hyper-parameters for algo
#           2 arrays with Y values (actual, predicted) for training dataset
#           2 arrays with Y values (actual, predicted) for test dataset
#
def learn_knn(df, df_TEST):

    best_idx = 0
    best_idx_parm = 0
    best_idx_score = 0
  
    for idx in range(5):

        df = df_list[idx]

        # partition dataset
        # will tune using stratified kfold and get final score on test set
        X_train, X_test, y_train, y_test = train_test_split(
               df.iloc[:,1:].values, df.iloc[:,0:1].values, test_size=0.2, random_state=0)

        best_parm  = 0
        best_score = 0
    
        for parm_value in np.arange(26,step=5)+1: 
    
            # create nvaive bayes classifier object with max_depth parameter
            clf = KNeighborsClassifier(n_neighbors=parm_value)
        
            # will train model using kfold validation and compute average score
            scores = []
            kf = KFold(n_splits=10)
        
            # split "Generate indices to split data into training and test set" test = validation
            for train_idx, valid_idx in kf.split(X_train, y_train):
        
                # load train and validation partition for this iteration of score computing
                X_train_part = X_train[np.ravel(train_idx)] 
                y_train_part = y_train[np.ravel(train_idx)] 
                X_valid_part = X_train[np.ravel(valid_idx)] 
                y_valid_part = y_train[np.ravel(valid_idx)] 
    
                # learn/fit model for this fold
                clf = clf.fit( X_train_part, np.ravel(y_train_part))
     
                # calculate f1 score to select best hyperparameter 
                y_pred = clf.predict( X_valid_part) 
                scores.append( f1_score(y_valid_part, y_pred) )
 
            # track best score and corresponding hyperparameter value
            if best_score < np.mean(scores):
                best_score = np.mean(scores)
                best_parm  = parm_value 

        # test model against entire validation partition of training dataset for this version
        clf = KNeighborsClassifier(n_neighbors=best_parm)
        clf = clf.fit( X_train, np.ravel(y_train) )
        y_pred = clf.predict( X_test )
        idx_f1_score = f1_score(y_test, y_pred)

        # track best score and corresponding hyperparameter value
        if best_idx_score < idx_f1_score:
            best_idx_score = idx_f1_score
            best_idx_parm  = best_parm
            best_idx = idx
                    
    # calculate test scores 
    #
    #print('\nRESULTS FOR KNN CLASSIFIER')
    #print('--------------------------')
    #print('Best pre-processing option: %i' % best_idx)
    #print("Best neighbors value = %i" % best_parm)

    # report results on full training dataset and then on test dataset

    # retrive best training dataset
    X_train = df_list[best_idx].copy()
    y_train = df.income
    X_train.drop('income', axis=1, inplace=True)

    # removing column which doesn't exist in TEST dataset
    col_to_del = 'native_country_ Holand-Netherlands'
    if col_to_del in X_train.columns.tolist():
        X_train.drop(col_to_del, axis=1, inplace=True)

    # repeat learning on full train partition using the best parameters
    clf = KNeighborsClassifier(n_neighbors=best_idx_parm)
    clf = clf.fit( X_train.values, y_train.values )
    y_pred = clf.predict(X_train)

    # select test dataset with went through proper preprocessing
    X_TEST = df_TEST_list[best_idx].copy()
    y_TEST = np.array(X_TEST.income)
    X_TEST.drop('income', axis=1, inplace=True)
    y_pred_TEST = clf.predict(X_TEST)

    return (best_idx, best_idx_score, 'Neighbors = ' + str(best_parm), np.array(y_train), y_pred, y_TEST, y_pred_TEST) 

#------------------------------------------------------------------------------------------------------ 
# Learn logistic regression 
# in:  list of 5 dataframes for logistic regression tuninng and selection
# out: 
#      tuple with 6 elements:
#           best dataset version
#           best metric score obtained for algo
#           string with description of best hyper-parameters for algo
#           2 arrays with Y values (actual, predicted) for training dataset
#           2 arrays with Y values (actual, predicted) for test dataset
#
def learn_logit(df, df_TEST):

    best_idx = 0
    best_idx_parm_l = '' 
    best_idx_parm_C = 0.0
    best_idx_score = 0
 
    for idx in range(5):

        df = df_list[idx]

        # partition dataset
        # will tune using stratified kfold and get final score on test set
        X_train, X_test, y_train, y_test = train_test_split(
               df.iloc[:,1:].values, df.iloc[:,0:1].values, test_size=0.2, random_state=0)

        best_parm_l = 'l1'
        best_parm_C = 1.0 
        best_score = 0
    
        l = ['l1'] * 5 + ['l2'] * 5 
        C = [1.0, 10.0, 100.0, 1000.0, 1000000.0] * 2
    
        for l_, C_ in zip(l,C):
    
            # create logistic regression classifier object
            clf = LogisticRegression(penalty=l_, C=C_)
        
            # will train model using kfold validation and compute average score
            scores = []
            kf = KFold(n_splits=10)
        
            # split "Generate indices to split data into training and test set" test = validation
            for train_idx, valid_idx in kf.split(X_train, y_train):
        
                # load train and validation partition for this iteration of score computing
                X_train_part = X_train[np.ravel(train_idx)] 
                y_train_part = y_train[np.ravel(train_idx)] 
                X_valid_part = X_train[np.ravel(valid_idx)] 
                y_valid_part = y_train[np.ravel(valid_idx)] 
    
                # learn/fit model for this fold
                clf = clf.fit(X_train_part, np.ravel(y_train_part))
                
                # calculate f1 score to select best hyperparameter 
                y_pred = clf.predict( X_valid_part) 
                scores.append( f1_score(y_valid_part, y_pred) )
     
            # track best score and corresponding hyperparameter value
            if best_score < np.mean(scores):
                best_score = np.mean(scores)
                best_parm_l = l_
                best_parm_C = C_

        # test model against entire validation partition of training dataset for this version
        clf = LogisticRegression(penalty=best_parm_l, C=best_parm_C)
        clf = clf.fit( X_train, np.ravel(y_train) )
        y_pred = clf.predict( X_test )
        idx_f1_score = f1_score(y_test, y_pred)

        # track best score and corresponding hyperparameter value
        if best_idx_score < idx_f1_score:
            best_idx_score = idx_f1_score
            best_idx_parm_l  = best_parm_l
            best_idx_parm_C  = best_parm_C
            best_idx = idx
                
    # calculate test scores 
    #
    #print('\nRESULTS FOR LOGISTIC REGRESSION CLASSIFIER') 
    #print('------------------------------------------')
    #print('Best pre-processing option: %i' % best_idx)
    #print("Best penalty= " + best_idx_parm_l) 
    #print("Best C value= " + str(best_idx_parm_C)) 
   
    # report results on full training dataset and then on test dataset

    # retrive best training dataset
    X_train = df_list[best_idx].copy()
    y_train = df.income
    X_train.drop('income', axis=1, inplace=True)

    # removing column which doesn't exist in TEST dataset
    col_to_del = 'native_country_ Holand-Netherlands'
    if col_to_del in X_train.columns.tolist():
        X_train.drop(col_to_del, axis=1, inplace=True)

    # repeat learning on full train partition using the best parameters
    clf = LogisticRegression(penalty=best_idx_parm_l, C=best_idx_parm_C)
    clf = clf.fit( X_train.values, y_train.values )
    y_pred = clf.predict(X_train)

    # select test dataset with went through proper preprocessing
    X_TEST = df_TEST_list[best_idx].copy()
    y_TEST = np.array(X_TEST.income)
    X_TEST.drop('income', axis=1, inplace=True)
    y_pred_TEST = clf.predict(X_TEST)

    txt = 'Penalty = ' + best_idx_parm_l + '; C = ' + str(best_idx_parm_C)
    return (best_idx, best_idx_score, txt, np.array(y_train), y_pred, y_TEST, y_pred_TEST) 


#------------------------------------------------------------------------------------------------------ 
# Learn decision tree 
# in:  list of 5 dataframes for decision tree tuninng and selection
# out: 
#      tuple with 6 elements:
#           best dataset version
#           best metric score obtained for algo
#           string with description of best hyper-parameters for algo
#           2 arrays with Y values (actual, predicted) for training dataset
#           2 arrays with Y values (actual, predicted) for test dataset
#
def learn_decision_tree(df, df_TEST):
    
    best_idx = 0
    best_idx_parm = 0
    best_idx_score = 0  
   
    for idx in range(5):

        df = df_list[idx]

        # partition dataset
        # will tune using stratified kfold and get final score on test set
        X_train, X_test, y_train, y_test = train_test_split( 
               df.iloc[:,1:].values, df.iloc[:,0:1].values, test_size=0.2, random_state=0)

        best_parm = 0
        best_score = 0

        for parm_value in np.arange(15)+1: 
    
            # create tree classifier object with max_depth parameter
            clf = tree.DecisionTreeClassifier(max_depth=parm_value)
        
            # will train model using kfold validation and compute average score
            scores = []
            kf = KFold(n_splits=10)
        
            # split "Generate indices to split data into training and test set" test = validation
            for train_idx, valid_idx in kf.split(X_train, y_train):
        
                # load train and validation partition for this iteration of score computing
                X_train_part = X_train[np.ravel(train_idx)] 
                y_train_part = y_train[np.ravel(train_idx)] 
                X_valid_part = X_train[np.ravel(valid_idx)] 
                y_valid_part = y_train[np.ravel(valid_idx)] 
    
                # learn/fit model for this fold
                clf = clf.fit( X_train_part, np.ravel(y_train_part))
                
                # calculate f1 score to select best hyperparameter 
                y_pred = clf.predict( X_valid_part) 
                scores.append( f1_score(y_valid_part, y_pred) )
     
            # track best score and corresponding hyperparameter value
            if best_score < np.mean(scores):
                best_parm = parm_value 
                best_score = np.mean(scores)

        # test model against entire validation partition of training dataset for this version
        clf = tree.DecisionTreeClassifier(max_depth=best_parm)
        clf = clf.fit( X_train, np.ravel(y_train) )
        y_pred = clf.predict( X_test )
        idx_f1_score = f1_score(y_test, y_pred)

        # track best score and corresponding hyperparameter value
        if best_idx_score < idx_f1_score:
            best_idx_score = idx_f1_score 
            best_idx_parm  = best_parm 
            best_idx = idx 
    
    # report results on full training dataset and then on test dataset

    # retrive best training dataset
    X_train = df_list[best_idx].copy()
    y_train = df.income
    X_train.drop('income', axis=1, inplace=True)

    # removing column which doesn't exist in TEST dataset
    col_to_del = 'native_country_ Holand-Netherlands'
    if col_to_del in X_train.columns.tolist():
        X_train.drop(col_to_del, axis=1, inplace=True)

    # repeat learning on full train partition using the best parameters
    clf = tree.DecisionTreeClassifier(max_depth=best_idx_parm)
    clf = clf.fit( X_train.values, y_train.values )
    y_pred = clf.predict(X_train)

    # select test dataset with went through proper preprocessing
    X_TEST = df_TEST_list[best_idx].copy()
    y_TEST = np.array(X_TEST.income)
    X_TEST.drop('income', axis=1, inplace=True)
    y_pred_TEST = clf.predict(X_TEST)

    txt = 'Max Tree Depth = ' + str(best_idx_parm)
    return (best_idx, best_idx_score, txt, np.array(y_train), y_pred, y_TEST, y_pred_TEST) 


#------------------------------------------------------------------------------------------------------ 
# Learn SVM 
# in:  list of 5 dataframes for SVM tuninng and selection 
# out: 
#      tuple with 6 elements:
#           best dataset version
#           best metric score obtained for algo
#           string with description of best hyper-parameters for algo
#           2 arrays with Y values (actual, predicted) for training dataset
#           2 arrays with Y values (actual, predicted) for test dataset
#
def learn_svm(df, df_TEST):
    
    best_idx = 0
    best_idx_kernel = '' 
    best_idx_C = 0
    best_idx_score = 0  
    # create list of parameters as list of tuples (tree_depth, no_of_estimators)
    parms = list()
    for i in ['rbf']:   # attempted other kernels (linear and poly) but program hang
        for j in [0.5, 0.75, 1]: 
            parms.append([i, j])

    for idx in range(5):

        df = df_list[idx]

        # partition dataset
        # will tune using stratified kfold and get final score on test set
        X_train, X_test, y_train, y_test = train_test_split( 
               df.iloc[:,1:].values, df.iloc[:,0:1].values, test_size=0.2, random_state=0)

        best_parm_kernel = '' 
        best_parm_C = 0
        best_score = 0

        for parm_kernel, parm_C in parms: 
    
            # create nvaive bayes classifier object with max_depth parameter
            clf = SVC(kernel=parm_kernel, C=parm_C)
        
            # will train model using kfold validation and compute average score
            scores = []
            kf = KFold(n_splits=10)
        
            # split "Generate indices to split data into training and test set" test = validation
            for train_idx, valid_idx in kf.split(X_train, y_train):
        
                # load train and validation partition for this iteration of score computing
                X_train_part = X_train[np.ravel(train_idx)] 
                y_train_part = y_train[np.ravel(train_idx)] 
                X_valid_part = X_train[np.ravel(valid_idx)] 
                y_valid_part = y_train[np.ravel(valid_idx)] 
    
                # learn/fit model for this fold
                clf = clf.fit( X_train_part, np.ravel(y_train_part))
                
                # calculate f1 score to select best hyperparameter 
                y_pred = clf.predict( X_valid_part) 
                scores.append( f1_score(y_valid_part, y_pred) )
     
            # track best score and corresponding hyperparameter value
            if best_score < np.mean(scores):
                best_parm_kernel = parm_kernel
                best_parm_C = parm_C
                best_score = np.mean(scores)

        # test model against entire validation partition of training dataset for this version
        clf = SVC(kernel=parm_kernel, C=best_parm_C)
        clf = clf.fit( X_train, np.ravel(y_train) )
        y_pred = clf.predict( X_test )
        idx_f1_score = f1_score(y_test, y_pred) 

        # track best score, corresponding hyperparameter value, and best dataset version
        if best_idx_score < idx_f1_score:
            best_idx_score = idx_f1_score 
            best_idx_kernel = best_parm_kernel
            best_idx_C = best_parm_C
            best_idx = idx 
    
    # retrive best training dataset 
    X_train = df_list[best_idx].copy()
    y_train = df.income
    X_train.drop('income', axis=1, inplace=True)

    # removing column which doesn't exist in TEST dataset
    col_to_del = 'native_country_ Holand-Netherlands'
    if col_to_del in X_train.columns.tolist():
        X_train.drop(col_to_del, axis=1, inplace=True)

    # repeat learning on full train partition using the best parameters
    clf = SVC(kernel=best_idx_kernel, C=best_idx_C)
    clf = clf.fit( X_train.values, y_train.values )
    y_pred = clf.predict(X_train)

    # select test dataset with went through proper preprocessing
    X_TEST = df_TEST_list[best_idx].copy()
    y_TEST = np.array(X_TEST.income)
    X_TEST.drop('income', axis=1, inplace=True)
    y_pred_TEST = clf.predict(X_TEST)
    
    txt = 'C = ' + str(best_idx_C)
    return (best_idx, best_idx_score, txt, np.array(y_train), y_pred, y_TEST, y_pred_TEST) 


#------------------------------------------------------------------------------------------------------ 
# Learn AdaBoost 
# in:  list of 5 dataframes for adaboost tuninng and selection 
# out: 
#      tuple with 6 elements:
#           best dataset version
#           best metric score obtained for algo
#           string with description of best hyper-parameters for algo
#           2 arrays with Y values (actual, predicted) for training dataset
#           2 arrays with Y values (actual, predicted) for test dataset
#
def learn_adaboost(df, df_TEST):
    
    best_idx = 0
    best_idx_depth = 0
    best_idx_n_est = 0
    best_idx_score = 0  

    # create list of parameters as list of tuples (tree_depth, no_of_estimators)
    parms = list()
    for i in np.arange(4,13,2): 
        for j in [1, 10, 50]: 
            parms.append([i, j])

    for idx in range(5):

        df = df_list[idx]

        # partition dataset
        # will tune using stratified kfold and get final score on test set
        X_train, X_test, y_train, y_test = train_test_split( 
               df.iloc[:,1:].values, df.iloc[:,0:1].values, test_size=0.2, random_state=0)

        best_parm_depth = 0
        best_parm_n_est = 0
        best_score = 0

        for parm_depth, parm_n_est in parms: 
    
            # create nvaive bayes classifier object with max_depth parameter
            clf = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=parm_depth), 
                                     n_estimators=parm_n_est)
        
            # will train model using kfold validation and compute average score
            scores = []
            kf = KFold(n_splits=10)
        
            # split "Generate indices to split data into training and test set" test = validation
            for train_idx, valid_idx in kf.split(X_train, y_train):
        
                # load train and validation partition for this iteration of score computing
                X_train_part = X_train[np.ravel(train_idx)] 
                y_train_part = y_train[np.ravel(train_idx)] 
                X_valid_part = X_train[np.ravel(valid_idx)] 
                y_valid_part = y_train[np.ravel(valid_idx)] 
    
                # learn/fit model for this fold
                clf = clf.fit( X_train_part, np.ravel(y_train_part))
                
                # calculate f1 score to select best hyperparameter 
                y_pred = clf.predict( X_valid_part) 
                scores.append( f1_score(y_valid_part, y_pred) )
     
            # track best score and corresponding hyperparameter value
            if best_score < np.mean(scores):
                best_parm_depth = parm_depth
                best_parm_n_est = parm_n_est
                best_score = np.mean(scores)

        # test model against entire validation partition of training dataset for this version
        clf = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=best_parm_depth), 
                                 n_estimators=best_parm_n_est)
        clf = clf.fit( X_train, np.ravel(y_train) )
        y_pred = clf.predict( X_test )
        idx_f1_score = f1_score(y_test, y_pred) 

        # track best score, corresponding hyperparameter value, and best dataset version
        if best_idx_score < idx_f1_score:
            best_idx_score = idx_f1_score 
            best_idx_depth = best_parm_depth 
            best_idx_n_est = best_parm_n_est
            best_idx = idx 
    
    # report results on full training dataset and then on test dataset 

    # retrive best training dataset 
    X_train = df_list[best_idx].copy()
    y_train = df.income
    X_train.drop('income', axis=1, inplace=True)

    # removing column which doesn't exist in TEST dataset
    col_to_del = 'native_country_ Holand-Netherlands'
    if col_to_del in X_train.columns.tolist():
        X_train.drop(col_to_del, axis=1, inplace=True)

    # repeat learning on full train partition using the best parameters
    clf = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=best_idx_depth), 
                                     n_estimators=best_idx_n_est)
    clf = clf.fit( X_train.values, y_train.values )
    y_pred = clf.predict(X_train)

    # select test dataset which went through proper preprocessing
    X_TEST = df_TEST_list[best_idx].copy()
    y_TEST = np.array(X_TEST.income)
    X_TEST.drop('income', axis=1, inplace=True)
    y_pred_TEST = clf.predict(X_TEST)

    txt = 'Estimators = ' + str(best_idx_n_est) + '; Max Tree Depth = ' + str(best_idx_depth)
    return (best_idx, best_idx_score, txt, np.array(y_train), y_pred, y_TEST, y_pred_TEST) 


#------------------------------------------------------------------------------------------------------ 
# Learn Random Forest 
# in:  list of 5 dataframes for adaboost tuninng and selection 
# out: 
#      tuple with 6 elements:
#           best dataset version
#           best metric score obtained for algo
#           string with description of best hyper-parameters for algo
#           2 arrays with Y values (actual, predicted) for training dataset
#           2 arrays with Y values (actual, predicted) for test dataset
#
def learn_random_forest(df, df_TEST):
    
    best_idx = 0
    best_idx_depth = 0
    best_idx_n_est = 0
    best_idx_score = 0  

    # create list of parameters as list of tuples (tree_depth, no_of_estimators)
    parms = list()
    for i in np.arange(4,13,2): 
        for j in [1, 10, 50]: 
            parms.append([i, j])

    for idx in range(5):

        df = df_list[idx]

        # partition dataset
        # will tune using stratified kfold and get final score on test set
        X_train, X_test, y_train, y_test = train_test_split( 
               df.iloc[:,1:].values, df.iloc[:,0:1].values, test_size=0.2, random_state=0)

        best_parm_depth = 0
        best_parm_n_est = 0
        best_score = 0

        for parm_depth, parm_n_est in parms: 
    
            # create random forest classifier object with max_depth parameter
            clf = RandomForestClassifier(n_estimators=parm_n_est, max_depth=parm_depth)
        
            # will train model using kfold validation and compute average score
            scores = []
            kf = KFold(n_splits=10)
        
            # split "Generate indices to split data into training and test set" test = validation
            for train_idx, valid_idx in kf.split(X_train, y_train):
        
                # load train and validation partition for this iteration of score computing
                X_train_part = X_train[np.ravel(train_idx)] 
                y_train_part = y_train[np.ravel(train_idx)] 
                X_valid_part = X_train[np.ravel(valid_idx)] 
                y_valid_part = y_train[np.ravel(valid_idx)] 
    
                # learn/fit model for this fold
                clf = clf.fit( X_train_part, np.ravel(y_train_part))
                
                # calculate f1 score to select best hyperparameter 
                y_pred = clf.predict( X_valid_part) 
                scores.append( f1_score(y_valid_part, y_pred) )
     
            # track best score and corresponding hyperparameter value
            if best_score < np.mean(scores):
                best_parm_depth = parm_depth
                best_parm_n_est = parm_n_est
                best_score = np.mean(scores)

        # test model against entire validation partition of training dataset for this version
        clf = RandomForestClassifier(n_estimators=best_parm_n_est, max_depth=best_parm_depth)
        clf = clf.fit( X_train, np.ravel(y_train) )
        y_pred = clf.predict( X_test )
        idx_f1_score = f1_score(y_test, y_pred) 

        # track best score, corresponding hyperparameter value, and best dataset version
        if best_idx_score < idx_f1_score:
            best_idx_score = idx_f1_score 
            best_idx_depth = best_parm_depth 
            best_idx_n_est = best_parm_n_est
            best_idx = idx 
    
    # report results on full training dataset and then on test dataset 

    # retrive best training dataset 
    X_train = df_list[best_idx].copy()
    y_train = df.income
    X_train.drop('income', axis=1, inplace=True)

    # removing column which doesn't exist in TEST dataset
    col_to_del = 'native_country_ Holand-Netherlands'
    if col_to_del in X_train.columns.tolist():
        X_train.drop(col_to_del, axis=1, inplace=True)

    # repeat learning on full train partition using the best parameters
    clf = RandomForestClassifier(n_estimators=best_idx_n_est, max_depth=best_idx_depth)
    clf = clf.fit( X_train.values, y_train.values )
    y_pred = clf.predict(X_train)

    # select test dataset with went through proper preprocessing
    X_TEST = df_TEST_list[best_idx].copy()
    y_TEST = np.array(X_TEST.income)
    X_TEST.drop('income', axis=1, inplace=True)
    y_pred_TEST = clf.predict(X_TEST)

    txt = 'Estimators = ' + str(best_idx_n_est) + '; Max Tree Depth = ' + str(best_idx_depth)
    return (best_idx, best_idx_score, txt, y_train, np.array(y_pred), y_TEST, y_pred_TEST) 

#------------------------------------------------------------------------------------------------------ 
# Learn Ensemble 
# Will make prediction based on all other algos, using score to weigh up and down each algo prediction
# in:  dictionary with output from all learning algorithms  
# out: 
#      tuple with 6 elements:
#           best dataset version
#           best metric score obtained for algo
#           string with description of best hyper-parameters for algo
#           2 arrays with Y values (actual, predicted) for training dataset
#           2 arrays with Y values (actual, predicted) for test dataset
#
def learn_ensemble( other_algo_results ):
    #
    # training predictions 
    arr_l = list()
    for k, v in other_algo_results.items():
        # create array with prediction for this algo
        arr = other_algo_results[k][4].copy()
        # replace zeros with -1 to work out the math 
        ser = pd.Series(arr) 
        ser.replace(0,-1,inplace=True)
        # calculate the weight of the prediction, positive, or negative
        arr = np.multiply(arr, other_algo_results[k][1])
        # add array to list of predictions
        arr_l.append(arr)
    # totalize predictions
    pred_arr = np.zeros(len(arr_l[0]))
    for a in arr_l:
        np.add( pred_arr, a, pred_arr )
    # finalize prediction 
    for x in np.nditer(pred_arr, op_flags=['readwrite']): 
        if x > 0:
            x[...] = 1
        else:
            x[...] = 0
    y_pred = np.copy(pred_arr)

    #
    # test predictions 
    arr_l = list()
    for k, v in other_algo_results.items():
        # create array with prediction for this algo
        arr = other_algo_results[k][6].copy()
        # replace zeros with -1 to work out the math 
        ser = pd.Series(arr) 
        ser.replace(0,-1,inplace=True)
        # calculate the weight of the prediction, positive, or negative
        arr = np.multiply(arr, other_algo_results[k][1])
        # add array to list of predictions
        arr_l.append(arr)
    # totalize predictions
    pred_arr = np.zeros(len(arr_l[0]))
    for a in arr_l:
        np.add( pred_arr, a, pred_arr )
    # finalize prediction 
    for x in np.nditer(pred_arr, op_flags=['readwrite']): 
        if x > 0:
            x[...] = 1
        else:
            x[...] = 0
    y_pred_TEST = np.copy(pred_arr)

    # y_train and y_TEST are the same for all algos, so set them from NAIVE BAYES (could be any)
    y_train = other_algo_results['NAIVE BAYES'][3].copy()
    y_TEST  = other_algo_results['NAIVE BAYES'][5].copy()
    return (-1, -1, 'N/A', y_train, y_pred, y_TEST, y_pred_TEST) 


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

    # will set dictionary "output_from_learning" to capture output from each algorithm
    # key: algo name
    # values: tuple with 6 elements:
    #         0 - best dataset version
    #         1 - best metric score for the algorithm
    #         2 - string with description of best hyper-parameters for algo
    #         3 - array with actual Y values from training dataset (same for all algos)
    #         4 - array with predicted Y values from training dataset 
    #         5 - array with actual Y values from test dataset (same for all algos) 
    #         6 - array with predicted Y values from test dataset 
    output_from_learning = {}
    output_from_learning['NAIVE BAYES'] = learn_naive_bayes( df_list[:], df_TEST_list[:] )
    output_from_learning['KNN'] = learn_knn( df_list[:], df_TEST_list[:] )
    output_from_learning['LOGIT'] = learn_logit( df_list[:], df_TEST_list[:] )
    output_from_learning['DECISION TREE'] = learn_decision_tree( df_list[:], df_TEST_list[:] )
    output_from_learning['SVM'] = learn_svm( df_list[:], df_TEST_list[:] )
    output_from_learning['ADABOOST'] = learn_adaboost( df_list[:], df_TEST_list[:] )
    output_from_learning['RANDOM ROREST'] = learn_random_forest( df_list[:], df_TEST_list[:] )
    output_from_learning['ENSEMBLE'] = learn_ensemble( output_from_learning )

    # save and print metrics for each algorithm to csv datafile

    f = open('adult_metrics.csv', 'w')
    header = 'Algo,Dataset,BestVersion,BestScore,BestParm,Accuracy,Precision,Recall,F1_Score,ROC_AUC,TN,FP,FN,TP'
    f.write( header + '\n' )
    for k, v in output_from_learning.items(): 
        #
        # to be clear....
        # k is algo name
        # v[0] is dataset version
        # v[1] is best metric score obtained for algo 
        # v[2] is description of best hyper-parameter setting
        # v[3] and v[4] are arrays (actual and predicted) with Y values for train dataset 
        # v[5] and v[6] are arrays (actual and predicted) with Y values for train dataset 
        
        # print algo header  
        print_algo_header(k, v[0], v[2])
 
        # save metrics for results obtained on training dataset 
        (ac, pr, rc, f1, roc, TN, FP, FN, TP) = get_metrics(v[3], v[4]) 
        f.write(k + ',' + 'TRAIN' + ',' + str(v[0]) + ',' + str(v[1]) + ',' + v[2] + ',' + 
                str(ac) + ',' + str(pr) + ',' + str(rc) + ',' + str(f1) + ',' + str(roc) + ',' + 
                str(TN) + ',' + str(FP) + ',' + str(FN) + ',' + str(TP) + '\n') 

        # print training metrics
        print_metrics('TRAIN', ac, pr, rc, f1, roc, TN, FP, FN, TP)
         
        # save metrics for results obtained on test dataset 
        (ac, pr, rc, f1, roc, TN, FP, FN, TP) = get_metrics(v[5], v[6])
        f.write(k + ',' + 'TEST' + ',' + str(v[0]) + ',' + str(v[1]) + ',' + v[2] + ',' + 
                str(ac) + ',' + str(pr) + ',' + str(rc) + ',' + str(f1) + ',' + str(roc) + ',' + 
                str(TN) + ',' + str(FP) + ',' + str(FN) + ',' + str(TP) + '\n') 

        # print test metrics
        print_metrics('TEST', ac, pr, rc, f1, roc, TN, FP, FN, TP)

    f.close()
