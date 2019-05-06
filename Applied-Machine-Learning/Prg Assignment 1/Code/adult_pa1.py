import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import tree
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale


#------------------------------------------------------------------------------------------------------ 
# show_detailed_results
# Show detailed score results for algo, along with confustion matrix
# in: accuracy score, true target values, predicitons 
# out:
#
def show_detailed_results(msg, ac, y_test, y_pred):

    f1  = f1_score(y_test, y_pred)
    rc  = recall_score(y_test, y_pred)
    pr  = precision_score(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)

    print('\n           ' + msg)
    print("           Accuracy........: %.3f" % ac)
    print("           Precision Score.: %.3f" % f1)
    print("           Recall Score....: %.3f" % rc)
    print("           F1 Score........: %.3f" % pr)
    print("           ROC AUC.........: %.3f" % roc)
    print("\n           Confusion Matrix:")

    print("           +----------------------------+")
    print("           |         Prediction         |")
    print("           +----------------------------+")
    print("           |      0      |       1      |")
    print("       +---|----------------------------|")
    print("       | 0 |    %5d    |     %5d    |" % (cm[0][0],cm[0][1]))
    print(" Truth +   |----------------------------|")
    print("       | 1 |    %5d    |     %5d    |" % (cm[1][0],cm[1][1]))
    print("       +---+----------------------------+")

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
    #df_saved = df.copy()
    
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
# general precessing
# Performs the following preprocessing to benefit all algorithms
# 1 - converts all columns in dataset to float 
#     all categorical and ordinal features are converted
# 2 - remove rows with nan's 
# 3 - remove redundant features
#
# in:  dataset as dataframe 
# out: tuple with three elements
#      1 - source dataframe w/o nan's and redundant features 
#      2 - source dataframe used to create np.arrays - all features as floats
#      3 - dictionary with mapping between original feature values and new number values 
# 
def general_preprocess(df):

    # found 2399 rows with NaN in the following columns: workclass, occupation, native_country 
    # cannot think of a way to impute values, so will dropna
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

    return (df_source, df, col_map)

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
# Replace continous feature with discrete value from 1-10 corresponding to it's percentile
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
# logit_preprocess 
# Pre-processing for logistic regression algo
# will do the following:
# - standardize continuous variables
# - discretize income variable
# - create dummy variable for each categorial feature
# in: dataframe not ready for logit algo 
# out: dataframe ready for logit algo 
#
def logit_preprocess(df):

    # standardize continous variables
    df.age            = scale(df.age)
    df.fnlwgt         = scale(df.fnlwgt)
    df.education_num  = scale(df.education_num)
    df.capital_gain   = scale(df.capital_gain)
    df.capital_loss   = scale(df.capital_loss)
    df.hours_per_week = scale(df.hours_per_week)

    # discretize income variable
    col_map = {}
    income_dict       = pd.Series(df.income.unique()).to_dict()
    income_dict_inv   = invert_dict( income_dict )
    col_map['income'] = [ income_dict, income_dict_inv ]
    df.income.replace(income_dict_inv, inplace=True )
   
    # create dummy variables for categorical features
    df_dummies = pd.get_dummies(df)  
  
    return df_dummies


#------------------------------------------------------------------------------------------------------ 
# nbayes_preprocess 
# Naive Bayes specific preprocessing 
# in: dataframe not ready for Naive Bayes algo 
# out: dataframe ready for Naive Bayes algo 
#
def nbayes_preprocess(df):

    # Discretize all continous columns 
    df = discretize( df, 'age'            )
    df = discretize( df, 'fnlwgt'         )
    df = discretize( df, 'capital_gain'   )
    df = discretize( df, 'capital_loss'   )
    df = discretize( df, 'hours_per_week' )

    return df


#------------------------------------------------------------------------------------------------------ 
# Learn naive_bayes 
# in:  dataframe ready for naive bayes algo 
# out: 
#
def learn_naive_bayes(df, df_TEST):

    # partition dataset
    # will tune using stratified kfold and get final score on test set
    X_train, X_test, y_train, y_test = train_test_split( 
           df.iloc[:,:13].values, df.iloc[:,13:14].values, test_size=0.2, random_state=0)

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
        clf = clf.fit( X_train_part, y_train_part) 
        
        # calculate f1 score 
        y_pred = clf.predict( X_valid_part) 
        scores.append( f1_score(y_valid_part, y_pred) )

    # display results 
    #
    print('\nRESULTS FOR NAIVE BAYES CLASSIFIER')
    print('----------------------------------')

    # first results on training dataset - validation partition
    #
    ac = clf.score( X_test, y_test )
    y_pred = clf.predict(X_test)
    show_detailed_results('TRAIN VALIDATION PARTITION', ac, y_test, y_pred)

    # then results on test dataset 
    #
    X_train = df.copy() 
    X_train.drop('income', axis=1, inplace=True)
    y_train = df.income
    clf = clf.fit( X_train.values, y_train.values )
    #
    X_TEST = df_TEST.copy()
    X_TEST.drop('income', axis=1, inplace=True)
    y_TEST = df_TEST.income
    ac = clf.score( X_TEST, y_TEST)
    y_pred_TEST = clf.predict(X_TEST)
    show_detailed_results('TEST DATASET', ac, y_TEST, y_pred_TEST)



#------------------------------------------------------------------------------------------------------ 
# Learn decision tree 
# in:  dataframe ready for decision tree algo 
# out: 
#
def learn_decision_tree(df, df_TEST):

    # partition dataset
    # will tune using stratified kfold and get final score on test set
    X_train, X_test, y_train, y_test = train_test_split( 
           df.iloc[:,:13].values, df.iloc[:,13:14].values, test_size=0.2, random_state=0)

    best_parm  = 0
    best_score = 0

    for parm_value in np.arange(15)+1: 

        # create nvaive bayes classifier object with max_depth parameter
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
            clf = clf.fit(X_train_part, np.ravel(y_train_part))
            
            # calculate f1 score to select best hyperparameter 
            y_pred = clf.predict( X_valid_part) 
            scores.append( f1_score(y_valid_part, y_pred) )
 
        # track best score and corresponding hyperparameter value
        if best_score < np.mean(scores):
            best_score = np.mean(scores)
            best_parm  = parm_value 
                
    # repeat learning on full train partition with best parm value 
    clf = tree.DecisionTreeClassifier(max_depth=best_parm)
    #clf = tree.DecisionTreeClassifier(max_depth=8)
    clf = clf.fit( X_train, np.ravel(y_train))

    # calculate test scores 
    #
    print('\nRESULTS FOR DECISION TREE CLASSIFIER')
    print('------------------------------------')
    print("Best tree depth = %i" % best_parm)
   
    # first results on training dataset - validation partition
    #
    ac = clf.score( X_test, y_test )
    y_pred = clf.predict(X_test)
    show_detailed_results('TRAIN VALIDATION PARTITION', ac, y_test, y_pred)

    # then results on test dataset 
    #
    X_train = df.copy() 
    X_train.drop('income', axis=1, inplace=True)
    y_train = df.income
    clf = clf.fit( X_train.values, y_train.values )
    #
    X_TEST = df_TEST.copy()
    X_TEST.drop('income', axis=1, inplace=True)
    y_TEST = df_TEST.income
    ac = clf.score( X_TEST, y_TEST)
    y_pred_TEST = clf.predict(X_TEST)
    show_detailed_results('TEST DATASET', ac, y_TEST, y_pred_TEST)


#------------------------------------------------------------------------------------------------------ 
# Learn knn algorithm 
# in:  dataframe ready for knn algo 
# out: 
#
def learn_knn(df, df_TEST):

    # partition dataset
    # will tune using stratified kfold and get final score on test set
    X_train, X_test, y_train, y_test = train_test_split( 
           df.iloc[:,:13].values, df.iloc[:,13:14].values, test_size=0.2, random_state=0)

    best_parm  = 0
    best_score = 0

    for parm_value in np.arange(31,step=3)+1: 

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
            clf = clf.fit(X_train_part, np.ravel(y_train_part))
 
            # calculate f1 score to select best hyperparameter 
            y_pred = clf.predict( X_valid_part) 
            scores.append( f1_score(y_valid_part, y_pred) )
 
        # track best score and corresponding hyperparameter value
        if best_score < np.mean(scores):
            best_score = np.mean(scores)
            best_parm  = parm_value 
                
    # repeat learning on full train partition with best parm value 
    clf = KNeighborsClassifier(n_neighbors=best_parm)
    clf = clf.fit( X_train, np.ravel(y_train) )

    # calculate test scores 
    #
    print('\nRESULTS FOR KNN CLASSIFIER')
    print('--------------------------')
    print("Best neighbors value = %i" % best_parm)
  
    # first results on training dataset - validation partition
    #
    ac = clf.score( X_test, y_test )
    y_pred = clf.predict(X_test)
    show_detailed_results('TRAIN VALIDATION PARTITION', ac, y_test, y_pred)

    # then results on test dataset 
    #
    X_train = df.copy() 
    X_train.drop('income', axis=1, inplace=True)
    y_train = df.income
    clf = clf.fit( X_train.values, y_train.values )
    #
    X_TEST = df_TEST.copy()
    X_TEST.drop('income', axis=1, inplace=True)
    y_TEST = df_TEST.income
    ac = clf.score( X_TEST, y_TEST)
    y_pred_TEST = clf.predict(X_TEST)
    show_detailed_results('TEST DATASET', ac, y_TEST, y_pred_TEST)


#------------------------------------------------------------------------------------------------------ 
# Learn logistic regression 
# in:  dataframe ready for logit algo 
# out: 
#
def learn_logit(df, df_TEST):

    X = df.copy()
    X.drop('income', axis=1, inplace=True)
    y = df.income

    # partition dataset
    # will tune using stratified kfold and get final score on test set
    X_train, X_test, y_train, y_test = train_test_split( X.values, y.values, 
                                                test_size=0.2, random_state=0)

    best_parm  = [ 'l1', 1.0 ] 
    best_score = 0

    l = ['l1'] * 5 + ['l2'] * 5 
    C = [1.0, 10.0, 100.0, 1000.0, 1000000.0] * 2

    for l_, C_ in zip(l,C):

        # create nvaive bayes classifier object with max_depth parameter
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
            best_parm  = [ l_, C_ ]
                
    # repeat learning on full train partition with best parm value 
    clf = LogisticRegression(penalty=best_parm[0], C=best_parm[1])
    clf = clf.fit( X_train, np.ravel(y_train) ) 

    # calculate test scores 
    #
    print('\nRESULTS FOR LOGISTIC REGRESSION CLASSIFIER') 
    print('------------------------------------------')
    print("Best penalty= " + best_parm[0]) 
    print("Best C value= " + str(best_parm[1])) 
   
    # first results on training dataset - validation partition
    #
    ac = clf.score( X_test, y_test )
    y_pred = clf.predict(X_test)
    show_detailed_results('TRAIN VALIDATION PARTITION', ac, y_test, y_pred)

    # then results on test dataset 
    #
    X_train = df.copy() 
    X_train.drop('income', axis=1, inplace=True)
    # removing column which doesn't exist in TEST dataset
    X_train.drop('native_country_ Holand-Netherlands', axis=1, inplace=True)
    y_train = df.income
    clf = clf.fit( X_train.values, y_train.values )
    #
    X_TEST = df_TEST.copy()
    X_TEST.drop('income', axis=1, inplace=True)
    y_TEST = df_TEST.income
    ac = clf.score( X_TEST, y_TEST)
    y_pred_TEST = clf.predict(X_TEST)
    show_detailed_results('TEST DATASET', ac, y_TEST, y_pred_TEST)



#####################################################################################################
#
if __name__ == '__main__':
    
    # load data
    df = load_data('adult.data')
    df_TEST = load_data('adult.test')

    # general preprocessing - converts every column to float
    # called "general" because all algorithms will have this preprocessing in common
    # - get df_source back without nan rows and without redundant features
    # - get df_floats with all categorical and nominal features converted to float
    # - get dictionary with mapping info for each converted field
    #
    (df, df_floats, df_col_map) = general_preprocess( df.copy() ) 
    (df_TEST, df_floats_TEST, df_col_map_TEST) = general_preprocess( df_TEST.copy() ) 

    # performs pre-processing for naive bayes algorithm 
    # - get df_nbayes with continous features properly discretized
    df_nbayes = nbayes_preprocess( df_floats.copy() )
    df_nbayes_TEST = nbayes_preprocess( df_floats_TEST.copy() )
    learn_naive_bayes( df_nbayes, df_nbayes_TEST )

    # performs pre-processing for decision tree algorithm 
    # same as naive bayes
    df_dtree = df_nbayes.copy() 
    df_dtree_TEST = df_nbayes_TEST.copy() 
    #learn_decision_tree( df_dtree, df_dtree_TEST )
    learn_decision_tree( df_floats, df_floats_TEST )     # performed better with continous features 

    # performs pre-processing for knn algorithm
    # will use discretized functions to increase clustering of datapoints
    learn_knn( df_nbayes, df_nbayes_TEST )               # performed better with discretized features 
    # also trying using continous feature values 
    df_knn = df_floats.copy() 
    df_knn_TEST = df_floats_TEST.copy() 
    #learn_knn( df_knn, df_knn_TEST )
    
    # performs pre-processing for logistic regression algorithm
    df_logit = logit_preprocess( df.copy() )
    df_logit_TEST = logit_preprocess( df_TEST.copy() )
    learn_logit( df_logit, df_logit_TEST )

