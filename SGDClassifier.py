# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 00:06:25 2019

@author: nimit
"""

from sklearn.linear_model import SGDClassifier
import sys
import pandas as pd
from sklearn.metrics import accuracy_score
pmName = 'models'
pm = __import__(pmName)



def SGDClasify():
    
    trainPath = sys.argv[1]
    testPath = sys.argv[2]
    
    files_list_train = pm.files(trainPath)
    files_list_test = pm.files(testPath)
    
    list_dict_train, words_list_train = pm.text_processing(files_list_train)

    bagOfWords_train,bernoulli_train = pm.generate_models(list_dict_train, words_list_train)
    
    list_dict_test, words_list_test = pm.text_processing(files_list_test)
    
    bagOfWords_test,bernoulli_test = pm.generate_models(list_dict_test, words_list_train)
     
    frame_train_BOW = pd.DataFrame(bagOfWords_train[1:]) 
    frame_test_BOW = pd.DataFrame(bagOfWords_test[1:])
    
    
    X_train_BOW = frame_train_BOW.iloc[:,:-1]
    y_train_BOW = frame_train_BOW.iloc[:,-1]
    
    X_test_BOW = frame_test_BOW.iloc[:,:-1]
    y_test_BOW = frame_test_BOW.iloc[:,-1]
    

    frame_train_Ber = pd.DataFrame(bernoulli_train[1:]) 
    frame_test_Ber = pd.DataFrame(bernoulli_test[1:])
    
    
    X_train_Ber = frame_train_Ber.iloc[:,:-1]
    y_train_Ber = frame_train_Ber.iloc[:,-1]
    
    X_test_Ber = frame_test_Ber.iloc[:,:-1]
    y_test_Ber = frame_test_Ber.iloc[:,-1]
    


    #sklearn.linear_model.SGDClassifier(loss='hinge’, penalty=’l2’, alpha=0.0001,l1_ratio=0.15, fit_intercept=True,
    #                                   max_iter=None,tol=None, shuffle=True, verbose=0, epsilon=0.1,n_jobs=None,
    #                                   random_state=None, learning_rate='optimal’, eta0=0.0, power_t=0.5,
    #                                   early_stopping=False, validation_fraction=0.1,n_iter_no_change=5,
    #                                   class_weight=None,warm_start=False, average=False, n_iter=None)
    
    
    SGDClassifierModelBOW = SGDClassifier(max_iter = 1500)
    print(SGDClassifierModelBOW)
    #print(SGDClassifierModelBOW)
    SGDClassifierModelBOW.fit(X_train_BOW, y_train_BOW)
    
    
    
    #Calculating Prediction
    y_pred_BOW = SGDClassifierModelBOW.predict(X_test_BOW)
    
    accuracyBOW=accuracy_score(y_test_BOW,y_pred_BOW)
    print('The Accuracy with BOW :',accuracyBOW*100)
    
    SGDClassifierModelBer = SGDClassifier(max_iter = 1500)
    SGDClassifierModelBer.fit(X_train_Ber, y_train_Ber)
    
    
    
    #Calculating Prediction
    y_pred_Ber = SGDClassifierModelBer.predict(X_test_Ber)
    
    accuracyBer=accuracy_score(y_test_Ber,y_pred_Ber)
    print('The Accuracy with Berboulli Model :',accuracyBer*100)
    
    
    # Applying Grid Search to find the best model and the best parameters
    from sklearn.model_selection import GridSearchCV
    parameters = [{'alpha': [0.0001,0.001,0.01,0.1,1,10,100,1000]},]
    grid_search_BOW = GridSearchCV(estimator = SGDClassifierModelBOW,
                               param_grid = parameters,
                               scoring = 'accuracy',
                               cv = 10,
                               n_jobs = -1)
    grid_search_BOW = grid_search_BOW.fit(X_train_BOW, y_train_BOW)
    
    accuracyBOW = grid_search_BOW.best_score_
    print('The Accuracy with BOW with grid search :',accuracyBOW*100)
    
    
    grid_search_Ber = GridSearchCV(estimator = SGDClassifierModelBer,
                               param_grid = parameters,
                               scoring = 'accuracy',
                               cv = 10,
                               n_jobs = -1)
    grid_search_Ber = grid_search_Ber.fit(X_train_BOW, y_train_BOW)
    
    accuracyBer = grid_search_Ber.best_score_
    print('The Accuracy with Bernoulli with grid search :',accuracyBer*100)
    
    from sklearn.metrics import classification_report
    
    print('Classification Report for BOW')
    print(classification_report(y_test_BOW,y_pred_BOW.round()))
    
    print('Classification Report for Bernoulli')
    print(classification_report(y_test_Ber,y_pred_Ber.round()))
    
    
    
    
if __name__ == '__main__':
   SGDClasify()