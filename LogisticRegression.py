# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 15:43:43 2019

@author: nimitha jammula
"""
import numpy as np
pmName = 'models'
pm = __import__(pmName)
import sys
import pandas as pd

class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
    
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        # weights initialization
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient
            
            if(self.verbose == True and i % 10000 == 0):
                z = np.dot(X, self.theta)
                h = self.__sigmoid(z)
                print(f'loss: {self.__loss(h, y)} \t')
    
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
    
        return self.__sigmoid(np.dot(X, self.theta))
    
    def predict(self, X, threshold):
        return self.predict_prob(X) >= threshold
    
def logistic_main():
    
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
    

    
    model1 = LogisticRegression(lr=0.1, num_iter=1000)
    model1.fit(X_train_BOW, y_train_BOW)
    
    preds1 = model1.predict(X_test_BOW, 0.5)
    #print(preds1)
    # accuracy
    print('The Accuracy for Bag Of Words Model :',((preds1 == y_test_BOW).mean())*100,'%')
    
    
    frame_train_Ber = pd.DataFrame(bernoulli_train[1:]) 
    frame_test_Ber = pd.DataFrame(bernoulli_test[1:])
    
    
    X_train_Ber = frame_train_Ber.iloc[:,:-1]
    y_train_Ber = frame_train_Ber.iloc[:,-1]
    
    X_test_Ber = frame_test_Ber.iloc[:,:-1]
    y_test_Ber = frame_test_Ber.iloc[:,-1]
    

    
    model2 = LogisticRegression(lr=0.1, num_iter=1000)
    model2.fit(X_train_Ber, y_train_Ber)
    
    preds2 = model2.predict(X_test_Ber, 0.5)
    # accuracy
    print('The accuracy with Bernoulli Model :',((preds2 == y_test_Ber).mean())*100,'%')
    
    from sklearn.metrics import classification_report
    
    print('Classification Report for BOW')
    print(classification_report(y_test_BOW,preds1.round()))
    
    print('Classification Report for Bernoulli')
    print(classification_report(y_test_Ber,preds2.round()))

    
if __name__ == '__main__':
   logistic_main()