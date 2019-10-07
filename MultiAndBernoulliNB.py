# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 20:38:58 2019

@author: nimitha jammula
"""

'''
    creating the instance of a module in the same folder to access the methods declared in that module
'''
pmName = 'models'
pm = __import__(pmName)

'''
    Importing the libraries
'''
import os
import sys
import math

'''
    Training the Multinomial Naive Byes model By  taking the class List and Documents list
'''

def trainMultinomialNB(Class, Documents):
    V = ExtractVocabulary(Documents)
    N = len(Documents)
    #print(N)
    
    prior = []
    Tct = {}
    len_V = len(V)
    countprob = {}
    for t in V:
        countprob[t] = [0,0]
    for c in Class:
        #print(c)
        textc, Nc = ConcatenateTextAndCountDocsInClass(Documents, c)
        #print(Nc)
        len_textc = len(textc.split())
        den = len_textc + len_V
        prior.append(Nc/N)
        for t in V:
            Tct[t] = [0,0]
        for t in V:
            Tct[t][c] = textc.count(t)
        for t in V:
            countprob[t][c] = (Tct[t][c]+1)/ den
    return V,prior,countprob
 
'''
    Getting the list of vocabulary (unique words) from the list of documents
'''           
def ExtractVocabulary(Documents):
    x, V = pm.text_processing(Documents)
    V.remove('class-label')
    return V

'''
    appends the data of all documents from the list of docs into a string from a particular class
    And return the count of the documents in the class
'''
def ConcatenateTextAndCountDocsInClass(Documents, c):
    t = 'ham'
    if(c == 0):
        t = 'spam'
    count = 0
    s = ''
    for d in Documents:
        folder = os.path.dirname(d)
        #print(folder)
        if(folder.find(t) != -1):
            count= count+1
            with open(d, 'r', errors = 'ignore') as file:
                s = s + file.read()
            
    return s,count

'''
    Takes the class, Vocabulary, Conditional probabilities and document and apply Multinomial Naive Bayes
    Calculates the scores of each class and finally returns the class that gives the maximum score
'''
def ApplyMultinomialNB(C,V, prior, countprob, doc):
    W = extractWordsFromDocument(V,doc)
    score = [0]*len(C)
    for c in C:
        score[c] = math.log(prior[c])
        for t in W:
            score[c] = score[c] + math.log(countprob[t][c])
    return score.index(max(score))

'''
    Gives the list of words from the documents that are alos present in the existing Vocabulary
'''

def extractWordsFromDocument(V,d):
    with open(d, 'r', errors = 'ignore') as file:
        data= file.read() 
    W = data.split()
    d = []
    for w in W:
        if(w in V):
            d.append(w)
    #print(W)
    return d 

'''
    Calculate the metrix parameters like f1-score, precision, recall and accuracy by diving the count of accurately classified documents by total count of documents
'''
def findAccuracy(C,V, prior, countprob, documents):
    true_positive_m = 0
    true_negative_m = 0
    false_positive_m = 0
    false_negative_m = 0
    
    true_positive_b = 0
    true_negative_b = 0
    false_positive_b = 0
    false_negative_b = 0
    

    doc_count = len(documents)
    for d in documents:
        folder = os.path.dirname(d)
        cm = ApplyMultinomialNB(C,V,prior,countprob,d)
        cb = ApplyBernoulliNB(C,V,prior,countprob,d)
        labelm = 'ham'
        labelb = 'ham'
        if(cm == 0):
            labelm = 'spam'
        if(cb == 0):
            labelb = 'spam'
            
        if(labelm == 'ham' and folder.find('ham') != -1):
            true_positive_m = true_positive_m +1
        elif(labelm == 'ham' and folder.find('spam') != -1):
            false_negative_m = false_negative_m +1
        elif(labelm == 'spam' and folder.find('spam') != -1):
            true_negative_m = true_negative_m +1
        else:
            false_positive_m = false_positive_m+1
            
        if(labelb == 'ham' and folder.find('ham') != -1):
            true_positive_b = true_positive_b +1
        elif(labelb == 'ham' and folder.find('spam') != -1):
            false_negative_b = false_negative_b +1
        elif(labelb == 'spam' and folder.find('spam') != -1):
            true_negative_b = true_negative_b +1
        else:
            false_positive_b = false_positive_b + 1
            
    precision_m = true_positive_m/(true_positive_m + false_positive_m)
    recall_m = true_positive_m/(true_positive_m + false_negative_m)
    f1_score_m = 2*precision_m*recall_m/(precision_m + recall_m)
    
    precision_b = true_positive_b/(true_positive_b + false_positive_b)
    recall_b = true_positive_b/(true_positive_b + false_negative_b)
    f1_score_b = 2*precision_b*recall_b/(precision_b + recall_b)
    
    print('The precision with Multi NB :', precision_m)
    print('The recall with Multi NB :', recall_m)
    print('The F1 Score with multi NB :', f1_score_m)
    
    print('The precision with Bernoulli NB :', precision_b)
    print('The recall with Bernoulli  NB :', recall_b)
    print('The F1 Score with Bernoulli  NB :', f1_score_b)
    
    return (((true_positive_m+true_negative_m)/doc_count)*100), (((true_positive_b+true_negative_b)/doc_count)*100)

'''
    Takes the class, Vocabulary,priors, Conditional probabilities and document and apply Bernoulli Naive Bayes
    Calculates the scores of each class and finally returns the class that gives the maximum score
'''
def ApplyBernoulliNB(C,V, prior, countprob, doc):
    W = extractWordsFromDocument(V,doc)
    score = [0]*len(C)
    for c in C:
        score[c] = math.log(prior[c])
        for t in V:
            if(t in W):
                score[c] = score[c] + math.log(countprob[t][c])
            else:
                score[c] = score[c] + math.log(1-countprob[t][c])
    return score.index(max(score))

'''
    takes the list of documents and the class and returns the count of documemts in that class
'''

def CountDocsInClass(Documents, c):
    t = 'ham'
    docInClass = []
    if(c == 0):
        t = 'spam'
    count = 0
    for d in Documents:
        folder = os.path.dirname(d)
        #print(folder)
        if(folder.find(t) != -1):
            count= count+1
            docInClass.append(d)
    return docInClass,count
'''
    Generate the bernoulli model with the list of dictionaries and list of features from the Vocabulary
'''
def generate_bernoulli_model(listOfDict, listOfFeatures):
    bernoulli = []
    length = len(listOfFeatures)
    
    bernoulli.append(listOfFeatures)
    
    for dnry in listOfDict:
        rowBL = [0] * length
        for i in range(length):
            if(listOfFeatures[i] in dnry.keys()):
                rowBL[i] = 1
            else:
                rowBL[i] = 0

        bernoulli.append(rowBL) 
    return bernoulli

'''
    Training the Bernoulli Naive Byes model By  taking the class List and Documents list
'''
def trainBernoulliNB(Class, Documents):
    V = ExtractVocabulary(Documents)
    N = len(Documents)
    #print(N)
    
    prior = []
    Tct = {}
    countprob = {}
    for t in V:
        countprob[t] = [0,0]
    for c in Class:
        #print(c)
        docInClass, Nc = CountDocsInClass(Documents, c)
        #print(Nc)
        den = Nc+2
        feature_matrix = findFeatureMatrix(docInClass, V)
        prior.append(Nc/N)
        for t in V:
            Tct[t] = [0,0]
        for t in V:
            Tct[t][c] = countDocsInClassContainingTerm(feature_matrix, t)
        for t in V:
            countprob[t][c] = (Tct[t][c]+1)/ den
    return V,prior,countprob

'''
    Takes in the feature matrix and term and returns the count of documents containing the term
'''

def countDocsInClassContainingTerm(feature_matrix, t):
    count = 0    
    inn = feature_matrix[0].index(t)
    for i in range(1,len(feature_matrix)):
        count = count+ feature_matrix[i][inn]
        #print(count)
    return count

'''
    Generate the feature matrix from the list of documents and vocabulary
'''
def findFeatureMatrix(docInClass, V):
    list_dict, words_list = pm.text_processing(docInClass)

    bernoulli = generate_bernoulli_model(list_dict, V)
    
    return bernoulli

'''
     main method to take the traing and testing datasets and print the accuracy for Multinomial Naive Bayes and Bernoulli Naive Bayes
'''           
def multi_main():
    
    trainPath = sys.argv[1]
    trainDocuments = pm.files(trainPath)
    labels = [0,1]

    testPath = sys.argv[2]
    testDocuments = pm.files(testPath)
    
    
    
    Vm, priorm, countprobm = trainMultinomialNB(labels,trainDocuments)
    
    #print(countprob)

    V, prior, countprob = trainBernoulliNB(labels,trainDocuments)
    x,y = findAccuracy(labels ,V, prior, countprob, testDocuments)
    print('The Accuracy with Multinomial Algorithm is :', x)
    print('The Accuracy with Bernoulli Algorithm is :', y)
    
if __name__ == '__main__':
    multi_main()