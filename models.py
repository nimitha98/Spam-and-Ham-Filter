# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 15:25:19 2019

@author: nimitha jammula
"""

import sys
import os

'''
 code for preprocessing text 
 '''
def unique_words_set(input_filename):
    input_file = open(input_filename, 'r', errors = 'ignore')
    file_contents = input_file.read()
    input_file.close()
    word_list = file_contents.split()
    unique_words = set(word_list)
    return unique_words

'''
 for each email, getting the dictionary of words and its frequencies
 '''
def wordsCount(filePath):
    counts = dict()
    folder = os.path.dirname(filePath)
    #print(folder)
    label = 0
    if(folder.find('spam') == -1):
        label = 1
    
    input_file = open(filePath, 'r', errors = 'ignore')
    file_contents = input_file.read()
    input_file.close()
    word_list = file_contents.split()
    for word in word_list:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1
    counts['class-label'] = label
    #print(counts['class-label'])
    return counts

'''
 Takes the folder path as input and returns the list of file names in the folder
 '''
def files(folderPath):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(folderPath):
        #print(d)
        for file in f:
            if '.txt' in file:
                files.append(os.path.join(r, file))

    return files

'''
    Generates the BagOfWords model and Bernoulli model
'''
def generate_models(listOfDict, listOfFeatures):
    bagOfWords = []
    bernoulli = []
    length = len(listOfFeatures)
    
    bagOfWords.append(listOfFeatures)
    bernoulli.append(listOfFeatures)
    
    for dnry in listOfDict:
        rowBOW = [0] * length
        rowBL = [0] * length
        for i in range(length):
            if(listOfFeatures[i] in dnry.keys()):
                rowBOW[i] = dnry[listOfFeatures[i]]
                rowBL[i] = 1
            else:
                rowBOW[i] = 0
                rowBL[i] = 0
        bagOfWords.append(rowBOW)
        rowBL[length-1] = dnry[listOfFeatures[length-1]]
        bernoulli.append(rowBL) 
    return bagOfWords,bernoulli

'''
 Convert the email document to tokens of words 
    and returns unique words along with the dictionaries of frequencies
'''
def text_processing(filesList):
    uniqueWordSet = set()
    list_dict = list(dict())
    
    for f in filesList:
        words_set = unique_words_set(f)
        uniqueWordSet = words_set.union(uniqueWordSet)
        list_dict.append(wordsCount(f))
        
    words_list = list(uniqueWordSet)
    words_list.append('class-label')
    
    return list_dict, words_list

'''
    main method to take the input and give the bag of Words Model and Bernoulli Model from the given datasets 
'''        
def main():
    
    folderPath = sys.argv[1]
    files_list = files(folderPath)
    
    list_dict, words_list = text_processing(files_list)

    bagOfWords,bernoulli = generate_models(list_dict, words_list)
     
    print(bagOfWords)
    print(bernoulli)

    
    
if __name__ == '__main__':
    main()