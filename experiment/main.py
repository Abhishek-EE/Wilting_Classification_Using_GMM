# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 22:23:15 2020

@author: abhis
"""
import sys
import pandas as pd
import argparse
import cv2
import numpy as np
from sklearn.decomposition import PCA

sys.path.append('../')
Data_Path = ('..//Data//')

import src.FeatureExtraction as feature
import src.Models.MixtureofGaussian.Mixture_of_Gaussian_Model as MOG
import src.activation as act
import time



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', action='store_true', help='Check data loading.')
    parser.add_argument('--trainMOG', action='store_true', help='Train Mixture of Gaussian')
    parser.add_argument('--trainFA', action='store_true',help='Factor Analysis')
    parser.add_argument('--traingaussian', action='store_true',help='Train the Gaussian model')
    parser.add_argument('--trainT', action='store_true',help='Train T dist')
    return parser.parse_args()

def load_data_train(i):
    train_data_path = Data_Path + 'TrainData//'
    dfData = pd.read_csv(Data_Path + 'TrainAnnotations.csv')
    df = dfData.loc[dfData['annotation']==i]
    
    df_copy = df.copy()
    train_set = df_copy.sample(frac=0.75, random_state=0)
    val_set = df_copy.drop(train_set.index)
    print(train_set.shape)
    print(val_set.shape)
    
    dfTrainData = image_to_feature(train_set,train_data_path)
    dfValidationData = image_to_feature(val_set,train_data_path)
    
#    dfFace = pd.read_csv(Data_Path + 'faceData.csv')
#    dfTrainFace = image_to_feature(dfFace[0:1000].copy())
#    dfTestFace = image_to_feature(dfFace[1520:1620].copy())
#    
#    dfNonFace = pd.read_csv(Data_Path + 'NonFaceData.csv')
#    dfTrainNF = image_to_feature(dfNonFace[0:1000].copy())
#    dfTestNF = image_to_feature(dfNonFace[1520:1620].copy())
#    
#    dfTrainFace.to_csv('predictions/trainF_feature.csv',index = False)
#    dfTrainNF.to_csv('predictions/trainNF_features.csv', index = False)
#    
#  
#    print('Number of training data for face: {}'.format(len(dfTrainFace)))
#    print('Number of testing data for face: {}'.format(len(dfTestFace))) 
#    print('Number of training data for Nonface: {}'.format(len(dfTrainNF)))
#    print('Number of testing data for Nonface: {}'.format(len(dfTestNF)))
#    dfTrain = pd

    return dfTrainData,dfValidationData

def image_to_feature(df,path):
    'takes a df and converts it to images'
    #f = np.zeros((1000))
    f = []
    #f = pd.DataFrame(np.zeros((1,921600)))
    #f = np.zeros((1,921600))
    
    start = time.time()
    for i in df.file_name:
        
        print('its happening')
        image = cv2.imread(path+i)
       
        #dimg = pd.DataFrame(feature.feature_extraction(img = image))
        dimg = feature.feature_extraction(img = image)
        f.append(dimg)
    #df['Feature'] = f  
    print(time.time()-start)
    imgarray = np.zeros((len(f),f[0].shape[1]))
    for i in range(len(f)):
        imgarray[i,:] = f[i]
    
    print(imgarray.shape)    
    df = pd.DataFrame(imgarray)
    return df

def PCA_to_processing(df):
    
    return df

def MOGapproach():
    dfTrainData , dfValData = load_data_train()
    

if __name__ == '__main__':
    
#    pimg = '..//Data//TrainData//018892.jpg'
#    
#    x = cv2.imread('..//Data//TainData//018892.jpg')
#    print(type(x))
    traindata, valdata = load_data_train(0)
    print(traindata.shape)
    pca = PCA(0.9)
    train_img_pca = pca.fit(traindata)
    print(type(train_img_pca))
    print(train_img_pca.shape)
    
    
    
    
#    FLAGS = get_args()
#  
#    if FLAGS.input:
#        load_data()
#    if FLAGS.trainMOG:
#       ' MixtureOfGaussian()'
#    if FLAGS.traingaussian:
#       ' gaussian()'
#    if FLAGS.trainFA:
#        'factoranalyser()'
#    if FLAGS.trainT:
#       ' StudnetT()'