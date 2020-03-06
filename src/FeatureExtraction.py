# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 01:29:16 2020

@author: Abhishek Ranjan Singh
"""
import cv2
import numpy as np

def hist_equal(img,flag = False):
    '''takes image and an output space [0,L] as an input and gives an equalized image(in float) as output'''
    if flag:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    if len(img.shape) != 2:
        R, G, B = cv2.split(img)
        
        output1_R = cv2.equalizeHist(R)
        output1_G = cv2.equalizeHist(G)
        output1_B = cv2.equalizeHist(B)
        
        equ = cv2.merge((output1_R, output1_G, output1_B))
        return equ
    
    equ = cv2.equalizeHist(img)
    res = np.hstack((img,equ)) #stacking images side-by-side

    return res

def feature_extraction(img = None, flag = True, flag2 = False):
    '''Takes an image as an input and gives a 400x1 vector as output'''
    if type(img) != type(None):
#        if (img.shape[0],img.shape[1]) != Shape: #Resizing the image to match the length of feature vector
#            img = cv2.resize(img,Shape,interpolation = cv2.INTER_AREA)
        #Performing histogram equalization and also transfoming the image space to [0,1] as computational load will be reduced
        #
        if flag:
            img = hist_equal(img,flag2)
            
        return img.reshape(1,-1)
    return None
        
    

def main():
    lena = cv2.imread('20069.jpg') 
    img2 = cv2.imread('20110.jpg')
    
    lenagray = cv2.cvtColor(lena,cv2.COLOR_BGR2GRAY)
    x = feature_extraction(lena,0)
    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    #check whether the feature extraction works or not
    print(x.shape)
    print(type(x))
    
    cv2.imshow('Output',hist_equal(lenagray))
    cv2.imshow('img2',hist_equal(img2gray))
    cv2.waitKey(0)
    print('HelloWorld')

if __name__ == '__main__':
    main()
    