#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 15:49:07 2019

@author: caglayantuna
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from spatial_extent_func import *
from project_functions import *
import siamxt 
from sklearn.model_selection import cross_val_score

def extinc_profile_single(imsingle):
    
    Bc = np.zeros((3,3),dtype = bool)
    Bc[1,:] = True
    Bc[:,1] = True
    
    # Parameters used to compute the extinction profile
    nextrema =  [int(2**jj) for jj in range(15)][::-1]
    
    # Array to store the profile
    H,W = imsingle.shape
    Z = 2*len(nextrema)+1
    ep = np.zeros((H,W,Z))
    #Min tree profile
    #Negating the image
    max_value =imsingle.max()
    data_neg = (max_value - imsingle)
 

    # Building the max-tree of the negated image, i.e. min-tree
    mxt = siamxt.MaxTreeAlpha(data_neg,Bc)

    # Area attribute extraction and computation of area extinction values
    area = mxt.node_array[3,:]
    ext = mxt.computeExtinctionValues(area,"area")
    
    # Height attribute extraction and computation of area extinction values
    #height = mxt.computeHeight()
    #ext = mxt.computeExtinctionValues(height,"height")
    
    # Volume attribute extraction and computation of area extinction values
    #volume = mxt.computeVolume()
    #ext = mxt.computeExtinctionValues(volume,"volume")
    
    # Min-tree profile
    i = len(nextrema) - 1
    for n in nextrema:
      mxt2 = mxt.clone()
      mxt2.extinctionFilter(ext,n)
      ep[:,:,i] = max_value - mxt2.getImage()
      i-=1
    i = len(nextrema)
    ep[:,:,i] = imsingle
    i +=1
    #Building the max-tree
    mxtmax = siamxt.MaxTreeAlpha(imsingle,Bc)

    # Area attribute extraction and computation of area extinction values
    area = mxtmax.node_array[3,:]
    ext = mxtmax.computeExtinctionValues(area,"area")
    
    # Height attribute extraction and computation of area extinction values
    #height = mxtmax.computeHeight()
    #ext = mxtmax.computeExtinctionValues(height,"height")
    
    # Volume attribute extraction and computation of area extinction values
    #volume = mxtmax.computeVolume()
    #Vext = mxtmax.computeExtinctionValues(volume,"volume")
    
    # Max-tree profile
    for n in nextrema:
      mxt2 = mxtmax.clone()
      mxt2.extinctionFilter(ext,n)
      ep[:,:,i] = mxt2.getImage()
      i+=1
    # Putting the original image in the profile    
    #image_array= np.concatenate((ep,imarray),axis=2)
    return ep

def data_prepare(gt,input):
    # class colors values
    #firstclass=[255,255,0]
    #secondclass = [255, 0, 255]
    #thirdclass = [255, 0,0]
    #forthtclass = [0, 102, 0]
    #fifthclass = [0, 255, 0]
    # class colors values for morbihan
    firstclass=2
    secondclass = 13
    thirdclass = 16
    forthtclass = 19
    fifthclass = 6

    #coordinates
    firstindices = np.where(gt == firstclass)
    secondindices = np.where(gt == secondclass)
    thirdindices = np.where(gt == thirdclass)
    forthindices = np.where(gt == forthtclass)
    fifthindices = np.where(gt == fifthclass)

    #data
    testone=input[firstindices[0],firstindices[1],:]
    testtwo = input[secondindices[0], secondindices[1],:]
    testthree = input[thirdindices[0],thirdindices[1],:]
    testfour = input[forthindices[0],forthindices[1],:]
    testfive = input[fifthindices[0],fifthindices[1],:]

    test = np.concatenate((testone,testtwo,testthree,testfour,testfive))

    # test labels
    testlabelone = np.full((testone.shape[0]), 1, dtype=np.uint8)
    testlabeltwo = np.full((testtwo.shape[0]), 2, dtype=np.uint8)
    testlabelthree = np.full((testthree.shape[0]), 3, dtype=np.uint8)
    testlabelfour = np.full((testfour.shape[0]), 4 ,dtype=np.uint8)
    testlabelfive = np.full((testfive.shape[0]), 5, dtype=np.uint8)

    testlabel = np.concatenate((testlabelone,testlabeltwo,testlabelthree,testlabelfour,testlabelfive))

    return test,testlabel
def RFclassification(train,test,trainlabel,testlabel):
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(train, trainlabel)

    y_pred = clf.predict(test)
    print("Accuracy:", metrics.accuracy_score(testlabel, y_pred))
    
    #cross validation
    scores = cross_val_score(clf, test, testlabel, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
if __name__ == "__main__":
    Image = geoimread('data/morbihangt.tif')
    #Image = geoimread('data/2018ndvimergedgt.tif')
    #Image = geoimread('data/kalideoscutten.tif')

    gt = geoImToArray(Image)
    gt = gt.astype(np.uint8)
    gt=gt[:,:,0]
    Image = geoimread('data/morbihanndvi.tif')    
    #Image = geoimread('data/2018ndvimerged.tif')
    #Image = geoimread('data/kaldieoscutten.tif')
    imarray = geoImToArray(Image)
    
    #train and test
    imarraytrain= imarray[:,0:450,:] 
    imarraytest=imarray[:,450:-1,:]
    gttrain=gt[:,0:450]  
    gttest=gt[:,450:-1]
    #spatial hierarchy train
    imsingletrain=lexsortnew(imarraytrain)
    imsingletrain=im_normalize(imsingletrain,16)
    #imsingletrain=meanSITS(imarraytrain)
    #imsingletrain=dtw_image(imarraytrain)
    imsingletrain= imsingletrain.astype(np.uint16)
    
    #spatial hierarchy test
    imsingletest=lexsortnew(imarraytest)
    imsingletest=im_normalize(imsingletest,16)
    #imsingletest=meanSITS(imarraytest)
    #imsingletest=dtw_image(imarraytest)
    imsingletest= imsingletest.astype(np.uint16)
    
    #extinction profile
    eptrain=extinc_profile_single(imsingletrain)
    eptest=extinc_profile_single(imsingletest)
    #data prepare
    inputtrain= np.concatenate((eptrain,imarraytrain),axis=2)
    inputtest= np.concatenate((eptest,imarraytest),axis=2)

    train,trainlabel=data_prepare(gttrain, inputtrain)
    test,testlabel=data_prepare(gttest, inputtest)

    #classification

    RFclassification(train, test, trainlabel, testlabel)    