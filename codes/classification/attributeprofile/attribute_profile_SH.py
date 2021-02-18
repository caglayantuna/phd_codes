# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 13:34:51 2019

@author: caglayan
"""

from project_functions import *
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import siamxt
def ap_single(im,r,c):
    immin=im.max()-im
    Bc = np.ones((3,3),dtype = bool)
    thresholds =  [100,200,300]
    mxt = siamxt.MaxTreeAlpha(im, Bc)
    mxtmin=siamxt.MaxTreeAlpha(immin, Bc)
    ap = np.zeros((r,c,6))
    i = 0
    #max tree profile
    for t in thresholds:
        mxt2 = mxt.clone()
        a= attribute_area_filter(mxt2, (t))
        a=np.reshape(a,[a.shape[0],a.shape[1],1])
        ap[:,:,0+i:1+i]=a
        i+=1
    #min tree profile
    i=0    
    for t in thresholds:
        mxt2 = mxtmin.clone()
        a=  im.max() -attribute_area_filter(mxt2, (t))
        a=np.reshape(a,[a.shape[0],a.shape[1],1])
        ap[:,:,3+i:4+i] = a
        i+=1 
    im=np.reshape(im,[im.shape[0],im.shape[1],1])
    approfile= np.concatenate((ap,im),axis=2)
    return approfile
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
    
    #train data preperation 
    gttrain=gt[:,0:450]  
    r, c = tuple(imsingletrain.shape)
    imsingletrain =ap_single(imsingletrain, r, c)
    aptrain = imsingletrain.astype(np.uint16)
    inputtrain= np.concatenate((aptrain,imarraytrain),axis=2)
    train, trainlabel=data_prepare(gttrain, inputtrain)

    #test data preperation 
    gttest=gt[:,450:-1]
    r, c = tuple(imsingletest.shape)
    imsingletest = ap_single(imsingletest, r, c)
    aptest = imsingletest.astype(np.uint16)
    inputtest= np.concatenate((aptest,imarraytest),axis=2)
    test, testlabel=data_prepare(gttest, inputtest)
    
    
    #classification
    RFclassification(train, test, trainlabel, testlabel)