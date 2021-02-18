# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 15:24:35 2019

@author: caglayan
"""
from os.path import dirname, abspath
path = dirname(dirname(dirname(dirname(abspath(__file__)))))
import sys
sys.path.insert(0,path)

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
#from spatial_extent_func import *
from functions.project_functions import *
import siamxt 
from sklearn.metrics import f1_score


def fp_area(im):
    r,c,b=im.shape
    immin=im.max()-im
    Bc = np.ones((3,3,3),dtype = bool)
    thresholds =  [500,1000,1500]
    mxt = siamxt.MaxTreeAlpha(im, Bc)
    mxtmin=siamxt.MaxTreeAlpha(immin, Bc)
    fp = np.zeros((r,c,6*b))
    i = 0
    #max tree profile
    for t in thresholds:
        mxt1 = mxt.clone()
        mxt1.areaOpen(t)
        area=mxt1.node_array[3,:]
        fp[:,:,0+i:b+i] = area[mxt1.node_index]
        
        mxt2 = mxtmin.clone()
        mxt2.areaOpen(t)
        area=mxt2.node_array[3,:]
        fp[:,:,3*b+i:4*b+i] = area[mxt2.node_index]     
        i+=b  
    fprofile= np.concatenate((im,fp),axis=2)
    return fprofile
def fp_height(im):
    r,c,b=im.shape
    immin=im.max()-im
    Bc = np.zeros((3,3,3), dtype=bool)
    Bc[1,:,1]=1
    Bc[:,1,1]=1
    Bc[1,1,:]=1
    thresholds =  [10,20,30]
    mxt = siamxt.MaxTreeAlpha(im, Bc)
    mxtmin=siamxt.MaxTreeAlpha(immin, Bc)
    fp = np.zeros((r,c,6*b))
    i = 0
    #max tree profile
    for t in thresholds:
        mxt1 = mxt.clone()
        mxt1.hmax(t)
        fp[:,:,0+i:b+i] = mxt1.computeHeight()[mxt1.node_index]
        
        mxt2 = mxtmin.clone()
        mxt2.hmax(t)
        fp[:,:,3*b+i:4*b+i] = mxt2.computeHeight()[mxt2.node_index]     
        i+=b  
    fprofile= np.concatenate((im,fp),axis=2)
    return fprofile
def fp_volume(im):
    r,c,b=im.shape
    immin=im.max()-im
    Bc = np.zeros((3,3,3), dtype=bool)
    Bc[1,:,1]=1
    Bc[:,1,1]=1
    Bc[1,1,:]=1
    thresholds =  [10,20,30]
    mxt = siamxt.MaxTreeAlpha(im, Bc)
    mxtmin=siamxt.MaxTreeAlpha(immin, Bc)
    fp = np.zeros((r,c,6*b))
    i = 0
    #max tree profile
    for t in thresholds:
        mxt1 = mxt.clone()
        mxt1.vmax(t)
        fp[:,:,0+i:b+i] = mxt1.computeVolume()[mxt1.node_index]
        
        mxt2 = mxtmin.clone()
        mxt2.vmax(t)
        fp[:,:,3*b+i:4*b+i] = mxt2.computeVolume()[mxt2.node_index]     
        i+=b  
    fprofile= np.concatenate((im,fp),axis=2)
    return fprofile
def volume_cube(imarray):
    Bc = np.ones((3,3,3), dtype=bool)
    mxt = siamxt.MaxTreeAlpha(imarray, Bc)
    volume=mxt.computeVolume()
    volume_img =volume[mxt.node_index]
    return volume_img
def mean_gray_cube(imarray):
    Bc = np.ones((3,3,3), dtype=bool)
    mxt = siamxt.MaxTreeAlpha(imarray, Bc)
    mean=mxt.computeNodeGrayAvg()
    mean_img =mean[mxt.node_index]
    return mean_img
def height_gray_cube(imarray):
    Bc = np.ones((3,3,3), dtype=bool)
    mxt = siamxt.MaxTreeAlpha(imarray, Bc)
    mean=mxt.computeHeight()
    height_img =mean[mxt.node_index]
    return height_img

def data_prepare(gt,im):
     
    classes=np.unique(gt)
    classes=classes[classes!=0]
    
    data=[]
    datalabel=np.array([])
    
    for i in range(len(classes)):
        indices=np.where(gt == classes[i])
        newdata=im[indices[0],indices[1]]
        data.append(newdata)
        datalabel=np.append(datalabel,np.full((newdata.shape[0]), classes[i], dtype=np.uint8),axis=0)
    
    return np.uint64(np.vstack(data)),np.uint8(datalabel)
def RFclassification(train,test,trainlabel,testlabel):
    clf = RandomForestClassifier(n_estimators=500, max_features='sqrt',
				max_depth=25, min_samples_split=2,n_jobs=12)
    clf.fit(train, trainlabel)
    y_pred = clf.predict(test)
    print("Accuracy:", metrics.accuracy_score(testlabel, y_pred))
    print(f1_score(testlabel, y_pred, average=None)) 
    return metrics.accuracy_score(testlabel, y_pred),f1_score(testlabel, y_pred, average=None)
def MLPclassification(train,test,trainlabel,testlabel):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(train, trainlabel)
    y_pred = clf.predict(test)
    print("Accuracy:", metrics.accuracy_score(testlabel, y_pred))
    print(f1_score(testlabel, y_pred, average=None)) 
    return metrics.accuracy_score(testlabel, y_pred)
    
if __name__ == "__main__":
    Image = geoimread(path+'/dataset/land_cover_mapping/gtdordogne.tif')
    #Image = geoimread(path+'/dataset/dataset_charlotte/crop_labels.tif')

    gt = geoImToArray(Image)
    gt = gt.astype(np.uint8)

    Image = geoimread(path+'/dataset/land_cover_mapping/ndvimergeddordogne.tif') 
    #Image = geoimread(path+'/dataset/dataset_charlotte/T30UVU_Image_NDVI_60.tif')  
    imarray = geoImToArray(Image)

    r, c, b = imarray.shape
    
    #vertical case
    c_half=int(np.round(c/2))
    imarraytrain=imarray[:,0:c_half-100,:] 
    gttrain=gt[:,0:c_half-100]
    imarraytest=imarray[:,c_half+100:-1,:] 
    gttest=gt[:,c_half+100:-1]
    fptrain=fp_volume(imarraytrain)#profile
    fptest=fp_volume(imarraytest)#profile
    train, trainlabel=data_prepare(gttrain,fptrain)
    test, testlabel=data_prepare(gttest, fptest)
    rf1,f11=RFclassification(train, test, trainlabel, testlabel)
    rf2,f12=RFclassification(test, train, testlabel, trainlabel)
    #mlp1=MLPclassification(train,test,trainlabel,testlabel)
    #mlp2=MLPclassification(test, train, testlabel, trainlabel)
    
    
    #horizontal calse case
    b_half=int(np.round(b/2))
    imarraytrain=imarray[:,0:b_half-100,:] 
    gttrain=gt[:,0:b_half-100]
    imarraytest=imarray[:,b_half+100:-1,:] 
    gttest=gt[:,b_half+100:-1]
    fptrain=fp_volume(imarraytrain)#profile
    fptest=fp_volume(imarraytest)#profile
    train, trainlabel=data_prepare(gttrain,fptrain)
    test, testlabel=data_prepare(gttest, fptest)
    rf3,f13=RFclassification(train, test, trainlabel, testlabel)
    rf4,f14=RFclassification(test, train, testlabel, trainlabel)
    #mlp3=MLPclassification(train,test,trainlabel,testlabel)
    #mlp4=MLPclassification(test, train, testlabel, trainlabel)       
    
    
    np.savetxt('fpvolumesth.txt', (rf1,f11,rf2,f12,rf3,f13,rf4,f14),fmt='%8s',delimiter=',')
    
    print(np.mean((rf1,rf2,rf3,rf4))) 
    print(np.mean((f11,f12,f13,f14),axis=0)) 
    