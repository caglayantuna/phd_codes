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
from functions.project_functions import *
import siamxt 
from sklearn.metrics import f1_score




def fp_marginal(imarray):
    r,c,b=imarray.shape
    imarraymin=imarray.max()-imarray
    Bc = np.ones((3, 3), dtype=bool)
    attributes = np.zeros([r, c, 6 * b], dtype=float)
    for i in range(b):
        mxt = siamxt.MaxTreeAlpha(imarray[:, :, i], Bc)
        mxtmin=siamxt.MaxTreeAlpha(imarraymin[:, :, i], Bc)
        mxt.areaOpen(100)
        attributes[:, :, i] = mxt.node_array[3,:][mxt.node_index]
        mxt.areaOpen(300)
        attributes[:, :, b + i] = mxt.node_array[3,:][mxt.node_index]
        mxt.areaOpen(500)
        attributes[:, :, 2 * b + i] =mxt.node_array[3,:][mxt.node_index]
        mxtmin.areaOpen(100)
        attributes[:, :, 3 * b+i] = mxtmin.node_array[3,:][mxtmin.node_index]
        mxtmin.areaOpen(300)
        attributes[:, :, 4*  b + i] = mxtmin.node_array[3,:][mxtmin.node_index]
        mxtmin.areaOpen(500)
        attributes[:, :, 5 * b + i] =mxtmin.node_array[3,:][mxtmin.node_index]
    image_array= np.concatenate((imarray,attributes),axis=2)
    return image_array
def fp_marginal_height(imarray):
    r,c,b=imarray.shape
    imarraymin=imarray.max()-imarray
    Bc = np.zeros((3, 3), dtype=bool)
    Bc[1,:]=1
    Bc[:,1]=1
    attributes = np.zeros([r, c, 6 * b], dtype=float)
    for i in range(b):
        mxt = siamxt.MaxTreeAlpha(imarray[:, :, i], Bc)
        mxtmin=siamxt.MaxTreeAlpha(imarraymin[:, :, i], Bc)
        mxt.vmax(10)
        attributes[:, :, i] = mxt.computeHeight()[mxt.node_index]
        mxt.vmax(20)
        attributes[:, :, b + i] = mxt.computeHeight()[mxt.node_index]
        mxt.vmax(30)
        attributes[:, :, 2 * b + i] =mxt.computeHeight()[mxt.node_index]
        mxtmin.vmax(10)
        attributes[:, :, 3 * b+i] = mxtmin.computeHeight()[mxtmin.node_index]
        mxtmin.vmax(20)
        attributes[:, :, 4*  b + i] = mxtmin.computeHeight()[mxtmin.node_index]
        mxtmin.vmax(30)
        attributes[:, :, 5 * b + i] =mxtmin.node_array[3,:][mxtmin.node_index]
    image_array= np.concatenate((imarray,attributes),axis=2)
    return image_array
def mean_gray_image(imarray):
    Bc = np.ones((3, 3), dtype=bool)
    r, c, b = tuple(imarray.shape)
    mean_img = np.zeros([r, c, b], dtype=float)
    for i in range(b):
        mxt = siamxt.MaxTreeAlpha(imarray[:, :, i], Bc)
        mean=mxt.computeNodeGrayAvg()
        mean_img[:, :, i] = mean[mxt.node_index]
        mean_img=np.array(mean_img,dtype=np.uint16)
    return mean_img
def volume_image(imarray):
    Bc = np.ones((3, 3), dtype=bool)
    r, c, b = tuple(imarray.shape)
    volume_img = np.zeros([r, c, b], dtype=float)
    for i in range(b):
        mxt = siamxt.MaxTreeAlpha(imarray[:, :, i], Bc)
        volume=mxt.computeVolume()
        volume_img[:, :, i] =volume[mxt.node_index]
        volume_img=np.array(volume_img,dtype=np.uint16)
    return volume_img
def height_image(imarray):
    Bc = np.ones((3, 3), dtype=bool)
    r, c, b = tuple(imarray.shape)
    height_img = np.zeros([r, c, b], dtype=float)
    for i in range(b):
        mxt = siamxt.MaxTreeAlpha(imarray[:, :, i], Bc)
        height=mxt.computeHeight()
        height_img[:, :, i] =height[mxt.node_index]
        height_img=np.array(height_img,dtype=np.uint16)
    return height_img
def area_image(imarray):
    Bc = np.ones((3, 3), dtype=bool)
    r, c, b = tuple(imarray.shape)
    area_img = np.zeros([r, c, b], dtype=float)
    for i in range(b):
        mxt = siamxt.MaxTreeAlpha(imarray[:, :, i], Bc)
        area=mxt.node_array[3,:]
        area_img[:, :, i] =area[mxt.node_index]
        #area_img=np.array(area_img,dtype=np.uint16)
    return area_img
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
    fptrain=fp_marginal_height(imarraytrain)#profile
    fptest=fp_marginal_height(imarraytest)#profile
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
    fptrain=fp_marginal_height(imarraytrain)#profile
    fptest=fp_marginal_height(imarraytest)#profile
    train, trainlabel=data_prepare(gttrain,fptrain)
    test, testlabel=data_prepare(gttest, fptest)
    rf3,f13=RFclassification(train, test, trainlabel, testlabel)
    rf4,f14=RFclassification(test, train, testlabel, trainlabel)
    #mlp3=MLPclassification(train,test,trainlabel,testlabel)
    #mlp4=MLPclassification(test, train, testlabel, trainlabel)       
    
    
    np.savetxt('fpvolumeth.txt', (rf1,f11,rf2,f12,rf3,f13,rf4,f14),fmt='%8s',delimiter=',')
    print(np.mean((rf1,rf2,rf3,rf4))) 
    print(np.mean((f11,f12,f13,f14),axis=0)) 

