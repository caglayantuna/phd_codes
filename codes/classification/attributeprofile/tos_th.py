#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 11:45:53 2020

@author: caglayantuna
"""

from os.path import dirname, abspath
path = dirname(dirname(dirname(dirname(abspath(__file__)))))
import sys
sys.path.insert(0,path)

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from functions.project_functions import *
import higra as hg 
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from imageio import imwrite
import siamxt 


def sap_marginal(imarray):
    r,c,b=imarray.shape
    imarraymin=imarray.max()-imarray
    Bc = np.ones((3, 3), dtype=bool)
    attributes = np.zeros([r, c, 6 * b], dtype=np.uint8)

    for i in range(b):
        tree, altitudes = hg.component_tree_tree_of_shapes_image2d(imarray[:,:,i])
        area = hg.attribute_area(tree)
        attributes[:, :, i] = hg.reconstruct_leaf_data(tree, altitudes, area < 100)
        attributes[:, :, b + i] = hg.reconstruct_leaf_data(tree, altitudes, area <500 )
        attributes[:, :, 2 * b + i] = hg.reconstruct_leaf_data(tree, altitudes, area < 2500)
        attributes[:, :, 3 * b + i] = hg.reconstruct_leaf_data(tree, altitudes, area < 12500)
        attributes[:, :, 4 * b + i] = hg.reconstruct_leaf_data(tree, altitudes, area < 60000)
        attributes[:, :, 5 * b + i] = hg.reconstruct_leaf_data(tree, altitudes, area < 300000)
    image_array= np.concatenate((imarray,attributes),axis=2)
    return image_array
def ap_cube_duration(im):
    r,c,b=im.shape
    immin=im.max()-im
    Bc = np.zeros((3,3,3), dtype=bool)
    Bc[1,:,1]=1
    Bc[:,1,1]=1
    Bc[1,1,:]=1
    thresholds =  [3,6,9]
    mxt = siamxt.MaxTreeAlpha(im, Bc)
    mxtmin=siamxt.MaxTreeAlpha(immin, Bc)
    ap = np.zeros((r,c,6*b),dtype=np.uint8)
    i = 0
    #max tree profile
    for t in thresholds:
        mxt1 = mxt.clone()
        ap[:,:,0+i:b+i] = duration_filter(mxt1, (t))
        mxt2 = mxtmin.clone()
        ap[:,:,3*b+i:4*b+i] = im.max()-duration_filter(mxt2, (t))
        
        i+=b  
    return ap
def sap_marginal_height(imarray):
    r,c,b=imarray.shape
    imarraymin=imarray.max()-imarray
    Bc = np.ones((3, 3), dtype=bool)
    attributes = np.zeros([r, c, 6 * b], dtype=float)

    for i in range(b):
        tree, altitudes = hg.component_tree_tree_of_shapes_image2d(imarray[:,:,i])
        area = hg.attribute_height(tree,altitudes)
        attributes[:, :, i] = hg.reconstruct_leaf_data(tree, altitudes, area < 20)
        attributes[:, :, b + i] = hg.reconstruct_leaf_data(tree, altitudes, area < 40)
        attributes[:, :, 2 * b + i] = hg.reconstruct_leaf_data(tree, altitudes, area < 60)
        attributes[:, :, 3 * b + i] = hg.reconstruct_leaf_data(tree, altitudes, area < 80)
        attributes[:, :, 4 * b + i] = hg.reconstruct_leaf_data(tree, altitudes, area < 100)
        attributes[:, :, 5 * b + i] = hg.reconstruct_leaf_data(tree, altitudes, area < 120)
    image_array= np.concatenate((imarray,attributes),axis=2)
    return image_array
def sap_marginal_compactness(imarray):
    r,c,b=imarray.shape
    imarraymin=imarray.max()-imarray
    Bc = np.ones((3, 3), dtype=bool)
    attributes = np.zeros([r, c, 6 * b], dtype=float)

    for i in range(b):
        tree, altitudes = hg.component_tree_tree_of_shapes_image2d(imarray[:,:,i])
        area = hg.attribute_compactness(tree,altitudes)
        attributes[:, :, i] = hg.reconstruct_leaf_data(tree, altitudes, area < 0.3)
        attributes[:, :, b + i] = hg.reconstruct_leaf_data(tree, altitudes, area < 0.4)
        attributes[:, :, 2 * b + i] = hg.reconstruct_leaf_data(tree, altitudes, area < 0.5)
        attributes[:, :, 3 * b + i] = hg.reconstruct_leaf_data(tree, altitudes, area < 0.6)
        attributes[:, :, 4 * b + i] = hg.reconstruct_leaf_data(tree, altitudes, area < 0.7)
        attributes[:, :, 5 * b + i] = hg.reconstruct_leaf_data(tree, altitudes, area < 0.8)       
    image_array= np.concatenate((imarray,attributes),axis=2)
    return image_array
def data_prepare(gt,im,samples=None):
    #apim=ap_cube_duration(im)
    classes=np.unique(gt)
    classes=classes[classes!=0]
    
    data=[]
    datalabel=np.array([])
    
    for i in range(len(classes)):
        indices=np.where(gt == classes[i])
        if samples is not None:
            indices=np.array(indices)
            indices=indices.T
            np.random.shuffle(indices)
            indices=indices.T
            B = np.array(indices)[:,:samples]
            indices=tuple(B) 
           
        newdata=im[indices[0],indices[1]]
        data.append(newdata)
        datalabel=np.append(datalabel,np.full((newdata.shape[0]), classes[i], dtype=np.uint8),axis=0)         
    return np.uint8(np.vstack(data)),np.uint8(datalabel)

def visualize(y_pred,imindex,ref):
    values=np.unique(ref)
    values=values[values!=0]
    indices=None
    for i in range(len(values)):
           newind = np.array(np.where(ref == values[i]))
           
           if indices is None:
               indices = newind
           else:
              indices = np.concatenate(([indices , newind ]), axis=1)
    indices=np.transpose(indices)    
    
    res=np.zeros((ref.shape[0],ref.shape[1],3))
    res[indices[y_pred==1][:,0],indices[y_pred==1][:,1],:]=[255,255,0]
    res[indices[y_pred==2][:,0],indices[y_pred==2][:,1],:]=[255,0,0]
    res[indices[y_pred==3][:,0],indices[y_pred==3][:,1],:]=[0,255,255]
    res[indices[y_pred==4][:,0],indices[y_pred==4][:,1],:]=[0,255,0]
    res[indices[y_pred==5][:,0],indices[y_pred==5][:,1],:]=[255,0,255]
    res[indices[y_pred==6][:,0],indices[y_pred==6][:,1],:]=[127,0,0]
    res[indices[y_pred==7][:,0],indices[y_pred==7][:,1],:]=[0,127,0]
    res[indices[y_pred==8][:,0],indices[y_pred==8][:,1],:]=[0,127,127]
    res[indices[y_pred==9][:,0],indices[y_pred==9][:,1],:]=[0,0,127]
    res[indices[y_pred==10][:,0],indices[y_pred==10][:,1],:]=[0,127,255]
    res[indices[y_pred==11][:,0],indices[y_pred==11][:,1],:]=[210,105,0]
    res[indices[y_pred==12][:,0],indices[y_pred==12][:,1],:]=[255,69,0]
    res= res.astype(np.uint8)
    imwrite('aptosthbrittany'+str(imindex)+'.png',res)     


def RFclassification(train,test,trainlabel,testlabel,i,ref):
    clf = RandomForestClassifier(n_estimators=100, max_features='sqrt',
				max_depth=25, min_samples_split=2,n_jobs=12)
    clf.fit(train, trainlabel)
    y_pred = clf.predict(test)
    visualize(y_pred,i,ref)
    print("Accuracy:", metrics.accuracy_score(testlabel, y_pred))
    print(f1_score(testlabel, y_pred, average=None)) 
    print(confusion_matrix(testlabel, y_pred))
    return metrics.accuracy_score(testlabel, y_pred),f1_score(testlabel, y_pred, average=None)

if __name__ == "__main__":
    #Image = geoimread(path+'/dataset/land_cover_mapping/gtrasterdordogne.tif')
    Image = geoimread(path+'/dataset/dataset_charlotte/crop_labels.tif')

    gt = geoImToArray(Image)
    gt = gt.astype(np.uint8)

    #Image = geoimread(path+'/dataset/land_cover_mapping/ndvi_dordogne_int.tif') 
    Image = geoimread(path+'/dataset/dataset_charlotte/ndvi_brittany_gapfilled.tif')   
    imarray = geoImToArray(Image)
    r, c, b = imarray.shape
    #imarray=np.concatenate((sap_marginal(imarray),ap_cube_duration(imarray)),axis=2)
    imarray=sap_marginal(imarray)
    
    #vertical case
    c_half=int(np.round(c/2))
    imarraytrain=imarray[:,0:c_half-100,:] 
    gttrain=gt[:,0:c_half-100]
    imarraytest=imarray[:,c_half+100:-1,:] 
    gttest=gt[:,c_half+100:-1]
    trsamples=10000
    train, trainlabel=data_prepare(gttrain,imarraytrain,trsamples)
    test, testlabel=data_prepare(gttest, imarraytest)
    rf1,f11=RFclassification(train, test, trainlabel, testlabel,1,gttest)
    train, trainlabel=data_prepare(gttrain,imarraytrain)
    test, testlabel=data_prepare(gttest, imarraytest,trsamples)
    rf2,f12=RFclassification(test, train, testlabel, trainlabel,2,gttrain)


    
    
    #horizontal calse case
    r_half=int(np.round(r/2))
    imarraytrain=imarray[0:r_half-100,:,:] 
    gttrain=gt[0:r_half-100,:]
    imarraytest=imarray[r_half+100:-1,:,:] 
    gttest=gt[r_half+100:-1:]

    train, trainlabel=data_prepare(gttrain,imarraytrain,trsamples)
    test, testlabel=data_prepare(gttest, imarraytest)
    rf3,f13=RFclassification(train, test, trainlabel, testlabel,3,gttest)
    train, trainlabel=data_prepare(gttrain,imarraytrain)
    test, testlabel=data_prepare(gttest, imarraytest,trsamples)
    rf4,f14=RFclassification(test, train, testlabel, trainlabel,4,gttrain)   
    


    
    np.savetxt('tosthareafirst.txt', (rf1,f11,rf2,f12,rf3,f13,rf4,f14),fmt='%8s',delimiter=',')
    
    print(np.mean((rf1,rf2,rf3,rf4))) 
    print(np.mean((f11,f12,f13,f14),axis=0)) 
    print(np.std((rf1,rf2,rf3,rf4))) 