# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:38:06 2019

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


def ep_area(im):
    
    Bc = np.zeros((3, 3,3), dtype=bool)
    Bc[1,:,1]=1
    Bc[:,1,1]=1
    Bc[1,1,:]=1

    r=3
    # Parameters used to compute the extinction profile
    nextrema =  [4096,1024,256]
    
    # Array to store the profile
    H,W,b = im.shape
    Z = 2*b*len(nextrema)
    ep = np.zeros((H,W,Z))
    #Min tree profile
    #Negating the image
    max_value =im.max()
    data_neg = (max_value - im)
 
    # Trees
    mxt = siamxt.MaxTreeAlpha(data_neg,Bc)
    mxtmax = siamxt.MaxTreeAlpha(im,Bc)

    # Area attribute extraction and computation of area extinction values
    #area = mxtmax.node_array[3,:]
    #extmax = mxtmax.computeExtinctionValues(area,"area")
    #area = mxt.node_array[3,:]
    #extmin = mxt.computeExtinctionValues(area,"area")  
    #height attribute
    #feature = mxtmax.computeHeight()
    #extmax = mxtmax.computeExtinctionValues(feature,"height")
    #feature = mxt.computeHeight()
    #extmin = mxt.computeExtinctionValues(feature,"height") 
    #Volume
    feature = mxtmax.computeVolume()
    extmax = mxtmax.computeExtinctionValues(feature,"volume")
    feature = mxt.computeVolume()
    extmin = mxt.computeExtinctionValues(feature,"volume")
    i = 0
    # Min-tree profile
    for n in nextrema:
      mxt1 = mxtmax.clone()
      mxt1.extinctionFilter(extmax,n)
      ep[:,:,i:b+i]= mxt1.getImage()
      
      mxt2 = mxt.clone()
      mxt2.extinctionFilter(extmin,n)
      ep[:,:,b*r+i:b*(r+1)+i]= max_value - mxt2.getImage()
      
      i+=b
    eprofile= np.concatenate((im,ep),axis=2)
    return eprofile

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
    
    return np.uint8(np.vstack(data)),np.uint8(datalabel)
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
    eptrain=ep_area(imarraytrain)#profile
    eptest=ep_area(imarraytest)#profile
    train, trainlabel=data_prepare(gttrain,eptrain)
    test, testlabel=data_prepare(gttest, eptest)
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
    eptrain=ep_area(imarraytrain)#profile
    eptest=ep_area(imarraytest)#profile
    train, trainlabel=data_prepare(gttrain,eptrain)
    test, testlabel=data_prepare(gttest, eptest)
    rf3,f13=RFclassification(train, test, trainlabel, testlabel)
    rf4,f14=RFclassification(test, train, testlabel, trainlabel)
    #mlp3=MLPclassification(train,test,trainlabel,testlabel)
    #mlp4=MLPclassification(test, train, testlabel, trainlabel)       
    
    
    np.savetxt('ep_st_volume.txt', (rf1,f11,rf2,f12,rf3,f13,rf4,f14),fmt='%8s',delimiter=',')
    
    
    print(np.mean((rf1,rf2,rf3,rf4))) 
    print(np.mean((f11,f12,f13,f14),axis=0))