# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 13:06:59 2019

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
from sklearn.metrics import confusion_matrix
from imageio import imwrite



def ap_cube(im):
    r,c,b=im.shape
    immin=im.max()-im
    Bc = np.zeros((3,3,3), dtype=bool)
    Bc[1,:,1]=1
    Bc[:,1,1]=1
    Bc[1,1,:]=1
    thresholds =  [100000,1000000,10000000,50000000,100000000,500000000]
    mxt = siamxt.MaxTreeAlpha(im, Bc)
    mxtmin=siamxt.MaxTreeAlpha(immin, Bc)
    ap = np.zeros((r,c,2*len(thresholds)*b),dtype=np.uint8)
    i = 0
    #max tree profile
    for t in thresholds:
        mxt1 = mxt.clone()
        ap[:,:,i:b+i] = attribute_area_filter(mxt1, (t))
        mxt2 = mxtmin.clone()
        ap[:,:,len(thresholds)*b+i:(len(thresholds)+1)*b+i] = im.max() -attribute_area_filter(mxt2, (t))
        
        i+=b  
    approfile= np.concatenate((im,ap),axis=2)
    return approfile
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
    #approfile= np.concatenate((im,ap),axis=2)
    return ap

def ap_cube_centroid(im):
    r,c,b=im.shape
    immin=im.max()-im
    Bc = np.zeros((3,3,3), dtype=bool)
    Bc[1,:,1]=1
    Bc[:,1,1]=1
    Bc[1,1,:]=1
    thresholds =  [1,4,7]
    mxt = siamxt.MaxTreeAlpha(im, Bc)
    mxtmin = siamxt.MaxTreeAlpha(immin, Bc)
    nodesmax=np.ones(mxt.node_array.shape[1])
    nodesmin=np.ones(mxtmin.node_array.shape[1])
    nodesmax=np.array(nodesmax, dtype=bool)
    nodesmin=np.array(nodesmin, dtype=bool)
    centroidmax=mxt.computeNodeCentroid()[:,2]
    centroidmin=mxtmin.computeNodeCentroid()[:,2]
    ap = np.zeros((r,c,6*b))
    i = 0
    #max tree profile
    for t in thresholds:
        mxt1 = mxt.clone()
        node=np.where(centroidmax<t)
        nodesmax[node]=False
        mxt1.prune(nodesmax)
        ap[:,:,0+i:b+i] =mxt1.getImage()
        mxt2 = mxtmin.clone()
        node=np.where(centroidmin<t)
        nodesmin[node]=False
        mxt2.prune(nodesmin)
        ap[:,:,3*b+i:4*b+i] = im.max() -mxt2.getImage()    
        i+=b  

    approfile= np.concatenate((im,ap),axis=2)
    return approfile

def ap_cube_height(im):
    r,c,b=im.shape
    immin=im.max()-im
    Bc = np.ones((3,3,3),dtype = bool)
    thresholds =  [50,100,150]
    mxt = siamxt.MaxTreeAlpha(im, Bc)
    mxtmin=siamxt.MaxTreeAlpha(immin, Bc)
    ap = np.zeros((r,c,2*len(thresholds)*b))
    i = 0
    #max tree profile
    for t in thresholds:
        mxt1 = mxt.clone()
        mxt1.hmax(t)
        ap[:,:,0+i:b+i] = mxt1.getImage()
        mxt2 = mxtmin.clone()
        mxt2.hmax(t)
        ap[:,:,len(thresholds)*b+i:(len(thresholds)+1)*b+i] = im.max() -mxt2.getImage()
        
        i+=b  
    approfile= np.concatenate((im,ap),axis=2)
    return approfile
def ap_cube_begin(im):
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
        ap[:,:,0+i:b+i] = begin_filter(mxt1, (t))
        mxt2 = mxtmin.clone()
        ap[:,:,3*b+i:4*b+i] = im.max()-begin_filter(mxt2, (t))
        
        i+=b  
    #approfile= np.concatenate((im,ap),axis=2)
    return ap
def ap_cube_end(im):
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
        ap[:,:,0+i:b+i] = end_filter(mxt1, (t))
        mxt2 = mxtmin.clone()
        ap[:,:,3*b+i:4*b+i] = im.max()-end_filter(mxt2, (t))
        
        i+=b  
    #approfile= np.concatenate((im,ap),axis=2)
    return ap
def ap_stability(im):
    r,c,b=im.shape
    immin=im.max()-im
    Bc = np.zeros((3,3,3), dtype=bool)
    Bc[1,:,1]=1
    Bc[:,1,1]=1
    Bc[1,1,:]=1
    thresholds =  [0.3,0.6,0.9]
    mxt = siamxt.MaxTreeAlpha(im, Bc)
    attribute_area_filter(mxt, 100)
    duration_filter(mxt, 9)
    mxtmin=siamxt.MaxTreeAlpha(immin, Bc)
    attribute_area_filter(mxtmin, 100)
    duration_filter(mxt, 9)
    stabilitymax=temp_stability_ratio(mxt)
    stabilitymin=temp_stability_ratio(mxtmin)
    nodesmax=np.ones(mxt.node_array.shape[1])
    nodesmin=np.ones(mxtmin.node_array.shape[1])
    nodesmax=np.array(nodesmax, dtype=bool)
    nodesmin=np.array(nodesmin, dtype=bool)

    ap = np.zeros((r,c,6*b))
    i = 0
    #max tree profile
    for t in thresholds:
        mxt1 = mxt.clone()
        node=np.where(stabilitymax<t)
        nodesmax[node]=False
        mxt1.prune(nodesmax)
        ap[:,:,0+i:b+i] =mxt1.getImage()
        mxt2 = mxtmin.clone()
        node=np.where(stabilitymin<t)
        nodesmin[node]=False
        mxt2.prune(nodesmin)
        ap[:,:,3*b+i:4*b+i] = im.max() -mxt2.getImage()    
        i+=b  
    approfile= np.concatenate((im,ap),axis=2)
    return approfile
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
    imwrite('apsthdordogne'+str(imindex)+'.png',res)    
def RFclassification(train,test,trainlabel,testlabel,i,ref):
    clf = RandomForestClassifier(n_estimators=100, max_features='sqrt',
				max_depth=25, min_samples_split=2,n_jobs=12)
    clf.fit(train, trainlabel)
    y_pred = clf.predict(test)
    #visualize(y_pred,i,ref)
    print("Accuracy:", metrics.accuracy_score(testlabel, y_pred))
    print(f1_score(testlabel, y_pred, average=None)) 
    print(confusion_matrix(testlabel, y_pred))
    return metrics.accuracy_score(testlabel, y_pred),f1_score(testlabel, y_pred, average=None)
def MLPclassification(train,test,trainlabel,testlabel):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(train, trainlabel)
    y_pred = clf.predict(test)
    print("Accuracy:", metrics.accuracy_score(testlabel, y_pred))
    print(f1_score(testlabel, y_pred, average=None)) 
    return metrics.accuracy_score(testlabel, y_pred)

if __name__ == "__main__":
    #Image = geoimread(path+'/dataset/land_cover_mapping/gt_raster_dordogne_cp.tif')
    Image = geoimread(path+'/dataset/dataset_charlotte/crop_labels.tif')

    gt = geoImToArray(Image)
    gt = gt.astype(np.uint8)

    #Image = geoimread(path+'/dataset/land_cover_mapping/ndvi_dordogne_int.tif') 
    Image = geoimread(path+'/dataset/dataset_charlotte/ndvi_brittany_gapfilled.tif')    

    #gt[maskarray!=0]=0
    imarray = geoImToArray(Image)
    r, c, b = imarray.shape
    #imarray=np.concatenate((ap_cube_duration(imarray),ap_cube_begin(imarray),ap_cube_end(imarray)),axis=2)
    imarray=ap_cube_centroid(imarray)
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




    
    np.savetxt('apstarea_first.txt', (rf1,f11,rf2,f12,rf3,f13,rf4,f14),fmt='%8s',delimiter=',')
    
    print(np.mean((rf1,rf2,rf3,rf4))) 
    print(np.mean((f11,f12,f13,f14),axis=0)) 
    print(np.std((rf1,rf2,rf3,rf4))) 