# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 11:57:38 2019

@author: caglayan
"""

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
from functions.project_functions import *
import siamxt 
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from imageio import imwrite
def extinc_profile_single(im):
    
    Bc = np.zeros((3, 3), dtype=bool)
    Bc[1,:]=1
    Bc[:,1]=1
    r=3
    # Parameters used to compute the extinction profile
    nextrema =  [4192,2096,1048,512,256,128]
    H,W=im.shape
    # Array to store the profile
    Z = 2*len(nextrema)
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
      ep[:,:,i]= mxt1.getImage()
      
      mxt2 = mxt.clone()
      mxt2.extinctionFilter(extmin,n)
      ep[:,:,i+r]= max_value - mxt2.getImage()
      
      i+=1
    #eprofile= np.concatenate((im,ep),axis=2)
    return ep

def ep_marginal(imarray):
    r,c,b=imarray.shape
    
    ep=np.zeros([r,c,12*b])
    d=0
    for i in range(b):
        epsingle=extinc_profile_single(imarray[:,:,i])
        ep[:,:,d:d+12]=epsingle
        d+=12
    
    result= np.concatenate((imarray,ep),axis=2)
 
    return result

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
    imwrite('epthdordogne'+str(imindex)+'.png',res)    
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
    Image = geoimread(path+'/dataset/land_cover_mapping/gt_raster_dordogne_cp.tif')
    #Image = geoimread(path+'/dataset/dataset_charlotte/crop_labels.tif')

    gt = geoImToArray(Image)
    gt = gt.astype(np.uint8)

    Image = geoimread(path+'/dataset/land_cover_mapping/ndvi_dordogne_new.tif') 
    #Image = geoimread(path+'/dataset/dataset_charlotte/ndvi_brittany_gapfilled.tif')    

    #gt[maskarray!=0]=0
    imarray = geoImToArray(Image)
    r, c, b = imarray.shape
    imarray=ep_marginal(imarray)
    #vertical case
    c_half=int(np.round(c/2))
    imarraytrain=imarray[:,0:c_half-100,:] 
    gttrain=gt[:,0:c_half-100]
    imarraytest=imarray[:,c_half+100:-1,:] 
    gttest=gt[:,c_half+100:-1]

    train, trainlabel=data_prepare(gttrain,imarraytrain,3000)
    test, testlabel=data_prepare(gttest, imarraytest)
    rf1,f11=RFclassification(train, test, trainlabel, testlabel,1,gttest)
    train, trainlabel=data_prepare(gttrain,imarraytrain)
    test, testlabel=data_prepare(gttest, imarraytest,3000)
    rf2,f12=RFclassification(test, train, testlabel, trainlabel,2,gttrain)


    
    
    #horizontal calse case
    r_half=int(np.round(r/2))
    imarraytrain=imarray[0:r_half-100,:,:] 
    gttrain=gt[0:r_half-100,:]
    imarraytest=imarray[r_half+100:-1,:,:] 
    gttest=gt[r_half+100:-1:]

    train, trainlabel=data_prepare(gttrain,imarraytrain,3000)
    test, testlabel=data_prepare(gttest, imarraytest)
    rf3,f13=RFclassification(train, test, trainlabel, testlabel,3,gttest)
    train, trainlabel=data_prepare(gttrain,imarraytrain)
    test, testlabel=data_prepare(gttest, imarraytest,3000)
    rf4,f14=RFclassification(test, train, testlabel, trainlabel,4,gttrain)




    
    np.savetxt('epstarea_first.txt', (rf1,f11,rf2,f12,rf3,f13,rf4,f14),fmt='%8s',delimiter=',')
    
    print(np.mean((rf1,rf2,rf3,rf4))) 
    print(np.mean((f11,f12,f13,f14),axis=0)) 
    print(np.std((rf1,rf2,rf3,rf4))) 