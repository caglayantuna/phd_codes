import siamxt
import numpy as np
from functions.project_functions import *

def data_prepare(Image):
    imarray=geoImToArray(Image)
    imarray=np.where(imarray>1, 0, imarray)
    imarray=im_normalize(imarray,16)
    imarray=np.array(imarray,dtype=np.uint16)
    r,c,b=imarray.shape
    imarray=np.reshape(imarray,[r,c])
    return imarray
def min_tree(im,bc):
    imarray=im.max()-im
    mxt = siamxt.MaxTreeAlpha(imarray,bc)
    return mxt
def bigareainleaf(mxt,leafnodes,area):
     node = np.where(mxt.node_array[3,leafnodes] >= area)[0]
     return node
def find_node_interval(mxt,interval,area):
    leaf_interval = np.where(np.logical_and(interval<=mxt.node_array[2,:],mxt.node_array[2,:] <= 65535))[0]
    nodeininterval=bigareainleaf(mxt,leaf_interval,area)
    node=leaf_interval[nodeininterval]
    return node
def sum_of_nodes(mxt,node):
    a=np.zeros(mxt.shape)
    for i in range(node.size):
        a=a+mxt.recConnectedComponent(node[i],bbonly = False) 
    a[a>0]=255
    return a
def node_show(a):
    plt.figure()
    plt.imshow(a, cmap='gray')
    plt.show()
def create_change_map(im1,im2):
    result=im1-im2
    result= result.clip(min=0)
    result=np.array(result,dtype=np.uint8)
    plt.figure()
    plt.imshow(result, cmap='gray')
    plt.show()
    return result
def stability(mxt,node):
    sm = mxt.computeStabilityMeasure()
    stability_of_nodes=sm[node]
    return stability_of_nodes
def stability_threshold(tree,node,threshold):
    sm = tree.computeStabilityMeasure()
    stability=sm[node]
    nodes = np.where(stability<threshold)[0]
    nodes=node[nodes]
    return nodes
def accuracy(im):
    #Image = geoimread('grountruhrasterized.png')
    
    Image = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/flood_detection/gtrastersecond.png')
    imarray=geoImToArray(Image)
    gt=np.reshape(imarray,[imarray.shape[0],imarray.shape[1]])
    im[im==0]=1
    im=np.float64(im)
    res=im-gt
    TP=np.count_nonzero(res == 0)/np.count_nonzero(gt == 255)
    TN=np.count_nonzero(res == 1)/np.count_nonzero(gt == 0)
    FP=np.count_nonzero(res == 255)/np.count_nonzero(gt == 0)
    FN=np.count_nonzero(res == -254)/np.count_nonzero(gt == 255)
    return TP, TN, FP, FN
def parent_level_difference(tree):
    parent=tree.node_array[0,:]
    leveldiff=tree.node_array[2,:]-tree.node_array[2,parent]
    return leveldiff
def parent_area_difference(tree):
    parent=tree.node_array[0,:]
    areadiff=tree.node_array[3,parent]-tree.node_array[3,:]
    return areadiff


#level different  with parent and find the threshold
def level_difference_for_each_level(tree):
    leveldiff=parent_level_difference(tree)
    leveldifference=np.zeros([65535])
    for  i in range(65535):
        nodes = np.where(tree.node_array[2,:] == i)[0]
        leveldifference[i]=np.sum(leveldiff[nodes])
    return leveldifference
def level_threshold(tree):
    levdiff=level_difference_for_each_level(tree)
    threshold=np.argmax(levdiff)
    return threshold

#level different  with parent and find the threshold
def variance_for_each_level(tree):
    variance=tree.computeNodeGrayVar()
    varianceforlevel=np.zeros([65535])
    for  i in range(65535):
        nodes = np.where(tree.node_array[2,:] == i)[0]
        varianceforlevel[i]=np.sum(variance[nodes])
    return varianceforlevel
def level_threshold_with_variance(tree):
    levdiff=variance_for_each_level(tree)
    threshold=np.argmin(levdiff)
    return threshold

#next three fucntions for parent variance difference for each level and find threshold
def parent_variance_difference(tree):
    variance=tree.computeNodeGrayVar()
    parent=tree.node_array[0,:]
    variancediff=variance[parent]-variance
    return variancediff
def variance_difference_for_each_level(tree):
    variancediff=parent_variance_difference(tree)
    variancedifference=np.zeros([255])
    for  i in range(255):
        nodes = np.where(tree.node_array[2,:] == i)[0]
        variancedifference[i]=np.sum(variancediff[nodes])
    return variancedifference
def level_threshold_with_variance_differencre(tree):
    vardiff=variance_difference_for_each_level(tree)
    threshold=np.argmax(vardiff)
    return threshold
def imshow(result):
    plt.figure()
    plt.imshow(result, cmap='gray')
    plt.show()
def resultwithouttree(imarray1,imarray2,threshold):
    threshold=65535-threshold
    imarray1[imarray1<threshold]=255
    imarray1[imarray1>=threshold]=0
    imarray2[imarray2<threshold]=255
    imarray2[imarray2>=threshold]=0
    result=imarray1-imarray2
    result=result.astype(np.uint8)
    imshow(result)
   

#first dataset   
#Image1 = geoimread('images/sar1DD4clipnew.tif')
#Image2 = geoimread('images/sarB757clipnew.tif')
#Image2= geoimread('images/sarC36Aclipnew.tif')
#Image2 = geoimread('images/sard22dclipnew.tif')

#second dataset   
Image1 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/flood_detection/secondflooddata1.png')
Image2 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/flood_detection/secondflooddata2.png')
#Image2= geoimread('images2/sarC36Aclipsecond.tif')


imarray1=geoImToArray(Image1)
imarray2=geoImToArray(Image2)
#imarray3=data_prepare(Image3)
imarray=np.dstack([imarray1,imarray2])

Bc = np.zeros((3,3,3), dtype = bool)
Bc[1,1,:] = True
Bc[:,1,1] = True
Bc[1,:,1] = True

tree =min_tree(imarray, Bc)

threshold=level_threshold_with_variance_differencre(tree)
#node=find_node_interval(tree,threshold,100)
#node1=stability_threshold(tree,node,0.01)
sum=tree.getImage()
sum[sum<=threshold+9]=0
sum[sum>threshold+9]=255
#sum=sum_of_nodes(tree,node)

node_show(sum[:,:,0])

node_show(sum[:,:,1])

#node_show(sum[:,:,2])


res=create_change_map(sum[:,:,0],sum[:,:,1])
bc = np.zeros((3,3), dtype = bool)
bc[1,:] = True
bc[:,1] = True
mxt = siamxt.MaxTreeAlpha(res,bc)
res=attribute_area_filter(mxt,20)
node_show(res)
TP, TN, FP, FN=accuracy(res)

print((TP, TN, FP, FN )) 


Precision=(TP)/(TP+FP )
 
Recall=(TP)/(TP+FN )
 
F1=(2*Precision*Recall)/(Precision+Recall)

#are filtering
#bc = np.zeros((3,3), dtype = bool)
#bc[1,:] = True
#bc[:,1] = True
