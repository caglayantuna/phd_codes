#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 13:10:46 2020

@author: caglayantuna
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 11:54:10 2020

@author: caglayantuna
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 08:33:35 2020

@author: caglayantuna
"""

import siamxt
from functions.project_functions import *
from imageio import imsave

from sklearn.metrics import mean_squared_error
from scipy.interpolate import interp1d


def rmse_for_all(im,res):  
    gt=np.copy(im)
    r,c,b=gt.shape
    result=np.zeros(b)
    for i in range(b):
        result[i]=np.sqrt(mean_squared_error(gt[:,:,i], res[:,:,i]))
    return result
def create_gt():
   Image1 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/synthetic_images/gtsent1.png')
   Image2 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/synthetic_images/gtsent2.png')
   Image3 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/synthetic_images/gtsent3.png')
   Image4 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/synthetic_images/gtsent4.png')
   Image5 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/synthetic_images/gtsent5.png')
   Image6 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/synthetic_images/gtsent6.png')

   #im prepare
   imarray1=geoImToArray(Image1)[:,:,3]
   imarray2=geoImToArray(Image2)[:,:,3]
   imarray3=geoImToArray(Image3)[:,:,3]
   imarray4=geoImToArray(Image4)[:,:,3]
   imarray5=geoImToArray(Image5)[:,:,3]
   imarray6=geoImToArray(Image6)[:,:,3]
   merged=np.dstack((imarray1,imarray2,imarray3,imarray4,imarray5,imarray6))
   return merged

def filter_per_single(merged,identified,t):
    im=merged[:,:,t]
    bc = np.ones((3,3), dtype=bool) 
    tree = siamxt.MaxTreeAlpha(im, bc)
    coord=np.where(identified[:,:,t]>0)
    nodespruned=tree.node_index[coord]
    nodespruned=np.unique(nodespruned)
    nodes=np.zeros(tree.node_array.shape[1])
    nodes[nodespruned]=True
    nodes=np.array(nodes,dtype=int)
    nodes=np.array(nodes, dtype=bool)
    tree.prune(nodes)  
    res=tree.getImage()
    return res

Image1 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/synthetic_images/outlier1.png')
Image2 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/synthetic_images/outlier2.png')
Image3 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/synthetic_images/outlier3.png')
Image4 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/synthetic_images/outlier4.png')
Image5 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/synthetic_images/outlier5.png')
Image6 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/synthetic_images/outlier6.png')

#im prepare
imarray1=geoImToArray(Image1)[:,:,3]
imarray2=geoImToArray(Image2)[:,:,3]
imarray3=geoImToArray(Image3)[:,:,3]
imarray4=geoImToArray(Image4)[:,:,3]
imarray5=geoImToArray(Image5)[:,:,3]
imarray6=geoImToArray(Image6)[:,:,3]

merged=np.dstack((imarray1,imarray2,imarray3,imarray4,imarray5,imarray6))


#max tree
Bc = np.ones((3,3,6), dtype=bool) 
treemax = siamxt.MaxTreeAlpha(merged, Bc)

#filtering
filtered=duration_filter(treemax,0)
identified=merged-filtered

res1=filter_per_single(merged,identified,0)
res2=filter_per_single(merged,identified,1)
res3=filter_per_single(merged,identified,2)
res4=filter_per_single(merged,identified,3)
res5=filter_per_single(merged,identified,4)
res6=filter_per_single(merged,identified,5)


result=np.dstack((res1,res2,res3,res4,res5,res6))
#node_show(filtered)
#node_show(result)

gt=create_gt()
rmse_tree=rmse_for_all(gt,result)