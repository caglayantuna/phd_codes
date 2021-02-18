#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 09:17:40 2019

@author: caglayantuna
"""

import siamxt
from functions.project_functions import *
from sklearn.metrics import mean_squared_error
def rmse_for_all(gt,res):  
    r,c,b=gt.shape
    result=np.zeros(b)
    for i in range(b):
        result[i]=mean_squared_error(gt[:,:,i], res[:,:,i])
    return result
def projected_tree(tree,t):
     node= np.unique(tree.node_index[:,:,t])
     nodesize=node.shape[0]
     tree_up=tree.clone()
     for i in range(nodesize):
        a=tree.recConnectedComponent(node[i],bbonly = False)
        area=np.count_nonzero(a[:,:,t])
        tree_up.node_array[3,node[i]]=area
     nodes=np.ones(tree_up.node_array.shape[1])
     nodes[node]=False
     nodes=np.array(nodes,dtype=int)
     nodes=np.array(nodes, dtype=bool)
     tree_up.prune(nodes)
     tree_up.node_index=tree_up.node_index[:,:,t]
     tree_up.node_array[0,0]=0
     return tree_up
def detect_moving_nodes(tree):
    nodes=np.where(tree.node_array[13,:]-tree.node_array[12,:]==0)
    nodespruned=np.ones(tree.node_array.shape[1])
    nodespruned[nodes]=False
    nodespruned[0]=False #root 
    nodespruned=np.array(nodespruned, dtype=bool)

    tree.prune(nodespruned)
    nodespruned=np.array(nodespruned, dtype=bool)
    image=tree.getImage()
    return image

#data pleiades   
Image1 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/orignew1.png')
Image2=geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/orignew2.png')
Image3 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/orignew3.png')
Image4 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/orignew4.png')
Image5 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/orignew5.png')
Image6 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/orignew6.png')
#im prepare
imarray1=geoImToArray(Image1)
imarray2=geoImToArray(Image2)
imarray3=geoImToArray(Image3)
imarray4=geoImToArray(Image4)
imarray5=geoImToArray(Image5)
imarray6=geoImToArray(Image6)

imarray1car=np.copy(imarray1)
imarray2car=np.copy(imarray2)
imarray3car=np.copy(imarray3)
imarray4car=np.copy(imarray4)
imarray5car=np.copy(imarray5)
imarray6car=np.copy(imarray6)

car=imarray1[13:19,256:266]
imarray1car[35:41,254:264]=car
imarray2car[130:140,62:68]=np.transpose(car)
imarray3car[220:230,62:68]=np.transpose(car)
imarray4car[173:179,183:193]=car
imarray5car[180:186,180:190]=car
imarray6car[180:186,13:23]=car

merged=np.dstack((imarray1car,imarray2car,imarray3car,imarray4car,imarray5car,imarray6car))
ref=np.dstack((imarray1,imarray2,imarray3,imarray4,imarray5,imarray6))


#max tree
Bc = np.ones((3,3,6), dtype=bool) 
treemax = siamxt.MaxTreeAlpha(merged, Bc)


tree1=projected_tree(treemax,0)
tree2=projected_tree(treemax,1)
tree3=projected_tree(treemax,2)
tree4=projected_tree(treemax,3)
tree5=projected_tree(treemax,4)
tree6=projected_tree(treemax,5)

filtered1=duration_filter(tree1,0)
filtered2=duration_filter(tree2,0)
filtered3=duration_filter(tree3,0)
filtered4=duration_filter(tree4,0)
filtered5=duration_filter(tree5,0)
filtered6=duration_filter(tree6,0)

filtered=np.dstack((filtered1,filtered2,filtered3,filtered4,filtered5,filtered6))

rmse_tree=rmse_for_all(ref,filtered)


