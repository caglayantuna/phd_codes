#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:39:53 2019

@author: caglayantuna
"""

import siamxt
from functions.project_functions import *
from scipy import stats
from scipy.misc import imsave
from itertools import combinations 


def parent_level_difference_nodes(tree,nodes):
    parent=tree.node_array[0,nodes]
    leveldiff=tree.node_array[2,nodes]-tree.node_array[2,parent]
    return leveldiff
def lowest_common_ancestor(tree,nodes):
    ancestors=tree.getAncestors(np.int(nodes[0]))
    #ancestors=ancestors[ancestors!=nodes[0]]
    for i in range(nodes.shape[0]):
        anc=tree.getAncestors(np.int(nodes[i]))
        ancestors=np.intersect1d(ancestors,anc)
    return np.max(ancestors)


    
def dasgupta_cost_pairnew(tree):
    index=tree.node_index
    x,y=index.shape
    totalcost=0
    padded = np.pad(index, pad_width=1, mode='constant', constant_values=index.max()+1)
    for i in range(x):
        for j in range(y):
              node=padded[i+1,j+1]
              node1=padded[i,j+1]
              node2=padded[i,j]
              node3=padded[i+1,j]
              node4=padded[i+2,j]
              neighbors=np.stack((node1,node2,node3,node4))
              neighbors=neighbors[neighbors!=index.max()+1]
              cost=0
              for n in range(len(neighbors)):
                  pairs=np.array((node,neighbors[n]))
                  lca=lowest_common_ancestor(tree,pairs)
                  weights=np.abs(tree.node_array[2,pairs[0]]-tree.node_array[2,pairs[1]])
                  #costtemp=tree.node_array[3,lca]*weights
                  leaves=tree.getDescendants(int(lca)).size
                  costtemp=leaves*weights
                  cost+=costtemp
              totalcost+=cost
    return totalcost
def dasgupta_cost_pair(tree):
    nodes=np.where(tree.node_array[1,:]==0)[0]
    nodes=nodes[nodes!=0]
    pairs=list(combinations(nodes,2))
    totalcost=0
    for i in range(len(pairs)):
         nodes=np.array(pairs[i])
         lca=lowest_common_ancestor(tree,nodes)
         #weights=np.abs(tree.node_array[2,nodes]-tree.node_array[2,lca])
         weights=np.abs(tree.node_array[2,nodes[0]]-tree.node_array[2,nodes[1]])
         cost=np.sum(tree.node_array[3,lca]*weights)
         totalcost+=cost
    return totalcost
def tsd_pair(tree):
    nodes=np.where(tree.node_array[1,:]==0)[0]
    nodes=nodes[nodes!=0]
    pairs=list(combinations(nodes,2))
    all_weights=np.sum(parent_level_difference(tree))
    p=np.zeros(len(pairs))
    q=np.zeros(len(pairs))
    for i in range(len(pairs)):
         nodepairs=np.array(pairs[i])
         lca=lowest_common_ancestor(tree,nodepairs)
         weights=np.abs(tree.node_array[2,nodepairs]-tree.node_array[2,lca])
         p[i]=np.sum(weights/all_weights)
         q[i]=np.product(weights/all_weights)
    index=np.where(p)
    res=np.sum(p[index] * np.log(p[index] / q[index]))
    #res=np.sum(np.sqrt(np.abs(p-q)))
    return res
def cost_for_all(tree1,tree2,imarray):
    costall=np.zeros([6,3])
    bc = np.ones((3,3), dtype=bool)
    for i in range(6):
        treepr=projected_tree(tree1,i)
        treepr2=projected_tree(tree2,i)
        costall[i,1]=dasgupta_cost_pairnew(treepr)
        costall[i,2]=dasgupta_cost_pairnew(treepr2)
        tree2d=siamxt.MaxTreeAlpha(imarray[:,:,i],bc)
        costall[i,0]=dasgupta_cost_pairnew(tree2d)
    return costall/costall.max()

def tsd(tree):
    nodes=np.where(tree.node_array[1,:]==0)[0]
    leaf_weights=parent_level_difference_nodes(tree,nodes)
    all_weights=np.sum(parent_level_difference(tree))
    p=leaf_weights/all_weights
    q=findSumofProduct(p)
    index=np.where(p)
    res=np.sum(p[index] * np.log(p[index] / q[index]))
    #res=np.sum(np.sqrt(np.abs(p-q)))
    return res
def tree_for_each(imarray):
    Bc = np.zeros((3,3), dtype = bool)
    Bc[1,:] = True
    Bc[:,1] = True
    r, c, b = tuple(imarray.shape)
    amount_of_nodes=np.zeros(b, dtype=float)
    leveldiff=[]
    for i in range(b):
        mxt = siamxt.MaxTreeAlpha(imarray[:, :, i], Bc)
        amount_of_nodes[i]=mxt.node_array.shape[1]
        leveldiff.append(parent_level_difference(mxt))
    return amount_of_nodes, leveldiff
def projected_tree(tree,t):
     node= np.unique(tree.node_index[:,:,t])
     #node=np.array(np.where(np.logical_and(tree.node_array[13,:]>= t, tree.node_array[12,:]<= t)))[0]
     nodesize=node.shape[0]
     tree_up=tree.clone()
     for i in range(nodesize):
        a=tree.recConnectedComponent(node[i],bbonly = False)
        area=np.count_nonzero(a[:,:,t])
        tree_up.node_array[3,node[i]]=area
     nodes=np.ones(tree_up.node_array.shape[1])
     nodes[node]=False
     nodes[tree_up.node_array[3,:]==0]=True
     nodes=np.array(nodes,dtype=int)
     nodes=np.array(nodes, dtype=bool)
     tree_up.prune(nodes)
     tree_up.node_index=tree_up.node_index[:,:,t]
     tree_up.node_array[0,0]=0

     return tree_up
def projected_tree_th(tree,imarray):
    r, c, b = tuple(imarray.shape)
    amount_of_nodes=np.zeros(b, dtype=float)
    leveldiff=[]

    for i in range(b):
        nodes= np.unique(tree.node_index[:,:,i])
        amount_of_nodes[i]=nodes.shape[0]
        leveldiff.append(parent_level_difference_nodes(tree,nodes))
    return amount_of_nodes,leveldiff
def node_signature(mxt,x,y):
    # Node corresponding to a regional maximum
    node = mxt.node_index[x,y]
    # Extracting area attribute from NA
    area = mxt.node_array[3,:]
    #levels=np.zeros([230])
    #signature=np.zeros([230])
    #for i in range(len(x)):
    #    levels1,signature1 =  mxt.getSignature(area, mxt.node_index[x[i],y[i]])
    #    levels+=levels1
    #    signature+=signature1
    #levels=levels/len(x)
    #signature=signature/len(x)
    
    #same lcation
    levels,signature=mxt.getSignature(area, mxt.node_index[x,y])
    
    #Gradient of the area signature
    gradient = signature[0:-1] - signature[1:]

    # Display area signature
    fig = plt.figure(figsize = (12,6))
    plt.subplot(121)
    plt.plot(levels,signature,'k')
    plt.grid()
    plt.xlabel("Gray-level")
    plt.ylabel("Area")
    #plt.title("Area signature")
    # Display gradient of the area signature
    plt.subplot(122)
    plt.grid()
    plt.plot(levels[0:-1],gradient,'k')
    plt.xlabel("Gray-level")
    plt.ylabel("Gradient")
    #plt.title("Gradient Curve")
    return levels,signature, gradient

Image = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/tide_observation/morbihanndvi.tif')
#Image = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/land_cover_mapping/ndvimergeddordogne.tif')
imarray=geoImToArray(Image)
imarray=np.array(imarray,dtype=np.uint8)
imarray=imarray[200:500,100:400,0:6]

#data pleiades   
#Image1 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrsmall1.tif')
#Image2=geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrsmall2.tif')
#Image3 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrsmall3.tif')
#Image4 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrsmall4.png')
#Image5 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrsmall5.png')
#Image6 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrsmall6.png')

#im prepare
#imarray1=geoImToArray(Image1)
#imarray2=geoImToArray(Image2)
#imarray3=geoImToArray(Image3)
#imarray4=geoImToArray(Image4)
#imarray5=geoImToArray(Image5)
#imarray6=geoImToArray(Image6)


#imarray=np.dstack((imarray1,imarray2,imarray3,imarray4,imarray5,imarray6))


bc = np.ones((3,3), dtype=bool)


Bc = np.ones((3,3,3), dtype = bool)

#max tree
tree3d= siamxt.MaxTreeAlpha(imarray, Bc)
treepr=projected_tree(tree3d,3)
tree2d=siamxt.MaxTreeAlpha(imarray[:,:,3],bc)


Bc = np.ones((3,3,7), dtype = bool)
#new max tree
treenew= siamxt.MaxTreeAlpha(imarray, Bc)
treeprnew=projected_tree(treenew,3)

#amount of nodes
#nodes2d,level1=tree_for_each(imarray)
#nodespr,level2=projected_tree_th(tree3d,imarray)
#nodesprnew,level3=projected_tree_th(treenew,imarray)


#Tree quality cost
#costpr=dasgupta_cost_pair(treepr)
#cost2d=dasgupta_cost_pair(tree2d)
#costprnew=dasgupta_cost_pair(treeprnew)

#costfor all
#costall=cost_for_all(tree3d,treenew,imarray)




#pleiades
#x=193
#y=187
#sentinel
x=21
y=213
#x,y=np.where(imarray[:,:,3]==255)

#NODE SIGNATURE
levels1,sign1,gr1=node_signature(tree2d,x,y)
levels2,sign2,gr2=node_signature(treepr,x,y)
levels3,sign3,gr3=node_signature(treeprnew,x,y)


#filtering
#im2d=attribute_area_filter(tree2d,20)
#impr=attribute_area_filter(treepr,20)
#imprnew=attribute_area_filter(treeprnew,20)

#changed1=imarray4-im2d
#changed1[changed1>0]=255
#changed2=imarray4-impr
#changed2[changed2>0]=255
#changed3=imarray4-imprnew
#changed3[changed3>0]=255

#count1=np.count_nonzero(changed1)
#count2=np.count_nonzero(changed2)
#count3=np.count_nonzero(changed3)