# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 10:23:29 2019

@author: caglayan
"""
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt


class patternspectra:
    def __init__(self,tree,attribute1,attribute2,attribute3):
        self.tree=tree
        
           
        self.attribute1=self.getattribute(attribute1)
        self.attribute2=self.getattribute(attribute2)
        self.attribute3=self.getattribute(attribute3)
        
    def getattribute(self,attribute):
        if attribute=='area':
            attribute=self.area_vector()
        elif attribute=='mean':
            attribute=self.mean_vector()
        elif attribute=='volume':
            attribute=self.volume_vector()
        if attribute=='ecc':
            attribute=self.eccentricity_vector()
        elif attribute=='rect':
            attribute=self.rectangularity_vector()
        elif attribute=='lasttime':
            attribute=self.lasttime()
        if attribute=='duration':
            attribute=self.duration() 
        return attribute
    def parent_level_difference(self):
        parent=self.tree.node_array[0,:]
        leveldiff=self.tree.node_array[2,:]-self.tree.node_array[2,parent]
        #leveldiff =leveldiff[tree.node_index]
        return leveldiff
    def area_vector(self):      
      area=self.tree.node_array[3,:]
        #area_img=np.array(area_img,dtype=np.uint16)
      return area
    def eccentricity_vector(self):      
      eccen=self.tree.computeEccentricity()[2]
        #area_img=np.array(area_img,dtype=np.uint16)
      return eccen
    def lasttime(self):      
      #time=tree.node_array[13,:]-tree.node_array[12,:]
      time=self.tree.node_array[13,:]
      return time
    def duration(self):      
      time=self.tree.node_array[13,:]-self.tree.node_array[12,:]
      return time
    def rectangularity_vector(self):      
      shape=self.tree.computeRR()
      return shape
    def mean_vector(self):
      mean=self.tree.computeNodeGrayAvg()
      return mean
    def volume_vector(self):
      volume=self.tree.computeVolume()
      return volume
    def area_weighted(self):
      multiply=np.multiply(self.parent_level_difference(),self.area_vector())
      return multiply
    def ps_statistics2d(self,a,b):
           ps=self.area_weighted()
           statistic,binedges1,binedges2,res=stats.binned_statistic_2d(self.attribute1,self.attribute2,ps,statistic='sum', bins=[a,b])
           return statistic,binedges1,binedges2,res
    def ps_statistics3d(self,a,b,c):
           ps=self.area_weighted()
           data=np.column_stack((self.attribute1,self.attribute2,self.attribute3)) 
           statistic,edges,binnumber=stats.binned_statistic_dd(data,ps,statistic='sum', bins=[a,b,c])
           self.edges=edges
           return statistic,edges,binnumber
    def getps(self,a,b,c):
        statistic,edges,binnumber=self.ps_statistics3d(a,b,c)
        statistic = statistic.astype(np.uint16)
        return statistic
    def getedges(self,a,b,c):
        statistic,edges,binnumber=self.ps_statistics3d(a,b,c)
        return edges
    def find_node(self,interval1,interval2,vector):
        node = np.where(np.logical_and(interval1<=vector,vector <= interval2))[0]
        return node
    def stability_threshold(self,node,threshold):
       sm = self.tree.computeStabilityMeasure()
       stability=sm[node]
       nodes = np.where(stability<threshold)[0]
       nodes=node[nodes]
       return nodes
    def sum_of_nodes(self,node):
       a=np.zeros(self.tree.shape)
       for i in range(node.size):
          a=a+self.tree.recConnectedComponent(node[i],bbonly = False) 
       a[a>0]=255
       return a
    def node_show(self,a):
      b=a.shape[2]
      for i in range(b):
       plt.figure()
       plt.imshow(a[:,:,i], cmap='gray')
       plt.show()       
    def pattern_spectra_nodes3d(self,bin1,bin2,bin3):
      edges=self.edges  
      interval1=edges[0][bin1]
      interval2=edges[0][bin1+1]
      nodes1=self.find_node(interval1,interval2,self.attribute1)
    
      interval1=edges[1][bin2]
      interval2=edges[1][bin2+1]
      nodes2=self.find_node(interval1,interval2,self.attribute2)
    
      interval1=edges[2][bin3]
      interval2=edges[2][bin3+1]
      node3=self.find_node(interval1,interval2,self.attribute3)
    
      node=(np.intersect1d(nodes1, nodes2))
      node=(np.intersect1d(node, node3))
    
      a=self.sum_of_nodes(node)
      self.node_show(a)
      return node
    def pattern_spectra_nodes2d(self,binedges1,binedges2,bin1,bin2):
      interval1=binedges1[bin1]
      interval2=binedges1[bin1+1]
      nodes1=find_node(interval1,interval2,self.attribute1)
    
      interval1=binedges2[bin2]
      interval2=binedges2[bin2+1]
      nodes2=find_node(interval1,interval2,self.attribute2)
      node=(np.intersect1d(nodes1, nodes2))
    
      a=sum_of_nodes(tree,node)
      node_show(a)
      return node