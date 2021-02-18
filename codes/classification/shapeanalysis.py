#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 14:53:26 2020

@author: caglayantuna
"""

from os.path import dirname, abspath
path = dirname(dirname(dirname(abspath(__file__))))
import sys
sys.path.insert(0,path)

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from functions.project_functions import *




    
file = "/Users/caglayantuna/Desktop/charlotte_dinodataset/SOURCE_VECTOR/TRAINING_SUP-4PXL-MS.shp"

driver = ogr.GetDriverByName("ESRI Shapefile")

dataSource = driver.Open(file, 0)

layer = dataSource.GetLayer()
featureCount = layer.GetFeatureCount()

layerDefinition=layer.GetLayerDefn()
schema = []
ldefn = layer.GetLayerDefn()
for n in range(ldefn.GetFieldCount()):
    fdefn = ldefn.GetFieldDefn(n)
    schema.append(fdefn.name)
    
#feature = layer.GetFeature(31)
classes=[]

for feature in layer:
    classes.append(feature.GetField("OCCID"))

deneme=np.array(classes)

for i in range(8):
    print (np.count_nonzero(deneme==i+1))
#for feature in layer:
 #   geom = feature.GetGeometryRef()
 #   #print (geom.Centroid().ExportToWkt())
