#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 17:20:07 2019

@author: caglayantuna
"""

import higra as hg
from functions.project_functions import *


#data pleiades   
Image1 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/orignew1.png')

#im prepare
image=geoImToArray(Image1)

size = image.shape[:2]


graph = hg.get_4_adjacency_graph(size)

edge_weights = hg.weight_graph(graph, image, hg.WeightFunction.L2)

gradient = hg.graph_4_adjacency_2_khalimsky(graph, edge_weights)
tree, altitudes = hg.constrained_connectivity_hierarchy_strong_connection(graph, edge_weights)


leaf_graph=hg.CptHierarchy.get_leaf_graph(tree)
cost=hg.dasgupta_cost(tree, edge_weights, leaf_graph)

tsd=hg.tree_sampling_divergence(tree, edge_weights, leaf_graph)