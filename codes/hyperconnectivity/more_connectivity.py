#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 16:09:51 2019

@author: caglayantuna
"""

import siamxt
from functions.project_functions import *
from scipy import stats
from scipy.misc import imsave



Image = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/tide_observation/morbihanndvi.tif')
imarray=geoImToArray(Image)
imarray=np.array(imarray,dtype=np.uint16)

Bc = np.ones((3,3,7), dtype = bool)



#max tree
#tree= siamxt.MaxTreeAlpha(imarray, Bc)

#min tree
tree1 = siamxt.MaxTreeAlpha(imarray.max()-imarray, Bc)


tempstabil=temp_stability_nodes(tree1)
