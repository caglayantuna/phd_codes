#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 17:58:47 2019

@author: caglayantuna
"""

import siamxt
from functions.project_functions import *
import functions.patternspectra

Image1 = geoimread('psdata/2018ndvimerged.tif')

imarray1=geoImToArray(Image1)
imarray1=np.array(imarray1,dtype=np.uint16)

Bc = np.ones((3,3,3), dtype = bool)

tree = siamxt.MaxTreeAlpha(imarray1.max()-imarray1, Bc)

a=patternspectra.patternspectra(tree,'area','rect','duration')



c=a.getps(100,100,7)


a.pattern_spectra_nodes3d(1,27,6)