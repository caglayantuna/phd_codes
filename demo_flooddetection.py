#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 18:03:26 2019

@author: caglayantuna
"""

import siamxt
import numpy as np
from project_functions import *


#first dataset   
Image1 = geoimread('images/sar1DD4clipnew.tif')
Image2 = geoimread('images/sarB757clipnew.tif')
#Image2= geoimread('images/sarC36Aclipnew.tif')
#Image3 = geoimread('images/sard22dclipnew.tif')

#second dataset   
#Image1 = geoimread('images2/sar1DD4clipsecond')
#Image2 = geoimread('images2/sarB757clipsecond')
#Image2= geoimread('images2/sarC36Aclipsecond.tif')


imarray1=data_prepare(Image1)
imarray2=data_prepare(Image2)
#imarray3=data_prepare(Image3)


imarray=np.dstack([imarray1,imarray2])

Bc = np.zeros((3,3,3), dtype = bool)
Bc[1,1,:] = True
Bc[:,1,1] = True
Bc[1,:,1] = True

tree =min_tree(imarray, Bc)