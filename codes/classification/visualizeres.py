#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 11:22:30 2020

@author: caglayantuna
"""

from imageio import imread, imwrite


a=imread('/Users/caglayantuna/Desktop/visualize results/justndvibrittany1.png')

b=imread('/Users/caglayantuna/Desktop/visualize results/ref_brittany1.png')

c=imread('/Users/caglayantuna/Desktop/visualize results/apthbrittany2.png')

d=imread('/Users/caglayantuna/Desktop/visualize results/mp2.png')

e=imread('/Users/caglayantuna/Desktop/visualize results/apthdurbrittany2.png')

f=imread('/Users/caglayantuna/Desktop/visualize results/aptosthdurbrittany2.png')


#imwrite('ndviclipped.png',a[5000:5500,2250:2500,:])
#imwrite('refclipped.png',b[5000:5500,2250:2500,:])

#imwrite('apthclipped.png',c[5000:5500,2250:2500,:])

#imwrite('mpclipped.png',d[5000:5500,2250:2500,:])

imwrite('apthdurclipped.png',e[5000:5500,2250:2500,:])

imwrite('tosthdurclipped.png',f[5000:5500,2250:2500,:])