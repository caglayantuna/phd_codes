#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 21:27:48 2020

@author: caglayantuna
"""
import numpy as np


import matplotlib.pyplot as plt
from collections import Counter

array=np.loadtxt('/Users/caglayantuna/Desktop/attribute_analysis/brittany/st_heightmaxbrittany.txt')


res=Counter(array)

plt.plot(array)