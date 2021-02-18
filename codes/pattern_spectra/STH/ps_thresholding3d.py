#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 11:22:54 2019

@author: caglayantuna
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 00:14:35 2019

@author: caglayantuna
"""
import siamxt
from functions.project_functions import *
from scipy import stats
from scipy.misc import imsave


#synthetic data 
#Image1 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/synthetic_images/cd1.png')
#imarray1=geoImToArray(Image1)[:,:,3]
#imarray1= imarray1.astype(np.uint8)

#Image2 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/synthetic_images/cd2.png')
#imarray2=geoImToArray(Image2)[:,:,3]
#imarray2= imarray2.astype(np.uint8)

#Image3 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/synthetic_images/cd3.png')
#imarray3=geoImToArray(Image3)[:,:,3]
#imarray3= imarray3.astype(np.uint8)


#data pleiades   
Image1 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrsmall1.tif')
Image2=geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrsmall2.tif')
Image3 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrsmall3.tif')
Image4 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrsmall4.png')
Image5 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/pleiades/phrsmall5.png')
#data sentinel1   
#Image1 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/flood_detection/firstflooddata1.tif')
#Image2=geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/flood_detection/firstflooddata2.tif')
#Image3 = geoimread('/Users/caglayantuna/Desktop/Thesis_codes/dataset/flood_detection/firstflooddata3.tif')

#im prepare
imarray1=geoImToArray(Image1)
imarray2=geoImToArray(Image2)
imarray3=geoImToArray(Image3)
imarray4=geoImToArray(Image4)
imarray5=geoImToArray(Image5)

merged=np.dstack((imarray1,imarray2,imarray3,imarray4,imarray5))
if merged.max()>255:
        merged=np.array(merged,dtype=np.uint16)
else: 
         merged=np.array(merged,dtype=np.uint8)
#max tree
Bc = np.ones((3,3,3), dtype=bool) 
tree1 = siamxt.MaxTreeAlpha(merged, Bc)


#mIN tree
#tree1 = siamxt.MaxTreeAlpha(merged.max()-merged, Bc)
#tree1.node_array[2,:]=merged.max()-tree1.node_array[2,:]

#area weighted parent level difference
ps1=np.abs(area_weighted(tree1))
#ps1vector=np.reshape(ps1,imarray1.size)
#ps2vector=np.reshape(ps2,[30*31*6])

#area attribute
area1=area_vector(tree1)

#shape attribute
#shape1=eccentricity_vector(tree1)
#shape1=tree1.computeVolume()
shape1=mean_vector(tree1)
#shape1=tree1.computeStabilityMeasure()

#time attribute
#time1=temp_stability_range(tree1)
time1=tree1.computeNodeCentroid()[:,2]
#time1=time_duration(tree1)

#3D PATTERN SPECTRA
data1=np.column_stack((area1,shape1,time1)) 
#data2=np.column_stack((area2,shape2,time2)) 

#logaritmic edges
arealog= np.logspace(np.log10(np.min(area1)), np.log10(np.max(area1) ), num=50)
shapelog = np.logspace(np.log10(np.min(shape1)), np.log10(np.max(shape1) ), num=50)
res11,res12,res13=stats.binned_statistic_dd(data1,ps1,statistic='sum', bins=[arealog,shapelog,5])





#extract all bns
#bins=np.nonzero(res11>5000)#min
bins=np.nonzero(res11>2600)
bins=np.array(bins)

nodes=get_nodes_from_bins3d(tree1,res12[0],res12[1],res12[2],bins,area1, shape1,time1)
result=reconstruct_nodes(tree1,nodes)


node_show(merged)
#max
node_show(result)

#min
#node_show(merged.max()-result)
