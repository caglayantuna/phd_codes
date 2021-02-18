import gdal
import numpy as np
from osgeo import ogr
import matplotlib.pyplot as plt
#from scipy.misc import imread as imread
import siamxt
import ot
from scipy import stats
from datetime import datetime

def geoimread(filename):
    Image = gdal.Open(filename)
    return Image
def NDVI_to_array(Image):
    row=Image.RasterYSize
    col = Image.RasterXSize
    band = Image.RasterCount

    imarray = np.zeros([row, col, band], dtype=np.float)
    for b in range(band):
        imarray[:, :, b] = np.array(Image.GetRasterBand(b + 1).ReadAsArray())

    return imarray
def geoImToArray(Image):
    row=Image.RasterYSize
    col = Image.RasterXSize
    band = Image.RasterCount

    imarray = np.zeros([row, col, band], dtype=np.float)
    for b in range(band):
        imarray[:, :, b] = np.array(Image.GetRasterBand(b + 1).ReadAsArray())
    if band==1:
       imarray=np.reshape(imarray,[row,col])
    if imarray.max()>255:
        imarray=np.array(imarray,dtype=np.uint16)
    else: 
         imarray=np.array(imarray,dtype=np.uint8)
    return imarray
def acquisiton_time(Image):
    d=Image.GetMetadata()
    str=d.get('TIFFTAG_DATETIME')
    date=datetime.fromisoformat(str).timestamp()
    return date
def meanSITS(imarray):
    r, c, d = imarray.shape
    mean_img = np.zeros([r, c], dtype=float)
    for x in range(r):
        for y in range(c):
            mean = np.mean(imarray[x, y, :])
            mean_img[x, y] = mean
    #mean_img=np.array(mean_img,dtype=np.uint16)
    return mean_img
def stdSITS(imarray):
    r, c, d = imarray.shape
    std_img = np.zeros([r, c], dtype=float)
    for x in range(r):
        for y in range(c):
            std = np.std(imarray[x, y, :])
            std_img[x, y] = std

    return std_img
def distanceSITS(imarray):
    r, c, d = imarray.shape
    dist_img = np.zeros([r, c], dtype=float)
    for x in range(r):
        for y in range(c):
            dist=np.max(imarray[x, y, :])-np.min(imarray[x, y, :])
            dist_img[x, y] = dist
    return dist_img
def quartilecoeff(imarray):
    r, c, d = imarray.shape
    quartilecoefficient = np.zeros([r, c], dtype=float)
    for x in range(r):
        for y in range(c):
            Q1=np.percentile(imarray[x, y, :],25)
            Q3 = np.percentile(imarray[x, y, :], 75)
            quartilecoefficient[x, y]=(Q3-Q1)/(Q3+Q1)
    return quartilecoefficient
def entropySITS(imarray):
    r, c, d = imarray.shape
    entropy_image = np.zeros([r, c], dtype=float)
    for x in range(r):
        for y in range(c):
            value, counts = np.unique(imarray[x, y, :],return_counts=True)
            norm_counts = counts / counts.sum()
            entropy = -(norm_counts * np.log(norm_counts))
            entropy_image[x,y]=entropy
    return entropy_image
def interqrange(imarray):
    r, c, d = imarray.shape
    dist_img = np.zeros([r, c], dtype=float)
    for x in range(r):
        for y in range(c):
            q75, q25 = np.percentile(imarray[x,y], [75, 25])
            iqr = q75 - q25
            dist_img[x, y] = iqr
    return dist_img
def shape_feature_read(filename,feature):
    file = ogr.Open(filename)
    shape = file.GetLayer(0)
    feature = shape.GetFeature(feature)
    first = feature.ExportToJson()
    print (first)
    return first
def writenodeindex(mxt,data,filename):
    result = mxt.node_index
    result = np.array(result, dtype=np.float32)

    [cols, rows] = result.shape
    trans = data.GetGeoTransform()
    proj = data.GetProjection()
    outfile = filename
    outdriver = gdal.GetDriverByName("GTiff")
    outdata = outdriver.Create(str(outfile), rows, cols, 1, gdal.GDT_Float32)
    # Write the array to the file, which is the original array in this example
    outdata.SetProjection(proj)
    outdata.SetGeoTransform(trans)
    outdata.GetRasterBand(1).WriteArray(result)
    outdata.FlushCache()

def max_tree_example(node,mxt):
    A = mxt.node_array[3, :]

    anc = mxt.getAncestors(int(node))[::-1]

    area = A[anc]

    gradient = area[0:-1] - area[1:]
    indexes = np.argsort(gradient)

    max1 = indexes[-1]
    anc_max1 = [anc[max1], anc[max1 + 1]]

    result = mxt.recConnectedComponent(anc_max1[0])

    result = result + 1
    result = result - 1
    result = result * 255
    return result
def attribute_area_filter(mxt,area):
    mxt.areaOpen(area)
    ao = mxt.getImage()
    return ao
def array_to_raster(array,data,filename):
    rows = []
    cols = []
    band = []
    if   array.ndim==2:
          [cols, rows]= array.shape
          band=1
    elif array.ndim==3:
         [cols, rows, band] = array.shape

    trans = data.GetGeoTransform()
    proj = data.GetProjection()
    outfile = filename
    outdriver = gdal.GetDriverByName("GTiff")
    outdata = outdriver.Create(str(outfile), rows, cols, band, gdal.GDT_Int16)
    # Write the array to the file, which is the original array in this example
    outdata.SetProjection(proj)
    outdata.SetGeoTransform(trans)
    if array.ndim == 2:
        outdata.GetRasterBand(1).WriteArray(array)
    elif array.ndim == 3:
      for b in range(band):
        outdata.GetRasterBand(b + 1).WriteArray(array[:, :,b])
    outdata.FlushCache()
def max_tree_signature(filename,node):

    a = imread(filename)
    Bc = np.ones((3, 3), dtype=bool)
    mxt = siamxt.MaxTreeAlpha(a, Bc)
    result = max_tree_example(node, mxt)
    return result
def im_normalize(image,bit):
    image=np.array(image,dtype=float)
    max=image.max()

    imagenew=(image/max)*(2**bit-1)
    return imagenew
def data_prepare(Image):
    imarray=geoImToArray(Image)
    imarray=np.where(imarray>1, 0, imarray)
    imarray=im_normalize(imarray,16)
    imarray=np.array(imarray,dtype=np.uint16)
    r,c,b=imarray.shape
    imarray=np.reshape(imarray,[r,c])
    return imarray
def min_tree(im,bc):
    imarray=im.max()-im
    mxt = siamxt.MaxTreeAlpha(imarray,bc)
    return mxt
def bigareainleaf(mxt,leafnodes,area):
     node = np.where(mxt.node_array[3,leafnodes] >= area)[0]
     return node
def find_node_interval(mxt,interval,area):
    leaf_interval = np.where(np.logical_and(interval<=mxt.node_array[2,:],mxt.node_array[2,:] <= 65535))[0]
    nodeininterval=bigareainleaf(mxt,leaf_interval,area)
    node=leaf_interval[nodeininterval]
    return node

def create_change_map(im1,im2):
    result=im1-im2
    result= result.clip(min=0)
    result=np.array(result,dtype=np.uint8)
    plt.figure()
    plt.imshow(result, cmap='gray')
    plt.show()
    return result
def accuracy(im):
    Image = geoimread('grountruhrasterized.png')
    #Image = geoimread('images2/gtrastersecond.png')
    imarray=geoImToArray(Image)
    gt=np.reshape(imarray,[imarray.shape[0],imarray.shape[1]])
    im[im==0]=1
    res=im-gt
    TP=np.count_nonzero(res == 0)/np.count_nonzero(gt == 255)
    TN=np.count_nonzero(res == 1)/np.count_nonzero(gt == 0)
    FP=np.count_nonzero(res == 255)/np.count_nonzero(gt == 0)
    FN=np.count_nonzero(res == -254)/np.count_nonzero(gt == 255)
    return TP, TN, FP, FN
def parent_level_difference(tree):
    parent=tree.node_array[0,:]
    leveldiff=tree.node_array[2,:]-tree.node_array[2,parent]
    #leveldiff =leveldiff[tree.node_index]
    return leveldiff
def mean_vector(tree):
    mean=tree.computeNodeGrayAvg()
    return mean

def area_weighted(tree):
    multiply=np.multiply(parent_level_difference(tree),area_vector(tree) )
    return multiply
def area_vector(tree):      
      area=tree.node_array[3,:]
        #area_img=np.array(area_img,dtype=np.uint16)
      return area
def eccentricity_vector(tree):      
      eccen=tree.computeEccentricity()[2]
      where_are_NaNs = np.isnan(eccen)
      eccen[where_are_NaNs] = 0
      return eccen
def im_show(a):
    plt.figure()
    plt.imshow(a, cmap='gray')
    plt.show()
def im_edit(im,tree):
    c=tree.getImage()
    c[im==0]=0
    Bc = np.ones((3,3), dtype=bool)
    treelast = siamxt.MaxTreeAlpha(c.max()-c, Bc)
    
    treelast.node_array[3,:]=treelast.node_array[2,:].max()-treelast.node_array[2,:]
    treelast.areaOpen(0)
    ao = treelast.getImage()
    c=c.max()-ao
    return c

def find_node(interval1,interval2,vector):
    node = np.where(np.logical_and(interval1<=vector,vector <= interval2))[0]
    return node
def stability_threshold(tree,node,threshold):
    #lower
    sm = tree.computeStabilityMeasure()
    stability=sm[node]
    nodes = np.where(stability<threshold)[0]
    nodes=node[nodes]
    return nodes
def stability_threshold_interval(stability,nodes,threshold1,threshold2):
    #interval
    node = np.where(np.logical_and(threshold1<=stability,stability <= threshold2))[0] 
    node=nodes[node]
    return node
def sum_of_nodes(mxt,node):
    a=np.zeros(mxt.shape)
    for i in range(node.size):
        a=a+mxt.recConnectedComponent(node[i],bbonly = False) 
    a[a>0]=255
    c=mxt.getImage()
    c[a==0]=0
    return c

def node_show(a):
    b=a.shape[2]
    for i in range(b):
       plt.figure()
       plt.imshow(a[:,:,i], cmap='gray',vmin=0,vmax=255)
       plt.show()
    
def pattern_spectra_nodes(tree,binedges1,binedges2,bin1,bin2,attributevector1, attributevector2):
    interval1=binedges1[bin1]
    interval2=binedges1[bin1+1]
    nodes1=find_node(interval1,interval2,attributevector1)
    
    interval1=binedges2[bin2]
    interval2=binedges2[bin2+1]
    nodes2=find_node(interval1,interval2,attributevector2)
    node=(np.intersect1d(nodes1, nodes2))
    
    a=sum_of_nodes(tree,node)
    #node_show(a)
    return a
def get_nodes_from_bins(binedges1,binedges2,bins,attributevector1, attributevector2):
    nodes=[]
    for i in range(len(bins[0,:])):
        bins=np.array(bins)
        x,y=bins[:,i]
        interval1=binedges1[x]
        interval2=binedges1[x+1]
        nodes1=find_node(interval1,interval2,attributevector1)
    
        interval1=binedges2[y]
        interval2=binedges2[y+1]
        nodes2=find_node(interval1,interval2,attributevector2)
        
        node=(np.intersect1d(nodes1, nodes2))
        nodes=np.hstack((nodes,node))
    return nodes
def get_nodes_from_bins3d(tree,binedges1,binedges2,binedges3,bins,attributevector1, attributevector2, attributevector3):
    nodes=[]
    for i in range(len(bins[0,:])):
        bins=np.array(bins)
        x,y,z=bins[:,i]
        interval1=binedges1[x]
        interval2=binedges1[x+1]
        nodes1=find_node(interval1,interval2,attributevector1)
    
        interval1=binedges2[y]
        interval2=binedges2[y+1]
        nodes2=find_node(interval1,interval2,attributevector2)
        
        interval1=binedges3[z]
        interval2=binedges3[z+1]
        node3=find_node(interval1,interval2,attributevector3)
    
        node=(np.intersect1d(nodes1, nodes2))
        node=(np.intersect1d(node, node3))       
        
        nodes=np.hstack((nodes,node))
    return nodes
def reconstruct_nodes(tree,node):
    newtree=tree.clone()
    node=np.array(node,dtype=int)
    nodes=np.ones(newtree.node_array.shape[1])
    nodes[node]=False
    #nodes=np.array(nodes,dtype=int)
    nodes=np.array(nodes, dtype=bool)

    a=newtree.prune(nodes)
    res=a.getImage()
    return res

def im_prepare(im):
    r,c,b=im.shape
    im=np.reshape(im,[r,c])
    if im.max()>255:
        im=np.array(im,dtype=np.uint16)
    else: 
         im=np.array(im,dtype=np.uint8)
    return im

def temp_stability_range(tree):
    #with some existed nodes
    nodesize=tree.node_array.shape[1]
    tempstability=np.zeros(nodesize)
    r,c,b=tree.shape
    area=np.zeros(b)
    for i in range(nodesize):
        a=tree.recConnectedComponent(i,bbonly = False)
        for j in range(b):
            area[j]=np.count_nonzero(a[:,:,j])
        areanew=area[area!=0]
        tempstability[i]=np.max(areanew)-np.min(areanew)
        tempstability[i]=tempstability[i]/np.max(areanew)
        #tempstability[i]=np.std((area))
    return tempstability
def temp_stability_std(tree):
    #with some existed nodes
    nodesize=tree.node_array.shape[1]
    tempstability=np.zeros(nodesize)
    r,c,b=tree.shape
    area=np.zeros(b)
    for i in range(nodesize):
        a=tree.recConnectedComponent(i,bbonly = False)
        for j in range(b):
            area[j]=np.count_nonzero(a[:,:,j])
        areanew=area[area!=0]
        tempstability[i]=np.std(areanew)/areanew.max()
        #tempstability[i]=np.std((area))
    return tempstability
def temp_stability_ratio(tree):
    #with some existed nodes
    nodesize=tree.node_array.shape[1]

    r,c,b=tree.shape
    area=np.zeros([nodesize,b])
    tempstability=np.zeros(nodesize)
    for i in range(nodesize):
        a=tree.recConnectedComponent(i,bbonly = False)
        for j in range(b):
            area[i,j]=np.count_nonzero(a[:,:,j])
    for i in range(nodesize):
        for j in range(b-1):
            if area[i,j+1]==0 or area[i,j]==0:
                c=0
            else:
                c=min(area[i,j],area[i,j+1])/max(area[i,j],area[i,j+1])
            tempstability[i]+=c

    tempstability=tempstability/(b-1)
    return tempstability
def stability_filter(tree,t):
     stability=temp_stability_ratio(tree)
     node=np.where(stability<t)
     nodes=np.ones(tree.node_array.shape[1])
     nodes[node]=False
     nodes=np.array(nodes, dtype=bool)
     tree.prune(nodes)
     result=tree.getImage()
     return result 
def time_end(tree):      
      time=tree.node_array[13,:]
      return time
def time_begin(tree):      
      time=tree.node_array[12,:]
      return time
def time_duration(tree):      
      time=tree.node_array[13,:]-tree.node_array[12,:] 
      return time 
def pattern_spectra_nodes3d(tree,binedges1,binedges2,binedges3,bin1,bin2,bin3,attributevector1, attributevector2, attributevector3):
    interval1=binedges1[bin1]
    interval2=binedges1[bin1+1]
    nodes1=find_node(interval1,interval2,attributevector1)
    
    interval1=binedges2[bin2]
    interval2=binedges2[bin2+1]
    nodes2=find_node(interval1,interval2,attributevector2)
    
    interval1=binedges3[bin3]
    interval2=binedges3[bin3+1]
    node3=find_node(interval1,interval2,attributevector3)
    
    node=(np.intersect1d(nodes1, nodes2))
    node=(np.intersect1d(node, node3))
    
    a=sum_of_nodes(tree,node)
    #node_show(a)
    return a
def ot_distance_wozero(a,b):
    acoord=np.where(a!=0)
    bcoord=np.where(b!=0)
    acoordarr=np.array(acoord)
    bcoordarr=np.array(bcoord)
    acoordarr=acoordarr.transpose()
    bcoordarr=bcoordarr.transpose()
    M = ot.dist(acoordarr,bcoordarr)
    M = M/M.max()

    dist= ot.emd(a[acoord]/a.sum(), b[bcoord]/b.sum(), M)
    maxplace = np.flip(np.dstack(np.unravel_index(np.argsort(dist.ravel()), (dist.shape[0], dist.shape[1]))))[0]
    bincount=np.count_nonzero(dist)
    bins1=acoordarr[maxplace[:,1]]
    bins2=bcoordarr[maxplace[:,0]]
    bins1=bins1[0:bincount,:]
    bins2=bins2[0:bincount,:]
    return bins1,bins2, dist
def ot_distance_all(a,b):
    acoord=np.where(a>=0)
    bcoord=np.where(b>=0)
    acoordarr=np.array(acoord)
    bcoordarr=np.array(bcoord)
    acoordarr=acoordarr.transpose()
    bcoordarr=bcoordarr.transpose()
    M = ot.dist(acoordarr,bcoordarr)
    M = M/M.max()

    dist= ot.emd(a[acoord]/a.sum(), b[bcoord]/b.sum(), M)
    maxplace = np.flip(np.dstack(np.unravel_index(np.argsort(dist.ravel()), (dist.shape[0], dist.shape[1]))))[0]
    bincount=np.count_nonzero(dist)
    bins1=acoordarr[maxplace[:,1]]
    bins2=bcoordarr[maxplace[:,0]]
    bins1=bins1[0:bincount,:]
    bins2=bins2[0:bincount,:]
    return bins1,bins2, dist
def area_threshold(tree,threshold):
    node = np.where(threshold<=tree.node_array[3])[0]    
    return node
def wass_distance(a,b):
    r,c=a.shape
    a=np.reshape(a,[r*c])
    b=np.reshape(b,[r*c])
    dist=stats.wasserstein_distance(a,b)
    return dist
def kolmogorov_distance(a,b):
    r,c=a.shape
    a=np.reshape(a,[r*c])
    b=np.reshape(b,[r*c])
    dist=stats.ks_2samp(a,b)
    return dist
def area_node_signature(tree):
    area = tree.node_array[3,:]

    # Area signature computation
    levels,signature =  tree.getSignature(area, node)

    #Gradient of the area signature
    gradient = signature[0:-1] - signature[1:]

    # Display area signature
    fig = plt.figure(figsize = (12,6))
    plt.subplot(121)
    plt.plot(levels,signature)
    plt.grid()
    plt.xlabel("Gray-level")
    plt.ylabel("Area")
    plt.title("Area signature")


    # Display gradient of the area signature
    plt.subplot(122)
    plt.grid()
    plt.plot(levels[0:-1],gradient)
    plt.xlabel("Gray-level")
    plt.ylabel("Gradient")
    plt.title("Gradient signature")
def area_image(tree):
      area_img = np.zeros(tree.node_index.shape, dtype=float)
      
      area=tree.node_array[3,:]
      area_img =area[tree.node_index]
        #area_img=np.array(area_img,dtype=np.uint16)
      return area_img
def mean_image(tree):
    mean=tree.computeNodeGrayAvg()
    mean_img =mean[tree.node_index]
    return mean_img
def rect_image(tree):
    rect=tree.computeRR()
    rect_img =rect[tree.node_index]
    return rect_img
def height_image(tree):
    height=tree.computeHeight()
    height_img =height[tree.node_index]
    return height_img
def volume_image(tree):
    volume=tree.computeVolume()
    volume_img =volume[tree.node_index]
    return volume_img
def area_image_sits(imarray):
    Bc = np.ones((3, 3), dtype=bool)
    r, c, b = tuple(imarray.shape)
    area_img = np.zeros([r, c, b], dtype=float)
    for i in range(b):
        mxt = siamxt.MaxTreeAlpha(imarray[:, :, i], Bc)
        attribute_area_filter(mxt, 100)
        area=mxt.node_array[3,:]
        area_img[:, :, i] =area[mxt.node_index]
        #area_img=np.array(area_img,dtype=np.uint16)
    return area_img
def filtered_area_image(imarray,mxt,t):
    a= attribute_area_filter(mxt, (t))
    area_im=area_image(imarray)
    return area_im
def mean_gray_image_sits(imarray):
    Bc = np.ones((3, 3), dtype=bool)
    r, c, b = tuple(imarray.shape)
    mean_img = np.zeros([r, c, b], dtype=float)
    for i in range(b):
        mxt = siamxt.MaxTreeAlpha(imarray[:, :, i], Bc)
        attribute_area_filter(mxt, 100)
        mean=mxt.computeNodeGrayAvg()
        mean_img[:, :, i] = mean[mxt.node_index]
        mean_img=np.array(mean_img,dtype=np.uint16)
    return mean_img
def volume_image_sits(imarray):
    Bc = np.ones((3, 3), dtype=bool)
    r, c, b = tuple(imarray.shape)
    volume_img = np.zeros([r, c, b], dtype=float)
    for i in range(b):
        mxt = siamxt.MaxTreeAlpha(imarray[:, :, i], Bc)
        volume=mxt.computeVolume()
        volume_img[:, :, i] =volume[mxt.node_index]
        volume_img=np.array(volume_img,dtype=np.uint16)
    return volume_img
def height_image_sits(imarray):
    Bc = np.ones((3, 3), dtype=bool)
    r, c, b = tuple(imarray.shape)
    height_img = np.zeros([r, c, b], dtype=float)
    for i in range(b):
        mxt = siamxt.MaxTreeAlpha(imarray[:, :, i], Bc)
        attribute_area_filter(mxt, 500)
        height=mxt.computeHeight()
        height_img[:, :, i] =height[mxt.node_index]
        height_img=np.array(height_img,dtype=np.uint16)
    return height_img
def temsptabil_image(tree):
    temp_stability=temp_stability_nodes(tree,node)
    temp_img=temp_stability[tree1.node_index]
    return temp_img
def begin_filter(tree,t):
    time=tree.node_array[12,:]
    tree.node_array[3,:]=t-time
    filtered=attribute_area_filter(tree,t)
    return filtered
def end_filter(tree,t):
    time=tree.node_array[13,:]
    tree.node_array[3,:]=time
    filtered=attribute_area_filter(tree,t)
    return filtered
def duration_filter(tree,t):
    time=time_duration(tree)
    tree.node_array[3,:]=time
    filtered=attribute_area_filter(tree,t)
    return filtered