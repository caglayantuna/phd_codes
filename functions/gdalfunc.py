import os, sys, osr
import shutil
import rtree
import math
import random
from osgeo import gdal, ogr, osr

import numpy as np


# --------------------------------------------------------------
def lyrFeatinXY(shp_in, minX, maxX, minY, maxY, shp_out):
    """
        Does a spatial selection of features. Selects the features that are into the X Y values,
        and creates a shapefile with this selection
        (From script_marcela/function.py --> replace CreateNewLayer(layer, shp_out) )
    """
    print
    "Selecting polygons"
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(shp_in, 0)
    layer = dataSource.GetLayer()
    layer.SetSpatialFilterRect(float(minX), float(minY), float(maxX), float(maxY))

    # -- Create output file
    # CreateNewLayer(layer, shp_out)
    if os.path.exists(shp_out):
        os.remove(shp_out)
    try:
        output = driver.CreateDataSource(shp_out)
    except:
        print
        'Could not create output datasource ', shp_out
        sys.exit(1)
    output.CopyLayer(layer, shp_out.split('.')[0])


# --------------------------------------------------------------
def clipVectorData(shp_in, shp_mask, shp_out):
    """
        Cuts a shapefile with another shapefile
        ARGs:
            INPUT:
                - shp_in: the shapefile to be cut
                - shp_mask: the other shapefile, mask
            OUTPUT:
                - shp_out: output shapefile
        (From script_marcela/Decoupe.py)
    """
    if os.path.exists(shp_out):
        os.remove(shp_out)
    cmd = "ogr2ogr "
    cmd += "-clipsrc " + shp_mask + " "
    cmd += shp_out + " "
    cmd += shp_in + " "
    cmd += "-progress"
    print
    cmd
    os.system(cmd)


# --------------------------------------------------------------
def clipContains(shp_in, shp_mask, shp_out):
    """
        Cuts a shapefile within another shapefile
        ARGs:
            INPUT:
                - shp_in: the shapefile to be cut
                - shp_mask: the other shapefile, mask
            OUTPUT:
                - shp_out: output shapefile
    """
    driver = ogr.GetDriverByName("ESRI Shapefile")
    shp = driver.Open(shp_in, 0)
    shp_m = driver.Open(shp_mask, 0)

    if shp is None:
        print
        "Could not open file ", shp_in
        sys.exit(1)
    if shp_m is None:
        print
        "Could not open file ", shp_mask
        sys.exit(1)
    layer = shp.GetLayer()
    layer_mask = shp_m.GetLayer()

    # -- Create output file
    if os.path.exists(shp_out):
        os.remove(shp_out)
    try:
        output = driver.CreateDataSource(shp_out)
    except:
        print
        'Could not create output datasource ', shp_out
        sys.exit(1)
    newLayer = output.CreateLayer('CompletelyContains', geom_type=ogr.wkbPolygon, srs=layer.GetSpatialRef())
    if newLayer is None:
        print
        "Could not create output layer"
        sys.exit(1)
    newLayerDef = newLayer.GetLayerDefn()
    init_fields(layer, newLayer)

    # RTree Spatial Indexing with OGR
    # -- Index creation
    print
    "Index creation..."
    index = rtree.index.Indeinterleaved = False
    for fid in range(0, layer_mask.GetFeatureCount()):
        feat = layer_mask.GetFeature(fid)
        geom = feat.GetGeometryRef()
        if geom == None:
          continue
        xmin, xmax, ymin, ymax = geom.GetEnvelope()
        index.insert(fid, (xmin, xmax, ymin, ymax))
        feat.Destroy()


# -- Search for all features in layer that intersect feature in layer_mask
    print
    "Research..."
    fid_list = []
    for fid_in in range(0, layer.GetFeatureCount()):
      feat_in = layer.GetFeature(fid_in)
      geom_in = feat_in.GetGeometryRef()
      if geom_in == None:
        continue
      xmin, xmax, ymin, ymax = geom_in.GetEnvelope()
      for fid_mask in list(index.intersection((xmin, xmax, ymin, ymax))):
        feat_mask = layer_mask.GetFeature(fid_mask)
        geom_mask = feat_mask.GetGeometryRef()
        if geom_mask == None:
            continue
        if (geom_mask.Contains(geom_in)):
            fid_list.append(fid_in)
        feat_mask.Destroy()
      feat_in.Destroy()

    print
    len(fid_list)
    for fid_in in range(0, layer.GetFeatureCount()):
     feat_in = layer.GetFeature(fid_in)
     geom_in = feat_in.GetGeometryRef()
     if geom_in == None:
        continue
     if fid_in in fid_list:
        if geom_in.GetGeometryName() == 'MULTIPOLYGON':
            for geom_part in geom_in:
                addMultiPolygon(geom_part.ExportToWkb(), feat_in, newLayer)
        else:
            addMultiPolygon(geom_in.ExportToWkb(), feat_in, newLayer)
     feat_in.Destroy()


# --------------------------------------------------------------
def addMultiPolygon(simplePolygon, feature, out_lyr):
    """
        Link with multipart2singlepart (above)
    """
    featureDefn = out_lyr.GetLayerDefn()
    polygon = ogr.CreateGeometryFromWkb(simplePolygon)
    out_feat = ogr.Feature(featureDefn)
    out_feat.SetGeometry(polygon)
    # Loop over each field from the source, and copy onto the new feature
    for id in range(feature.GetFieldCount()):
        data = feature.GetField(id)
        out_feat.SetField(id, data)
    out_lyr.CreateFeature(out_feat)


# --------------------------------------------------------------
def init_fields(in_lyr, out_lyr):
    in_lyr_defn = in_lyr.GetLayerDefn()
    for id in range(in_lyr_defn.GetFieldCount()):
        # Get the Field Definition
        field = in_lyr_defn.GetFieldDefn(id)
        fname = field.GetName()
        ftype = field.GetTypeName()
        fwidth = field.GetWidth()
        # Copy field definitions from source
        if ftype == 'String':
            fielddefn = ogr.FieldDefn(fname, ogr.OFTString)
            fielddefn.SetWidth(fwidth)
        else:
            fielddefn = ogr.FieldDefn(fname, ogr.OFTInteger)
        out_lyr.CreateField(fielddefn)


# --------------------------------------------------------------
def rotate(point, theta, center):
    """
        Symmetric rotation of point around center of theta angle
        ARGs:
            INPUT:
                point: point to rotate
                theta: angle rotation in degrees
                center: symmetry center
    """
    thetaRad = math.radians(theta)
    return [(point[0] - center[0]) * math.cos(thetaRad) - (point[1] - center[1]) * math.sin(thetaRad) + center[0],
            (point[1] - center[1]) * math.cos(thetaRad) + (point[0] - center[0]) * math.sin(thetaRad) + center[1]]


# --------------------------------------------------------------
def createPolygonRectangle(shp_name, centerX=0.0, centerY=0.0, radiusX=1.0, radiusY=1.0, theta=0.0):
    """
        Create shp_name shapefile with a unique rectangle polygon whose center and radius as parameters, theta is rotation angle. Field table is composed of only identifier named FID
        ARGs:
            INPUT:
                shp_name: output shapefile name
                (centerX,centerY): circle center
                (radiusX,radiusY): circle radius
                theta: rotation angle in degrees
        Spatial reference is EPSG:2154
    """
    short_shp_name = shp_name.split('.')
    # -- Create output file
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(shp_name):
        os.remove(shp_name)
    try:
        output = driver.CreateDataSource(shp_name)
    except:
        print
        'Could not create output datasource ', shp_name
        sys.exit(1)

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(2154)
    newLayer = output.CreateLayer(short_shp_name[0], geom_type=ogr.wkbPolygon, srs=srs)
    if newLayer is None:
        print
        "Could not create output layer"
        sys.exit(1)
    newLayer.CreateField(ogr.FieldDefn("FID", ogr.OFTInteger))
    newLayerDef = newLayer.GetLayerDefn()

    # -- Create ring feature
    feature = ogr.Feature(newLayerDef)

    ring = ogr.Geometry(ogr.wkbLinearRing)
    A = [centerX - radiusX, centerY - radiusY]
    B = [centerX - radiusX, centerY + radiusY]
    C = [centerX + radiusX, centerY + radiusY]
    D = [centerX + radiusX, centerY - radiusY]
    if theta != 0:
        A = rotate(A, theta, [centerX, centerY])
        B = rotate(B, theta, [centerX, centerY])
        C = rotate(C, theta, [centerX, centerY])
        D = rotate(D, theta, [centerX, centerY])
    ring.AddPoint(A[0], A[1])
    ring.AddPoint(B[0], B[1])
    ring.AddPoint(C[0], C[1])
    ring.AddPoint(D[0], D[1])
    ring.CloseRings()

    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)

    feature.SetGeometry(poly)
    feature.SetField("FID", 1)
    ring.Destroy()
    poly.Destroy()
    newLayer.CreateFeature(feature)

    output.Destroy()


# -----------------------------------------------------------------------
def getRasterExtent(raster_in):
    """
        Get raster extent of raster_in from GetGeoTransform()
        ARGs:
            INPUT:
                - raster_in: input raster
            OUTPUT
                - ex: extent with [minX,maxX,minY,maxY]
    """
    if not os.path.isfile(raster_in):
        return []
    raster = gdal.Open(raster_in, GA_ReadOnly)
    if raster is None:
        return []
    geotransform = raster.GetGeoTransform()
    originX = geotransform[0]
    originY = geotransform[3]
    spacingX = geotransform[1]
    spacingY = geotransform[5]
    r, c = raster.RasterYSize, raster.RasterXSize

    minX = originX
    maxY = originY
    maxX = minX + c * spacingX
    minY = maxY + r * spacingY
    return [minX, maxX, minY, maxY]

def bbox2ix(bbox,gt):
    xo = int(round((bbox[0] - gt[0])/gt[1]))
    yo = int(round((gt[3] - bbox[3])/gt[1]))
    xd = int(round((bbox[1] - bbox[0])/gt[1]))
    yd = int(round((bbox[3] - bbox[2])/gt[1]))
    return(xo,yo,xd,yd)

def rasclip(ras,shp):
    ds = gdal.Open(ras)
    gt = ds.GetGeoTransform()

    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(shp, 0)
    layer = dataSource.GetLayer()

    for feature in layer:

        xo,yo,xd,yd = bbox2ix(feature.GetGeometryRef().GetEnvelope(),gt)
        arr = ds.ReadAsArray(xo,yo,xd,yd)
        yield arr

    layer.ResetReading()
    ds = None
    dataSource = None