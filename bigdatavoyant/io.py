from osgeo import gdal
import os
import contextlib
import sys
from bigdatavoyant.raster_data import RasterData

def read_file(file, output='ogr', targetCRS=None, point_cols=None, type='vector'):
    if type == 'vector':
        print('Not implemented yet')
        return
    elif type == 'raster':
        return read_raster_file(file)
    else:
        raise Exception('ERROR: Recognized types are "raster" and "vector"')

def read_raster_file(file):
    try:
        dataSource = gdal.Open(file)
    except:
        print(sys.exc_info()[1])
    else:
        return RasterData(dataSource)

def get_crs(layer):
    spatialRef = layer.GetSpatialRef()
    crs = "%s:%s" % (spatialRef.GetAttrValue("AUTHORITY", 0), spatialRef.GetAttrValue("AUTHORITY", 1)) if spatialRef is not None else None
    return crs

def getDefinition(layer):
    schema = []
    ldefn = layer.GetLayerDefn()
    for n in range(ldefn.GetFieldCount()):
        fdefn = ldefn.GetFieldDefn(n)
        schema.append(fdefn.name)
    return schema

@contextlib.contextmanager
def writeLayerXML(csv_file, point_cols, crs):
    import tempfile
    name = os.path.basename(csv_file)
    name = os.path.splitext(name)[0]
    xml = '<OGRVRTDataSource><OGRVRTLayer name="%s"><SrcDataSource>%s</SrcDataSource><SrcLayer>%s</SrcLayer><GeometryType>wkbPoint</GeometryType><LayerSRS>%s</LayerSRS><GeometryField encoding="PointFromColumns" x="%s" y="%s"/></OGRVRTLayer></OGRVRTDataSource>'
    try:
        tf = tempfile.NamedTemporaryFile(mode='w+', suffix=".vrt", delete=False)
        filename = tf.name
        tf.write(xml % (name, csv_file, name, crs, point_cols['x'], point_cols['y']))
        tf.close()
        yield filename
    finally:
        os.unlink(filename)
