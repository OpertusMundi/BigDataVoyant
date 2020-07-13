from osgeo import gdal
import os
import contextlib
import sys
from bigdatavoyant.raster_data import RasterData
import geovaex

def read_file(file, targetCRS=None, point_cols=None, type='vector'):
    if type == 'vector':
        return read_vector_file(file)
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

def read_vector_file(file):
    filename = os.path.basename(file)
    filename = os.path.splitext(filename)[0]
    arrow_file = os.path.dirname(file) + '/' + filename + '.arrow'
    if os.path.exists(arrow_file):
        print('Found arrow file %s, using this instead.' % (arrow_file))
    else:
        geovaex.io.to_arrow(file, arrow_file)
    return geovaex.open(arrow_file)
