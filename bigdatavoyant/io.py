from osgeo import gdal
import os
import contextlib
import sys
from bigdatavoyant.raster_data import RasterData
from .netcdf_profiler import NetCDFProfiler
import geovaex

def read_file(file, type='vector', targetCRS=None, point_cols=None, **kwargs):
    if type == 'vector':
        output_path = kwargs.pop('output_path', None)
        return read_vector_file(file, output_path)
    elif type == 'raster':
        return read_raster_file(file)
    elif type == 'netcdf':
        lat_attr = kwargs.pop('lat_attr', 'lat')
        lon_attr = kwargs.pop('lon_attr', 'lon')
        time_attr = kwargs.pop('time_attr', 'time')
        return read_netcdf(file, lat_attr=lat_attr, lon_attr=lon_attr, time_attr=time_attr)
    else:
        raise Exception('ERROR: Recognized types are "raster", "vector" and "netcdf"')

def read_raster_file(file):
    try:
        dataSource = gdal.Open(file)
    except:
        print(sys.exc_info()[1])
    else:
        return RasterData(dataSource)

def read_vector_file(file, output_path=None):
    filename = os.path.basename(file)
    filename = os.path.splitext(filename)[0]
    arrow_file = os.path.dirname(file) + '/' + filename + '.arrow' if output_path is None else output_path + '/' + filename + '.arrow'
    if output_path is not None and not os.path.exists(output_path):
        os.makedirs(output_path)
    if os.path.exists(arrow_file):
        print('Found arrow file %s, using this instead.' % (arrow_file))
    else:
        geovaex.io.to_arrow(file, arrow_file)
    return geovaex.open(arrow_file)

def read_netcdf(file, lat_attr='lat', lon_attr='lon', time_attr='time'):
    return NetCDFProfiler.from_file(file, lat_attr=lat_attr, lon_attr=lon_attr, time_attr=time_attr)
