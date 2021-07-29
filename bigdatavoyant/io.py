from osgeo import gdal
import os
import contextlib
import sys
from bigdatavoyant.raster_data import RasterData
from .netcdf_profiler import NetCDFProfiler
import geovaex

def read_file(file, type='vector', targetCRS=None, point_cols=None, **kwargs):
    """Reads a file into a DataFrame.
    Parameters:
        file (string): The full path of the spatial file.
        type (string): The type of the file: vector (default), raster or netcdf.
        lat_attr (string): (netcdf/vector ONLY) The variable name containing the latitude coordinate.
        lon_attr (string): (netcdf/vector ONLY) The variable name containing the longitude coordinate.
        time_attr (string): (netcdf ONLY) The variable name containing the time coordinate.
    Returns:
        (object) A dataset/datasource with the file's data.
    """
    if type == 'vector':
        output_path = kwargs.pop('output_path', None)
        return read_vector_file(file, output_path, **kwargs)
    elif type == 'raster':
        return read_raster_file(file)
    elif type == 'netcdf':
        lat_attr = kwargs.pop('lat', 'lat')
        lon_attr = kwargs.pop('lon', 'lon')
        time_attr = kwargs.pop('time_attr', 'time')
        return read_netcdf(file, lat_attr=lat_attr, lon_attr=lon_attr, time_attr=time_attr)
    else:
        raise Exception('ERROR: Recognized types are "raster", "vector" and "netcdf"')

def read_raster_file(file):
    """Reads a raster file into a DataFrame.
    Returns:
        (object) A datasource with the file's data.
    """
    try:
        dataSource = gdal.Open(file, gdal.GA_ReadOnly)
    except:
        print(sys.exc_info()[1])
    else:
        return RasterData(dataSource)

def read_vector_file(file, output_path=None, **kwargs):
    """Reads a vector file into a DataFrame.
    For vector files an intermediate ARROW file is created.
    Parameters:
        output_path (string): The path into which the arrow file will be stored. If None, it will be stored in the same path of source file.
    Returns:
        (object) A dataset with the file's data.
    """
    filename = os.path.basename(file)
    filename = os.path.splitext(filename)[0]
    arrow_file = os.path.dirname(file) + '/' + filename + '.arrow' if output_path is None else output_path + '/' + filename + '.arrow'
    if output_path is not None and not os.path.exists(output_path):
        os.makedirs(output_path)
    return geovaex.read_file(file, convert=arrow_file, **kwargs)

def read_netcdf(file, lat_attr='lat', lon_attr='lon', time_attr='time', crs='WGS 84'):
    """Reads a NetCDF file into a DataFrame.
    Parameters:
        lat_attr (string): The variable name containing the latitude coordinate.
        lon_attr (string): The variable name containing the longitude coordinate.
        time_attr (string): The variable name containing the time coordinate.
    Returns:
        (object) A DataFrame with the file's data.
    """
    return NetCDFProfiler.from_file(file, lat_attr=lat_attr, lon_attr=lon_attr, time_attr=time_attr)
