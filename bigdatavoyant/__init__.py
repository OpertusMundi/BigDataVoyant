import sys
try:
    from osgeo import ogr, gdal
except:
    sys.exit('ERROR: cannot find GDAL/OGR modules')
import bigdatavoyant.io
from .raster_data import RasterData
from .profiler import Profiler

def gdal_error_handler(err_class, err_num, err_msg):
    errtype = {
        gdal.CE_None:'None',
        gdal.CE_Debug:'Debug',
        gdal.CE_Warning:'Warning',
        gdal.CE_Failure:'Failure',
        gdal.CE_Fatal:'Fatal'
    }
    err_msg = err_msg.replace('\n',' ')
    err_class = errtype.get(err_class, 'None')
    print('%s: %s' % (err_class.upper(), err_msg))