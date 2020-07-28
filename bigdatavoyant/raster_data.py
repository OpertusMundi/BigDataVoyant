import bigdatavoyant.io
from osgeo import gdal
from bigdatavoyant.aux.cog_validator import validate_cog
from bigdatavoyant.report import Report
import os

class RasterData(object):
    """docstring for RasterData"""
    def __init__(self, ds):
        self._ds = ds
        self._count = ds.RasterCount
        self._crs = ds.GetSpatialRef()
        if self._crs is not None:
            self._short_crs = "%s:%s" % (self._crs.GetAttrValue("AUTHORITY", 0), self._crs.GetAttrValue("AUTHORITY", 1))
            self._unit = self._crs.GetAttrValue("UNIT")
        else:
            self._short_crs = None
            self._unit = None
        self._dimensions = [self._ds.RasterXSize, self._ds.RasterYSize]

    @property
    def count(self):
        return self._count

    @property
    def crs(self):
        return self._crs

    @property
    def short_crs(self):
        return self._short_crs

    @property
    def unit(self):
        return self._unit

    @property
    def dimensions(self):
        return self._dimensions

    @property
    def width(self):
        return self._dimensions[0]

    @property
    def height(self):
        return self._dimensions[1]

    @classmethod
    def from_file(self, file):
        return bigdatavoyant.io.read_file(file, type="raster")

    def _getBandInfo(self, scope, **kwargs):
        def _apply(band, scope, **kwargs):
            if scope == 'statistics':
                bApproxOK = kwargs.pop('bApproxOK', False)
                bForce = kwargs.pop('bForce', True)
                return dict(zip(['min', 'max', 'mean', 'std'], band.GetStatistics(bApproxOK, bForce, **kwargs)))
            elif scope == 'defaultHistogram':
                return band.GetDefaultHistogram(**kwargs)
            elif scope == 'histogram':
                return band.GetHistogram(**kwargs)
            elif scope == 'datatypes':
                datatype = band.DataType
                return gdal.GetDataTypeName(datatype)
            elif scope == 'noDataValue':
                return band.GetNoDataValue(**kwargs)
            elif scope == 'colorInterpretation':
                color = band.GetColorInterpretation()
                return gdal.GetColorInterpretationName(color)
            else:
                raise Exception('ERROR: Method %s not supported' % (scope))

        data = []
        for i in range(self._count):
            i += 1
            band = self._ds.GetRasterBand(i)
            if band is None:
                continue
            data.append(_apply(band, scope, **kwargs))

        return data

    def statistics(self, **kwargs):
        return self._getBandInfo('statistics', **kwargs)

    def defaultHistogram(self, **kwargs):
        return self._getBandInfo('defaultHistogram', **kwargs)

    def histogram(self, **kwargs):
        return self._getBandInfo('histogram', **kwargs)

    def datatypes(self):
        return self._getBandInfo('datatypes')

    def noDataValue(self, **kwargs):
        return self._getBandInfo('noDataValue')

    def colorInterpretation(self):
        return self._getBandInfo('colorInterpretation')

    def affineTransform(self, Xpixel, Yline):
        gt = self._ds.GetGeoTransform()
        Xgeo = gt[0] + Xpixel*gt[1] + Yline*gt[2]
        Ygeo = gt[3] + Xpixel*gt[4] + Yline*gt[5]
        return [Xgeo, Ygeo]

    def mbr(self):
        from shapely import geometry
        dim = self._dimensions
        nw = geometry.Point(*self.affineTransform(0, 0))
        ne = geometry.Point(*self.affineTransform(dim[0], 0))
        se = geometry.Point(*self.affineTransform(dim[0], dim[1]))
        sw = geometry.Point(*self.affineTransform(0, dim[1]))
        mbr = geometry.Polygon([nw, ne, se, sw])
        return mbr

    def resolution(self):
        gt = self._ds.GetGeoTransform()
        return {'x': gt[1], 'y': gt[5], 'unit': self._unit}

    def info(self):
        metadata = self._ds.GetMetadata()
        image_structure = self._ds.GetMetadata('IMAGE_STRUCTURE')
        driver = self._ds.GetDriver().LongName
        files = [os.path.basename(filename) for filename in self._ds.GetFileList()]
        width = self.width
        height = self.height
        bands = self.colorInterpretation()
        return {'metadata': metadata, 'image_structure': image_structure, 'driver': driver, 'files': files, 'width': width, 'height': height, 'bands': bands}

    def is_cog(self):
        if self._ds.GetDriver().ShortName != 'GTiff':
            return False
        errors, _ = validate_cog(self._ds)
        return len(errors) == 0

    def report(self):
        report = {}
        report['info'] = self.info()
        report['statistics'] = self.statistics()
        report['histogram'] = self.defaultHistogram()
        report['mbr'] = self.mbr().wkt
        report['resolution'] = self.resolution()
        report['cog'] = self.is_cog()
        report['number_of_bands'] = self._count
        report['datatypes'] = self.datatypes()
        report['noDataValue'] = self.noDataValue()
        report['crs'] = self._short_crs
        report['color_interpetation'] = self.colorInterpretation()

        return Report(report)
