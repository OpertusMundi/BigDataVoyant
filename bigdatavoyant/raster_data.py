import bigdatavoyant.io
from osgeo import gdal
from bigdatavoyant.aux.cog_validator import validate_cog
from bigdatavoyant.report import Report
import os

class RasterData(object):
    """Raster Profiler class using GDAL datasource methods.
    Attributes:
        _ds (object): The GDAL datasource.
        _count (int): The number of the raster bands.
        _crs (object): A GDAL object of the Spatial Reference of the datasource.
        _short_crs (string): The short name of the native CRS.
        _unit (string): The unit of the CRS.
        _dimensions (list): The x, y dimensions of the raster.
    """
    def __init__(self, ds):
        """Creates the RasterData object from a datasource.
        Parameters:
            ds (object): The GDAL datasource.
        """
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
        """Returns the number of bands."""
        return self._count

    @property
    def crs(self):
        """Returns the native CRS (GDAL Spatial Reference) of the datasource."""
        return self._crs

    @property
    def short_crs(self):
        """Returns a short name of the native CRS."""
        return self._short_crs

    @property
    def unit(self):
        """Returns the unit of the raster."""
        return self._unit

    @property
    def dimensions(self):
        """Returns the x, y dimensions of the raster."""
        return self._dimensions

    @property
    def width(self):
        """Returns the width (x-dimension) of the raster."""
        return self._dimensions[0]

    @property
    def height(self):
        """Returns the height (y-dimension) of the raster."""
        return self._dimensions[1]

    @classmethod
    def from_file(self, file):
        """Creates the RasterData object from a raster file.
        Parameters:
            file (string): The full path of the raster file.
        """
        return bigdatavoyant.io.read_file(file, type="raster")

    def _getBandInfo(self, scope, **kwargs):
        """Computes various kinds of metadata for each band.
        Parameters:
            scope (string): One of statistics, defaultHistogram, histogram, datatypes, noDataValue or colorInterpretation.
        Returns:
            (list) The computed values for each band.
        """
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
        """Computes descriptive statistics for each band.
        Parameters:
            **kwargs: GDALRasterBand.GetStatistics kwargs.
        Returns:
            (list) The computed values for each band.
        """
        return self._getBandInfo('statistics', **kwargs)

    def defaultHistogram(self, **kwargs):
        """Computes the default histogram for each band.
        Parameters:
            **kwargs: GDALRasterBand.GetDefaultHistogram kwargs.
        Returns:
            (list) The default histogram values for each band.
        """
        return self._getBandInfo('defaultHistogram', **kwargs)

    def histogram(self, **kwargs):
        """Computes the histogram for each band.
        Parameters:
            **kwargs: GDALRasterBand.GetHistogram kwargs.
        Returns:
            (list) The histogram values for each band.
        """
        return self._getBandInfo('histogram', **kwargs)

    def datatypes(self):
        """Retrieves the datatypes for each band.
        Parameters:
            **kwargs: GDALRasterBand.GetDataTypeName kwargs.
        Returns:
            (list) The data types for each band.
        """
        return self._getBandInfo('datatypes')

    def noDataValue(self, **kwargs):
        """Retrieves the no data value for each band.
        Parameters:
            **kwargs: GDALRasterBand.GetNoDataValue kwargs.
        Returns:
            (list) The no data value for each band.
        """
        return self._getBandInfo('noDataValue')

    def colorInterpretation(self):
        """Retrieves the color interpretation for each band.
        Parameters:
            **kwargs: GDALRasterBand.GetColorInterpretation kwargs.
        Returns:
            (list) The color interpretation for each band.
        """
        return self._getBandInfo('colorInterpretation')

    def affineTransform(self, Xpixel, Yline):
        """Calculated an affine transform of pixels.
        Parameters:
            Xpixel (int): The x pixel position.
            Yline (int): The y pixel position.
        Returns:
            (list): The spatial coordinates of the pixel.
        """
        gt = self._ds.GetGeoTransform()
        Xgeo = gt[0] + Xpixel*gt[1] + Yline*gt[2]
        Ygeo = gt[3] + Xpixel*gt[4] + Yline*gt[5]
        return [Xgeo, Ygeo]

    def mbr(self):
        """Computes the MBR of the raster.
        Returns:
            (object) The shapely geometry representing the MBR.
        """
        from shapely import geometry
        dim = self._dimensions
        nw = geometry.Point(*self.affineTransform(0, 0))
        ne = geometry.Point(*self.affineTransform(dim[0], 0))
        se = geometry.Point(*self.affineTransform(dim[0], dim[1]))
        sw = geometry.Point(*self.affineTransform(0, dim[1]))
        mbr = geometry.Polygon([nw, ne, se, sw])
        return mbr

    def resolution(self):
        """Calculates the x,y resolution of the raster.
        Returns:
            (dict) x, y resolution along with their measurement unit.
        """
        gt = self._ds.GetGeoTransform()
        return {'x': gt[1], 'y': gt[5], 'unit': self._unit}

    def info(self):
        """Various types of the raster file information.
        Returns:
            (dict) The type and information.
        """
        metadata = self._ds.GetMetadata()
        image_structure = self._ds.GetMetadata('IMAGE_STRUCTURE')
        driver = self._ds.GetDriver().LongName
        files = [os.path.basename(filename) for filename in self._ds.GetFileList()]
        width = self.width
        height = self.height
        bands = self.colorInterpretation()
        return {'metadata': metadata, 'image_structure': image_structure, 'driver': driver, 'files': files, 'width': width, 'height': height, 'bands': bands}

    def is_cog(self):
        """ Checks whether the raster is a cloud optimized GeoTiff.
        Returns:
            (bool) True if is COG.
        """
        if self._ds.GetDriver().ShortName != 'GTiff':
            return False
        errors, _ = validate_cog(self._ds)
        return len(errors) == 0

    def sample(self, file, bbox):
        """Creates a sample raster file.
        Parameters:
            file (string): Full path of the sample raster file.
            bbox (list): The bounding box of the sample.
        """
        sample_ds = gdal.Translate(file, self._ds, projWin=bbox)
        sample_ds = None

    def report(self):
        """Creates a report with a collection of metadata.
        Returns:
            (object) A report object.
        """
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
