import vaex
from geovaex import GeoDataFrame
import pygeos as pg
import pandas as pd
import uuid
import warnings
from .distribution import Distribution
from .report import Report
from .plots import heatmap, map_choropleth
from .clustering import Clustering
from .heatmap import Heatmap

def custom_formatwarning(msg, *args, **kwargs):
    """Ignore everything except the message."""
    return str(msg) + '\n'
warnings.formatwarning = custom_formatwarning

@vaex.register_dataframe_accessor('profiler', override=True)
class Profiler(object):
    """Vector profiler class.
    Attributes:
        df (object): The vector (geo)dataframe.
        _count (int): The number of features in dataframe.
        _categorical (list): A list of categorical fields.
    """

    def __init__(self, df):
        """Initiates the Profiles class.
        Parameters:
            df (object): The vector (geo)dataframe.
        """
        self.df = df
        self._count = None
        self._categorical = None
        self._has_geometry = isinstance(df, GeoDataFrame)

    def mbr(self):
        """Returns the Minimum Bounding Rectangle.
        Returns:
            (string) The WKT representation of the MBR.
        """
        if not self._has_geometry:
            warnings.warn('DataFrame is not spatial.')
            return None
        return pg.to_wkt(self.df.geometry.total_bounds())

    @property
    def featureCount(self):
        """Property containing the original length of features in the dataframe."""
        return self.df._length_original

    def count(self):
        """Counts the length of dataframe and caches the value in the corresponding object attribute.
        Returns:
            (int) The number of features.
        """
        if self._count is not None:
            return self._count
        self._count = {col: self.df.count(col, array_type='list') for col in self.df.get_column_names(virtual=False)}
        return self._count

    def convex_hull(self, chunksize=50000, max_workers=None):
        """Returns the convex hull of all geometries in the dataframe.
        Parameters:
            chunksize (int): The chunksize (number of features) for each computation.
            max_workers (int): The number of workers to be used, if None equals to number of available cores.
        Returns:
            (string) The WKT representation of convex hull.
        """
        if not self._has_geometry:
            warnings.warn('DataFrame is not spatial.')
            return None
        return pg.to_wkt(self.df.geometry.convex_hull_all(chunksize=chunksize, max_workers=max_workers))

    def thumbnail(self, file=None, maxpoints=100000, **kwargs):
        """Creates a thumbnail of the dataset.
        Parameters:
            file (string): The full path to save the thumbnail. If None, the thumbnail is not saved to file.
            maxpoints (int): The maximum number of points to used for the thumbnail.
        Raises:
            Exception: if cannot write to filesystem.
        Returns:
            (string) base64 encoded png.
        """
        from .static_map import StaticMap
        if not self._has_geometry:
            warnings.warn('DataFrame is not spatial.')
            return None

        static_map = StaticMap(**kwargs)
        df = self.df.sample(n=maxpoints) if maxpoints < len(self.df) else self.df.copy()
        df = df.to_geopandas_df()
        static_map.addGeometries(df)

        if (file is not None):
            static_map.toFile(file)
        else:
            return static_map.base64()

    @property
    def crs(self):
        """Returns the native CRS (proj4 object) of the dataframe."""
        if not self._has_geometry:
            warnings.warn('DataFrame is not spatial.')
            return None
        return self.df.geometry.crs

    @property
    def short_crs(self):
        """Returns the short CRS of the dataframe."""
        if not self._has_geometry:
            warnings.warn('DataFrame is not spatial.')
            return None
        return self.df.geometry.crs.to_string()

    def attributes(self):
        """The attributes of the (geo)dataframe.
        Returns:
            (list) The attributes of the df.
        """
        return self.df.get_column_names(virtual=False)

    def data_types(self):
        """Calculates the datatype of each attribute.
        Returns:
            (dict) The datatype of each attribute.
        """
        datatypes = {col: self.df.data_type(col) for col in self.df.get_column_names(virtual=False)}
        for col in datatypes:
            try:
                datatypes[col] = datatypes[col].__name__
            except:
                datatypes[col] = datatypes[col].name
        return datatypes

    def categorical(self, min_frac=0.01, sample_length=10000):
        """Checks whether each attribute holds categorical data, using a sample.
        Parameters:
            min_frac (float): The minimum fraction of unique values, under which the attribute is considered categorical.
            sample_length (int): The length of sample to be used.
        Returns:
            (list): A list of the categorical attributes.
        """
        if self._categorical is not None:
            return self._categorical
        df = self.df.to_vaex_df() if isinstance(self.df, GeoDataFrame) else self.df
        df = df.sample(n=sample_length) if sample_length < len(df) else df
        categorical = []
        for col in df.get_column_names(virtual=False):
            nunique = df[col].nunique()
            if nunique/df[col].count() <= min_frac:
                categorical.append(col)
                # self.df.ordinal_encode(col, inplace=True)
        self._categorical = categorical
        return self._categorical

    def distribution(self, attributes=None, n_obs=5, dropmissing=True):
        """Creates the distribution of values for each attribute.
        By default, it calculates the distribution only for the categorical attributes,
        returning the 5 most frequent values for each attribute, dropping the missing values.
        Parameters:
            attributes (list): A list of the attributes to create the distribution.
            n_obs (int): The number of most frequent values to return.
            dropmissing (bool): Whether to drop missing values or not.
        Returns:
            (object) A distribution object.
        """
        attributes = self.categorical() if attributes is None else attributes
        return Distribution(self.df, attributes, n_obs)

    def distribution_ml(self, attributes=None, n_obs=5, dropmissing=True):
        """Creates the distribution of values for each attribute using machine learning techniques.
        By default, it calculates the distribution only for the categorical attributes,
        returning the 5 most frequent values for each attribute, dropping the missing values.
        Parameters:
            attributes (list): A list of the attributes to create the distribution.
            n_obs (int): The number of most frequent values to return.
            dropmissing (bool): Whether to drop missing values or not.
        Returns:
            (object) A distribution object.
        """
        attributes = self.categorical() if attributes is None else attributes
        return Distribution(self.df, attributes, n_obs, dropmissing=dropmissing, method='ml')

    def get_sample(self, n_obs=None, frac=None, method="first", bbox=None, random_state=None):
        """Creates a sample of the dataframe.
        Parameters:
            n_obs (int): The number of features contained in the sample.
            frac (float): The fraction of the total number of features contained in the sample. It overrides n_obs.
            method (string): The method it will be used to extract the sample. One of: first, last, random.
            bbox (list): The desired bounding box of the sample.
            random_state (int): Seed or RandomState for reproducability, when None a random seed it chosen.
        Returns:
            (object): A sample dataframe.
        """
        df = self.df
        if bbox is not None:
            if not self._has_geometry:
                warnings.warn('DataFrame is not spatial.')
            else:
                df = self.df.within(pg.box(*bbox))
        length = len(df)
        if n_obs is None and frac is None:
            n_obs = min(round(0.05*length), 100000)
        if (method == "first"):
            if frac is not None:
                n_obs = round(frac*length)
            sample = df.head(n_obs)
        elif (method == "random"):
            sample = df.sample(n=n_obs, frac=frac, random_state=random_state)
        elif (method == "last"):
            if frac is not None:
                n_obs = round(frac*length)
            sample = df.tail(n_obs)
        else:
            raise Exception('ERROR: Method %s not supported' % (method))
        return sample

    def quantiles(self):
        """Calculates the 5, 25, 50, 75, 95 quantiles.
        Returns:
            (object) A pandas dataframe with the calculated values.
        """
        columns=[5, 25, 50, 75, 95]
        df = pd.DataFrame(columns=columns)
        for col in self.df.get_column_names(virtual=False):
            quantiles = []
            try:
                for percentage in columns:
                    percentage = float(percentage)
                    quantiles.append(self.df.percentile_approx(col, percentage=percentage))
            except:
                pass
            else:
                row = pd.DataFrame([quantiles], columns=columns, index=[col])
                df = df.append(row)
        return df

    def distinct(self, attributes=None, n_obs=50):
        """Retrieves the distinct values for each attribute.
        By default, it retrieves all distinct values only for the categorical attributes in the dataframe.
        Parameters:
            attributes (list): A list of the attributes to create the distribution.
            n_obs (int): If given, only the first n_obs values for each attribute are returned.
        Returns:
            (dict) A dictionary with the list of distinct values for each attribute.
        """
        attributes = self.categorical() if attributes is None else attributes
        if n_obs is None:
            distinct = {col: self.df.unique(col, dropna=True).tolist() for col in attributes}
        else:
            distinct = {col: self.df.unique(col, dropna=True)[0:n_obs].tolist() for col in attributes}
        return distinct

    def recurring(self, attributes=None, n_obs=5):
        """Retrieves the most frequent values of each attribute.
        By default, it calculates the most frequent values only for the categorical attributes,
        returning the top 5.
        Parameters:
            attributes (list): A list of the attributes to create the distribution.
            n_obs (int): The maximum number of most frequent values for each attribute.
        Returns:
            (dict) A dictionary with the list of most frequent values for each attribute.
        """
        attributes = self.categorical() if attributes is None else attributes
        return {col: self.df[col].value_counts(dropna=True).index[:n_obs].tolist() for col in attributes}

    def statistics(self):
        """Calculates general descriptive statistics for each attribute.
        The statistics calculated are: minimum and maximum value, mean, median, standard deviation
        and the sum of all values, in case the attribute contains numerical values.
        Returns:
            (object) A pandas dataframe with the statistics.
        """
        columns = ['min', 'max', 'mean', 'median', 'std', 'sum']
        df = pd.DataFrame(columns=columns)
        for col in self.df.get_column_names(virtual=False):
            statistics = []
            try:
                minimum, maximum = self.df.limits(col)
                statistics.append(minimum)
                statistics.append(maximum)
                statistics.append(float(self.df.mean(col)))
                statistics.append(self.df.median_approx(col))
                statistics.append(self.df.std(col))
                statistics.append(float(self.df.sum(col)))
            except:
                pass
            else:
                row = pd.DataFrame([statistics], columns=columns, index=[col])
                df = df.append(row)
        return df

    def leaflet_heatmap(self, radius=10, maxpoints=100000, **kwargs):
        """Creates a heatmap plot of the dataframe, using by default a maximum sample of 100000 features.
        Parameters:
            radius (int): The radius of each point.
            maxpoints (int): The maximum number of features that will be used for the creation of the heatmap.
            **kwargs: Additional arguments for heatmap plot. See plot.heatmap.
        Returns:
            (object) A Folium plot obejct.
        """
        if not self._has_geometry:
            warnings.warn('DataFrame is not spatial.')
            return None
        pois = self.df.centroid()
        if maxpoints is not None and maxpoints < len(self.df):
            pois = pois.sample(n=maxpoints)
        return heatmap(pois, radius=radius, **kwargs)

    def heatmap(self, **kwargs):
        """Creates a heatmap plot of the dataframe, using by default the whole dataset.
        Parameters:
            **kwargs: Optional arguments for heatmap. See heatmap.Heatmap.
        Returns:
            (object) A Heatmap obejct.
        """
        if not self._has_geometry:
            warnings.warn('DataFrame is not spatial.')
            return None
        heatmap = Heatmap(self.df, **kwargs)
        return heatmap

    def clusters(self, maxpoints=1000000, **kwargs):
        """Computes clusters using centroids of geometries.
        Parameters:
            max_points (int): The maximum number of features that will be used for the cluster computation.
            **kwargs: Additional optional arguments for clustering. See clustering.Clustering.
        Returns:
            (object) A Clustering object.
        """
        if not self._has_geometry:
            warnings.warn('DataFrame is not spatial.')
            return None
        if maxpoints is not None and maxpoints < len(self.df):
            pois = self.df.sample(n=maxpoints)
        else:
            pois = self.df.copy()
        pois.constructive.centroid(inplace=True)
        geom = pois.geometry.to_pygeos().values()
        filt = pg.get_coordinate_dimension(geom)
        pois.add_column('tmp', filt, dtype=int)
        pois = pois[pois.tmp == 2]
        pois.drop('tmp', inplace=True)
        pois = pois.extract()
        return Clustering(pois, **kwargs)

    def report(self, **kwargs):
        """Creates a report with a collection of metadata.
        Parameters:
            **kwargs: See StaticMap class
        Returns:
            (object) A report object.
        """
        from .static_map import StaticMap
        from json import loads

        if self._has_geometry:
            static_map = StaticMap(**kwargs)

            mbr = self.mbr()
            try:
                static_map.addWKT(mbr, self.short_crs)
                mbr_static = static_map.base64()
            except:
                mbr_static = None

            convex_hull = self.convex_hull()
            try:
                static_map.addWKT(convex_hull, self.short_crs)
                convex_hull_static = static_map.base64()
            except Exception as e:
                convex_hull_static = None

            try:
                thumbnail = self.thumbnail(**kwargs)
            except:
                thumbnail = None
            clusters = self.clusters()
            shapes = clusters.shapes()
            try:
                static_map.addGeometries(shapes, weight='size')
                clusters_static = static_map.base64()
            except:
                clusters_static = None
            shapes = loads(shapes.to_json())
            try:
                heatmap = self.heatmap()
                heatmap_static = heatmap.to_static_map(**kwargs).base64()
                heatmap_geojson = heatmap.geojson
            except:
                heatmap_geojson = None
                heatmap_static = None

            short_crs = self.short_crs
            asset_type = 'vector'
        else:
            mbr = None
            mbr_static = None
            convex_hull = None
            convex_hull_static = None
            thumbnail = None
            short_crs = None
            heatmap_geojson = None
            heatmap_static = None
            shapes = None
            clusters_static = None
            asset_type = 'tabular'

        report = {
            'assetType': asset_type,
            'mbr': mbr,
            'mbrStatic': mbr_static,
            'featureCount': self.featureCount,
            'count': self.count(),
            'convexHull': convex_hull,
            'convexHullStatic': convex_hull_static,
            'thumbnail': thumbnail,
            'crs': short_crs,
            'attributes': self.attributes(),
            'datatypes': self.data_types(),
            'distribution': self.distribution().to_dict(),
            'quantiles': self.quantiles().to_dict(),
            'distinct': self.distinct(),
            'recurring': self.recurring(),
            'heatmap': heatmap_geojson,
            'heatmapStatic': heatmap_static,
            'clusters': shapes,
            'clustersStatic': clusters_static,
            'statistics': self.statistics().to_dict()
        }
        return Report(report)
