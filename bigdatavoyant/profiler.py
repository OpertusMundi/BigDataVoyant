import vaex
import pygeos as pg
import pandas as pd
import uuid
from .distribution import Distribution
from .report import Report
from .plots import heatmap, map_choropleth
from .clustering import compute_clusters, cluster_shapes

@vaex.register_dataframe_accessor('profiler', override=True)
class Profiler(object):
    def __init__(self, df):
        self.df = df
        self._count = None
        self._categorical = None

    def mbr(self):
        return pg.to_wkt(self.df.geometry.total_bounds())

    @property
    def featureCount(self):
        return self.df._length_original

    def count(self):
        if self._count is not None:
            return self._count
        self._count = {col: self.df.count(col, array_type='list') for col in self.df.get_column_names(virtual=False)}
        return self._count

    def convex_hull(self, chunksize=50000, max_workers=None):
        return pg.to_wkt(self.df.geometry.convex_hull_all(chunksize=chunksize, max_workers=max_workers))

    def thumbnail(self, file, maxpoints=100000):
        import contextily as ctx
        import matplotlib.pyplot as plt
        from pathlib import Path
        df = self.df.sample(n=maxpoints) if maxpoints < len(self.df) else self.df.copy()
        df.geometry.to_crs('EPSG:3857')
        df = df.to_geopandas_df()
        ax = df.plot(figsize=(10, 10), alpha=0.5, edgecolor='k')
        ctx.add_basemap(ax)
        if (file is not None):
            try:
                plt.savefig(file)
                print("Wrote file %s, %d bytes." % (file, Path(file).stat().st_size))
            except:
                raise Exception('ERROR: Could not write to filesystem.')

    @property
    def crs(self):
        return self.df.geometry.crs

    @property
    def short_crs(self):
        return self.df.geometry.crs.to_string()

    def attributes(self):
        return self.df.get_column_names(virtual=False)

    def data_types(self):
        datatypes = {col: self.df.data_type(col) for col in self.df.get_column_names(virtual=False)}
        for col in datatypes:
            try:
                datatypes[col] = datatypes[col].__name__
            except:
                datatypes[col] = datatypes[col].name
        return datatypes

    def categorical(self, min_frac=0.1, sample_lentgh=100000):
        if self._categorical is not None:
            return self._categorical
        df = self.df.to_vaex_df().sample(n=sample_lentgh) if sample_lentgh < len(self.df) else self.df
        categorical = []
        for col in df.get_column_names(virtual=False):
            nunique = df[col].nunique()
            if nunique/df[col].count() <= min_frac:
                categorical.append(col)
                # self.df.ordinal_encode(col, inplace=True)
        self._categorical = categorical
        return self._categorical

    def distribution(self, attributes=None, n_obs=5, dropmissing=True):
        attributes = self.categorical() if attributes is None else attributes
        return Distribution(self.df, attributes, n_obs)

    def distribution_ml(self, attributes=None, n_obs=5, dropmissing=True, method='brute'):
        attributes = self.categorical() if attributes is None else attributes
        return Distribution(self.df, attributes, n_obs, dropmissing=dropmissing, method='ml')

    def get_sample(self, n_obs=None, frac=None, method="first", bbox=None, random_state=None):
        df = self.df if bbox is None else self.df.within(pg.box(*bbox))
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
        columns=['5', '25', '50', '75', '95']
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

    def distinct(self, attributes=None, n_obs=None):
        attributes = self.categorical() if attributes is None else attributes
        if n_obs is None:
            distinct = {col: self.df.unique(col, dropna=True).tolist() for col in attributes}
        else:
            distinct = {col: self.df.unique(col, dropna=True)[0:n_obs].tolist() for col in attributes}
        return distinct

    def recurring(self, attributes=None, n_obs=5):
        attributes = self.categorical() if attributes is None else attributes
        return {col: self.df[col].value_counts(dropna=True).index[:n_obs].tolist() for col in attributes}

    def statistics(self):
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

    def heatmap(self, tiles='OpenStreetMap', width='100%', height='100%', radius=10, maxpoints=100000):
        pois = self.df.centroid()
        if maxpoints is not None and maxpoints < len(self.df):
            pois = pois.sample(n=maxpoints)
        return heatmap(pois, tiles, width, height, radius)

    def compute_clusters(self, alg='dbscan', min_pts=None, eps=None, n_jobs=-1, maxpoints=1000000):
        pois = self.df.centroid()
        if maxpoints is not None and maxpoints < len(self.df):
            pois = pois.sample(n=maxpoints)
        # pois = pois.to_geopandas_df()
        return compute_clusters(pois, alg=alg, min_pts=min_pts, eps=eps, n_jobs=n_jobs)

    def cluster_borders(self, **kwargs):
        pois_in_clusters = kwargs.pop('pois_in_clusters', None)
        eps_per_cluster = kwargs.pop('eps_per_cluster', None)
        shape_type = kwargs.pop('shape_type', 1)
        if pois_in_clusters is None or (eps_per_cluster is None and shape_type != 1):
            pois_in_clusters, eps_per_cluster, info = self.compute_clusters(
                alg=kwargs.pop('alg', 'dbscan'),
                min_pts=kwargs.pop('min_pts', None),
                eps=kwargs.pop('eps', None),
                n_jobs=kwargs.pop('n_jobs', -1),
                maxpoints=kwargs.pop('maxpoints', 1000000)
            )
        cluster_borders = cluster_shapes(
            pois_in_clusters,
            shape_type=shape_type,
            eps_per_cluster=eps_per_cluster
        )
        return cluster_borders

    def plot_clusters(self, cluster_borders=None, **kwargs):
        if cluster_borders is None:
            pois_in_clusters, eps_per_cluster = self.compute_clusters(
                alg=kwargs.pop('alg', 'dbscan'),
                min_pts=kwargs.pop('min_pts', None),
                eps=kwargs.pop('eps', None),
                n_jobs=kwargs.pop('n_jobs', -1),
                maxpoints=kwargs.pop('maxpoints', 1000000)
            )
            cluster_borders = cluster_shapes(
                pois_in_clusters,
                shape_type=kwargs.pop('shape_type', 1),
                eps_per_cluster=kwargs.pop('eps_per_cluster', None)
            )
        return map_choropleth(cluster_borders, id_field='cluster_id', value_field='size')

    def report(self, thumbnail=None, sample_method='random', sample_bbox=None, destination=None):
        thumbnail = str(uuid.uuid4()) + '.jpg' if thumbnail is None else thumbnail
        self.thumbnail(thumbnail)
        report = {
            'mbr': self.mbr(),
            'featureCount': self.featureCount,
            'count': self.count(),
            'convex_hull': self.convex_hull(),
            'thumbnail': thumbnail,
            'crs': self.crs.__str__(),
            'attributes': self.attributes(),
            'datatypes': self.data_types(),
            'distribution': self.distribution().to_dict(),
            'quantiles': self.quantiles().to_dict(),
            'distinct': self.distinct(),
            'recurring': self.recurring(),
            'statistics': self.statistics().to_dict()
        }
        return Report(report)
