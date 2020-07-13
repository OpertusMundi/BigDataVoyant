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
