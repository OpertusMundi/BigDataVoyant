from scipy import stats
import numpy as np
import pygeos as pg
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from json import loads
import geopandas as gpd
import geojsoncontour
from .static_map import StaticMap

class Heatmap(object):
    """Creates a heatmap from the geometries of a geoDataframe.

    It calculate the Kernel Density Estimation (KDE) using gaussian kernels
    from the centroids of the geometries and optionally weighted by the values
    of an additional column. It includes automatic bandwidth determination,
    either using a modified silverman method for not weighted data either
    scotts method as implemented in scipy for weighted data.
    """

    def __init__(self, df, bw_method=None, weights=None, alpha=0.4, grid=100, number_of_levels=100, ignore_levels=1, cmap=None, epsg=3857, padding=5000.):
        """Computes the KDE for a given dataset and creates the heatmap.

        Parameters:
            df (object): A dataframe with geometries and optionally weights.
            bw_method (string, optional): The method used to calculate the estimator bandwidth. This can be ‘scott’, ‘silverman’, a scalar constant or a callable. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
            weights (string): The attribute name of the weights in the dataframe.
            alpha (float): The alpha blending value, between 0 (transparent)
                and 1 (opaque) (default: 0.4).
            grid (int): The number of points in the grid (default: 100).
            number_of_levels (int): The number of levels in the generated heatmap (default: 100).
            ignore_levels (int): The number of first levels to ignore (default: 1).
            cmap (obect|string): A matplotlib Colormap instance or registered colormap name.
                The colormap maps the level values to colors.
                (default: A custom colormap blue-green-red weighted to the right.)
            epsg (int): The EPSG code to reproject the geometries (default: 3857).
            padding (float, optional): Padding around the MBR in meters.
        """
        self.df = df
        self.weights = weights
        self.alpha = alpha
        self.number_of_levels = number_of_levels
        self.ignore_levels = ignore_levels
        self.epsg = epsg

        # Create the data
        pois = df.centroid()
        if epsg is not None:
            pois.geometry.to_crs(epsg)
        data = pg.get_coordinates(pois.geometry.to_pygeos().values())
        xmin, ymin = data.min(axis=0) - padding
        xmax, ymax = data.max(axis=0) + padding

        m1, m2 = data.T
        x, y = np.mgrid[xmin:xmax:grid*1j, ymin:ymax:grid*1j]
        positions = np.vstack([x.ravel(), y.ravel()])
        values = np.vstack([m1, m2])

        # Calculate the bandwidth
        if bw_method is None and weights is None:
            std = m1.std(ddof=1)
            n = len(m1)
            iqr = stats.iqr(m1)
            bw_method = 0.9*(min(std, iqr/1.34)*n**(-0.2))/std
        elif bw_method is None and weights is not None:
            bw_method = 'scotts'

        # Compute Kernel Density Estimation and levels
        kde = stats.gaussian_kde(values, bw_method=bw_method, weights=df[weights])
        levels = np.reshape(kde(positions).T, x.shape)

        # Create the colormap
        if cmap is None:
            colors = ['#0000ff', '#00ff00', '#20e000', '#40c000', '#60a000', '#808000', '#a06000', '#c04000', '#e02000', '#ff0000']
            cmap = LinearSegmentedColormap.from_list('BuGrRd', colors, N=100)

        # Create the filled contour
        fig, ax = plt.subplots()
        cset = ax.contourf(x, y, levels, number_of_levels, cmap=cmap, alpha=alpha, antialiased=True)
        plt.close(fig)

        # Extract the geojson from contour
        for i in range(0, ignore_levels):
            del cset.collections[i]
        geojson = geojsoncontour.contourf_to_geojson(
            contourf=cset,
            ndigits=3,
            fill_opacity=alpha
        )
        geojson = loads(geojson)
        # Transform to EPSG:4326
        geojson = gpd.GeoDataFrame.from_features(geojson['features'], crs="epsg:3857").to_crs(epsg=4326).to_json()
        # Store computed values
        self.kde = kde
        self.grid = (x, y)
        self.cmap = cmap
        self.geojson = loads(geojson)
        self._levels = levels


    @property
    def pdf(self):
        """Returns the calculated probability density function."""
        return self._levels


    @property
    def bw(self):
        """Returns the bandwidth value."""
        return self.kde.covariance_factor()


    def evaluate_pdf(self, points):
        """Evaluates the estimated pdf on a set of points.
        Parameters:
            points (array): The set of points.
        Returns:
            (array) The pdf value at each point.
        """
        return self.kde(points)


    def plot(self, **kwargs):
        """Plots the heatmap into a Folium map.
        Parameters:
            kwargs: see .plots.geojson
        Returns:
            (object) A Folium Map object displaying the heatmap.
        """
        from .plots import geojson as geojsonPlot

        return geojsonPlot(self.geojson, **kwargs, styled=True)


    def to_geopandas_df(self, epsg=None):
        """Exports heatmap to GeoPandas dataframe.
        Parameters:
            epsg (int): The EPSG code to project the heatmap (default: no projection).
        Returns:
            (object) A Geopandas dataframe.
        """
        gdf = gpd.GeoDataFrame.from_features(self.geojson['features'], crs="epsg:4326")
        if epsg is not None:
            gdf = gdf.to_crs(epsg=epsg)
        return gdf


    def to_static_map(self, **kwargs):
        """Plots the heatmap into a static map.
        Parameters:
            **kwargs: Additional arguments for StaticMap.
        Returns:
            (object) A StaticMap object.
        """
        static = StaticMap(**kwargs)
        static.addHeatmap(
            *self.grid, self._levels,
            number_of_levels=self.number_of_levels,
            ignore_levels=self.ignore_levels,
            cmap=self.cmap,
            alpha=self.alpha,
            epsg=self.epsg
        )
        return static
