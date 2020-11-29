"""Code adapted from: https://github.com/SLIPO-EU/loci.git"""
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.ops import cascaded_union
from shapely.geometry import MultiPoint

from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN, OPTICS
from .plots import map_choropleth

class Clustering(object):
    """
    Computes clusters using DBSCAN or OPTICS algorithm as implemented in sklearn or HDBSCAN.
    """

    def __init__(self, pois, alg="hdbscan", min_samples=None, eps=None, n_jobs=-1, **kwargs):
        """Computes clusters using the sklearn algorithms or HDBSCAN.
        Parameters:
            pois (GeoDataFrame): A POI GeoDataFrame.
            alg (string): The clustering algorithm to use (hdbscan, dbscan or optics; default: hdbscan).
            min_samples (float|integer): The number of samples in a neighborhood for a point
                to be considered as a core point. Expressed as an absolute number (int > 1) or
                a fraction of the number of samples (float between 0 and 1).
            eps (float): The neighborhood radius (used only in dbscan).
            n_jobs (integer): Number of parallel jobs to run in the algorithm (default: -1)
            **kwargs: Optional arguments depending on the algorithm.
        """
        if min_samples is None:
            min_samples = int(round(np.log(len(pois))))
        if alg == 'dbscan':
            assert eps is not None
        self.pois = pois
        self.alg = alg
        self.min_samples = min_samples
        self.eps = eps
        self.n_jobs = n_jobs

        # Prepare list of coordinates
        data_arr = pois.geometry.get_coordinates()

        # Compute the clusters
        if alg == 'hdbscan':
            min_cluster_size = kwargs.pop('min_cluster_size', 50)
            core_dist_n_jobs = kwargs.pop('core_dist_n_jobs', n_jobs)
            clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, core_dist_n_jobs=core_dist_n_jobs, **kwargs)
            labels = clusterer.fit_predict(data_arr)

            tree = clusterer.condensed_tree_.to_pandas()
            cluster_tree = tree[tree.child_size > 1]
            chosen_clusters = clusterer.condensed_tree_._select_clusters()

            eps_per_cluster = cluster_tree[cluster_tree.child.isin(chosen_clusters)].\
                drop("parent", axis=1).drop("child", axis=1).reset_index().drop("index", axis=1)
            eps_per_cluster['lambda_val'] = eps_per_cluster['lambda_val'].apply(lambda x: 1 / x)
            eps_per_cluster.rename(columns={'lambda_val': 'eps', 'child_size': 'cluster_size'}, inplace=True)

        else:
            if alg == 'dbscan':
                clusterer = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=n_jobs, **kwargs).fit(data_arr)
            elif alg == 'optics':
                clusterer = OPTICS(min_samples=min_samples, eps=eps, n_jobs=n_jobs, **kwargs).fit(data_arr)
            else:
                raise Exception('Implemented algoriths are hdbscan, dbscan and optics.')
            labels = clusterer.labels_

            num_of_clusters_no_noise = set(labels)
            num_of_clusters_no_noise.discard(-1)
            num_of_clusters_no_noise = len(num_of_clusters_no_noise)

            eps_per_cluster = pd.DataFrame({'eps': [eps] * num_of_clusters_no_noise})
            eps_per_cluster['cluster_size'] = 0

        # Assign cluster labels to initial POIs
        pois['cluster_id'] = labels

        # Separate POIs that are inside clusters from those that are noise
        pois_in_clusters = pois[pois.cluster_id > -1]
        pois_noise = pois[pois.cluster_id == -1]

        self._num_of_clusters = len(set(labels))
        self._pois_in_clusters = pois_in_clusters
        self._eps_per_cluster = eps_per_cluster
        self._pois_noise = pois_noise
        self._shape_type = None


    @property
    def info(self):
        """Returns info about the clusters."""
        return dict(
            num_of_clusters=self._num_of_clusters,
            num_of_clustered_pois=len(self._pois_in_clusters),
            num_of_outlier_pois=len(self._pois_noise)
        )


    @property
    def clusters(self):
        """Returns the clusters and their eps."""
        return (self._pois_in_clusters, self._eps_per_cluster)


    @property
    def noise(self):
        """Returns the noise points."""
        return self._pois_noise


    def shapes(self, shape_type=1):
        """Computes cluster shapes.
        Parameters:
            shape_type (int): The methods to use for computing cluster
                shapes (allowed values: 1, 2 or 3; default: 1 - fastest).
        Returns:
            A GeoDataFrame containing the cluster shapes.
        """
        pois = self._pois_in_clusters.to_geopandas_df()
        eps_per_cluster = self._eps_per_cluster
        if shape_type == 2:
            cluster_borders = pois.groupby(['cluster_id'], sort=False)['geometry'].agg([list, np.size])
            join_df = pd.merge(cluster_borders, eps_per_cluster, left_index=True, right_index=True, how='inner')
            cluster_list = []
            for index, row in join_df.iterrows():
                eps = row['eps']
                cluster_i = []
                for p in row['list']:
                    cluster_i.append(p.buffer(eps))

                cluster_list.append(cascaded_union(cluster_i))

            join_df['geometry'] = cluster_list
            join_df['cluster_id'] = join_df.index
            join_df.reset_index(drop=True, inplace=True)
            join_df.drop(['list', 'cluster_size'], axis=1, inplace=True)

            cluster_borders = gpd.GeoDataFrame(join_df, crs=pois.crs, geometry='geometry')
            cluster_borders = cluster_borders[['cluster_id', 'size', 'geometry']]

        elif shape_type == 3:
            eps_dict = dict()
            for index, row in eps_per_cluster.iterrows():
                eps_dict[index] = row['eps']

            circles_from_pois = pois.copy()
            cid_size_dict = dict()
            circles = []
            for index, row in circles_from_pois.iterrows():
                cid = row['cluster_id']
                circles.append(row['geometry'].buffer(eps_dict[cid]))
                cid_size_dict[cid] = cid_size_dict.get(cid, 0) + 1

            circles_from_pois['geometry'] = circles

            s_index = pois.sindex

            pois_in_circles = gpd.sjoin(pois, circles_from_pois, how="inner", op='intersects')
            agged_pois_per_circle = pois_in_circles.groupby(['cluster_id_left', 'index_right'],
                                                            sort=False)['geometry'].agg([list])

            poly_list = []
            cluster_id_list = []
            for index, row in agged_pois_per_circle.iterrows():
                pois_in_circle = row['list']
                lsize = len(pois_in_circle)
                if lsize >= 3:
                    poly = MultiPoint(pois_in_circle).convex_hull
                    poly_list.append(poly)
                    cluster_id_list.append(index[0])

            temp_df = pd.DataFrame({
                'cluster_id': cluster_id_list,
                'geometry': poly_list
            })

            grouped_poly_per_cluster = temp_df.groupby(['cluster_id'], sort=False)['geometry'].agg([list])

            cluster_size_list = []
            poly_list = []
            for index, row in grouped_poly_per_cluster.iterrows():
                poly_list.append(cascaded_union(row['list']))
                cluster_size_list.append(cid_size_dict[index])

            grouped_poly_per_cluster['geometry'] = poly_list
            grouped_poly_per_cluster.drop(['list'], axis=1, inplace=True)

            cluster_borders = gpd.GeoDataFrame(grouped_poly_per_cluster, crs=pois.crs, geometry='geometry')
            cluster_borders['cluster_id'] = cluster_borders.index
            cluster_borders['size'] = cluster_size_list

        elif shape_type == 1:
            cluster_borders = pois.groupby(['cluster_id'], sort=False)['geometry'].agg([list, np.size])
            cluster_borders['list'] = [MultiPoint(l).convex_hull for l in cluster_borders['list']]
            cluster_borders.rename(columns={"list": "geometry"}, inplace=True)
            cluster_borders.sort_index(inplace=True)
            cluster_borders = gpd.GeoDataFrame(cluster_borders, crs=pois.crs, geometry='geometry')
            cluster_borders.reset_index(inplace=True)

        else:
            raise Exception('Argument for shape_type could be 1, 2 or 3')

        self._cluster_borders = cluster_borders
        self._shape_type = shape_type
        return cluster_borders


    def plot(self, shape_type=None, **kwargs):
        """Plots the clusters.

        If shapes have not been computed for this shape_type,
            shapes is called before plotting.

        Parameters:
            shape_type (int): The methods to use for computing cluster
                shapes (allowed values: 1, 2 or 3; default: 1 - fastest).
            **kwargs: Optional arguments for map_choropleth.
        Returns (object): A choropleth Folium map.
        """
        if shape_type is None:
            if self._shape_type is not None:
                shape_type = self._shape_type
            else:
                shape_type = 1
        if self._shape_type is None or self._shape_type != shape_type:
            cluster_borders = self.shapes(shape_type)
        cluster_borders = self._cluster_borders
        return map_choropleth(cluster_borders, **kwargs, id_field='cluster_id', value_field='size')
