"""Code adapted from: https://github.com/SLIPO-EU/loci.git"""
import folium
from folium.plugins import HeatMap
from mapclassify import NaturalBreaks
from shapely.geometry import box
import pygeos as pg

def bbox(gdf):
    """Computes the bounding box of a GeoDataFrame.

    Args:
        gdf (GeoDataFrame): A GeoDataFrame.

    Returns:
        A Polygon representing the bounding box enclosing all geometries in the GeoDataFrame.
    """

    minx, miny, maxx, maxy = gdf.geometry.total_bounds
    return box(minx, miny, maxx, maxy)

def heatmap(pois, tiles='OpenStreetMap', width='100%', height='100%', radius=10):
    """Generates a heatmap of the input POIs.

    Args:
        pois (GeoDataFrame): A POIs GeoDataFrame.
        tiles (string): The tiles to use for the map (default: `OpenStreetMap`).
        width (integer or percentage): Width of the map in pixels or percentage (default: 100%).
        height (integer or percentage): Height of the map in pixels or percentage (default: 100%).
        radius (float): Radius of each point of the heatmap (default: 10).

    Returns:
        A Folium Map object displaying the heatmap generated from the POIs.
    """

    # Set the crs to WGS84
    if pois.geometry.crs == 'EPSG:4326':
        pass
    else:
        pois.geometry.to_crs('EPSG:4326')

    # Automatically center the map at the center of the gdf's bounding box
    bb = pois.geometry.total_bounds()
    map_center = pg.get_coordinates(pg.centroid(bb))[0].tolist()

    heat_map = folium.Map(location=map_center, tiles=tiles, width=width, height=height)

    # Automatically set zoom level
    bounds = pg.total_bounds(bb)
    heat_map.fit_bounds(([bounds[1], bounds[0]], [bounds[3], bounds[2]]))

    # List comprehension to make list of lists
    heat_data = pois.geometry.get_coordinates(invert=True)
    # Plot it on the map
    HeatMap(heat_data, radius=radius).add_to(heat_map)

    return heat_map


def map_choropleth(areas, id_field, value_field, fill_color='YlOrRd', fill_opacity=0.6, num_bins=5,
                   tiles='OpenStreetMap', width='100%', height='100%'):
    """Returns a Folium Map showing the clusters. Map center and zoom level are set automatically.

    Args:
         areas (GeoDataFrame): A GeoDataFrame containing the areas to be displayed.
         id_field (string): The name of the column to use as id.
         value_field (string): The name of the column indicating the area's value.
         fill_color (string): A string indicating a Matplotlib colormap (default: YlOrRd).
         fill_opacity (float): Opacity level (default: 0.6).
         num_bins (int): The number of bins for the threshold scale (default: 5).
         tiles (string): The tiles to use for the map (default: `OpenStreetMap`).
         width (integer or percentage): Width of the map in pixels or percentage (default: 100%).
         height (integer or percentage): Height of the map in pixels or percentage (default: 100%).

    Returns:
        A Folium Map object displaying the given clusters.
    """

    # Set the crs to WGS84
    if areas.crs != 'EPSG:4326':
        areas = areas.to_crs('EPSG:4326')

    # Automatically center the map at the center of the bounding box enclosing the POIs.
    bb = bbox(areas)
    map_center = [bb.centroid.y, bb.centroid.x]

    # Initialize the map
    m = folium.Map(location=map_center, tiles=tiles, width=width, height=height)

    # Automatically set the zoom level
    m.fit_bounds(([bb.bounds[1], bb.bounds[0]], [bb.bounds[3], bb.bounds[2]]))

    threshold_scale = NaturalBreaks(areas[value_field], k=num_bins).bins.tolist()
    threshold_scale.insert(0, areas[value_field].min())

    choropleth = folium.Choropleth(areas, data=areas, columns=[id_field, value_field],
                                   key_on='feature.properties.{}'.format(id_field),
                                   fill_color=fill_color, fill_opacity=fill_opacity,
                                   threshold_scale=threshold_scale).add_to(m)

    # Construct tooltip
    fields = list(areas.columns.values)
    fields.remove('geometry')
    if 'style' in fields:
        fields.remove('style')
    tooltip = folium.features.GeoJsonTooltip(fields=fields)

    choropleth.geojson.add_child(tooltip)

    return m
