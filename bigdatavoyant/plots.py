"""Code partially adapted from: https://github.com/SLIPO-EU/loci.git"""
import folium
from folium.plugins import HeatMap
from mapclassify import NaturalBreaks
from shapely.geometry import box
import pygeos as pg
import geopandas as gpd

def bbox(gdf):
    """Computes the bounding box of a GeoDataFrame.

    Parameters:
        gdf (GeoDataFrame): A GeoDataFrame.

    Returns:
        A Polygon representing the bounding box enclosing all geometries in the GeoDataFrame.
    """

    minx, miny, maxx, maxy = gdf.geometry.total_bounds
    return box(minx, miny, maxx, maxy)

def heatmap(pois, basemap_provider='OpenStreetMap', basemap_name='Mapnik', width='100%', height='100%', radius=10):
    """Generates a heatmap of the input POIs.

    Parameters:
        pois (GeoDataFrame): A POIs GeoDataFrame.
        basemap_provider (string): The basemap provider.
        basemap_name: The basemap itself as named by the provider.
            List and preview of available providers and their basemaps can be found in https://leaflet-extras.github.io/leaflet-providers/preview/
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

    tiles, attribution, max_zoom = get_provider_info(basemap_provider, basemap_name)
    heat_map = folium.Map(location=map_center, tiles=tiles, attr=attribution, max_zoom=max_zoom, width=width, height=height)

    # Automatically set zoom level
    bounds = pg.total_bounds(bb)
    heat_map.fit_bounds(([bounds[1], bounds[0]], [bounds[3], bounds[2]]))

    # List comprehension to make list of lists
    heat_data = pois.geometry.get_coordinates(invert=True)
    # Plot it on the map
    HeatMap(heat_data, radius=radius).add_to(heat_map)

    return heat_map


def map_choropleth(areas, id_field, value_field, fill_color='YlOrRd', fill_opacity=0.6, num_bins=5,
                   basemap_provider='OpenStreetMap', basemap_name='Mapnik', width='100%', height='100%'):
    """Returns a Folium Map showing the clusters. Map center and zoom level are set automatically.

    Parameters:
        areas (GeoDataFrame): A GeoDataFrame containing the areas to be displayed.
        id_field (string): The name of the column to use as id.
        value_field (string): The name of the column indicating the area's value.
        fill_color (string): A string indicating a Matplotlib colormap (default: YlOrRd).
        fill_opacity (float): Opacity level (default: 0.6).
        num_bins (int): The number of bins for the threshold scale (default: 5).
        basemap_provider (string): The basemap provider.
        basemap_name: The basemap itself as named by the provider.
           List and preview of available providers and their basemaps can be found in https://leaflet-extras.github.io/leaflet-providers/preview/
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
    tiles, attribution, max_zoom = get_provider_info(basemap_provider, basemap_name)
    m = folium.Map(location=map_center, tiles=tiles, attr=attribution, max_zoom=max_zoom, width=width, height=height)

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


def get_provider_info(basemap_provider, basemap_name):
    """Gets provider info using contexily.
    Parameters:
        basemap_provider (string): The basemap provider.
        basemap_name: The basemap itself as named by the provider.
            List and preview of available providers and their basemaps can be found in https://leaflet-extras.github.io/leaflet-providers/preview/
    Returns:
        (tuple) Tiles url, attribution and max_zoom (if not provided, a default value 18 is returned).
    """
    from contextily import providers

    try:
        basemap = providers[basemap_provider]
        if basemap_name is not None:
            basemap = basemap[basemap_name]
    except KeyError:
        raise Exception('Basemap provider/name not found.')
    try:
        tiles = basemap['url']
        attribution = basemap['attribution']
    except KeyError:
        raise Exception('Basemap url not found (possible missing basemap_name).')
    if hasattr(basemap, 'max_zoom'):
        max_zoom = basemap['max_zoom']
    else:
        max_zoom = 18

    return (tiles, attribution, max_zoom)


def geojson(geojson, basemap_provider='OpenStreetMap', basemap_name='Mapnik', width='100%', height='100%', styled=False):
    """Plots into a Folium map.
    Parameters:
        geojson (dict): A geojson object.
        basemap_provider (string): The basemap provider.
        basemap_name: The basemap itself as named by the provider.
            List and preview of available providers and their basemaps can be found in https://leaflet-extras.github.io/leaflet-providers/preview/
         width (int|string): Width of the map in pixels or percentage (default: 100%).
         height (int|string): Height of the map in pixels or percentage (default: 100%).
         styled (bool): If True, follows the mapbox simple style, as proposed in https://github.com/mapbox/simplestyle-spec/tree/master/1.1.0.
    Returns:
        (object) A Folium Map object displaying the geoJSON.
    """
    df = gpd.GeoDataFrame.from_features(geojson['features'], crs="epsg:4326")

    bb = ymin, xmin, ymax, xmax = df.geometry.total_bounds
    map_center = pg.get_coordinates(pg.centroid(pg.box(xmin, ymin, xmax, ymax)))[0]
    tiles, attribution, max_zoom = get_provider_info(basemap_provider, basemap_name)
    m = folium.Map(location=map_center, tiles=tiles, attr=attribution, max_zoom=max_zoom, width=width, height=height)
    m.fit_bounds([[xmin, ymin], [xmax, ymax]])

    if styled:
        folium.GeoJson(
            df,
            name='geojson',
            style_function = lambda x: dict(
                color=x['properties']['stroke'],
                fillColor=x['properties']['fill'],
                fillOpacity=x['properties']['fill-opacity'],
                opacity=0.1,
                weight=x['properties']['stroke-width']
            )
        ).add_to(m)
    else:
        folium.GeoJson(df, name='geojson').add_to(m)

    return m
