from io import BytesIO
import base64
import contextily as ctx
import matplotlib.pyplot as plt
from pathlib import Path
from numpy import ndarray, generic

class StaticMap(object):
    """Creation of static maps."""
    def __init__(self, basemap_provider="OpenStreetMap", basemap_name="Mapnik", aspect_ratio=None, width=1920, height=None):
        """Initiates the StaticMap class, adjusting the canvas size and basemap.
        Parameters:
            basemap_provider (string): The basemap provider.
            basemap_name: The basemap itself as named by the provider.
                List and preview of available providers and their basemaps can be found in https://leaflet-extras.github.io/leaflet-providers/preview/
            aspect_ratio: The aspect ratio - width/height - of the map.
            width: The map width in pixels.
            height: The map height in pixels.
                Only two of aspect_ratio, width, height have to be defined. If all three are given, then aspect_ratio is ignored and recalculated
                from the values of the other two.
        """
        dpi = 100
        if height is None and aspect_ratio is None:
            aspect_ratio = 16/9
            height = width / aspect_ratio
        elif aspect_ratio is None:
            aspect_ratio = width / height
        else:
            height = width / aspect_ratio
        try:
            self.basemap = ctx.providers[basemap_provider]
            if basemap_name is not None:
                self.basemap = self.basemap[basemap_name]
        except KeyError:
            raise Exception('Basemap provider/name not found.')
        self.aspect_ratio = aspect_ratio
        self.width = width
        self.height = height
        self.dpi = dpi
        self.map = None

        self._width = width / dpi
        self._height = height / dpi


    def getBorders(self, df, aspect_ratio):
        """Calculates the map vertices for a geometric object, given its aspect ratio.
        Parameters:
            df (object): GeoPandas dataframe or numpy array.
            aspect_ratio (float): The aspect ratio of the map.
        Returns:
            (tuple) The coordinates of the map bounding rectangle in form (xmin, ymin, xmax, ymax).
        """
        if isinstance(df, (ndarray, generic)):
            minx, miny = df.min(axis=0)
            maxx, maxy = df.max(axis=0)
        else:
            minx, miny, maxx, maxy = df.total_bounds
        diffx = maxx - minx
        diffy = maxy - miny
        minx -= 0.05*diffx
        maxx += 0.05*diffx
        miny -= 0.05*diffy
        maxy += 0.05*diffy
        diffx = maxx - minx
        diffy = maxy - miny

        if diffy >= diffx/aspect_ratio:
            offset = 0.5 * (aspect_ratio*diffy - diffx)
            minx -= offset
            maxx += offset
        else:
            offset = 0.5 * (diffx/aspect_ratio - diffy)
            miny -= offset
            maxy += offset

        return (minx, miny, maxx, maxy)


    def addGeometries(self, gdf, weight=None, cmap='YlOrRd', colorbar_label=None, fontsize=18):
        """Add geometries to the map.
        Parameters:
            gdf (object): A GeoPandas dataframe.
            weight (string): The gdf attribute representing the weight of each geometry. If None, each geometry weights the same.
            cmap (string): The representation of color map.
            colorbar_label (string): The label of the colorbar axis.
            fontsize (int): The font size in colorbar.
        Returns:
            (obj) The matplotlib plot.
        """
        cb = gdf.to_crs('EPSG:3857')
        fig, ax = plt.subplots(figsize = (self._width, self._height), dpi=self.dpi)

        minx, miny, maxx, maxy = self.getBorders(cb, self.aspect_ratio)

        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

        plot_args = dict(
            ax=ax,
            legend=False,
            edgecolor='black'
        )
        if weight is not None:
            plot_args['column'] = weight
            plot_args['cmap'] = cmap

        cb.plot(**plot_args)
        plt.xticks([], [])
        plt.yticks([], [])
        ctx.add_basemap(ax, source=self.basemap)

        if weight is not None:
            vmin = cb['size'].min()
            vmax = cb['size'].max()

            cax = fig.add_axes([0.65, 0.95, 0.3, 0.01])
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
            cbr = fig.colorbar(sm, cax=cax, orientation='horizontal')
            if colorbar_label is not None:
                cbr.set_label(colorbar_label, fontsize=fontsize)
            cbr.ax.tick_params(labelsize=fontsize)

        self.map = plt
        return plt


    def base64(self):
        """Base64 png encoding of the map.
        Returns:
            (string) The base64 encoded png map.
        """
        plt = self.map
        img = BytesIO()
        plt.tight_layout()
        plt.savefig(img, format='png')
        img.seek(0)
        base64_img = base64.b64encode(img.read()).decode('utf8')

        return base64_img


    def toFile(self, file, **kwargs):
        """Saves static map to file.
        Parameters:
            file (string): The full path of the image file.
            **kwargs: Keyword arguments for matplotlib.pyplot.savefig
        """
        from pathlib import Path
        plt = self.map
        plt.tight_layout()
        try:
            plt.savefig(file, **kwargs)
            print("Wrote file %s, %d bytes." % (file, Path(file).stat().st_size))
        except:
            raise Exception('ERROR: Could not write to filesystem.')
        plt.savefig(file, **kwargs)
