from netCDF4 import Dataset
import numpy as np
from pygeos import to_wkt, box
from pathlib import Path
from .report import Report

def get_info(ds, key, scope='variable'):
    """Gets info about a specific attribute of the dataset.
    The returned ensemble of information depends on the type (scope) of the attribute.
    Parameters:
        ds (object): The (numpy) dataset.
        key (string): The name of the attribute.
        scope (string): The type of the attribute: variable (default) or dimension.
    """
    try:
        info = {attr: ds.variables[key].getncattr(attr).item() if hasattr(ds.variables[key].getncattr(attr), 'item') else ds.variables[key].getncattr(attr) for attr in ds.variables[key].ncattrs()}
        entity_type = ds.variables[key].dtype.name
    except KeyError:
        info = {}
        entity_type = 'NO VARIABLE ATTRIBUTES'
    if scope == 'dimension':
        info = {'type': entity_type, 'size': len(ds.dimensions[key]), **info}
    else:
        info = {'dimensions': ds.variables[key].dimensions, 'type': entity_type, 'size': ds.variables[key].size, **info}
    return info

class Dimensions(object):
    """A class gathering useful methods for dimensions.
    Attributes:
        _ds (object): The (numpy) dataset.
        _dimensions (list): A list of all dimensions contained in the dataset.
    """
    def __init__(self, ds):
        """Initiates the Dimensions class.
        Parameters:
            ds: The (numpy) dataset.
        """
        self._ds = ds
        self._dimensions = [dim for dim in ds.dimensions]

    @property
    def list(self):
        """Returns a list of all the dimensions in the dataset."""
        return self._dimensions

    @property
    def size(self):
        """Returns the number of dimensions."""
        return len(self._dimensions)

    def properties(self, dim=None):
        """Get the properties of a specific or all dimensions.
        Parameters:
            dim (string): A specific dimension. If None, properties of all dimensions in the dataset will be computed.
        Returns:
            (dict): The computed properties.
        """
        if dim is not None:
            return get_info(self._ds, dim, scope='dimension')
        return {dim: get_info(self._ds, dim, scope='dimension') for dim in self.list}


class Variables(object):
    """A class gathering useful methods for variables.
    Attributes:
        _ds (object): The (numpy) dataset.
        _variables (list): A list of all variables contained in the dataset.
    """
    def __init__(self, ds):
        """Initiates the Variables class.
        Parameters:
            ds: The (numpy) dataset.
        """
        self._ds = ds
        self._variables = [var for var in ds.variables]

    @property
    def list(self):
        """Returns a list of all the variables in the dataset."""
        return self._variables

    @property
    def size(self):
        """Returns the number of variables."""
        return len(self._variables)

    def properties(self, var=None):
        """Get the properties of a specific or all variables.
        Parameters:
            var (string): A specific variable. If None, properties of all variables in the dataset will be computed.
        Returns:
            (dict): The computed properties.
        """
        if var is not None:
            return get_info(self._ds, var, scope='variable')
        return {var: get_info(self._ds, var, scope='variable') for var in self.list}


class NetCDFProfiler(object):
    """NetCDF Profiler main class.
    Attributes:
        _ds (object): The (numpy) dataset.
        _lat_attr (string): The variable name containing the latitude coordinate.
        _lon_attr (string): The variable name containing the longitude coordinate.
        _time_attr (string): The variable name containing the time coordinate.
    """

    def __init__(self, ds, lat_attr='lat', lon_attr='lon', time_attr='time'):
        """Creates a NetCDF Profiler object from a (numpy) dataset.
        Parameters:
            ds (object): The (numpy) dataset.
            lat_attr (string): The variable name containing the latitude coordinate.
            lon_attr (string): The variable name containing the longitude coordinate.
            time_attr (string): The variable name containing the time coordinate.
        """
        assert isinstance(ds, Dataset)
        self._ds = ds
        self._lat_attr = lat_attr
        self._lon_attr = lon_attr
        self._time_attr = time_attr

    @classmethod
    def from_file(cls, filename, lat_attr='lat', lon_attr='lon', time_attr='time'):
        """Creates a NetCDF Profiler object from a netcdf file.
        Parameters:
            filename (string): The full path of the netcdf file.
            lat_attr (string): The variable name containing the latitude coordinate.
            lon_attr (string): The variable name containing the longitude coordinate.
            time_attr (string): The variable name containing the time coordinate.
        """
        ds = Dataset(filename, 'r')
        return cls(ds, lat_attr=lat_attr, lon_attr=lon_attr, time_attr=time_attr)

    @property
    def dataset(self):
        """Returns the raw dataset."""
        return self._ds

    def metadata(self):
        """
        Returns:
            (dict) The file's metadata for each attribute.
        """
        attrs = self.dataset.ncattrs()
        return {attr: self.dataset.getncattr(attr).item() if hasattr(self.dataset.getncattr(attr), 'item') else self.dataset.getncattr(attr) for attr in attrs}

    def dimensions(self):
        """Creates a dimension object for the dataset.
        Returns:
            (object) Dimensions object.
        """
        return Dimensions(self.dataset)

    def variables(self):
        """Creates a variable object for the dataset.
        Returns:
            (object) Variables object.
        """
        return Variables(self.dataset)

    def mbr(self):
        """Computes the MBR of the dataset.
        Returns:
            (string) The WKT representation of the MBR.
        """
        lat_min = self.dataset.variables[self._lat_attr][:].min()
        lat_max = self.dataset.variables[self._lat_attr][:].max()
        lon_min = self.dataset.variables[self._lon_attr][:].min()
        lon_max = self.dataset.variables[self._lon_attr][:].max()
        return to_wkt(box(lon_min, lat_min, lon_max, lat_max))

    def time_extent(self):
        """Computes the time extent of the dataset.
        Returns:
            (string) A string representation of the time extent (containing the unit).
        """
        time_min = self.dataset.variables[self._time_attr][:].min()
        time_max = self.dataset.variables[self._time_attr][:].max()
        unit = self.dataset.variables[self._time_attr].units
        return '%f - %f %s' % (time_min, time_max, unit)

    def no_data_values(self):
        """Retrieves the NO DATA VALUE for each attribute.
        Returns:
            (dict) The no data value for each attribute.
        """
        ds = self.dataset
        return {var: ds.variables[var][:].fill_value.item() for var in ds.variables}

    def _stats(self, variable):
        """Computes descriptive statistics for one variable.
        Parameters:
            variable (string): The variable name.
        Returns:
            (dict) A dictionary with the statistic measure and its value.
        """
        values = self.dataset.variables[variable][:]
        stats = {
            'count': values.count(),
            'missing': np.ma.count_masked(values).item(),
            'min': values.min().item(),
            'max': values.max().item(),
            'mean': values.mean().item(),
            'std': values.std().item(),
            'variance': values.var().item(),
            'contiguous': values.iscontiguous()
        }
        return stats

    def stats(self, variable=None):
        """Computes descriptive statistics for one or all variables contained in the dataset.
        Parameters:
            variable (string): The variable name. If None, statistics for all variables will be computed.
        Returns:
            (dict) A dictionary with the statistic measure and its value (for each variable, in case variable argument is None).
        """
        if variable is not None:
            return self._stats(variable)
        return {var: self._stats(var) for var in self.dataset.variables}

    def sample(self, bbox, filename='sample.nc', format='NETCDF4', description='Sample'):
        """Creates, and writes to a netcdf file, a sample of the dataset.
        Parameters:
            bbox (tuple): A bounding box for the sample.
            filename (string): The name of the file in which the sample will be written.
            format (string): The dataset format in the file (netcdf version; default NETCDF4).
            description (string): The description of the sample, written in the file metadata.
        """
        ds = self.dataset
        lon_min, lat_min, lon_max, lat_max = bbox
        lats = ds.variables[self._lat_attr][:]
        lons = ds.variables[self._lon_attr][:]
        lat_ids = np.fromiter((idx for idx in range(lats.shape[0]) if lats[idx] < lat_max and lats[idx] > lat_min), dtype=np.int)
        lon_ids = np.fromiter((idx for idx in range(lons.shape[0]) if lons[idx] < lon_max and lons[idx] > lon_min), dtype=np.int)
        # Open file
        sample_ds = Dataset(filename, 'w', format=format)
        # Write metadata
        sample_ds.description = description
        meta = self.metadata()
        for key in meta:
            sample_ds.setncattr(key, meta[key])
        # Create dimensions
        dims = Dimensions(ds).list
        for dim in dims:
            sample_ds.createDimension(dim, None)
            try:
                sample_ds_dim = sample_ds.createVariable(dim, ds.variables[dim].dtype, (dim))
            except KeyError:
                pass
            else:
                for attr in ds.variables[dim].ncattrs():
                    sample_ds_dim.setncattr(attr, ds.variables[dim].getncattr(attr))
                if dim == self._lat_attr:
                    sample_ds.variables[dim][:] = ds.variables[dim][:].take(lat_ids)
                elif dim == self._lon_attr:
                    sample_ds.variables[dim][:] = ds.variables[dim][:].take(lon_ids)
                else:
                    sample_ds.variables[dim][:] = ds.variables[dim][:]
        # Write variables
        for name in ds.variables:
            if name in dims:
                continue
            sample_ds_var = sample_ds.createVariable(name, ds.variables[name].dtype, dimensions=ds.variables[name].dimensions)
            var = ds.variables[name]
            sample_ds_var.setncatts({attr: ds.variables[name].getncattr(attr) for attr in ds.variables[name].ncattrs()})
            if self._lat_attr in var.dimensions and self._lon_attr in var.dimensions:
                lat_ax = var.dimensions.index(self._lat_attr)
                lon_ax = var.dimensions.index(self._lon_attr)
                sample_ds.variables[name][:] = ds.variables[name][:].take(lat_ids, lat_ax).take(lon_ids, lon_ax)
            else:
                sample_ds.variables[name][:] = ds.variables[name][:]
        print("Wrote file %s, %d bytes." % (filename, Path(filename).stat().st_size))
        sample_ds.close()

    def report(self):
        """Creates a report with a collection of metadata.
        Returns:
            (object) A report object.
        """
        dimensions = self.dimensions()
        variables = self.variables()
        report = {
            'assetType': 'NetCDF',
            'metadata': self.metadata(),
            'dimensionsSize': dimensions.size,
            'dimensionsList': dimensions.list,
            'dimensionsProperties': dimensions.properties(),
            'variablesSize': variables.size,
            'variablesList': variables.list,
            'variablesProperties': variables.properties(),
            'mbr': self.mbr(),
            'temporalExtent': self.time_extent(),
            'noDataValues': self.no_data_values(),
            'statistics': self.stats()
        }
        return Report(report)
