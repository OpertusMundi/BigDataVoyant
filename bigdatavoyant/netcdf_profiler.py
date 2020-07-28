from netCDF4 import Dataset
import numpy as np
from pygeos import to_wkt, box
from pathlib import Path
from .report import Report

def get_info(ds, key, scope='variable'):
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
    def __init__(self, ds):
        self._ds = ds
        self._dimensions = [dim for dim in ds.dimensions]

    @property
    def list(self):
        return self._dimensions

    @property
    def size(self):
        return len(self._dimensions)

    def properties(self, dim=None):
        if dim is not None:
            return get_info(self._ds, dim, scope='dimension')
        return {dim: get_info(self._ds, dim, scope='dimension') for dim in self.list}


class Variables(object):
    def __init__(self, ds):
        self._ds = ds
        self._variables = [var for var in ds.variables]

    @property
    def list(self):
        return self._variables

    @property
    def size(self):
        return len(self._variables)

    def properties(self, var=None):
        if var is not None:
            return get_info(self._ds, var, scope='variable')
        return {var: get_info(self._ds, var, scope='variable') for var in self.list}


class NetCDFProfiler(object):

    def __init__(self, ds, lat_attr='lat', lon_attr='lon', time_attr='time'):
        assert isinstance(ds, Dataset)
        self._ds = ds
        self._lat_attr = lat_attr
        self._lon_attr = lon_attr
        self._time_attr = time_attr

    @classmethod
    def from_file(cls, filename, lat_attr='lat', lon_attr='lon', time_attr='time'):
        ds = Dataset(filename, 'r')
        return cls(ds, lat_attr=lat_attr, lon_attr=lon_attr, time_attr=time_attr)

    @property
    def dataset(self):
        return self._ds

    def metadata(self):
        attrs = self.dataset.ncattrs()
        return {attr: self.dataset.getncattr(attr).item() if hasattr(self.dataset.getncattr(attr), 'item') else self.dataset.getncattr(attr) for attr in attrs}

    def dimensions(self):
        return Dimensions(self.dataset)

    def variables(self):
        return Variables(self.dataset)

    def mbr(self):
        lat_min = self.dataset.variables[self._lat_attr][:].min()
        lat_max = self.dataset.variables[self._lat_attr][:].max()
        lon_min = self.dataset.variables[self._lon_attr][:].min()
        lon_max = self.dataset.variables[self._lon_attr][:].max()
        return to_wkt(box(lon_min, lat_min, lon_max, lat_max))

    def time_extent(self):
        time_min = self.dataset.variables[self._time_attr][:].min()
        time_max = self.dataset.variables[self._time_attr][:].max()
        unit = self.dataset.variables[self._time_attr].units
        return '%f - %f %s' % (time_min, time_max, unit)

    def no_data_values(self):
        ds = self.dataset
        return {var: ds.variables[var][:].fill_value.item() for var in ds.variables}

    def _stats(self, variable):
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
        if variable is not None:
            return self._stats(variable)
        return {var: self._stats(var) for var in self.dataset.variables}

    def sample(self, bbox, filename='sample.nc', format='NETCDF4', description='Sample'):
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

    def report(self, sample_bbox, sample_filename):
        self.sample(sample_bbox, filename=sample_filename, format='NETCDF4', description='Sample')
        dimensions = self.dimensions()
        variables = self.variables()
        report = {
            'metadata': self.metadata(),
            'dimensions_size': dimensions.size,
            'dimensions_list': dimensions.list,
            'dimensions_properties': dimensions.properties(),
            'variables_size': variables.size,
            'variables_list': variables.list,
            'variables_properties': variables.properties(),
            'mbr': self.mbr(),
            'temporal_extent': self.time_extent(),
            'no_data_values': self.no_data_values(),
            'sample': sample_filename,
            'statistics': self.stats()
        }
        return Report(report)
