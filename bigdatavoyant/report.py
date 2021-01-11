import json
import numpy
from dicttoxml import dicttoxml
from xml.dom.minidom import parseString
import yaml
import os


def convert(o):
    if isinstance(o, numpy.int64) or isinstance(o, numpy.int32):
        return int(o)
    elif isinstance(o, numpy.float64) or isinstance(o, numpy.float32):
        return float(o)
    raise TypeError


class Report(dict):
    """A collection of useful methods for reporting."""

    def __init__(self, dictionary):
        """Creates a report object.
        Parameters:
            dictionary: A key-value dictionary with the name and the value of each attribute contained in the report.
        """
        super(Report, self).__init__(dictionary)

    def __str__(self):
        """Overrides the parent method."""
        return json.dumps(self, indent=2, default=convert)

    def to_json(self, indent=None):
        """Converts to JSON.
        Parameters:
            indent: The indentation spaces.
        Returns:
            (string) json representation of the report.
        """
        return json.dumps(self, indent=indent, default=convert)

    def to_xml(self):
        """Converts to xml.
        Returns:
            (string) xml representation of the report.
        """
        xml = dicttoxml(self, attr_type=False)
        dom = parseString(xml)
        return dom.toprettyxml()

    def to_yaml(self, indent=2):
        """Converts to yaml
        Parameters:
            indent: The indentation spaces (default: 2).
        Returns:
            (string) yaml representation of the report.
        """
        def noop(self, *args, **kw):
            pass
        yaml.emitter.Emitter.process_tag = noop
        return yaml.dump({**self}, indent=indent, default_flow_style=False, sort_keys=False)

    def to_file(self, filename, format=None, indent=2):
        """Writes report to file.
        Parameters:
            filename (string): The output filename (full path).
            format: The file format. One of json, xml or yml. If None, extracts format from file extension.
            indent: The indentation spaces (default: 2; for json and yaml formats only).
        """
        from pathlib import Path
        name = os.path.basename(filename)
        ext = os.path.splitext(name)[1].split('.')[1] if format is None else format
        if ext == 'json':
            content = self.to_json(indent)
        elif ext == 'xml':
            content = self.to_xml()
        elif ext == 'yml':
            content = self.to_yaml(indent)
        else:
            raise Exception('ERROR: Unrecognized file format.')

        file = open(filename, 'w')
        file.write(content)
        file.close()
        print("Wrote file %s, %d bytes." % (filename, Path(filename).stat().st_size))
