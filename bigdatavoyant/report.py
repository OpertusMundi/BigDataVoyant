import json
from dicttoxml import dicttoxml
from xml.dom.minidom import parseString
import yaml
import os

class Report(dict):
    """docstring for Report"""
    def __init__(self, dictionary):
        super(Report, self).__init__(dictionary)

    # def __repr__(self):
    #     return json.dumps(self['info'], indent=2)

    def __str__(self):
        return json.dumps(self, indent=2)

    def to_json(self, indent=None):
        return json.dumps(self, indent=indent)

    def to_xml(self):
        xml = dicttoxml(self, attr_type=False)
        dom = parseString(xml)
        return dom.toprettyxml()

    def to_yaml(self, indent=2):
        def noop(self, *args, **kw):
            pass
        yaml.emitter.Emitter.process_tag = noop
        return yaml.dump({**self}, indent=indent, default_flow_style=False, sort_keys=False)

    def to_file(self, filename, format=None, indent=2):
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
