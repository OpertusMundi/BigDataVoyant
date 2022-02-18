from collections import Counter
import os
import math

def counter_cosine_similarity(c1, c2):
    """Calculates the cosine distance between two lists.

    Parameters:
        c1 (collections.Counter): The first list
        c2 (collections.Counter): The second list

    Returns:
        (float) Magnitude of the cosine similarity.
    """
    terms = set(c1).union(c2)
    dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
    magA = math.sqrt(sum(c1.get(k, 0)**2 for k in terms))
    magB = math.sqrt(sum(c2.get(k, 0)**2 for k in terms))
    return dotprod / (magA * magB)

def length_similarity(c1, c2):
    """Calculate a length similarity between two lists.

    Parameters:
        c1 (collections.Counter): The first list.
        c2 (collections.Counter): The second list.

    Returns:
        (float) Magnitude of length similarity.
    """
    lenc1 = sum(c1.values())
    lenc2 = sum(c2.values())
    return min(lenc1, lenc2) / float(max(lenc1, lenc2))

def similarity_score(l1, l2):
    """Calculates a similarity score between two lists, combining cosine and length similarities.

    Parameters:
        l1 (list): The first list.
        l2 (list): The second list.

    Returns:
        (float) Similarity score
    """
    c1, c2 = Counter(l1), Counter(l2)
    return round(length_similarity(c1, c2) * counter_cosine_similarity(c1, c2), 3)

def _read_yaml(file):
    """Reads a YAML file.

    Parameters:
        file (str): The full path of the YAML file.

    Returns:
        (collections.OrderedDict) The YAML file content
    """
    import yaml
    with open(file, 'r') as stream:
        try:
            content = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            warnings.warn('Error while reading YAML file.')
            return None
    return content

def _filtered_listdir(path, ext):
    """Extends os.listdir with filter capabilities on file extension.

    Parameters:
        path (str): The directory path.
        ext (str): The file extension.

    Yields:
        (str) Full path of the file
    """
    for file in os.listdir(path):
        if not file.endswith(ext):
            continue
        yield os.path.join(path, file)

def read_yaml_files(path):
    """Reads the content of YAML files contained in a given directory.

    Parameters:
        path (str): The path of the directory.

    Returns:
        (list) list of the file contents.
    """
    return [_read_yaml(file) for file in _filtered_listdir(path, '.yml')]

def _normalize_datatype(name):
    """Normalizes numpy datatypes to generic names.

    Parameters:
        name (type): The numpy datatype.

    Returns:
        (str) The normalized datatype.
    """
    datatype = str(name)
    if 'str' in datatype:
        datatype = 'string'
    elif 'int' in datatype:
        datatype = 'integer'
    elif 'float' in datatype:
        datatype = 'float'
    return datatype

def _get_datatypes(df, attributes):
    """Get a DataFrame datatypes according to given attributes.

    Parameters:
        df (vaex.DataFrame)
        attributes (dict)

    Returns:
        (list) List of datatypes, sorted with the given attributes.
    """
    names = [attr['name'] for attr in attributes]
    datatypes = []
    for name in names:
        try:
            datatypes.append(_normalize_datatype(df.data_type(name)))
        except KeyError:
            datatypes.append(None)
        except NameError:
            datatypes.append(None)
    return datatypes

def get_similarity_scores(df, def_path):
    """Get similarity scores of a dataframe with schemata defined in YAML files contained in a given folder.

    Args:
        df (vaex.DataFrame)
        def_path (str): The full path of a folder containing the YAML files with schema definitions.

    Yields:
        (tuple) Schema name and similarity score
    """
    for schema in read_yaml_files(def_path):
        datatypes = [attr['type'] for attr in schema['attributes']]
        yield schema['name'], similarity_score(datatypes, _get_datatypes(df, schema['attributes']))
