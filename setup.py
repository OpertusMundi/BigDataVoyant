import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bigdatavoyant",
    version="1.2.4",
    author="Pantelis Mitropoulos",
    author_email="pmitropoulos@getmap.gr",
    description="Geodata profiling tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/OpertusMundi/BigDataVoyant.git",
    packages=setuptools.find_packages(),
    install_requires=[
        'gdal>=3.0.2,<3.2.0',
        'dicttoxml>=1.7.4,<1.8.0',
        'pandas>=1.0.3,<1.0.4',
        'geopandas>=0.7.0,<0.7.1',
        'numpy>=1.18.4,<1.18.5',
        'shapely>=1.7.0,<1.7.1',
        'hdbscan>=0.8.26,<0.8.27',
        'joblib>=0.17.0,<1.0.0',
        'scikit-learn>=0.23.1,<0.23.2',
        'matplotlib>=3.2.1,<3.2.2',
        'geovaex>=0.1.0',
        'folium>=0.11.0,<0.11.1',
        'mapclassify>=2.2.0,<2.2.1',
        'pygeos>=0.8.0,<1.0.0',
        'vaex>=3.0.0,<3.0.1',
        'contextily>=1.0.0,<1.0.1',
        'netCDF4>=1.5.3,<1.5.4',
        'scipy>=1.5.0,<1.5.9',
        'geojsoncontour>=0.4.0,<0.5.0',
        'descartes>=1.1.0,<1.1.1',
        'simplejson>=3.17.5,<3.17.6'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
