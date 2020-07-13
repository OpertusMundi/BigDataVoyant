import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bigdatavoyant",
    version="0.0.1",
    author="Pantelis Mitropoulos",
    author_email="pmitropoulos@getmap.gr",
    description="Geodata profiling tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/OpertusMundi/BigDataVoyant.git",
    packages=setuptools.find_packages(),
    install_requires=[
        'osgeo>=2.4.0,<3.2.0',
        'json>=2.0.9,<2.1.0',
        'dicttoxml>=1.7.4,<1.8.0',
        'yaml>=5.3.1,<5.4.0'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
