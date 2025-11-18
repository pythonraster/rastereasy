"""
rastereasy: A Python library for raster data processing.

This package provides tools to handle georeferenced raster images with functionalities
like resampling, reprojection, NDVI computation, and more.

Modules:
--------
- `rastereasy`: Core functionalities for raster data.
- `utils`: Helper functions for common operations.

Example:
--------
>>> import rastereasy
>>> image = rastereasy.Geoimage("example.tif")
>>> image.info()
>>> image.colorcomp(['4', '3', '2'])
>>> image.resample(2, inplace= True)
>>> im_reproject = image.reproject("EPSG:4326")
>>> im_reproject.save("output.tif")
"""

# Import essential classes and functions to simplify access for users
from .rastereasy import *
from .utils import *

# Define package metadata
__version__ = "0.3.2"
__author__ = "Thomas Corpetti"
__license__ = "MIT"

# Optional: Provide a __all__ variable to define the public API
