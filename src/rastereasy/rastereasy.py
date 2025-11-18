"""
Core module for the rastereasy package.
Contains the main Geoimage class and utility functions for raster processing.
"""

import os
# Google Colab detection and setup
import matplotlib


try:
    import google.colab
    IN_COLAB = True
    if IN_COLAB:
        from google.colab import output
        output.enable_custom_widget_manager()
        from google.colab import drive
        drive.mount('/content/drive')
        os.system('pip install rasterio')
        os.system('pip install ipympl')
except ImportError:
    IN_COLAB = False
    matplotlib.use('Qt5Agg') #  'Qt5Agg' 'TkAgg'
end_collect = False  # Flag used for collecting spectra

# Standard library imports
import glob
from itertools import product
import warnings
import copy
import datetime
import json

# Scientific and numerical libraries
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_webagg_core import NavigationToolbar2WebAgg

# Rasterio and geospatial imports
import rasterio as rio
from rasterio import windows
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.transform import Affine
from rasterio.plot import show_hist
from rasterio.features import rasterize
from rasterio.transform import from_bounds

# Geopandas for vector operations
import geopandas as gpd

# Local imports with relative path
from .utils import *
from scipy.ndimage import (
    gaussian_filter,
    median_filter,
    laplace,
    sobel
)
# Configure warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# Configure matplotlib backend based on environment
if os.environ.get('DISPLAY', '') == '':
    # Use 'agg' if no GUI is detected
    matplotlib.use('agg')
#else:
    # Use 'tkagg' for standard interactive display (commented out in original)
#    matplotlib.use('tkagg')

# Default constants
DEF_FIG_SIZE = (5, 5)  # Default figure size for visualizations
RANDOM_STATE = None    # Random state for reproducible results







def read_geoim(source_name, read_image=True, channel_first=True):
    """
    Read a geotiff image file.

    Parameters
    ----------
    source_name : str
        Path to the geotiff image file
    read_image : bool, optional
        If True, read both image data and metadata. If False, read only metadata.
        Default is True.
    channel_first : bool, optional
        If True, return image with shape (bands, rows, cols).
        If False, return image with shape (rows, cols, bands).
        Only relevant when read_image is True.
        Default is True.

    Returns
    -------
    numpy.ndarray or None
        Image data if read_image is True, None otherwise
    dict
        Metadata dictionary from rasterio
    names
        Metadata dictionary with names of the bands (if given)

    Examples
    --------
    >>> # Read image data and metadata
    >>> image, meta, names = read_geoim("path/to/image.tif")
    >>>
    >>> # Read only metadata
    >>> meta, names = read_geoim("path/to/image.tif", read_image=False)
    """
    src = rio.open(source_name)
    extra_tags = None
    tags = src.tags()
    if "EXTRA_TAGS" in tags:
        try:
            extra_tags = json.loads(tags["EXTRA_TAGS"])
        except json.JSONDecodeError:
            extra_tags = tags["EXTRA_TAGS"]
    if read_image:
        if channel_first:
            return src.read(), src.meta, extra_tags
        else:
            return np.rollaxis(src.read(), 0, 3), src.meta, extra_tags
    else:
        return src.meta, extra_tags


def write_geoim(im, meta_input, dest_name, channel_first=True, names=None):
    """
    Write a geotiff image to disk.

    Parameters
    ----------
    im : numpy.ndarray
        Image data to write
    meta_input : dict
        Metadata dictionary with rasterio metadata fields
    dest_name : str
        Output path for the geotiff file
    names : dict
        Names (in a dict) associated with spectral bands
    channel_first : bool, optional
        If True, assumes image has shape (bands, rows, cols).
        If False, assumes image has shape (rows, cols, bands).
        Default is True.

    Examples
    --------
    >>> write_geoim(image, metadata, "output.tif")
    """
    meta = meta_input.copy()

    # Create output directory if it doesn't exist
    folder = os.path.split(dest_name)[0]
    if os.path.exists(folder) is False and folder != '':
        os.makedirs(folder)

    # Remove existing file to avoid conflicts
    if os.path.exists(dest_name):
        os.remove(dest_name)
    if os.path.exists(f'{dest_name}.aux.xml'):
        os.remove(f'{dest_name}.aux.xml')

    # Set driver based on file extension
    if os.path.splitext(dest_name)[1] == '.tif':
        meta['driver'] = 'GTiff'
    elif os.path.splitext(dest_name)[1] == '.jp2':
        meta['driver'] = 'JP2OpenJPEG'

    # Handle boolean data type
    if meta['dtype'] == 'bool':
        im = im.astype(np.uint8)
        meta.update(dtype=rio.uint8)

    # Write the image
    if channel_first:
        with rio.open(dest_name, 'w', **meta) as dst:
            dst.write(im)
            if names is not None:
                dst.update_tags(EXTRA_TAGS=json.dumps(names))
    else:
        with rio.open(dest_name, 'w', **meta) as dst:
            dst.write(np.rollaxis(im, 2, 0))
            if names is not None:
                dst.update_tags(EXTRA_TAGS=json.dumps(names))

def diff_im(im1, im2):
    """
    Calculate the L1 norm (sum of absolute differences) between two Geoimages.

    Parameters
    ----------
    im1 : Geoimage
        First image
    im2 : Geoimage
        Second image

    Returns
    -------
    float
        L1 norm of the difference between the images
    """
    return np.abs(np.sum(im1.image - im2.image))


def im2tiles_sequence(source_name, dest_name, nb_row, nb_col, overlap=0, type_name='sequence', verbose=0):
    """
    Split a sequence of GeoTIFF images into tiles.

    Parameters
    ----------
    source_name : str
        Directory containing the GeoTIFF images.
    dest_name : str
        Destination directory for tiled images.
    nb_row : int
        Number of rows in each tiled image.
    nb_col : int
        Number of columns in each tiled image.
    overlap : int, optional
        Overlap between tiles in pixels. Default is 0.
    type_name : str, optional
        Naming convention for output tiles:
        - "sequence": Outputs as name_tif_001.tif, name_tif_002.tif, etc.
        - "coord": Outputs as name_tif_tiles_{row}-{col}.tif.
        Default is "sequence".
    verbose : int, optional
        If 1, prints information about processed images. Default is 0.

    Raises
    ------
    ValueError
        If type_name is not "sequence" or "coord".

    Examples
    --------
    >>> im2tiles_sequence(
    >>>      source_name="/path/to/source",
    >>>      dest_name="/path/to/destination",
    >>>      nb_row=256,
    >>>      nb_col=256,
    >>>      overlap=10,
    >>>      type_name="sequence",
    >>>      verbose=1)
    """
    if not (type_name == "sequence" or type_name == "coord"):
        raise ValueError("type_name must be either sequence or coord")

    # Get list of all tif files in the source directory
    list_im = glob.glob(f'{source_name}/*tif')

    # Create destination directory if it doesn't exist
    if not os.path.exists(dest_name):
        os.makedirs(dest_name)
        print('creation of folder ', dest_name)

    # Process each image
    for i in range(len(list_im)):
        if verbose == 1:
            print(f'process {list_im[i]}')
        split_image_to_tiles(
            list_im[i],
            dest_name,
            nb_col,
            nb_row,
            overlap=overlap,
            name=type_name,
            verbose=verbose,
            name_tile=None
        )


def im2tiles(source_name, dest_name, nb_row, nb_col, overlap=0, type_name='sequence', verbose=0, name_tile=None, reset=True):
    """
    Split a geotif image into tiles.

    Parameters
    ----------
    source_name : str
        A tif file to be split.
    dest_name : str
        Destination directory for tiled images.
    nb_row : int
        Number of rows in each tiled image.
    nb_col : int
        Number of columns in each tiled image.
    overlap : int, optional
        Overlap between tiles in pixels. Default is 0.
    type_name : str, optional
        Naming convention for output tiles. Either "sequence" or "coord".
        Default is "sequence".
    verbose : int, optional
        If 1, prints information about processed images. Default is 0.
    name_tile : str, optional
        Generic name for output tiles. If None, uses source_name without extension.
        Default is None.
    reset : bool, optional
        If True (default), delete all images inside the folder before creating tiles

    Raises
    ------
    ValueError
        If type_name is not "sequence" or "coord".

    Examples
    --------
    >>> im2tiles(
    >>>      source_name="input.tif",
    >>>      dest_name="/path/to/destination",
    >>>      nb_row=256,
    >>>      nb_col=256,
    >>>      overlap=10,
    >>>      type_name="sequence")
    >>>
    >>> im2tiles(
    >>>      source_name="input.tif",
    >>>      dest_name="/path/to/destination",
    >>>      nb_row=256,
    >>>      nb_col=256,
    >>>      overlap=10,
    >>>      type_name="sequence",
    >>>      reset=False)
    """
    if not (type_name == "sequence" or type_name == "coord"):
        raise ValueError("type_name must be either sequence or coord")

    # Create destination directory if it doesn't exist
    if not os.path.exists(dest_name):
        os.makedirs(dest_name)
        print('creation of folder ', dest_name)
    else:
        if reset:
            comm=('\\rm -rf %s'%dest_name)
            os.system(comm)
            os.makedirs(dest_name)
            print('remove folder ', dest_name)
        

    # Process the image
    split_image_to_tiles(
        source_name,
        dest_name,
        nb_col,
        nb_row,
        overlap=overlap,
        name=type_name,
        verbose=verbose,
        name_tile=name_tile
    )

def crop_rio(data, deb_row, end_row, deb_col, end_col, dest_name=None, meta=None, channel_first=True, names=None):
    """
    Crop a georeferenced image.

    Parameters
    ----------
    data : str or numpy.ndarray
        Path to a GeoTIFF image or a numpy array containing image data.
        If a numpy array is provided, meta must also be provided.
    deb_row : int
        Starting row coordinate (pixel).
    end_row : int
        Ending row coordinate (pixel).
    deb_col : int
        Starting column coordinate (pixel).
    end_col : int
        Ending column coordinate (pixel).
    dest_name : str, optional
        Path to save the cropped image. If None, the image is not saved.
        Default is None.
    meta : dict, optional
        Metadata dictionary (required if data is a numpy array).
        Default is None.
    names : dict, optional
        Band name dictionary
        Default is None.
    channel_first : bool, optional
        If True, assumes/returns image with shape (bands, rows, cols).
        If False, assumes/returns image with shape (rows, cols, bands).
        Default is True.

    Returns
    -------
    numpy.ndarray
        Cropped image data
    dict
        Updated metadata

    Raises
    ------
    ValueError
        If data is a numpy array but meta is not provided.

    Examples
    --------
    >>> # Crop from a file
    >>> im, meta = crop_rio("input.tif", 100, 500, 200, 600, "cropped.tif")
    >>>
    >>> # Crop from a numpy array
    >>> im, meta = crop_rio(image_array, 100, 500, 200, 600, meta=metadata)
    """
    if isinstance(data, str):
        # Case 1: data is a file path
        src = rio.open(data)
        win = windows.Window.from_slices((deb_row, end_row), (deb_col, end_col))
        big_window = windows.Window(col_off=0, row_off=0, width=src.meta['width'], height=src.meta['height'])
        wind = win.intersection(big_window)
        transform = windows.transform(wind, src.transform)
        meta = src.meta.copy()
        meta['transform'] = transform
        meta['driver'] = 'GTiff'
        im = src.read(window=win)
        meta['width'] = im.shape[2]
        meta['height'] = im.shape[1]

        if channel_first is False:
            im = np.rollaxis(im, 0, 3)

    elif (meta is None) and (isinstance(data, np.ndarray)):
        # Case 2: data is a numpy array but no metadata provided
        raise ValueError("You need to provide metadata with input data")

    elif isinstance(data, np.ndarray):
        # Case 3: data is a numpy array with metadata
        if channel_first is True:
            im = data[:, deb_row:end_row, deb_col:end_col]
        else:
            im = data[deb_row:end_row, deb_col:end_col, :]

        # Update metadata
        new_meta = meta.copy()
        new_meta['height'] = end_row - deb_row
        new_meta['width'] = end_col - deb_col

        # Calculate new transform
        transform = meta['transform']
        new_transform = transform * Affine.translation(deb_col, deb_row)

        # Update transform in metadata
        new_meta['transform'] = new_transform
        meta = new_meta.copy()
    else:
        raise ValueError("You need either to provide metadata with input numpy array or a path to a geotif image")

    # Save cropped image if destination name is provided
    if dest_name is not None:
        folder = os.path.split(dest_name)[0]
        if os.path.exists(folder) is False and folder != '':
            os.makedirs(folder)

        if os.path.exists(dest_name):
            os.remove(dest_name)
        if os.path.exists(f'{dest_name}.aux.xml'):
            os.remove(f'{dest_name}.aux.xml')

        if channel_first is True:
            with rio.open(dest_name, 'w', **meta) as outds:
                outds.write(im)
                if names is not None:
                    outds.update_tags(EXTRA_TAGS=json.dumps(names))
        else:
            with rio.open(dest_name, 'w', **meta) as outds:
                outds.write(np.rollaxis(im, 2, 0))
                if names is not None:
                    outds.update_tags(EXTRA_TAGS=json.dumps(names))

    return im, meta






def crop_rio_sequence(source_name, deb_row, end_row, deb_col, end_col, dest_name=None):
    """
    Crop a sequence of georeferenced images.

    Parameters
    ----------
    source_name : str
        Path to the directory containing GeoTIFF images.
    deb_row : int
        Starting row coordinate (pixel).
    end_row : int
        Ending row coordinate (pixel).
    deb_col : int
        Starting column coordinate (pixel).
    end_col : int
        Ending column coordinate (pixel).
    dest_name : str, optional
        Path to save the cropped images. If None, images are not saved.
        Default is None.

    Returns
    -------
    list of numpy.ndarray
        List of cropped images
    list of dict
        List of updated metadata

    Examples
    --------
    >>> images, metadatas = crop_rio_sequence("input_dir", 100, 500, 200, 600, "output_dir")
    """
    # Get list of all tif files in the source directory
    list_im = glob.glob(f'{source_name}/*tif')
    im_cropped = []
    meta_cropped = []

    # Create output directory if needed
    if dest_name is not None:
        if not os.path.exists(dest_name):
            os.makedirs(dest_name)
            print('creation of folder ', dest_name)

    # Process each image
    for i in range(len(list_im)):
        if dest_name is not None:
            name_out = f'{dest_name}/{os.path.split(list_im[i])[1]}'
        else:
            name_out = None

        im, meta = crop_rio(list_im[i], deb_row, end_row, deb_col, end_col, dest_name=name_out)
        im_cropped.append(im)
        meta_cropped.append(meta)

    return im_cropped, meta_cropped


def resampling(data, final_resolution, dest_name=None, method='cubic_spline', channel_first=True, meta=None, names = None):
    """
    Resample a georeferenced image to a new resolution.

    Parameters
    ----------
    data : str or numpy.ndarray
        The name of the input TIFF image or a numpy array with shape (N, row, col).
        If a numpy array is provided, an associated `meta` is required.
    final_resolution : float
        The desired resolution of the output image (in meters or degrees).
    dest_name : str, optional
        The name of the resampled image file to save. If None, the image is not saved.
        Default is None.
    method : str, optional
        The resampling method to use.
        Available methods: 'cubic_spline' (default), 'nearest', 'bilinear', 'cubic',
                          'lanczos', 'average', 'mode', 'max', 'min', 'med', 'sum',
                          'q1', 'q3'.
        Default is 'cubic_spline'.
    channel_first : bool, optional
        Whether to output the image in a shape of (bands, rows, cols).
        If False, the output shape will be (rows, cols, bands).
        Default is True.
    meta : dict, optional
        Metadata to use if `data` is a numpy array.
        Default is None.
    names : dict, optional
        Band name dictionary
        Default is None.

    Returns
    -------
    numpy.ndarray
        The resampled image.
    dict
        The metadata associated with the resampled image.

    Examples
    --------
    >>> # Resample a GeoTIFF image to a new resolution
    >>> resampled_image, meta = resampling('image.tif', 10)
    >>>
    >>> # Resample a numpy array with custom metadata
    >>> data = np.random.rand(4, 100, 100)  # Example data
    >>> meta = {'driver': 'GTiff', 'count': 4, 'dtype': 'float32', 'width': 100, 'height': 100}
    >>> resampled_image, meta = resampling(data, 10, meta=meta)
    """
    if isinstance(data, str):
        return resample_image_with_resolution(
            data,
            final_resolution,
            dest_name=dest_name,
            method=method,
            channel_first=channel_first
        )
    elif (meta is None) and (isinstance(data, np.ndarray)):
        raise ValueError("You need to provide metadata with input data")
    else:
        return resampling_image(
            data,
            meta,
            final_resolution,
            dest_name=dest_name,
            method=method,
            channel_first=channel_first,
            names = names
        )


def np2rio(image):
    """
    Convert a numpy array in channel-last format to channel-first format for rasterio.

    Parameters
    ----------
    image : numpy.ndarray
        Input image in channel-last format (rows, cols, bands).

    Returns
    -------
    numpy.ndarray
        Image in channel-first format (bands, rows, cols).

    Examples
    --------
    >>> rio_image = np2rio(numpy_image)
    """
    if len(image.shape) == 2:
        image = image.reshape(image.shape[0], image.shape[1], 1)

    return np.rollaxis(image, 2, 0)


def rio2np(image):
    """
    Convert a rasterio image in channel-first format to channel-last format for numpy.

    Parameters
    ----------
    image : numpy.ndarray
        Input image in channel-first format (bands, rows, cols).

    Returns
    -------
    numpy.ndarray
        Image in channel-last format (rows, cols, bands).

    Examples
    --------
    >>> numpy_image = rio2np(rio_image)
    """
    if len(image.shape) == 2:
        image = image.reshape(1, image.shape[0], image.shape[1])

    return np.rollaxis(image, 0, 3)


def image2table(image, channel_first=True):
    """
    Reshape an image into a 2D table of size (rows*cols, bands).

    Parameters
    ----------
    image : numpy.ndarray
        Input image array.
    channel_first : bool, optional
        If True, assumes input has shape (bands, rows, cols).
        If False, assumes input has shape (rows, cols, bands).
        Default is True.

    Returns
    -------
    numpy.ndarray
        A 2D array of shape (rows*cols, bands).

    Examples
    --------
    >>> table = image2table(image, channel_first=True)
    """
    if len(image.shape) == 3:
        if channel_first is False:
            new_shape = (image.shape[0] * image.shape[1], image.shape[2])
            return image.reshape(new_shape)
        else:
            image = np.rollaxis(image, 0, 3)
            new_shape = (image.shape[0] * image.shape[1], image.shape[2])
            return image.reshape(new_shape)
    else:
        new_shape = (image.shape[0] * image.shape[1],)
        return image.reshape(new_shape)


def table2image(table, size, channel_first=True):
    """
    Reshape a 2D table back into a 3D image.

    Parameters
    ----------
    table : numpy.ndarray
        Input table with shape (rows*cols, bands) or (rows*cols,).
    size : tuple
        Size of the output image as (rows, cols).
    channel_first : bool, optional
        If True, output will have shape (bands, rows, cols).
        If False, output will have shape (rows, cols, bands).
        Default is True.

    Returns
    -------
    numpy.ndarray
        Reshaped 3D image.

    Examples
    --------
    >>> image = table2image(table, (400, 600), channel_first=True)
    """
    if len(table.shape) == 1:
        if channel_first:
            return table.reshape(1,size[0], size[1])
        else:
            return table.reshape(size[0], size[1],1)
    elif channel_first is True:
        bands = table.shape[1]
        image = table.reshape(size[0], size[1], bands)
        return np.rollaxis(image, 2, 0)
    else:
        bands = table.shape[1]
        image = table.reshape(size[0], size[1], bands)
        return image


def extract_numpy_tables(data, outputs, label=None):
    """
    Extract paired NumPy tables (features and outputs) for machine learning.

    This function converts raster data stored in `Geoimage` objects into two
    NumPy arrays: one containing the input features (X), and one containing
    the corresponding outputs/labels (y). Optionally, it can filter the
    dataset to only include samples corresponding to one or multiple label
    values.

    Parameters
    ----------
    data : Geoimage
        A `rastereasy.Geoimage` containing the input features
        (e.g., multispectral bands).
    outputs : Geoimage
        A `rastereasy.Geoimage` containing the outputs/labels
        (e.g., classes or quantitative values).
    label : int or list of int, optional (default=None)
        If provided, only the samples corresponding to this label value
        (or list of values) in `outputs` will be extracted.

    Returns
    -------
    X : numpy.ndarray
        Input feature table of shape (N, f), where
        N is the number of extracted samples and
        f is the number of features (bands).
    y : numpy.ndarray
        Output array of shape (N, ), containing the labels/outputs
        associated with each sample.

    Examples
    --------
    >>> # Extract all data/labels from two Geoimages
    >>> X, y = extract_numpy_tables(data, labels)

    >>> # Extract only the samples where label == 1
    >>> X, y = extract_numpy_tables(data, outputs, label=1)

    >>> # Extract only the samples with labels in [1, 3, 5]
    >>> X, y = extract_numpy_tables(data, outputs, label=[1, 3, 5])
    """
    data_np = data.numpy_table()
    classes_np = outputs.numpy_table()

    if label is None:
        X = data_np
        y = classes_np
    else:
        # Assure qu'on gère int et liste
        labels = np.atleast_1d(label)
        mask = np.isin(classes_np.flatten(), labels)
        X = data_np[mask]
        y = classes_np.flatten()[mask]
    return X, y


def shp2geoim2(shapefile_path, attribute='code', resolution=10, nodata=0):
    """
    Convertit un shapefile en données raster et métadonnées géospatiales.

    Cette fonction transforme des données vectorielles d'un shapefile en une matrice
    raster et génère les métadonnées appropriées pour le géoréférencement.

    Parameters
    ----------
    shapefile_path : str
        Chemin d'accès au fichier shapefile (.shp).
    attribute : str, optional
        Attribut du shapefile à utiliser pour les valeurs des pixels dans le raster.
        Default is 'code'.
    resolution : float, optional
        Résolution spatiale (taille des pixels) du raster de sortie en unités
        du système de coordonnées (généralement mètres ou degrés).
        Default is 10.
    nodata : int or float, optional
        Valeur à attribuer aux zones situées en dehors des formes du shapefile.
        Default is 0.

    Returns
    -------
    tuple
        Un tuple contenant:
        - data (numpy.ndarray): Tableau 2D contenant les données rasterisées
        - meta (dict): Dictionnaire de métadonnées avec les informations de géoréférencement

    Raises
    ------
    ValueError
        Si le shapefile n'a pas de système de coordonnées (CRS) défini
        Si l'attribut spécifié n'existe pas dans le shapefile

    Examples
    --------
    >>> data, meta = shp2geoim2("landcover.shp", attribute="landtype", resolution=30)
    >>> im = Geoimage(data=data, meta=meta)
    >>> im.info()
    """
    import geopandas as gpd
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds
    from rasterio.enums import MergeAlg

    # Charger le shapefile
    gdf = gpd.read_file(shapefile_path)

    # Vérifier le CRS
    if gdf.crs is None:
        raise ValueError(f"The shapefile {shapefile_path} does not have a defined CRS.")

    # Vérifier l'attribut
    if attribute not in gdf.columns:
        available_attrs = ", ".join(gdf.columns)
        raise ValueError(f"Attribute '{attribute}' not found in shapefile '{shapefile_path}'. "
                        f"Available attributes are: {available_attrs}")

    # Calculer les limites (sans arrondi)
    minx, miny, maxx, maxy = gdf.total_bounds
    width = int((maxx - minx) / resolution)
    height = int((maxy - miny) / resolution)
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    # Créer les formes pour la rasterisation
    shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf[attribute]))

    # Rasterisation avec gestion d'erreurs
    try:
        data = rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=nodata,
            dtype='int32',
            merge_alg=MergeAlg.replace
        )
    except Exception as e:
        raise RuntimeError(f"Error during rasterization: {str(e)}")

    # Métadonnées
    meta = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': 1,
        'nodata': nodata,
        'dtype': 'int32',
        'crs': gdf.crs,
        'transform': transform
    }

    return data, meta


class shpfiles:
    """
    Utility class for working with shapefiles and converting them to raster formats.

    This class contains static methods for operations like:
    - Getting attribute names from shapefiles
    - Converting shapefiles to raster data
    - Converting shapefiles directly to Geoimage objects

    Examples
    --------
    >>> # Get attributes from a shapefile
    >>> attributes = shpfiles.get_shapefile_attributes("landcover.shp")
    >>>
    >>> # Convert a shapefile to a raster file
    >>> shpfiles.shp2raster("landcover.shp", "landcover.tif", attribute="landtype")
    >>>
    >>> # Convert a shapefile to a Geoimage object
    >>> landcover_img = shpfiles.shp2geoim("landcover.shp", attribute="landtype")
    """

    @staticmethod
    def get_shapefile_attributes(shapefile_path):
        """
        Get the attribute field names from a shapefile.

        Parameters
        ----------
        shapefile_path : str
            Path to the input shapefile.

        Returns
        -------
        list
            List of attribute field names in the shapefile.

        Examples
        --------
        >>> attributes = shpfiles.get_shapefile_attributes("landcover.shp")
        >>> print(attributes)
        >>> ['FID', 'landtype', 'area', 'perimeter']
        """
        try:
            # Load the shapefile using geopandas
            gdf = gpd.read_file(shapefile_path)

            # Get the column names
            attributes = list(gdf.columns)

            return attributes
        except Exception as e:
            print(f"Error loading shapefile: {e}")
            return []

    @staticmethod
    def __shp2geoim2(shapefile_path, attribute='code', resolution=10, nodata=0):
        """
        Convert a shapefile to a Geoimage object.

        Parameters
        ----------
        shapefile_path : str
            Path to the input shapefile.
        attribute : str, optional
            Attribute field in the shapefile to assign values to each pixel.
            Default is 'code'.
        resolution : float, optional
            Spatial resolution of the output raster in meters (or degrees).
            Default is 10.
        nodata : int or float, optional
            Value to assign to areas outside the shapes.
            Default is 0.

        Returns
        -------
        Geoimage
            A Geoimage object containing the rasterized data.

        Raises
        ------
        ValueError
            If the shapefile does not have a defined CRS
            If the specified attribute is not found in the shapefile

        Examples
        --------
        >>> landcover_img = shpfiles.shp2geoim2("landcover.shp", attribute="landtype")
        >>> landcover_img.visu()
        """
        import geopandas as gpd
        from rasterio.features import rasterize
        from rasterio.transform import from_bounds
        from rasterio.enums import MergeAlg

        # Load the shapefile
        gdf = gpd.read_file(shapefile_path)

        # Check if CRS is defined
        if gdf.crs is None:
            raise ValueError(f"The shapefile {shapefile_path} does not have a defined CRS.")

        # Check if attribute exists
        if attribute not in gdf.columns:
            raise ValueError(f"Attribute '{attribute}' not found in shapefile '{shapefile_path}'.")

        # Calculate bounds (without rounding)
        minx, miny, maxx, maxy = gdf.total_bounds
        width = int((maxx - minx) / resolution)
        height = int((maxy - miny) / resolution)
        transform = from_bounds(minx, miny, maxx, maxy, width, height)

        # Create shapes for rasterization
        shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf[attribute]))

        # Rasterize
        data = rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=nodata,
            dtype='int32',
            merge_alg=MergeAlg.replace
        )

        # Create metadata
        meta = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': 1,
            'nodata': nodata,
            'dtype': 'int32',
            'crs': gdf.crs,
            'transform': transform
        }

        return Geoimage(data=data, meta=meta)

    @staticmethod
    def shp2geoim(shapefile_path, attribute='code', resolution=10, nodata=0):
        """
        Convert a shapefile to a Geoimage object.

        Parameters
        ----------
        shapefile_path : str
            Path to the input shapefile.
        attribute : str, optional
            Attribute field in the shapefile to assign values to each pixel.
            Default is 'code'.
        resolution : float, optional
            Spatial resolution of the output raster in meters/degrees.
            Default is 10.
        nodata : int or float, optional
            Value to assign to areas outside the shapes.
            Default is 0.

        Returns
        -------
        Geoimage
            A Geoimage object containing the rasterized data.

        Notes
        -----
        - The `shapefile_path` should be the full path to a shapefile (.shp) on the disk.
        - The `attribute` field will be assigned to each pixel in the rasterized Geoimage.
        - To get the attributes of a shapefile, see :meth:`shpfiles.get_shapefile_attributes`
        - The `resolution` sets the size of each pixel in the output image.

        Examples
        --------
        >>> geo_img = shpfiles.shp2geoim("landcover.shp", attribute='landtype', resolution=5)
        """
        if attribute not in shpfiles.get_shapefile_attributes(shapefile_path):
            print('Attributes of shapefile', shapefile_path, ':', shpfiles.get_shapefile_attributes(shapefile_path))
            raise ValueError(f'Attribute {attribute} not in attributes of shapefile {shapefile_path}')

        gdf = gpd.read_file(shapefile_path)

        # Check if CRS is defined
        if gdf.crs is None:
            raise ValueError(f"The shapefile {shapefile_path} does not have a defined CRS.")

        # Calculate bounds of the shapefile
        minx, miny, maxx, maxy = gdf.total_bounds

        width = int((maxx - minx) / resolution)
        height = int((maxy - miny) / resolution)

        transform = from_bounds(minx, miny, maxx, maxy, width, height)

        shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf[attribute]))

        raster = rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=nodata,
            dtype='int32')

        meta = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': 1,
            'nodata': nodata,
            'dtype': 'int32',
            'crs': gdf.crs,
            'transform': transform
        }

        return Geoimage(data=raster, meta=meta)

    @staticmethod
    def shp2raster(shapefile_path, dest_name, attribute='code', resolution=10, nodata=0):
        """
        Convert a shapefile to a GeoTIFF raster file.

        Parameters
        ----------
        shapefile_path : str
            Path to the input shapefile.
        dest_name : str
            Path to save the output raster file.
        attribute : str, optional
            Attribute field in the shapefile to assign values to each pixel.
            Default is 'code'.
        resolution : float, optional
            Spatial resolution of the output raster in meters/degrees.
            Default is 10.
        nodata : int or float, optional
            Value to assign to areas outside the shapes.
            Default is 0.

        Notes
        -----
        - The `shapefile_path` should be the full path to a shapefile (.shp) on the disk.
        - To get the attributes of a shapefile, see :meth:`shpfiles.get_shapefile_attributes`
        - The output raster will be written in GeoTIFF format to the path specified by `dest_name`.

        Examples
        --------
        >>> shpfiles.shp2raster("landcover.shp", "landcover.tif", attribute='landtype', resolution=5)
        """
        gdf = gpd.read_file(shapefile_path)

        # Calculate bounds of the shapefile
        minx, miny, maxx, maxy = gdf.total_bounds

        width = int((maxx - minx) / resolution)
        height = int((maxy - miny) / resolution)

        transform = from_bounds(minx, miny, maxx, maxy, width, height)

        shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf[attribute]))

        raster = rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=nodata,
            dtype='int32')

        # Save the raster as GeoTIFF
        folder = os.path.split(dest_name)[0]
        if not os.path.exists(folder) and folder != '':
            os.makedirs(folder)

        if os.path.exists(dest_name):
            os.remove(dest_name)
        if os.path.exists(f'{dest_name}.aux.xml'):
            os.remove(f'{dest_name}.aux.xml')

        with rio.open(
            dest_name, 'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            nodata=nodata,
            dtype='int32',
            crs=gdf.crs,
            transform=transform
        ) as dst:
            dst.write(raster, 1)

class rasters:
    """
    Utility class for raster image operations.

    This class provides static methods for operations like stacking multiple images
    and removing bands from images.

    Examples
    --------
    >>> # Stack two images
    >>> combined_image = rasters.stack(image1, image2)
    >>>
    >>> # Remove bands from an image
    >>> reduced_image = rasters.remove_bands(image, bands=["NIR", "SWIR1"])
    """

    @staticmethod
    def stack(im1, im2, dtype='float64', dest_name=None, reformat_names=False):
        """
        Stack two Geoimage objects into a single image.

        Parameters
        ----------
        im1 : Geoimage
            First image to stack
        im2 : Geoimage
            Second image to stack
        dtype : str, optional
            Data type for the output image.
            Default is 'float64'.
        dest_name : str, optional
            Path to save the stacked image.
            Default is None.
        reformat_names : bool, optional
            How to handle band names:
            - If True: Reset all names like {"1":1, "2":2, ...}
            - If False: Adapt names like {"NIR_1":1, "R_1":2, "G_1":3, "R_2":4, ...}
            Default is False.

        Returns
        -------
        Geoimage
            A new Geoimage containing all bands from both input images

        Examples
        --------
        >>> combined = rasters.stack(sentinel2_img, landsat8_img)
        >>> combined.info()
        """
        return im1.stack(im2, dtype=dtype, dest_name=dest_name, reformat_names=reformat_names)

    @staticmethod
    def remove_bands(im, bands, reformat_names=True, dest_name=None):
        """
        Remove specified bands from an image.

        Parameters
        ----------
        im : Geoimage
            The input image
        bands : str, list, or numpy.ndarray
            The bands to remove, specified as either:
            - A string with a band name (e.g., 'SWIR1')
            - A list or array of band names (e.g., ['R', 'G', 'B'])
            - An integer representing the band index (e.g., 4)
            - A list or array of band indices (e.g., [4, 2, 3])
        reformat_names : bool, optional
            If True, the band names are renumbered from 1 after removal.
            If False, the original band names are preserved (gaps may remain).
            Default is True.
        dest_name : str, optional
            Path to save the modified image.
            Default is None.

        Returns
        -------
        Geoimage
            A new Geoimage with the specified bands removed

        Examples
        --------
        >>> reduced_img = rasters.remove_bands(image, bands=["NIR", "SWIR1"])
        >>> reduced_img.info()
        """
        return im.remove_bands(bands, reformat_names=reformat_names, dest_name=dest_name)


class Visualizer:
    """
    Utility class for visualizing raster image data.

    This class provides methods for interactive plotting and exploration of
    spectral data in georeferenced images.

    Examples
    --------
    >>> # Extract and plot spectral values from user-selected pixels
    >>> series, pixel_i, pixel_j = Visualizer.plot_spectra(image, bands=['R', 'G', 'B'])
    """

    @staticmethod
    def plot_spectra(im, bands=None, fig_size=(15, 5), percentile=2, title='',
                     title_im="Original image (click outside to stop)",
                     title_spectra="Spectra", xlabel="Bands", ylabel="Value", offset_i=0,offset_j=0):
        """
        Plots and extracts spectral values from user-selected pixels on a multispectral image.

        Parameters
        ----------
        im : Geoimage
            Multispectral georeferenced image to analyze.
        bands : list, optional
            List of bands to use for the color composition in the image plot.
            Default is None (uses the first three bands).
        fig_size : tuple, optional
            Size of the figure in inches, specified as (width, height).
            Default is (15, 5).
        percentile : int, optional
            Percentile value for the color composition scaling.
            Default is 2.
        title : str, optional
            Main title for the figure.
            Default is ''.
        title_im : str, optional
            Title for the image plot.
            Default is "Original image (click outside to stop)".
        title_spectra : str, optional
            Title for the spectra curves plot.
            Default is "Spectra".
        xlabel : str, optional
            X-axis label for the spectra curves plot.
            Default is "Bands".
        ylabel : str, optional
            Y-axis label for the spectra curves plot.
            Default is "Value".
        offset_i : int, optional
            Offset to add to i coordinates (in case of a zoom)
            Default is 0.
        offset_j : int, optional
            Offset to add to j coordinates (in case of a zoom)
            Default is 0.

        Returns
        -------
        list of lists
            Collection of spectral series extracted from the selected pixels.
        list of int
            List of row indices (i-coordinates) of the selected pixels.
        list of int
            List of column indices (j-coordinates) of the selected pixels.

        Notes
        -----
        The data collection stops when the user clicks outside the image area or
        clicks the "Finish" button.

        Examples
        --------
        >>> series, i_coords, j_coords = Visualizer.plot_spectra(
        >>>      image, bands=['B01', 'B02', 'B03'], fig_size=(10, 5))
        """
        imc = extract_colorcomp(im, bands=bands, percentile=percentile)

        def on_finish(series, val_i, val_j):
            print("Acquisition finished !")
            print(f"Collected Spectra : {series}")
            print(f"Rows : {val_i}")
            print(f"Cols : {val_j}")

        def on_end_collect(series, val_i, val_j):
            # Code to be executed after the user finishes selecting pixels
            print("User has finished selecting pixels.")
        series, val_i, val_j, end_collect = plot_clic_spectra(
            im.numpy_channel_last(),
            imc,
            figsize=fig_size,
            names=im.names,
            title_im=title_im,
            title_spectra=title_spectra,
            xlabel=xlabel,
            ylabel=ylabel,
            callback=on_end_collect,
            offset_i=offset_i,
            offset_j=offset_j,
            colab=IN_COLAB
        )

        return series, val_i, val_j


class InferenceTools:
    """
    Utility class for inference operations on raster images.

    This class provides methods for clustering, spectral adaptation
    fusion, ... of georeferenced images.

    Examples
    --------
    >>> # Perform K-means clustering on an image
    >>> classified_img, model = InferenceTools.kmeans(image, n_clusters=5)
    >>>
    >>> # Adapt spectral properties of one image to match another
    >>> adapted_img = InferenceTools.adapt(source_img, target_img, mapping='sinkhorn')
    """

    @staticmethod
    def kmeans(im, n_clusters=4, bands=None, random_state=None, dest_name=None, standardization=True):
        """
        Perform K-means clustering on a Geoimage.

        Parameters
        ----------
        im : Geoimage
            Input image to cluster
        n_clusters : int, optional
            Number of clusters (categories) to create.
            Default is 4.
        bands : list, optional
            List of bands to use for clustering. If None, all bands are used.
            Default is None.
        random_state : int, optional
            Random state for reproducible results.
            Default is None.
        dest_name : str, optional
            Path to save the clustered image.
            Default is None.
        standardization : bool, optional
            Whether to standardize bands before clustering.
            Default is True.

        Returns
        -------
        Geoimage
            A new Geoimage with clusters as pixel values
        tuple
            A tuple containing the KMeans model and the scaler (if standardization was applied)

        Examples
        --------
        >>> classified_img, model = InferenceTools.kmeans(image, n_clusters=3)
        >>> classified_img.visu()
        >>>
        >>> # Clustering with specific bands
        >>> classified_img, model = InferenceTools.kmeans(
        >>>     image, n_clusters=4, bands=["8", "2", "1"], random_state=42)
        """
        return im.kmeans(
            n_clusters=n_clusters,
            bands=bands,
            random_state=random_state,
            dest_name=dest_name,
            standardization=standardization
        )

    @staticmethod
    def fuse_dempster_shafer_2hypotheses(*images):
        """
        Fuse mass functions from multiple sources using Dempster-Shafer theory
        with two hypotheses: A and B.

        Parameters
        ----------
        *images : Geoimage
            Each input is a 3-band Geoimage.

            - Band 1: mass function m(A)

            - Band 2: mass function m(B)

            - Band 3: mass function m(A ∪ B)

        Returns
        -------
        Geoimage
            A new Geoimage with 3 bands containing the fused mass functions:
            m(A), m(B), and m(A ∪ B).
        Geoimage
            A new Geoimage with 1 band containing the conflict values.

        Examples
        --------
        >>> fused, conflict = fuse_dempster_shafer_2hypotheses(im1, im2, im3)
        >>> fused, conflict = fuse_dempster_shafer_2hypotheses(im1, im2, im3, im4)
        >>> fused, conflict = fuse_dempster_shafer_2hypotheses(im1, im2)
        """

        if len(images) < 2:
            raise ValueError("At least two Geoimages are required for fusion.")

        # Initialize outputs
        im_fusion = images[0].copy()
        im_conflict = images[0].select_bands(1)
        im_conflict.change_names({'conflict': 1})

        # Initial fusion
        fusion, conflict = fusion_2classes(images[0].numpy_table(), images[1].numpy_table())

        # Fuse remaining sources
        for img in images[2:]:
            fusion, conflict = fusion_2classes(fusion, img.numpy_table())

        # Update Geoimages
        im_fusion.upload_table(fusion)

        im_conflict.upload_table(conflict)

        return im_fusion, im_conflict

    @staticmethod
    def adapt(ims, imt, tab_source = None, nb=1000, mapping='gaussian', reg_e=1e-1, mu=1e0, eta=1e-2, bias=False, max_iter=20, verbose=True, sigma=1):
        """
        Adjusts the spectral characteristics of a source image to match those of a target image
        using optimal transport methods.

        This function normalizes the data, applies the chosen optimal transport algorithm to
        adapt the spectral characteristics, and then restores the original data scale.

        Parameters
        ----------
        ims : Geoimage
            Source image whose spectral characteristics will be adjusted.
        imt : Geoimage or numpy.ndarray
            Target image serving as a reference for spectral adjustment,
            or a NumPy array of shape (N, bands) containing N spectral samples.
        tab_source : numpy.ndarray, optional
            Required if `imt` is a NumPy array. Must be an array of shape (M, bands)
            containing spectral samples from the source image.
        nb : int, optional
            Number of random samples used to train the transport model.
            Default is 1000.
        mapping : str, optional
            Optimal transport method to use. Available options:
            - 'emd': Earth Mover's Distance (more precise but slower)
            - 'sinkhorn': Sinkhorn transport with regularization (good balance)
            - 'mappingtransport': Mapping-based transport (flexible)
            - 'gaussian': Transport with Gaussian assumptions (faster, robust)
            Default is 'gaussian'.
        reg_e : float, optional
            Regularization parameter for Sinkhorn transport.
            Default is 1e-1.
        mu : float, optional
            Regularization parameter for mapping-based methods.
            Default is 1e0.
        eta : float, optional
            Learning rate for mapping-based transport methods.
            Default is 1e-2.
        bias : bool, optional
            Adds a bias term to the transport model if enabled.
            Default is False.
        max_iter : int, optional
            Maximum number of iterations for iterative transport methods.
            Default is 20.
        verbose : bool, optional
            Enables progress messages during processing.
            Default is True.
        sigma : float, optional
            Standard deviation used for Gaussian transport methods.
            Default is 1.

        Returns
        -------
        Geoimage
            A new image where the spectral bands of the source image `ims` are
            adapted to match those of the target image `imt`.

        Raises
        ------
        ValueError
            If an unrecognized mapping method is specified.
        RuntimeError
            If the adaptation process fails.

        Notes
        -----
        - This function uses optimal transport tools (via the POT library).
        - Raster data is normalized before transport and then denormalized afterward.
        - Pixels with nodata values in both images are excluded from calculations.
        - Adjusted values are limited to remain within valid ranges.

        Examples
        --------
        >>> adapted_image = InferenceTools.adapt(source_image, target_image, mapping='sinkhorn', reg_e=0.01)
        >>> adapted_image.save('adapted_image.tif')
        >>>
        >>> # Adaptation using sample arrays
        >>> adapted_image = InferenceTools.adapt(source_image, tab_target, tab_source, mapping='sinkhorn', reg_e=0.01)
        >>> adapted_image.save('adapted_image.tif')
        >>> adapted_image.save('adapted_image.tif')
        >>>
        >>> # Adaptation using different methods
        >>> adapted_gaussian = InferenceTools.adapt(source_image, target_image, mapping='gaussian')
        >>> adapted_emd = InferenceTools.adapt(source_image, target_image, mapping='emd')
        """
        import ot  # Optimal transport library

        rng = np.random.RandomState(RANDOM_STATE)

        try:

            if isinstance(imt, Geoimage):
                im1 = ims.copy()
                im1s, scaler1 = im1.standardize(type='minmax')
                X1 = im1s.numpy_table()
                mask1 = ~np.any(X1 == im1s.nodata, axis=1)
                X1 = X1[mask1]
                if X1.shape[0] > nb:
                    idx1 = rng.randint(X1.shape[0], size=(nb,))
                    Xs = X1[idx1, :]
                else:
                    Xs = X1
                    im2 = imt.copy()

                im2 = imt.copy()
                im2s, scaler2 = im2.standardize(type='minmax')
                X2 = im2s.numpy_table()
                mask2 = ~np.any(X2 == im2s.nodata, axis=1)
                X2 = X2[mask2]
                if X2.shape[0] > nb:
                    idx2 = rng.randint(X2.shape[0], size=(nb,))
                    Xt = X2[idx2, :]
                else:
                    Xt = X2

            else:
                tab = tab_source.copy()
                scaler1 = MinMaxScaler().fit(tab)
                Xs = scaler1.transform(tab).astype(np.float64)


                tab = imt.copy()
                scaler2 = MinMaxScaler().fit(tab)
                Xt = scaler2.transform(tab).astype(np.float64)

            # Transport
            if mapping == "emd":
                transport_data = ot.da.EMDTransport()
            elif mapping == "sinkhorn":
                transport_data = ot.da.SinkhornTransport(reg_e=reg_e)
            elif mapping == "mappingtransport":
                transport_data = ot.da.MappingTransport(
                    mu=mu, eta=eta, bias=bias, max_iter=max_iter, verbose=verbose
                )
            elif mapping == 'gaussian':
                transport_data = ot.da.MappingTransport(
                    mu=mu, eta=eta, sigma=sigma, bias=bias, max_iter=max_iter, verbose=verbose
                )
            else:
                raise ValueError(f'Mapping type "{mapping}" is not recognized. '
                            f'Use "emd", "sinkhorn", "mappingtransport", or "gaussian".')

            if verbose:
                print(f"Fitting transport model using {mapping} method...")

            transport_data.fit(Xs=Xs, Xt=Xt)

            if verbose:
                print("Transforming data...")
            im1 = ims.copy()
            X1 = im1.numpy_table()
            X1 = scaler1.transform(X1).astype(np.float64)

            transp_Xs = transport_data.transform(Xs=X1)

#            Image = np.clip(table2image(transp_Xs, im1.shape, channel_first=True), 0., 1.)
            Image = table2image(transp_Xs, im1.shape, channel_first=True)

            im1.upload_image(Image, names=im1.get_names(),inplace=True)

            im1 = im1.inverse_standardize(scaler2)

            im1 = im1.where(ims == ims.nodata, ims.nodata, im1)

            if verbose:
                print("Adaptation complete.")

            return im1

        except Exception as e:
            raise RuntimeError(f"Error during spectral adaptation: {str(e)}") from e

def files2stack(imagefile_path, resolution=None, names="origin", dest_name=None, ext='jp2', history=False):
    """
    Create a stacked Geoimage from multiple single-band images.

    This function creates a multi-band Geoimage by stacking individual images,
    either from a list of image paths or from all images in a directory.
    All input images should have 1 band each.

    Parameters
    ----------
    imagefile_path : str or list of str
        - If a list of strings: paths to image files to stack (e.g., ['image1.jp2', 'image2.jp2', ...])
        - If a string: path to a directory containing images with the specified extension
    resolution : float, optional
        Resolution to which all images will be resampled. If None, all images must
        have the same resolution already.
        Default is None.
    names : dict or str, optional
        How to name the spectral bands in the stack:
        - If a dict: Maps band names to indices (e.g., {'B': 1, 'G': 2, 'R': 3, ...})
        - If "origin" (default): Uses the original filenames as band names
        - If None: Assigns numeric names ('1', '2', '3', ...)
        Default is "origin".
    dest_name : str, optional
        Path to save the stacked image as a TIFF file.
        Default is None (no file saved).
    ext : str, optional
        File extension of images to load if imagefile_path is a directory.
        Default is 'jp2'.
    history : bool, optional
        Whether to enable history tracking for the output Geoimage.
        Default is False.

    Returns
    -------
    Geoimage
        A stacked Geoimage containing all the input images as bands.

    Examples
    --------
    >>> # Stack from a list of image files
    >>> list_images = ['band1.jp2', 'band2.jp2', 'band3.jp2']
    >>> stacked_image = files2stack(list_images)
    >>> stacked_image.save('stacked.tif')
    >>>
    >>> # Stack all jp2 files from a directory with resolution resampling
    >>> folder_path = './my_bands_folder'
    >>> stacked_image = files2stack(folder_path, resolution=10)
    >>> stacked_image.info()

    Notes
    -----
    This function is particularly useful for combining separate band files (common in
    satellite imagery) into a single multi-band image for analysis.
    """
    # Handle input as string (directory) or list of files
    if isinstance(imagefile_path, str):
        imagefile_path = list_fles(imagefile_path, ext)
    elif not isinstance(imagefile_path, list):
        raise ValueError('imagefile_path should be a list of file paths or a string with a folder path')

    if len(imagefile_path) == 0:
        raise ValueError(f"No files with extension '{ext}' found in the specified path")

    # Case 1: No resampling needed (all images must have same resolution)
    if resolution is None:
        # Initialize with first image
        im = Geoimage(imagefile_path[0])
        if im.nb_bands!=1:
            message="im %s has %d bands"%(imagefile_path[0],im.nb_bands)
            warnings.warn(message,category=UserWarning)

        # Stack remaining images
        for i in range(len(imagefile_path) - 1):
            ims=Geoimage(imagefile_path[i + 1])
            if ims.nb_bands!=1:
                message="im %s has %d bands"%(imagefile_path[i + 1],ims.nb_bands)
                warnings.warn(message,category=UserWarning)

            im.stack(ims, reformat_names=True, inplace = True)

        # Handle band naming
        if names is not None:
            if names == 'origin':
                # Use filenames as band names
                names = {os.path.splitext(os.path.basename(file))[0]: i + 1
                         for i, file in enumerate(imagefile_path)}

                # Check for duplicated names
                if len(names) != im.nb_bands:
                    warnings.warn("Original filenames contain duplicates. Using sequential names instead. "
                                 "To suppress this warning, use names=None.",
                                 category=UserWarning)
                    names = {}
                    for j in range(len(imagefile_path)):
                        names[str(j + 1)] = j + 1

            # Apply band names
            im.change_names(names)

        # Save if requested
        if dest_name is not None:
            im.save(dest_name)

        # Enable history if requested
        if history:
            im.activate_history()

        return im

    # Case 2: Resampling required
    else:
        # Initialize with first image (resampled)
        im = Geoimage(imagefile_path[0])
        if im.nb_bands!=1:
            message="im %s has %d bands"%(imagefile_path[0],im.nb_bands)
            warnings.warn(message,category=UserWarning)
        if im.resolution != resolution:
            im.resample(resolution, inplace=True)

        # Process remaining images
        for i in range(len(imagefile_path) - 1):
            # Load and resample next image
            im_tmpo = Geoimage(imagefile_path[i + 1])
            if im_tmpo.resolution != resolution:
                im_tmpo.resample(resolution, inplace=True)

            if im_tmpo.nb_bands!=1:
                message="im %s has %d bands"%(imagefile_path[i + 1],im_tmpo.nb_bands)
                warnings.warn(message,category=UserWarning)
            # Extract common areas and stack
            im, im_tmpo = extract_common_areas(im, im_tmpo)
            im.stack(im_tmpo, reformat_names=True,inplace=True)

        # Handle band naming
        if names is not None:
            if names == 'origin':
                # Use filenames as band names
                names = {os.path.splitext(os.path.basename(file))[0]: i + 1
                         for i, file in enumerate(imagefile_path)}

                # Check for duplicated names
                if len(names) != im.nb_bands:
                    warnings.warn("Original filenames contain duplicates. Using sequential names instead. "
                                 "To suppress this warning, use names=None.",
                                 category=UserWarning)
                    names = {}
                    for j in range(len(imagefile_path)):
                        names[str(j + 1)] = j + 1

            # Apply band names
            im.change_names(names)

        # Save if requested
        if dest_name is not None:
            im.save(dest_name)

        # Enable history if requested
        if history:
            im.activate_history()

        return im


def extract_common_areas(im1, im2, resolution='min', projection=None):
    """
    Extract the overlapping area between two GeoImages.

    Parameters
    ----------
    im1 : Geoimage
        First input image
    im2 : Geoimage
        Second input image
    resolution : {'min', 'max', None}, optional
        How to handle resolution differences:
        - 'min': Use the more precise (smaller pixel size) resolution
        - 'max': Use the less precise (larger pixel size) resolution
        - None: Keep original resolutions
        Default is 'min'.
    projection : str, optional
        Projection system for output images. If None, uses im1's projection.
        Example: "EPSG:4326"
        Default is None.

    Returns
    -------
    Geoimage, Geoimage
        Two new Geoimages containing only the common overlapping area

    Examples
    --------
    >>> # Extract common area with smallest pixel size
    >>> common_im1, common_im2 = extract_common_areas(image1, image2)
    >>>
    >>> # Extract common area with specific projection
    >>> common_im1, common_im2 = extract_common_areas(image1, image2, projection="EPSG:4326")

    Notes
    -----
    This function is useful for preparing images for analysis or comparison when
    they cover different geographical areas but have an overlapping region.
    """
    # Handle different projections
    adapt = False
    if ((im1.get_meta()['crs'] != im2.get_meta()['crs']) or projection is not None):
        if projection is not None:
            im1 = im1.copy()
            im2 = im2.copy()
            im1.reproject(projection, inplace=True)
            im2.reproject(projection, inplace=True)
        else:
            im2 = im2.copy()
            im2.reproject(im1.get_meta()['crs'].to_string(), inplace=True)

    # Handle different resolutions
    if im1.resolution != im2.resolution:
        if resolution is not None:
            adapt = True

            if resolution == 'min':
                res = np.min((im1.resolution, im2.resolution))
            elif resolution == 'max':
                res = np.max((im1.resolution, im2.resolution))
            else:
                raise ValueError(f'Error: resolution {resolution} unknown. Use "min", "max", or None.')

            im1 = im1.resample(res)
            im2 = im2.resample(res)

    # Get corner coordinates
    lat1_d, lon1_d = im1.pixel2latlon(0, 0)
    lat1_e, lon1_e = im1.pixel2latlon(im1.shape[0], im1.shape[1])
    lat2_d, lon2_d = im2.pixel2latlon(0, 0)
    lat2_e, lon2_e = im2.pixel2latlon(im2.shape[0], im2.shape[1])


    # Calculate boundaries of overlapping area
    bound_lat_d = min(lat1_d, lat2_d)
    bound_lat_e = max(lat1_e, lat2_e)
    bound_lon_d = max(lon1_d, lon2_d)
    bound_lon_e = min(lon1_e, lon2_e)


    # Crop images to common bounds
    im1common = im1.crop(area=((bound_lon_d, bound_lon_e), (bound_lat_d, bound_lat_e)), pixel=False)
    im2common = im2.crop(area=((bound_lon_d, bound_lon_e), (bound_lat_d, bound_lat_e)), pixel=False)

    # Adjust for srowht differences in size after reprojection/resampling
    if adapt:
        im1common, im2common = ajust_sizes(im1common, im2common)

    return im1common, im2common


def extend_common_areas(image1, image2, nodata_value=0, resolution='min', projection=None):
    """
    Extend two images to cover their combined area, filling new areas with nodata value.

    Parameters
    ----------
    image1 : Geoimage
        First input image
    image2 : Geoimage
        Second input image
    nodata_value : int or float, optional
        Value to use for areas outside the original image bounds.
        Default is 0.
    resolution : {'min', 'max'}, optional
        How to handle resolution differences:
        - 'min': Use the more precise (smaller pixel size) resolution
        - 'max': Use the less precise (larger pixel size) resolution
        Default is 'min'.
    projection : str, optional
        Projection to use for output images. If None, uses image1's projection.
        Default is None.

    Returns
    -------
    Geoimage, Geoimage
        Two new Geoimages, each covering the combined area of both input images

    Examples
    --------
    >>> extended_im1, extended_im2 = extend_common_areas(image1, image2)

    Notes
    -----
    This function is useful for preparing images with different extents for comparison
    or mathematical operations that require the same dimensions.
    """
    # Handle different projections
    if ((image1.get_meta()['crs'] != image2.get_meta()['crs']) or projection is not None):
        if projection is not None:
            image1 = image1.copy()
            image2 = image2.copy()
            image1.reproject(projection, inplace=True)
            image2.reproject(projection, inplace=True)
        else:
            image2 = image2.copy()
            image2.reproject(image1.get_meta()['crs'].to_string(), inplace=True)

    # Handle different resolutions
    if image1.resolution != image2.resolution:
        if resolution == 'min':
            res = np.min((image1.resolution, image2.resolution))
        elif resolution == 'max':
            res = np.max((image1.resolution, image2.resolution))
        else:
            raise ValueError(f'Error: resolution {resolution} unknown. Use "min" or "max".')

        image1 = image1.resample(res)
        image2 = image2.resample(res)

    # Get the image data and metadata
    im1 = image1.image
    im2 = image2.image
    meta1 = image1.get_meta()
    meta2 = image2.get_meta()

    # Calculate common bounds
    bounds1 = rio.transform.array_bounds(meta1['height'], meta1['width'], meta1['transform'])
    bounds2 = rio.transform.array_bounds(meta2['height'], meta2['width'], meta2['transform'])

    # Create union bounds (combined area)
    union_bounds = (
        min(bounds1[0], bounds2[0]),  # xmin
        min(bounds1[1], bounds2[1]),  # ymin
        max(bounds1[2], bounds2[2]),  # xmax
        max(bounds1[3], bounds2[3])   # ymax
    )

    # Create new transformation with union bounds
    res = meta1['transform'][0]  # Use resolution from first image
    new_transform = from_bounds(
        *union_bounds,
        width=int((union_bounds[2] - union_bounds[0]) / res),
        height=int((union_bounds[3] - union_bounds[1]) / res)
    )

    # Calculate dimensions of new images
    new_height = int((union_bounds[3] - union_bounds[1]) / res)
    new_width = int((union_bounds[2] - union_bounds[0]) / res)

    # Create new arrays filled with nodata value
    im1_extend = np.full((meta1['count'], new_height, new_width), nodata_value, dtype=im1.dtype)
    im2_extend = np.full((meta2['count'], new_height, new_width), nodata_value, dtype=im2.dtype)

    # Reproject original data into the new arrays
    reproject(
        source=im1,
        destination=im1_extend,
        src_transform=meta1['transform'],
        dst_transform=new_transform,
        src_crs=meta1['crs'],
        dst_crs=meta1['crs'],
        nodata=nodata_value,
        resampling=Resampling.nearest
    )

    reproject(
        source=im2,
        destination=im2_extend,
        src_transform=meta2['transform'],
        dst_transform=new_transform,
        src_crs=meta2['crs'],
        dst_crs=meta2['crs'],
        nodata=nodata_value,
        resampling=Resampling.nearest
    )

    # Create new metadata
    new_meta = meta1.copy()
    new_meta.update({
        'height': new_height,
        'width': new_width,
        'transform': new_transform,
        'nodata': nodata_value
    })

    # Create and return new Geoimage objects
    return (
        Geoimage(data=im1_extend, meta=new_meta, names=image1.names),
        Geoimage(data=im2_extend, meta=new_meta, names=image2.names)
    )

def colorcomp(image, bands, name_save='', names=None, percentile=2, channel_first=True,
              meta=None, fig_size=DEF_FIG_SIZE, title='', extent=None):
    """
    Create a color composite visualization from a multi-band image.

    This function generates an RGB color composite from a multi-band image by selecting
    three bands to represent red, green, and blue channels. The image can be displayed
    and/or saved as a file.

    Parameters
    ----------
    image : numpy.ndarray
        Input image data as a 3D array.
    bands : list of str
        List of three band identifiers to use for the RGB composite. Can be either:
        - Band names (e.g., ["R", "G", "B"]) if `names` dictionary is provided
        - Band indices as strings (e.g., ["4", "3", "2"])
    name_save : str, optional
        Path to save the color composite image. If empty, image is not saved.
        Default is ''.
    names : dict, optional
        Dictionary mapping band names to band indices (e.g., {'R': 4, 'G': 3, 'B': 2}).
        Required if using band names in the `bands` parameter.
        Default is None.
    percentile : int, optional
        Percentile value for contrast stretching (e.g., 2 for a 2-98% stretch).
        Default is 2.
    channel_first : bool, optional
        Whether input image has shape (bands, rows, cols). If False, assumes
        shape (rows, cols, bands).
        Default is True.
    meta : dict, optional
        Metadata dictionary (required if saving the image).
        Default is None.
    fig_size : tuple, optional
        Size of the figure in inches as (width, height).
        Default is DEF_FIG_SIZE.
    title : str, optional
        Title for the visualization.
        Default is ''.
    extent : list or None, optional
        The extent of coordinates as [xmin, xmax, ymin, ymax]. If None, no extent is shown.
        Default is None.

    Returns
    -------
    None
        Function displays and/or saves the color composite but doesn't return any values.

    Raises
    ------
    ValueError
        If meta is None when trying to save the image.

    Examples
    --------
    >>> # Create and display a false color composite
    >>> colorcomp(image, bands=["NIR", "R", "G"], names={'R': 3, 'G': 2, 'B': 1, 'NIR': 4})
    >>>
    >>> # Create, display and save a natural color composite
    >>> colorcomp(image, bands=["3", "2", "1"], name_save="rgb_composite.tif", meta=metadata)
    """
    # Reset matplotlib settings to avoid interference from previous plots
    reset_matplotlib()

    # Convert image to channel_last format if needed
    if channel_first is True:
        image = np.rollaxis(image, 0, 3)

    # Create band name mapping if not provided
    if names is None:
        names = {}
        for i in range(image.shape[2]):
            names[str(i + 1)] = i + 1

    # Create empty RGB image
    im = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    # Populate RGB channels with normalized band values
    im[:, :, 0] = normalize(image[:, :, names[bands[0]] - 1], percentile)
    im[:, :, 1] = normalize(image[:, :, names[bands[1]] - 1], percentile)
    im[:, :, 2] = normalize(image[:, :, names[bands[2]] - 1], percentile)



# {'R': 1, 'TGT': 2, 'EZS': 3}

    # Save image if requested
    if name_save != '':
        band_dict = {name: i+1 for i, name in enumerate(bands)}
        if meta is None:
            raise ValueError("Error: you need to provide meta info to save the image")

        # Prepare for saving
        imt = np.transpose(im, (2, 0, 1))
        meta2 = meta.copy()
        meta2['count'] = 3
        meta2['dtype'] = 'uint8'
        meta2['nodata'] = 0

        # Create folder if it doesn't exist
        folder = os.path.split(name_save)[0]
        if os.path.exists(folder) is False and folder != '':
            os.makedirs(folder)

        # Remove existing files
        if os.path.exists(name_save):
            os.remove(name_save)
        if os.path.exists(f'{name_save}.aux.xml'):
            os.remove(f'{name_save}.aux.xml')

        # Write the color composite
        with rio.open(name_save, 'w', **meta2) as outds:
            outds.write(imt)
            outds.update_tags(EXTRA_TAGS=json.dumps(band_dict))

    # Display the image
    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_title(title)

    if extent is None:
        ax.imshow(im, interpolation='nearest')
        plt.axis('off')
    else:
        ax.imshow(im, interpolation='nearest', extent=extent)

    plt.show()
    #plt.close(fig)

class Geoimage:
    """
    Main class for manipulating georeferenced raster images.

    This class provides a comprehensive toolkit for working with geospatial raster data,
    supporting operations such as image creation, visualization, band manipulation,
    reprojection, resampling, cropping, and more.

    """



    def __init__(self, source_name=None, meta_only=False,names=None, history=False,
                 data=None, meta=None, georef=None, target_crs="EPSG:4326",area=None,extent='pixel'):
        """
        Initialize a Geoimage object from a file or data array with metadata.

        Parameters
        ----------
        source_name : str, optional
            Path to a geoimage (.tif, .jp2) image file to load.
            If provided, the image data and metadata
            will be read from this file.
        meta_only : bool, optional
            If True, do not read the image but just
            the meta information (useful for image.info()).
        names : dict, optional
            Dictionary mapping band names to
            band indices (e.g., {'NIR': 1, 'R': 2, 'G': 3}).
            If not provided, bands will be
            named numerically ('1', '2', '3', ...).
        area : tuple, optional
            To read  only a window of the image
                If based on pixel coordinates, you must indicate
                - the row/col coordinades of
                        the north-west corner (deb_row,deb_col)
                - the row/col coordinades of
                        the south-east corner (end_row,end_col)
                in a tuple  `area = ((deb_row,end_row),(deb_col,end_col))`

                If based on latitude/longitude coordinates, you must indicate
                - the lat/lon coordinades of the north-west corner (lat1,lon1)
                - the lat/lon coordinades of the south-east corner (lat2,lon2)
                `area = ((lon1,lon2),(lat1,lat2))`
            If not provide, read the entire image
        extent : str, optional
            if `area` is given, precise if the coordinates
            are in pixels (extent = "pixel", default)
            or latitude/longitude (extent = "latlon")
        history : bool, optional
            Whether to track modification history for the image.
            Default is False.
        data : numpy.ndarray, optional
            Image data to initialize the object with.
            Must be provided with `meta`.
            Shape should be (bands, rows, cols).
        meta : dict, optional
            Metadata dictionary containing rasterio
            metadata fields (e.g., crs, transform).
            Required if `data` is provided.
        georef : bool, optional
            Whether the image is georeferenced.
            If None, will be determined from metadata.
        target_crs : str, optional
            Target coordinate reference system
            if reprojection is needed during loading.
            Default is "EPSG:4326".

        Attributes
        ----------
        image : numpy.ndarray
            The image data array with shape (bands, rows, cols).
        shape : tuple
            The dimensions of the image as (rows, cols).
        nb_bands : int
            The number of spectral bands in the image.
        resolution : float
            The spatial resolution of the image (pixel size in map units).
        names : dict
            Dictionary mapping band names to band indices.
        nodata : float or int
            Value used to represent no data or invalid pixels.

        Examples
        --------
        >>> # Read only meta information
        >>> img = Geoimage("landsat_image.tif",meta_only=True)
        >>> img.info()
        >>>
        >>> # Read an entire Geoimage from a file
        >>> img = Geoimage("landsat_image.tif")
        >>> img.info()
        >>>
        >>> # Read a window of a file from pixel coordinates
        >>> You must indicate
        >>>      - the row/col coordinades of
        >>>            the north-west corner (deb_row,deb_col)
        >>>      - the row/col coordinades of
        >>>            the south-east corner (end_row,end_col)
        >>> in a tuple  `((deb_row,end_row),(deb_col,end_col))`
        >>> img = Geoimage("landsat_image.tif", area=((200,500),(240,600)))
        >>> img.info()
        >>>
        >>> # Read a window of a file from lat/lon coordinates (parameter extent='latlon')
        >>> You must indicate
        >>>      - the lat/lon coordinades of the north-west corner (lat1,lon1)
        >>>      - the lat/lon coordinades of the south-east corner (lat2,lon2)
        >>> in a tuple  `((lon1,lon2),(lat1,lat2))`
        >>> img = Geoimage("landsat_image.tif", area=((38.36,38.41),(7.06,7.02)),extent='latlon'))
        >>> img.info()
        >>>
        >>> # Create a Geoimage from a NumPy array with metadata
        >>> meta = {'driver': 'GTiff', 'width': 100, 'height': 100, 'count': 3,
        >>> ...         'crs': CRS.from_epsg(4326), 'transform': Affine(0.1, 0, 0, 0, -0.1, 0)}
        >>> data = np.zeros((3, 100, 100))
        >>> img = Geoimage(data=data, meta=meta)
        >>>
        >>> # Create a Geoimage with custom band names
        >>> img = Geoimage("landsat_image.tif", names={'R': 1, 'G': 2, 'B': 3, 'NIR': 4})
        >>>
        >>> # Create a Geoimage with custom band names
        >>> img = Geoimage("landsat_image.tif", names={'R': 1, 'G': 2, 'B': 3, 'NIR': 4})

        """
        extra_tags = None # To deal with names of the bands
        if meta_only:
            if source_name is None:
                raise ValueError("You must provide a source_name to get meta information")

            src = rio.open(source_name)
            if src.crs is None or src.transform is None:
                warnings.warn("Image not georeferenced. Some functions may not work.")
                self.__georef = False
            else:
                self.__georef = True
            self.image = None
            self.__meta = src.meta

            # Check if previous names have been saved
            tags = src.tags()
            if "EXTRA_TAGS" in tags:
                try:
                    extra_tags = json.loads(tags["EXTRA_TAGS"])
                except json.JSONDecodeError:
                    extra_tags = tags["EXTRA_TAGS"]
            # If not, usual names
            if extra_tags is None:
                self.names = {}
                for i in range(self.__meta['count']):
                    self.names[str(i + 1)] = i + 1
            else:
                if check_dict(extra_tags) and len(extra_tags)==self.__meta['count']:
                    self.names = extra_tags
                    self.__namesgiven = True
                else:
                    self.names = {}
                    for i in range(self.__meta['count']):
                        self.names[str(i + 1)] = i + 1

        else:

            if source_name is not None:
                # Case 1: Loading from rst/rdc file format
                if (os.path.splitext(source_name)[1].lower() in ['.rst', '.rdc']):
                    name_rst, name_rdc = find_rst_and_rdc(os.path.splitext(source_name)[0])
                    self.image, self.__meta = read_rst_with_rdc(name_rst, name_rdc, target_crs=target_crs)
                    self.__georef = True
                # Case 2: Loading from standard geotiff file
                else:
                    # Read the entire image
                    if area is None:
                        src = rio.open(source_name)
                        if src.crs is None or src.transform is None:
                            warnings.warn("Image not georeferenced. Some functions may not work.")
                            self.__georef = False
                        else:
                            self.__georef = True
                        self.image = src.read().copy()
                        self.__meta = src.meta

                        # Check if previous names have been saved

                        tags = src.tags()
                        if "EXTRA_TAGS" in tags:
                            try:
                                extra_tags = json.loads(tags["EXTRA_TAGS"])
                            except json.JSONDecodeError:
                                extra_tags = tags["EXTRA_TAGS"]
                        # If not, usual names
                        if extra_tags is not None:
                            if not (check_dict(extra_tags) and len(extra_tags)==self.__meta['count']):
                                extra_tags = None



                    else:
                        if extent=='pixel':
                            # coordinates in pixel
                            # area=((deb_row,end_row),(deb_col,end_col))
                            nb_lig=area[0][1]-area[0][0]
                            nb_col=area[1][1]-area[1][0]
                            offset_col = area[1][0]
                            offset_lig = area[0][0]
                            window = windows.Window(offset_col, offset_lig, nb_col, nb_lig)
                            with rio.open(source_name) as src:
                                self.__meta=src.meta.copy()
                                self.image = src.read(window=window).copy()
                                tags = src.tags()
                                if "EXTRA_TAGS" in tags:
                                    try:
                                        extra_tags = json.loads(tags["EXTRA_TAGS"])
                                    except json.JSONDecodeError:
                                        extra_tags = tags["EXTRA_TAGS"]
                                # If not, usual names
                                if extra_tags is not None:
                                    if not (check_dict(extra_tags) and len(extra_tags)==self.__meta['count']):
                                        extra_tags = None

                            self.__meta.update({
                                "height": window.height,
                                "width": window.width,
                                "transform": windows.transform(window, src.transform)
                            })

                            self.__georef = True
                            # Check if previous names have been saved

                        else:
                            # coordinates in latlon
                            # area=((deb_row,end_row),(deb_col,end_col))
                            src=rio.open(source_name)
                            try:
                                deb_row_lon = area[0][0]
                                end_row_lon = area[0][1]
                                deb_col_lat = area[1][0]
                                end_col_lat = area[1][1]
                                row_deb, col_deb = latlon_to_pixels(src.meta,
                                                                    deb_col_lat,
                                                                    deb_row_lon)
                                row_end, col_end = latlon_to_pixels(src.meta,
                                                                    end_col_lat,
                                                                    end_row_lon)
                            except Exception as e:
                                raise ValueError(f"Failed to convert geographic coordinates to pixel coordinates: {str(e)}")


                            nb_lig=row_end-row_deb
                            nb_col=col_end-col_deb
                            offset_col = col_deb
                            offset_lig = row_deb
                            window = windows.Window(offset_col, offset_lig, nb_col, nb_lig)
                            with rio.open(source_name) as src:
                                self.__meta=src.meta.copy()
                                self.image = src.read(window=window).copy()
                                tags = src.tags()
                                if "EXTRA_TAGS" in tags:
                                    try:
                                        extra_tags = json.loads(tags["EXTRA_TAGS"])
                                    except json.JSONDecodeError:
                                        extra_tags = tags["EXTRA_TAGS"]
                                # If not, usual names
                                if extra_tags is not None:
                                    if not (check_dict(extra_tags) and len(extra_tags)==self.__meta['count']):
                                        extra_tags = None

                                self.__meta.update({
                                    "height": window.height,
                                    "width": window.width,
                                    "transform": windows.transform(window, src.transform)
                                })

                                self.__georef = True



            # Case 3: Creating from provided data and metadata
            elif meta is not None:
                self.__meta = meta
                if georef is False:
                    self.__georef = False
                else:
                    self.__georef = True

                if data is not None:
                    # Ensure data is in the correct shape (bands, rows, cols)
                    if len(data.shape) == 2:
                        self.image = data.reshape((1, data.shape[0], data.shape[1]))
                    else:
                        self.image = data

                    # Validate dimensions match metadata
                    if ((meta['height'] != self.image.shape[1]) or
                        (meta['width'] != self.image.shape[2]) or
                        (meta['count'] != self.image.shape[0])):
                        raise ValueError("Error: metadata dimensions do not match data dimensions")
                else:
                    # Create empty image with dimensions from metadata
                    self.image = np.zeros((meta['count'], meta['height'], meta['width']))
            else:
                raise ValueError("Either source_name or both data and meta must be provided")

        # Set nodata value from metadata if available
        if "nodata" in self.__meta:
            self.nodata = self.__meta['nodata']
        else:
            self.nodata = None

        # Setup history tracking
        self.__history = history
        self.__listhistory = []

        if self.__history is True:
            now = datetime.datetime.now()
            now_str = now.strftime("%Y-%m-%d %H:%M:%S")
            if source_name is not None:
                self.__listhistory.append(f'[{now_str}] - Read image {source_name}')
            else:
                self.__listhistory.append(f'[{now_str}] - Created image from data array')

        # Setup band names
        if names is None:
            self.__namesgiven = False
            if meta_only is False:
                if extra_tags is None:
                    self.__update_names()
                else:
                    self.names = reorder_dict_by_values(extra_tags)
                    self.__namesgiven = True
        else:
            if len(names) != self.__meta['count']:
                raise ValueError(f"Error: the number of given names ({len(names)}) does not match "
                                f"the number of spectral bands ({self.__meta['count']})")
            elif lowest_value_dict(names) != 1:
                raise ValueError(f"Error: the lowest value of the names should be 1 "
                                f"(currently {lowest_value_dict(names)})")
            else:
                self.__namesgiven = True
                self.names = reorder_dict_by_values(names)

        # Update derived properties
        if meta_only is False:
            self.__update()
        self.__update()

    def __update(self):
        """
        Update derived properties after changes to the image or metadata.
        """
        self.shape = (self.__meta['height'], self.__meta['width'])
        self.nb_bands = self.__meta['count']
        self.resolution = self.__meta['transform'][0]
        self.names = reorder_dict_by_values(self.names)

        # Compute extent for visualization
        if self.__georef:
            limx, limy = self.pixel2latlon(0, 0)
            limxm, limym = self.pixel2latlon(self.shape[0], self.shape[1])

            if ((limx == limxm) or (limy == limym)):
#                self.__extent_latlon = [0, self.shape[1], self.shape[0], 0]
                self.__extent_latlon = [0, self.shape[1],  0, self.shape[0]]
                self.__extent_pixels = [0, self.shape[1], self.shape[0], 0]
            else:
#                self.__extent_latlon = [limy, limym, limx, limxm]
                self.__extent_latlon = [limy, limym, limxm, limx]
                self.__extent_pixels = [0, self.shape[1], self.shape[0], 0]
        else:
#            self.__extent_latlon = [0, self.shape[1], 0, self.shape[0]]
            self.__extent_latlon = [0, self.shape[1],  self.shape[0], 0]
            self.__extent_pixels = [0, self.shape[1], self.shape[0], 0]

        # Validate number of bands
        if self.__meta['count'] != len(self.names):
            raise ValueError(f'Number of band names given ({len(self.names)}) '
                            f'does not match number of bands ({self.__meta["count"]})')

        # Update nodata value
        self.nodata = self.__meta['nodata']

    def __update_names(self):
        """
        Initialize or update band names to default numeric sequence.
        """
        self.names = {}
        for i in range(self.image.shape[0]):
            self.names[str(i + 1)] = i + 1

    def __getitem__(self, index):
        """
        Enable indexing into the Geoimage to get pixel values.

        Parameters
        ----------
        index : tuple or slice
            - If a tuple (row, col): returns all band values at that pixel
            - If slice [:]: returns the entire image array

        Returns
        -------
        numpy.ndarray
            The selected pixel values or full image array

        Examples
        --------
        >>> # Get all band values for a specific pixel
        >>> values = image[100, 200]  # Values at row 100, col 200
        >>>
        >>> # Get the entire image array
        >>> array = image[:]
        """
        if index == slice(None):  # Corresponds to [:]
            return self.image[:]
        else:
            row, col = index
            return self.image[:, row, col]

    def __setitem__(self, condition, value):
        """
        Enable setting values in the image using boolean masks.

        This allows for conditional assignment operations like:
        image[image > 100] = 100  # Clip values above 100

        Parameters
        ----------
        condition : Geoimage or boolean array
            Boolean mask indicating which pixels to modify
        value : number, array, or Geoimage
            Value(s) to assign to the selected pixels

        Examples
        --------
        >>> # Clip values above 100
        >>> image[image > 100] = 100
        >>>
        >>> # Set all water pixels (value 1) to a different value
        >>> image[image == 1] = 0
        """
        im = self.where(condition, value, self)
        self.update_from(im)


    #------ Arithmetic and Comparison Operators ------#

    def __add__(self, other):
        """
        Add another Geoimage or a scalar value to this image.

        Parameters
        ----------
        other : Geoimage or scalar
            The image or value to add

        Returns
        -------
        Geoimage
            New Geoimage containing the result of the addition

        Examples
        --------
        >>> # Add two images
        >>> combined_image = image1 + image2
        >>>
        >>> # Add a constant to an image
        >>> brightened_image = image + 100
        """
        if isinstance(other, Geoimage):
            # Add two Geoimages
            data = self.image + other.image
            type_str = str(data.dtype)
            meta = self.__meta.copy()
            names = self.names.copy()
            meta['dtype'] = type_str
            meta['nodata'] = adapt_nodata(type_str, meta['nodata'])
            return Geoimage(data=data,
                          meta=meta, names=names,
                          georef=self.__georef)
        elif isinstance(other, (int, float)):
            # Add a scalar to a Geoimage
            data = self.image + other
            type_str = str(data.dtype)
            meta = self.__meta.copy()
            meta['dtype'] = type_str
            meta['nodata'] = adapt_nodata(type_str, meta['nodata'])
            names = self.names.copy()
            return Geoimage(data=data,
                          meta=meta, names=names,
                          georef=self.__georef)
        else:
            raise TypeError(f"Unsupported operand type: {type(other)}")

    def __sub__(self, other):
        """
        Subtract another Geoimage or a scalar value from this image.

        Parameters
        ----------
        other : Geoimage or scalar
            The image or value to subtract

        Returns
        -------
        Geoimage
            New Geoimage containing the result of the subtraction

        Examples
        --------
        >>> # Subtract one image from another
        >>> difference_image = image1 - image2
        >>>
        >>> # Subtract a constant from an image
        >>> darkened_image = image - 50
        """
        if isinstance(other, Geoimage):
            data = self.image - other.image
            type_str = str(data.dtype)
            meta = self.__meta.copy()
            names = self.names.copy()
            meta['dtype'] = type_str
            meta['nodata'] = adapt_nodata(type_str, meta['nodata'])
            return Geoimage(data=data,
                          meta=meta, names=names,
                          georef=self.__georef)
        elif isinstance(other, (int, float)):
            data = self.image - other
            type_str = str(data.dtype)
            meta = self.__meta.copy()
            meta['dtype'] = type_str
            meta['nodata'] = adapt_nodata(type_str, meta['nodata'])
            names = self.names.copy()
            return Geoimage(data=data,
                          meta=meta, names=names,
                          georef=self.__georef)
        else:
            raise TypeError(f"Unsupported operand type: {type(other)}")

    def __mul__(self, other):
        """
        Multiply this image by another Geoimage or a scalar value.

        Parameters
        ----------
        other : Geoimage or scalar
            The image or value to multiply by

        Returns
        -------
        Geoimage
            New Geoimage containing the result of the multiplication

        Examples
        --------
        >>> # Multiply two images (e.g., for masking)
        >>> masked_image = image1 * image2
        >>>
        >>> # Scale an image by a factor
        >>> scaled_image = image * 2.5
        """
        if isinstance(other, Geoimage):
            data = self.image.astype('float64') * other.image.astype('float64')
            type_str = str(data.dtype)
            meta = self.__meta.copy()
            names = self.names.copy()
            meta['dtype'] = type_str
            meta['nodata'] = adapt_nodata(type_str, meta['nodata'])
            return Geoimage(data=data,
                          meta=meta, names=names,
                          georef=self.__georef)
        elif isinstance(other, (int, float)):
            data = self.image.astype('float64') * other
            type_str = str(data.dtype)
            meta = self.__meta.copy()
            meta['dtype'] = type_str
            meta['nodata'] = adapt_nodata(type_str, meta['nodata'])
            names = self.names.copy()
            return Geoimage(data=data,
                          meta=meta, names=names,
                          georef=self.__georef)
        else:
            raise TypeError(f"Unsupported operand type: {type(other)}")

    def __truediv__(self, other):
        """
        Divide this image by another Geoimage or a scalar value.

        Parameters
        ----------
        other : Geoimage or scalar
            The image or value to divide by

        Returns
        -------
        Geoimage
            New Geoimage containing the result of the division

        Examples
        --------
        >>> # Divide one image by another (e.g., for ratio calculations)
        >>> ndvi = (nir - red) / (nir + red)  # Normalized Difference Vegetation Index
        >>>
        >>> # Scale an image by a divisor
        >>> halved_image = image / 2
        """
        if isinstance(other, Geoimage):
            data = self.image.astype('float64') / other.image.astype('float64')
            type_str = str(data.dtype)
            meta = self.__meta.copy()
            names = self.names.copy()
            meta['dtype'] = type_str
            meta['nodata'] = adapt_nodata(type_str, meta['nodata'])
            return Geoimage(data=data,
                          meta=meta, names=names,
                          georef=self.__georef)
        elif isinstance(other, (int, float)):
            data = self.image.astype('float64') / other
            type_str = str(data.dtype)
            meta = self.__meta.copy()
            meta['dtype'] = type_str
            meta['nodata'] = adapt_nodata(type_str, meta['nodata'])
            names = self.names.copy()
            return Geoimage(data=data,
                          meta=meta, names=names,
                          georef=self.__georef)
        else:
            raise TypeError(f"Unsupported operand type: {type(other)}")

    def __floordiv__(self, other):
        """
        Perform integer division of this image by another Geoimage or a scalar value.

        Parameters
        ----------
        other : Geoimage or scalar
            The image or value to divide by

        Returns
        -------
        Geoimage
            New Geoimage containing the result of the integer division

        Examples
        --------
        >>> # Integer division of images
        >>> result = image1 // image2
        >>>
        >>> # Integer division by a scalar
        >>> result = image // 10
        """
        if isinstance(other, Geoimage):
            data = self.image.astype('float64') // other.image.astype('float64')
            type_str = str(data.dtype)
            meta = self.__meta.copy()
            names = self.names.copy()
            meta['dtype'] = type_str
            meta['nodata'] = adapt_nodata(type_str, meta['nodata'])
            return Geoimage(data=data,
                          meta=meta, names=names,
                          georef=self.__georef)
        elif isinstance(other, (int, float)):
            data = self.image.astype('float64') // other
            type_str = str(data.dtype)
            meta = self.__meta.copy()
            meta['dtype'] = type_str
            meta['nodata'] = adapt_nodata(type_str, meta['nodata'])
            names = self.names.copy()
            return Geoimage(data=data,
                          meta=meta, names=names,
                          georef=self.__georef)
        else:
            raise TypeError(f"Unsupported operand type: {type(other)}")

    def __pow__(self, other):
        """
        Raise this image to the power of another Geoimage or a scalar value.

        Parameters
        ----------
        other : Geoimage or scalar
            The image or value to use as the exponent

        Returns
        -------
        Geoimage
            New Geoimage containing the result of the power operation

        Examples
        --------
        >>> # Square an image
        >>> squared_image = image ** 2
        >>>
        >>> # Raise image to a variable power
        >>> variable_power = image1 ** image2
        """
        if isinstance(other, Geoimage):
            data = self.image.astype('float64') ** other.image.astype('float64')
            type_str = str(data.dtype)
            meta = self.__meta.copy()
            names = self.names.copy()
            meta['dtype'] = type_str
            meta['nodata'] = adapt_nodata(type_str, meta['nodata'])
            return Geoimage(data=data,
                          meta=meta, names=names,
                          georef=self.__georef)
        elif isinstance(other, (int, float)):
            data = self.image.astype('float64') ** other
            type_str = str(data.dtype)
            meta = self.__meta.copy()
            meta['dtype'] = type_str
            meta['nodata'] = adapt_nodata(type_str, meta['nodata'])
            names = self.names.copy()
            return Geoimage(data=data,
                          meta=meta, names=names,
                          georef=self.__georef)
        else:
            raise TypeError(f"Unsupported operand type: {type(other)}")

    #------ Comparison Operators ------#

    def __lt__(self, other):
        """
        Element-wise less than comparison.

        Parameters
        ----------
        other : Geoimage or scalar
            The image or value to compare against

        Returns
        -------
        Geoimage
            New Geoimage containing boolean results of the comparison

        Examples
        --------
        >>> # Find pixels with values less than 100
        >>> mask = image < 100
        >>> mask.visu()  # Visualize the mask
        """
        if isinstance(other, Geoimage):
            data = self.image < other.image
        else:
            data = self.image < other
        type_str = str(data.dtype)
        meta = self.__meta.copy()
        names = self.names.copy()
        meta['dtype'] = type_str
        meta['nodata'] = adapt_nodata(type_str, meta['nodata'])
        return Geoimage(data=data,
                      meta=meta, names=names,
                      georef=self.__georef)

    def __le__(self, other):
        """
        Element-wise less than or equal comparison.

        Parameters
        ----------
        other : Geoimage or scalar
            The image or value to compare against

        Returns
        -------
        Geoimage
            New Geoimage containing boolean results of the comparison
        """
        if isinstance(other, Geoimage):
            data = self.image <= other.image
        else:
            data = self.image <= other
        type_str = str(data.dtype)
        meta = self.__meta.copy()
        names = self.names.copy()
        meta['dtype'] = type_str
        meta['nodata'] = adapt_nodata(type_str, meta['nodata'])
        return Geoimage(data=data,
                      meta=meta, names=names,
                      georef=self.__georef)

    def __gt__(self, other):
        """
        Element-wise greater than comparison.

        Parameters
        ----------
        other : Geoimage or scalar
            The image or value to compare against

        Returns
        -------
        Geoimage
            New Geoimage containing boolean results of the comparison

        Examples
        --------
        >>> # Find pixels with values greater than a threshold
        >>> high_values = image > 200
        """
        if isinstance(other, Geoimage):
            data = self.image > other.image
        else:
            data = self.image > other
        type_str = str(data.dtype)
        meta = self.__meta.copy()
        names = self.names.copy()
        meta['dtype'] = type_str
        meta['nodata'] = adapt_nodata(type_str, meta['nodata'])
        return Geoimage(data=data,
                      meta=meta, names=names,
                      georef=self.__georef)

    def __ge__(self, other):
        """
        Element-wise greater than or equal comparison.

        Parameters
        ----------
        other : Geoimage or scalar
            The image or value to compare against

        Returns
        -------
        Geoimage
            New Geoimage containing boolean results of the comparison
        """
        if isinstance(other, Geoimage):
            data = self.image >= other.image
        else:
            data = self.image >= other
        type_str = str(data.dtype)
        meta = self.__meta.copy()
        names = self.names.copy()
        meta['dtype'] = type_str
        meta['nodata'] = adapt_nodata(type_str, meta['nodata'])
        return Geoimage(data=data,
                      meta=meta, names=names,
                      georef=self.__georef)

    def __eq__(self, other):
        """
        Element-wise equality comparison.

        Parameters
        ----------
        other : Geoimage or scalar
            The image or value to compare against

        Returns
        -------
        Geoimage
            New Geoimage containing boolean results of the comparison

        Examples
        --------
        >>> # Create a mask for pixels with a specific value
        >>> water_mask = image == 1  # Assuming water is coded as 1
        """
        if isinstance(other, Geoimage):
            data = self.image == other.image
        else:
            data = self.image == other
        type_str = str(data.dtype)
        meta = self.__meta.copy()
        names = self.names.copy()
        meta['dtype'] = type_str
        return Geoimage(data=data,
                      meta=meta, names=names,
                      georef=self.__georef)

    def __ne__(self, other):
        """
        Element-wise not-equal comparison.

        Parameters
        ----------
        other : Geoimage or scalar
            The image or value to compare against

        Returns
        -------
        Geoimage
            New Geoimage containing boolean results of the comparison
        """
        if isinstance(other, Geoimage):
            data = (self.image != other.image)
        else:
            data = (self.image != other)
        type_str = str(data.dtype)
        meta = self.__meta.copy()
        names = self.names.copy()
        meta['dtype'] = type_str
        meta['nodata'] = adapt_nodata(type_str, meta['nodata'])
        return Geoimage(data=data,
                      meta=meta, names=names,
                      georef=self.__georef)

    def __neg__(self):
        """
        Negate the image (multiply by -1).

        Returns
        -------
        Geoimage
            New Geoimage containing the negated values

        Examples
        --------
        >>> # Negate all values in an image
        >>> negative_image = -image
        """
        data = -self.image
        type_str = str(data.dtype)
        meta = self.__meta.copy()
        names = self.names.copy()
        meta['dtype'] = type_str
        meta['nodata'] = adapt_nodata(type_str, meta['nodata'])
        return Geoimage(data=data,
                      meta=meta,
                      names=names,
                      georef=self.__georef)


    #------ Reversible Operators and Bitwise Operators ------#

    def __radd__(self, other):
        """
        Reverse addition operator (for cases like 5 + image).

        Parameters
        ----------
        other : scalar
            The value to add to this image

        Returns
        -------
        Geoimage
            New Geoimage containing the result of the addition

        Examples
        --------
        >>> brightened_image = 100 + image  # Same as image + 100
        """
        return self.__add__(other)

    def __rmul__(self, other):
        """
        Reverse multiplication operator (for cases like 2 * image).

        Parameters
        ----------
        other : scalar
            The value to multiply this image by

        Returns
        -------
        Geoimage
            New Geoimage containing the result of the multiplication

        Examples
        --------
        >>> doubled_image = 2 * image  # Same as image * 2
        """
        return self.__mul__(other)

    def __rsub__(self, other):
        """
        Reverse subtraction operator (for cases like 100 - image).

        Parameters
        ----------
        other : scalar or Geoimage
            The value to subtract this image from

        Returns
        -------
        Geoimage
            New Geoimage containing the result of the subtraction

        Examples
        --------
        >>> inverse_difference = 255 - image  # Image negative/inverse for 8-bit data
        """
        if isinstance(other, Geoimage):
            data = other.image - self.image
            type_str = str(data.dtype)
            meta = self.__meta.copy()
            names = self.names.copy()
            meta['dtype'] = type_str
            meta['nodata'] = adapt_nodata(type_str, meta['nodata'])
            return Geoimage(data=data,
                          meta=meta, names=names,
                          georef=self.__georef)
        elif isinstance(other, (int, float)):
            data = other - self.image
            type_str = str(data.dtype)
            meta = self.__meta.copy()
            meta['dtype'] = type_str
            meta['nodata'] = adapt_nodata(type_str, meta['nodata'])
            names = self.names.copy()
            return Geoimage(data=data,
                          meta=meta, names=names,
                          georef=self.__georef)
        else:
            raise TypeError(f"Unsupported operand type: {type(other)}")

    def __rtruediv__(self, other):
        """
        Reverse division operator (for cases like 1 / image).

        Parameters
        ----------
        other : scalar or Geoimage
            The value to divide by this image

        Returns
        -------
        Geoimage
            New Geoimage containing the result of the division

        Examples
        --------
        >>> reciprocal_image = 1 / image  # Reciprocal of each pixel value
        """
        if isinstance(other, Geoimage):
            data = other.image.astype('float64') / self.image.astype('float64')
            type_str = str(data.dtype)
            meta = self.__meta.copy()
            names = self.names.copy()
            meta['dtype'] = type_str
            meta['nodata'] = adapt_nodata(type_str, meta['nodata'])
            return Geoimage(data=data,
                          meta=meta, names=names,
                          georef=self.__georef)
        elif isinstance(other, (int, float)):
            data = other / self.image.astype('float64')
            type_str = str(data.dtype)
            meta = self.__meta.copy()
            meta['dtype'] = type_str
            meta['nodata'] = adapt_nodata(type_str, meta['nodata'])
            names = self.names.copy()
            return Geoimage(data=data,
                          meta=meta, names=names,
                          georef=self.__georef)
        else:
            raise TypeError(f"Unsupported operand type: {type(other)}")

    def __rfloordiv__(self, other):
        """
        Reverse integer division operator (for cases like 100 // image).

        Parameters
        ----------
        other : scalar or Geoimage
            The value to floor-divide by this image

        Returns
        -------
        Geoimage
            New Geoimage containing the result of the integer division

        Examples
        --------
        >>> result = 100 // image  # Integer division of 100 by each pixel value
        """
        if isinstance(other, Geoimage):
            data = other.image.astype('float64') // self.image.astype('float64')
            type_str = str(data.dtype)
            meta = self.__meta.copy()
            names = self.names.copy()
            meta['dtype'] = type_str
            meta['nodata'] = adapt_nodata(type_str, meta['nodata'])
            return Geoimage(data=data,
                          meta=meta, names=names,
                          georef=self.__georef)
        elif isinstance(other, (int, float)):
            data = other // self.image.astype('float64')
            type_str = str(data.dtype)
            meta = self.__meta.copy()
            meta['dtype'] = type_str
            meta['nodata'] = adapt_nodata(type_str, meta['nodata'])
            names = self.names.copy()
            return Geoimage(data=data,
                          meta=meta, names=names,
                          georef=self.__georef)
        else:
            raise TypeError(f"Unsupported operand type: {type(other)}")

    #------ Bitwise Operators ------#

    def __or__(self, other):
        """
        Bitwise OR operation between images.

        Parameters
        ----------
        other : Geoimage
            The image to OR with this image

        Returns
        -------
        Geoimage
            New Geoimage containing the result of the OR operation

        Examples
        --------
        >>> combined_mask = mask1 | mask2  # Pixels that are in either mask
        """
        data = self.image | other.image
        meta = self.__meta.copy()
        type_str = str(data.dtype)
        meta['dtype'] = type_str
        meta['nodata'] = adapt_nodata(type_str, meta['nodata'])
        names = self.names.copy()
        return Geoimage(data=data,
                      meta=meta, names=names,
                      georef=self.__georef)

    def __and__(self, other):
        """
        Bitwise AND operation between images.

        Parameters
        ----------
        other : Geoimage
            The image to AND with this image

        Returns
        -------
        Geoimage
            New Geoimage containing the result of the AND operation

        Examples
        --------
        >>> overlap_mask = mask1 & mask2  # Pixels that are in both masks
        """
        data = self.image & other.image
        meta = self.__meta.copy()
        type_str = str(data.dtype)
        meta['dtype'] = type_str
        meta['nodata'] = adapt_nodata(type_str, meta['nodata'])
        names = self.names.copy()
        return Geoimage(data=data,
                      meta=meta, names=names,
                      georef=self.__georef)

    def __xor__(self, other):
        """
        Bitwise XOR operation between images.

        Parameters
        ----------
        other : Geoimage
            The image to XOR with this image

        Returns
        -------
        Geoimage
            New Geoimage containing the result of the XOR operation

        Examples
        --------
        >>> exclusive_mask = mask1 ^ mask2  # Pixels that are in only one of the masks
        """
        data = self.image ^ other.image
        meta = self.__meta.copy()
        type_str = str(data.dtype)
        meta['dtype'] = type_str
        meta['nodata'] = adapt_nodata(type_str, meta['nodata'])
        names = self.names.copy()
        return Geoimage(data=data,
                      meta=meta, names=names,
                      georef=self.__georef)

    def __invert__(self):
        """
        Bitwise NOT operation (invert all bits).

        Returns
        -------
        Geoimage
            New Geoimage containing the inverted values

        Examples
        --------
        >>> inverted_mask = ~mask  # Invert a binary mask
        """
        data = ~(self.image.copy())
        meta = self.__meta.copy()
        type_str = str(data.dtype)
        meta['dtype'] = type_str
        meta['nodata'] = adapt_nodata(type_str, meta['nodata'])
        names = self.names.copy()
        return Geoimage(data=data,
                      meta=meta, names=names,
                      georef=self.__georef)

    def where(self, condition, value1, value2):
        """
        Select values based on a condition, similar to numpy.where().

        This method allows for conditional operations, selecting values from
        `value1` where `condition` is True, and from `value2` where it's False.

        Parameters
        ----------
        condition : Geoimage
            Boolean mask indicating where to select values from `value1`
        value1 : Geoimage or scalar
            Values to use where condition is True
        value2 : Geoimage or scalar
            Values to use where condition is False

        Returns
        -------
        Geoimage
            New Geoimage containing the result of the conditional selection

        Examples
        --------
        >>> # Create a cloud-free composite from two images
        >>> cloud_free = image1.where(cloud_mask, image2, image1)
        >>>
        >>> # Threshold an image
        >>> thresholded = image.where(image > 100, 255, 0)

        """
        tab_condition = match_dimensions(condition.image, self.image)

        if isinstance(value1, Geoimage) and isinstance(value2, Geoimage):
            data = np.where(tab_condition, value1.image, value2.image)
        elif isinstance(value1, Geoimage) and not isinstance(value2, Geoimage):
            data = np.where(tab_condition, value1.image, value2)
        elif not isinstance(value1, Geoimage) and isinstance(value2, Geoimage):
            data = np.where(tab_condition, value1, value2.image)
        else:
            data = np.where(tab_condition, value1, value2)

        meta = self.__meta.copy()
        type_str = str(data.dtype)
        meta['dtype'] = type_str
        meta['nodata'] = adapt_nodata(type_str, meta['nodata'])

        return Geoimage(data=data, meta=self.__meta.copy(),
                       names=self.names.copy(), georef=self.__georef)

    def update_from(self, other):
        """
        Update the current Geoimage with the attributes from another Geoimage.

        This method copies all attributes from the `other` Geoimage to this one,
        effectively replacing this image's content with that of `other`.

        Parameters
        ----------
        other : Geoimage
            The Geoimage to copy attributes from

        Returns
        -------
        None
            This method modifies the current object in place

        Examples
        --------
        >>> result = image1.where(mask, 0, image1)  # Create a masked copy
        >>> image1.update_from(result)  # Update the original with the masked version
        """
        for attr, value in other.__dict__.items():
            setattr(self, attr, value)

    def reset_names(self):
        """
        Reset the band names to sequential numbers ("1", "2", ...).

        This method is useful when multiple stacks, removals, or additions of bands
        have left the band naming confusing or inconsistent.

        Returns
        -------
        self : Geoimage
            The method returns the object itself to allow method chaining

        Examples
        --------
        >>> stacked_image = image1.apply_stack(image2)
        >>> stacked_image.reset_names()
        >>> stacked_image.info()  # Shows bands renamed to "1", "2", ...
        """
        self.names = initialize_dict(self.nb_bands)
        self.__namesgiven = False

        if self.__history is not False:
            now = datetime.datetime.now()
            now_str = now.strftime("%Y-%m-%d %H:%M:%S")
            self.__listhistory.append(f'[{now_str}] - Reset names of bands')

        return self

    def change_nodata(self, nodatavalue):
        """
        Modify the no-data value of the image.

        Parameters
        ----------
        nodatavalue : float or int
            The new no-data value to assign

        Returns
        -------
        self : Geoimage
            The method returns the object itself to allow method chaining

        Examples
        --------
        >>> image.change_nodata(np.nan)  # Use NaN as nodata
        >>> image.change_nodata(-9999)   # Use -9999 as nodata
        """
        self.nodata = nodatavalue
        self.__meta['nodata'] = nodatavalue

        if self.__history is not False:
            now = datetime.datetime.now()
            now_str = now.strftime("%Y-%m-%d %H:%M:%S")
            self.__listhistory.append(f'[{now_str}] - Changing nodata to {nodatavalue}')

        return self

    def change_names(self, names):
        """
        Modify the names of spectral bands.

        Parameters
        ----------
        names : dict
            Dictionary mapping band names to band indices
            (e.g., {'R': 1, 'G': 2, 'B': 3, 'NIR': 4})

        Returns
        -------
        self : Geoimage
            The method returns the object itself to allow method chaining

        Raises
        ------
        ValueError
            If the number of provided names doesn't match the number of bands

        Examples
        --------
        >>> sentinel2_names = {'B': 1, 'G': 2, 'R': 3, 'NIR': 4, 'SWIR1': 5, 'SWIR2': 6}
        >>> image.change_names(sentinel2_names)
        >>> image.info()  # Shows updated band names
        """
        if len(names) != self.__meta['count']:
            raise ValueError(f"Error: the number of given names ({len(names)}) does not match "
                           f"the number of spectral bands ({self.__meta['count']})")
        else:
            if check_dict(names):
                self.__namesgiven = True
                self.names = reorder_dict_by_values(names)
            else:
                raise ValueError(f"Error: inconsistent names given {names}")

        if self.__history is not False:
            now = datetime.datetime.now()
            now_str = now.strftime("%Y-%m-%d %H:%M:%S")
            self.__listhistory.append(f'[{now_str}] - Changed band names')

        # return self


    def activate_history(self):
        """
        Activate history tracking for the image object.

        This method enables logging of operations performed on the image,
        which can be useful for tracking processing steps and debugging.

        Returns
        -------
        self : Geoimage
            The method returns the object itself to allow method chaining

        Examples
        --------
        >>> image = Geoimage("landsat_image.tif")
        >>> image.activate_history()
        >>> image.resample(30,inplace=True)  # This operation will be tracked in history
        >>> image.info()  # The history section will show the resampling operation

        See Also
        --------
        deactivate_history : Disable history tracking
        info : Display image information including history
        """
        self.__history = True
        return self

    def deactivate_history(self):
        """
        Deactivate history tracking for the image object.

        This method disables logging of operations performed on the image.

        Returns
        -------
        self : Geoimage
            The method returns the object itself to allow method chaining

        Examples
        --------
        >>> image.deactivate_history()
        >>> image.resample(15)  # This operation won't be tracked in history

        See Also
        --------
        activate_history : Enable history tracking
        """
        self.__history = False
        return self

    def copy(self):
        """
        Create a deep copy of the Geoimage object.

        Returns
        -------
        Geoimage
            A new Geoimage instance that is a complete copy of the current object

        Examples
        --------
        >>> original = Geoimage("landsat_image.tif")
        >>> duplicate = original.copy()
        >>> duplicate.resample(15,inplace=True)  # This won't affect the original image

        Notes
        -----
        This method creates a completely independent copy. Changes to the copy
        won't affect the original image, and vice versa.
        """
        data = self.image.copy()
        meta = self.__meta.copy()
        names = self.names.copy()
        georef = self.__georef
        history = self.__history
        im = Geoimage(data=data, meta=meta, names=names, georef=georef)

        if history:
            im.activate_history()
            # Also copy history records if they exist
            if hasattr(self, '__listhistory') and self.__listhistory:
                im.__listhistory = self.__listhistory.copy()

        return im

    def info(self):
        """
        Print detailed information about the image.

        This method displays a comprehensive overview of the image's properties,
        including:
        - Dimensions (rows, columns, bands)
        - Spatial resolution
        - Geographic coordinates of the center
        - Projection system
        - Data type
        - Nodata value
        - Band names
        - Processing history (if history tracking is enabled)

        Examples
        --------
        >>> image = Geoimage("landsat_image.tif")
        >>> image.info()
        >>> - Size of the image:
        >>>    - Rows (height): 1024
        >>>    - Col (width): 1024
        >>>    - Bands: 4
        >>> - Spatial resolution: 30.0 meters / degree
        >>> - Central point latitude - longitude coordinates: (36.12345, -118.67890)
        >>> - Driver: GTiff
        >>> - Data type: uint16
        >>> - Projection system: EPSG:32611
        >>> - Nodata: 0
        >>> - Given names for spectral bands:
        >>>    {'B': 1, 'G': 2, 'R': 3, 'NIR': 4}
        >>> --- History of modifications---
        >>> [2023-09-15 10:30:22] - Read image landsat_image.tif
        >>> [2023-09-15 10:31:45] - Apply resampling at 30.000000 meters
        """
        print('- Size of the image:')
#        print('   - Rows (height):', self.shape[0])
#        print('   - Cols (width):', self.shape[1])
#        print('   - Bands:', self.nb_bands)
#        print('- Spatial resolution:', self.resolution, ' meters / degree (depending on projection system)')
        print('   - Rows (height):', self.__meta['height'])
        print('   - Cols (width):', self.__meta['width'])
        print('   - Bands:', self.__meta['count'])
        print('- Spatial resolution:', self.__meta['transform'][0], ' meters / degree (depending on projection system)')

        if self.__georef is True:
            center_lat, center_lon = self.get_latlon_coordinates()
            print(f'- Central point latitude - longitude coordinates: ({center_lat:.8f}, {center_lon:.8f})')

        print('- Driver:', self.__meta['driver'])
        print('- Data type:', self.__meta['dtype'])
        print('- Projection system:', self.__meta['crs'])

        if self.nodata is not None:
            print('- Nodata:', self.nodata)

        print('\n- Given names for spectral bands: ')
        print('  ', self.names)

#        if self.__history is not False and hasattr(self, '__listhistory') and self.__listhistory:
        if self.__history is not False and self.__listhistory:
            print('\n--- History of modifications---')
            for history_entry in self.__listhistory:
                print(history_entry)

        print('\n')

    def get_type(self):
        """
        Get the data type of the image.

        Returns
        -------
        str
            The NumPy data type of the image (e.g., 'uint8', 'float32')

        Examples
        --------
        >>> data_type = image.get_type()
        >>> print(f"Image has data type: {data_type}")
        """
        return self.__meta['dtype']

    def get_spatial_resolution(self):
        """
        Get the spatial resolution of the image.

        Returns
        -------
        float
            The spatial resolution in meters or degrees (depending on the projection)

        Examples
        --------
        >>> resolution = image.get_spatial_resolution()
        >>> print(f"Image has {resolution} meter resolution")
        """
        return self.__meta['transform'][0]

    def get_latlon_coordinates(self):
        """
        Get the latitude and longitude coordinates of the central point of the image.

        Returns
        -------
        tuple of float
            The (latitude, longitude) of the center of the image

        Examples
        --------
        >>> lat, lon = image.get_latlon_coordinates()
        >>> print(f"Image center is at latitude {lat}, longitude {lon}")
        """
        return (0.5 * (self.pixel2latlon(0, 0)[0] + self.pixel2latlon(self.shape[0], self.shape[1])[0]),
                0.5 * (self.pixel2latlon(0, 0)[1] + self.pixel2latlon(self.shape[0], self.shape[1])[1]))

    def get_size(self):
        """
        Get the size (dimensions) of the image.

        Returns
        -------
        tuple of int
            The (rows, columns) dimensions of the image

        Examples
        --------
        >>> rows, cols = image.get_size()
        >>> print(f"Image has {rows} rows and {cols} columns")
        """
        return (self.__meta['height'], self.__meta['width'])

    def get_nb_bands(self):
        """
        Get the number of spectral bands in the image.

        Returns
        -------
        int
            The number of bands

        Examples
        --------
        >>> nb_bands = image.get_nb_bands()
        >>> print(f"Image has {nb_bands} spectral bands")
        """
        return self.__meta['count']

    def get_meta(self):
        """
        Get the metadata dictionary.

        Returns
        -------
        dict
            A copy of the rasterio metadata dictionary

        Examples
        --------
        >>> metadata = image.get_meta()
        >>> print(f"Image CRS: {metadata['crs']}")
        """
        return self.__meta.copy()

    def get_nodata(self):
        """
        Get the nodata value of the image.

        Returns
        -------
        float, int, or None
            The nodata value if it exists, otherwise None

        Examples
        --------
        >>> nodata = image.get_nodata()
        >>> print(f"Nodata value: {nodata}")
        """
        if "nodata" in self.__meta:
            return self.__meta['nodata']
        else:
            return None

    def get_bounds(self):
        """
        Get the geographic bounds of the image.

        Returns
        -------
        rasterio.coords.BoundingBox
            The bounding box of the image (left, bottom, right, top)

        Examples
        --------
        >>> bounds = image.get_bounds()
        >>> print(f"Image covers from ({bounds.left}, {bounds.bottom}) to ({bounds.right}, {bounds.top})")
        """
        return calculate_bounds(self.__meta)

    def get_names(self):
        """
        Get the band names dictionary.

        Returns
        -------
        dict
            A copy of the dictionary mapping band names to band indices

        Examples
        --------
        >>> names = image.get_names()
        >>> print(f"Red band is at index {names.get('R', 'unknown')}")
        """
        return self.names.copy()

    def get_georef(self):
        """
        Check if the image is georeferenced.

        Returns
        -------
        bool
            True if the image is georeferenced, False otherwise

        Examples
        --------
        >>> is_georeferenced = image.get_georef()
        >>> if not is_georeferenced:
        ...     print("Warning: Image is not georeferenced")
        """
        return self.__georef

    def unique(self):
        """
        Get the unique values in the image.

        Returns
        -------
        numpy.ndarray
            Array of unique values found in the image

        Examples
        --------
        >>> unique_values = image.unique()
        >>> print(f"Image contains {len(unique_values)} unique values")
        >>> print(f"Values: {unique_values}")

        Notes
        -----
        This method is particularly useful for categorical data or classified images
        to see how many classes are present.
        """
        return np.unique(self.image)

    def isnan(self):
        """
        Return a boolean mask indicating which pixels in the image
        contain NaN values.

        Returns
        -------
        Geoimage
            A new `Geoimage` object with test of nan values

        Examples
        --------
        >>> im_isnan = image.isnan()

        Notes
        -----
        This method does not modify the current object. Instead, it returns
        a new `Geoimage` instance. The resulting boolean raster can be used
        as an independent mask or combined with logical operations to filter
        values.
        """
        data = np.isnan(self.image)
        meta = self.__meta.copy()

        meta['dtype'] = "bool"
        meta['nodata'] = None
        names = {f"{k}_isnan": v for k, v in self.names.items()}
        return Geoimage(data=data,
                      meta=meta, names=names,
                      georef=self.__georef)


    def abs(self, axis=None, inplace=False):
        """
        Calculate the absolute value of the image data.

        This method modifies the image content directly by replacing all values
        with their absolute values.

        Parameters
        ----------
        axis : str or None, optional
            Not used, kept for API consistency with other statistical methods.
            Default is None.

        inplace : bool, default False
            If False, return a copy. Otherwise, do absolute value in place and return None.

        Returns
        -------
        Geoimage
            The absolute value image or None if `inplace=True`

        Examples
        --------
        >>> difference = image1 - image2  # May contain negative values
        >>> diff_abs = difference.abs()  # Compute absolute values of `differences`
        >>> difference.abs(inplace = True)  # Directly convert differences to absolute values
        >>> difference.info()

        """

        if inplace:
            self.image = np.abs(self.image)

            if self.__history is not False:
                now = datetime.datetime.now()
                now_str = now.strftime("%Y-%m-%d %H:%M:%S")
                self.__listhistory.append(f'[{now_str}] - Applied absolute value to image data')
        else:
            return(self.__apply_abs(axis=axis))

    def __apply_abs(self, axis=None):
        """
        Create a new image with the absolute values of this image.

        Unlike the `abs()` method, this doesn't modify the original image.

        Parameters
        ----------
        axis : str or None, optional
            Not used, kept for API consistency with other statistical methods.
            Default is None.

        Returns
        -------
        Geoimage
            A new image containing the absolute values

        Examples
        --------
        >>> difference = image1 - image2  # May contain negative values
        >>> abs_difference = difference.apply_abs()  # New image with absolute values
        >>> difference.info()  # Original still has negative values
        >>> abs_difference.info()  # New image has only positive values

        See Also
        --------
        abs : Modify the image in-place with absolute values
        """
        im = self.copy()
        im.image = np.abs(im.image)

        if im.__history is not False:
            now = datetime.datetime.now()
            now_str = now.strftime("%Y-%m-%d %H:%M:%S")
            im.__listhistory.append(f'[{now_str}] - Created new image with absolute values')

        return im

    def sum(self, axis=None):
        """
        Calculate the sum of image values along a specified axis.

        Parameters
        ----------
        axis : {'band', 'row', 'col', None}, optional
            The axis along which to compute the sum:
            - 'band': Sum across spectral bands for each pixel
            - 'row': Sum across rows (lines) for each band and column
            - 'col': Sum across columns for each band and row
            - 'pixel': Sum across pixels for each bands
            - None: Sum of all values in the image
            Default is None.

        Returns
        -------
        float or numpy.ndarray
            - If axis=None: A single value representing the sum of the entire image
            - If axis='band': Array with shape (nb_rows,nb_cols) containing  sums along bands
            - If axis='row': Array with shape (nb_bands,nb_cols) containing sums along rows
            - If axis='col': Array with shape (nb_bands,nb_rows) containing  sums along cols
            - If axis='pixel': Array with shape (nb_bands) containing  sums  along all pixels for each band

        Raises
        ------
        ValueError
            If an invalid axis is specified

        Examples
        --------
        >>> total = image.sum()  # Sum of all pixel values
        >>> print(f"Total pixel sum: {total}")
        >>>
        >>> band_sums = image.sum(axis='pixel')  # Sum along all pixels for each band
        """
        if axis == 'band':
            #return np.nansum(self.image, axis=(1, 2))
            return np.nansum(self.image, axis=0)
        elif axis == 'row':
            #return np.nansum(self.image, axis=(0, 2))
            return np.nansum(self.image, axis=1)
        elif axis == 'col':
            #return np.nansum(self.image, axis=(0, 1))
            return np.nansum(self.image, axis=2)
        elif axis == 'pixel':
            return np.nansum(self.image, axis=(1, 2))
        elif axis is None:
            return np.nansum(self.image)
        else:
            raise ValueError(f'Error: axis "{axis}" undefined. Use "band", "row", "col", or None.')

    def min(self, axis=None):
        """
        Calculate the minimum value along a specified axis.

        Parameters
        ----------
        axis : {'band', 'row', 'col', None}, optional
            The axis along which to compute the minimum:
            - 'band': Minimum across spectral bands for each pixel
            - 'row': Minimum across rows (lines) for each band and column
            - 'col': Minimum across columns for each band and row
            - 'pixel': Minimum across pixels for each bands
            - None: Global minimum of the entire image
            Default is None.

        Returns
        -------
        float or numpy.ndarray
            - If axis=None: A single value representing the global minimum
            - If axis='band': Array with shape (nb_rows,nb_cols) containing  mins along bands
            - If axis='row': Array with shape (nb_bands,nb_cols) containing mins along rows
            - If axis='col': Array with shape (nb_bands,nb_rows) containing  mins along cols
            - If axis='pixel': Array with shape (nb_bands) containing  mins along all pixels for each band

        Raises
        ------
        ValueError
            If an invalid axis is specified

        Examples
        --------
        >>> min_value = image.min()  # Global minimum value
        >>> print(f"Minimum pixel value: {min_value}")
        >>>
        >>> band_mins = image.min(axis='pixel')  # Minimum along all pixels for each band
        """
        if axis == 'band':
            #return np.nanmin(self.image, axis=(1, 2))
            return np.nanmin(self.image, axis=0)
        elif axis == 'row':
            #return np.nanmin(self.image, axis=(0, 2))
            return np.nanmin(self.image, axis=1)
        elif axis == 'col':
            #return np.nanmin(self.image, axis=(0, 1))
            return np.nanmin(self.image, axis=2)
        elif axis == 'pixel':
            return np.nanmin(self.image, axis=(1, 2))
        elif axis is None:
            return np.nanmin(self.image)
        else:
            raise ValueError(f'Error: axis "{axis}" undefined. Use "band", "row", "col", or None.')

    def max(self, axis=None):
        """
        Calculate the maximum value along a specified axis.

        Parameters
        ----------
        axis : {'band', 'row', 'col', None}, optional
            The axis along which to compute the maximum:
            - 'band': Maximum across spectral bands for each pixel
            - 'row': Maximum across rows (lines) for each band and column
            - 'col': Maximum across columns for each band and row
            - 'pixel': Maximum across pixels for each bands
            - None: Global maximum of the entire image
            Default is None.

        Returns
        -------
        float or numpy.ndarray
            - If axis=None: A single value representing the global maximum
            - If axis='band': Array with shape (nb_rows,nb_cols) containing  max along bands
            - If axis='row': Array with shape (nb_bands,nb_cols) containing max along rows
            - If axis='col': Array with shape (nb_bands,nb_rows) containing  max along cols
            - If axis='pixel': Array with shape (nb_bands) containing  maxs along all pixels for each band

        Raises
        ------
        ValueError
            If an invalid axis is specified

        Examples
        --------
        >>> max_value = image.max()  # Global maximum value
        >>> print(f"Maximum pixel value: {max_value}")
        >>>
        >>> band_maxs = image.max(axis='pixel')  # Maximum along all pixels for each band
        """
        if axis == 'band':
            #return np.nanmax(self.image, axis=(1, 2))
            return np.nanmax(self.image, axis=0)
        elif axis == 'row':
            #return np.nanmax(self.image, axis=(0, 2))
            return np.nanmax(self.image, axis=1)
        elif axis == 'col':
            #return np.nanmax(self.image, axis=(0, 1))
            return np.nanmax(self.image, axis=2)
        elif axis == 'pixel':
            return np.nanmax(self.image, axis=(1, 2))
        elif axis is None:
            return np.nanmax(self.image)
        else:
            raise ValueError(f'Error: axis "{axis}" undefined. Use "band", "row", "col", or None.')

    def mean(self, axis=None):
        """
        Calculate the mean value along a specified axis.

        Parameters
        ----------
        axis : {'band', 'row', 'col', None}, optional
            The axis along which to compute the mean:
            - 'band': Mean across spectral bands for each pixel
            - 'row': Mean across rows (lines) for each band and column
            - 'col': Mean across columns for each band and row
            - 'pixel': Mean across pixels for each bands
            - None: Global mean of the entire image
            Default is None.

        Returns
        -------
        float or numpy.ndarray
            - If axis=None: A single value representing the global mean
            - If axis='band': Array with shape (nb_rows,nb_cols) containing  mean along bands
            - If axis='row': Array with shape (nb_bands,nb_cols) containing mean along rows
            - If axis='col': Array with shape (nb_bands,nb_rows) containing  mean along cols
            - If axis='pixel': Array with shape (nb_bands) containing  mean along all pixels for each band

        Raises
        ------
        ValueError
            If an invalid axis is specified

        Examples
        --------
        >>> mean_value = image.mean()  # Global mean value
        >>> print(f"Mean pixel value: {mean_value}")
        >>>
        >>> band_means = image.mean(axis='pixel')  # Mean along all pixels for each band

        Notes
        -----
        This method uses np.nanmean, which ignores NaN values in the calculation.
        If you have NaN values as nodata, they won't affect the mean calculation.
        """
        if axis == 'band':
            #return np.nanmean(self.image, axis=(1, 2))
            return np.nanmean(self.image, axis=0)
        elif axis == 'row':
            #return np.nanmean(self.image, axis=(0, 2))
            return np.nanmean(self.image, axis=1)
        elif axis == 'col':
            #return np.nanmean(self.image, axis=(0, 1))
            return np.nanmean(self.image, axis=2)
        elif axis == 'pixel':
            return np.nanmean(self.image, axis=(1, 2))
        elif axis is None:
            return np.nanmean(self.image)
        else:
            raise ValueError(f'Error: axis "{axis}" undefined. Use "band", "row", "col", or None.')

    def std(self, axis=None):
        """
        Calculate the standard deviation along a specified axis.

        Parameters
        ----------
        axis : {'band', 'row', 'col', None}, optional
            The axis along which to compute the standard deviation:
            - 'band': Std dev across spectral bands for each pixel
            - 'row': Std dev across rows (lines) for each band and column
            - 'col': Std dev across columns for each band and row
            - 'pixel': Std dev across pixels for each bands
            - None: Global standard deviation of the entire image
            Default is None.

        Returns
        -------
        float or numpy.ndarray
            - If axis=None: A single value representing the global std
            - If axis='band': Array with shape (nb_rows,nb_cols) containing  std along bands
            - If axis='row': Array with shape (nb_bands,nb_cols) containing std along rows
            - If axis='col': Array with shape (nb_bands,nb_rows) containing  std along cols
            - If axis='pixel': Array with shape (nb_bands) containing  std along all pixels for each band

        Raises
        ------
        ValueError
            If an invalid axis is specified

        Examples
        --------
        >>> std_value = image.std()  # Global standard deviation
        >>> print(f"Standard deviation of pixel values: {std_value}")
        >>>
        >>> band_stds = image.std(axis='pixel')  # Standard deviation along all pixels for each band

        """
        if axis == 'band':
            #return np.nanstd(self.image, axis=(1, 2))
            return np.nanstd(self.image, axis=0)
        elif axis == 'row':
            #return np.nanstd(self.image, axis=(0, 2))
            return np.nanstd(self.image, axis=1)
        elif axis == 'col':
            #return np.nanstd(self.image, axis=(0, 1))
            return np.nanstd(self.image, axis=2)
        elif axis == 'pixel':
            return np.nanstd(self.image, axis=(1, 2))
        elif axis is None:
            return np.nanstd(self.image)
        else:
            raise ValueError(f'Error: axis "{axis}" undefined. Use "band", "row", "col", or None.')

    def median(self, axis=None):
        """
        Calculate the median value along a specified axis.

        Parameters
        ----------
        axis : {'band', 'row', 'col', None}, optional
            The axis along which to compute the median:
            - 'band': Median across spectral bands for each pixel
            - 'row': Median across rows (lines) for each band and column
            - 'col': Median across columns for each band and row
            - 'pixel': Median across pixels for each bands
            - None: Global median of the entire image
            Default is None.

        Returns
        -------
        float or numpy.ndarray
            - If axis=None: A single value representing the global median
            - If axis='band': Array with shape (nb_rows,nb_cols) containing  median along bands
            - If axis='row': Array with shape (nb_bands,nb_cols) containing median along rows
            - If axis='col': Array with shape (nb_bands,nb_rows) containing  median along cols
            - If axis='pixel': Array with shape (nb_bands) containing  median along all pixels for each band

        Raises
        ------
        ValueError
            If an invalid axis is specified

        Examples
        --------
        >>> median_value = image.median()  # Global median value
        >>> print(f"Median pixel value: {median_value}")
        >>>
        >>> band_medians = image.median(axis='pixel')  # Median along all pixels for each band

        Notes
        -----
        The median is the value separating the higher half from the lower half of the data.
        It's less sensitive to outliers than the mean, making it useful for images with
        extreme values or noise.
        """
        if axis == 'band':
            #return np.median(self.image, axis=(1, 2))
            return np.median(self.image, axis=0)
        elif axis == 'row':
            #return np.median(self.image, axis=(0, 2))
            return np.median(self.image, axis=1)
        elif axis == 'col':
            #return np.median(self.image, axis=(0, 1))
            return np.median(self.image, axis=2)
        elif axis == 'pixel':
            return np.median(self.image, axis=(1, 2))
        elif axis is None:
            return np.median(self.image)
        else:
            raise ValueError(f'Error: axis "{axis}" undefined. Use "band", "row", "col", or None.')

    def replace_values(self, value_to_replace, new_value):
        """
        Replace all pixels that match a specified value across all bands.

        Parameters
        ----------
        value_to_replace : float, int, or array-like
            The value(s) to search for in each pixel across all bands:
            - If a single value: Looks for pixels where all bands equal this value
            - If an array: Looks for pixels where each band matches the corresponding value
              in the array (must have the same length as the number of bands)

        new_value : float, int, or array-like
            The value(s) to assign to the matching pixels:
            - If a single value: Assigns this value to all bands of matching pixels
            - If an array: Assigns each value to the corresponding band of matching pixels
              (must have the same length as the number of bands)

        Returns
        -------
        self : Geoimage
            The modified image, allowing method chaining

        Examples
        --------
        >>> # Replace all nodata (0) values with NaN
        >>> image.replace_values(0, np.nan)
        >>>
        >>> # Replace a specific RGB color [255, 0, 0] with [0, 0, 0] (black)
        >>> image.replace_values([255, 0, 0], [0, 0, 0])

        Notes
        -----
        This method is useful for replacing nodata values, specific classes in
        a classification, or adjusting specific spectral signatures in an image.
        """
        # Convert single number to array with the same value for each band
        if isinstance(value_to_replace, (int, float)):
            value_to_replace = np.full((self.image.shape[0],), value_to_replace)
        else:
            value_to_replace = np.array(value_to_replace)

        # Ensure new_value is a float array if it contains NaN
        if isinstance(new_value, (int, float)):
            new_value = np.full((self.image.shape[0],), new_value, dtype=float)
        else:
            new_value = np.array(new_value, dtype=float)

        # Create a mask where all bands of a pixel match `value_to_replace`
        mask = np.all((self.image == value_to_replace[:, None, None]) |
                      (np.isnan(self.image) & np.isnan(value_to_replace[:, None, None])), axis=0)

        # Replace matching pixels with `new_value` in all bands
        self.image[:, mask] = new_value[:, None]

        if self.__history is not False:
            now = datetime.datetime.now()
            now_str = now.strftime("%Y-%m-%d %H:%M:%S")
            self.__listhistory.append(f'[{now_str}] - Replaced values {value_to_replace} with {new_value}')

        return self

    def percentage_pixels(self, value=None):
        """
        Calculate the percentage of pixels with a specified value across all bands.

        This method calculates what percentage of the total pixels have the specified
        value in all bands. It's particularly useful for calculating coverage of
        nodata values, specific land cover classes, or other categorical values.

        Parameters
        ----------
        value : int, float, list, or array-like, optional
            The value to check for in each band of a pixel:
            - If a single number: Checks for pixels where all bands equal this value
            - If a vector: Checks for pixels where each band matches the corresponding
              value in the vector (must have same length as number of bands)
            - If None: Uses the image's nodata value (default)

        Returns
        -------
        float
            The percentage of pixels (from 0 to 100) where all bands have the specified value

        Examples
        --------
        >>> # Calculate percentage of nodata pixels
        >>> pct_nodata = image.percentage_pixels()  # Uses image's nodata value
        >>> print(f"Image contains {pct_nodata:.2f}% nodata pixels")
        >>>
        >>> # Calculate percentage of pixels with a specific class value
        >>> pct_water = image.percentage_pixels(1)  # Assuming 1 = water class
        >>> print(f"Water covers {pct_water:.2f}% of the image")
        >>>
        >>> # Calculate percentage of pixels with a specific spectral signature
        >>> pct_rgb_black = image.percentage_pixels([0, 0, 0])  # RGB black pixels
        >>> print(f"Black pixels (RGB=0,0,0) cover {pct_rgb_black:.2f}% of the image")

        Notes
        -----
        This method handles both single values and vectors for the `value` parameter.
        The calculation correctly handles NaN values by considering NaN equal to NaN.
        """
        # If no value specified, use the image's nodata value
        if value is None:
            value = self.nodata

        # Access the full data array (shape: bands x rows x cols)
        data = self.image[:]

        # Convert `value` to a numpy array if it's a single number
        if isinstance(value, (int, float)):
            value = np.full((data.shape[0],), value)
        else:
            value = np.array(value)

        # Create a mask where each pixel is True if all bands match `value`
        matching_pixels = np.all((data == value[:, None, None]) |
                                 (np.isnan(data) & np.isnan(value[:, None, None])), axis=0)

        # Calculate the percentage of pixels that meet this condition
        percentage = (100 * np.sum(matching_pixels)) / (data.shape[1] * data.shape[2])

        return percentage

    def hist(self, **args):
        """
        Display histograms of the image data.

        This method provides a flexible way to visualize the distribution of pixel values
        in one or more bands of the image. It supports various customization options for
        the histogram display.

        Parameters
        ----------
        bands : str, int, list, optional
            The bands to visualize. If not specified, all bands are included.
            This can be band names (e.g., ["NIR", "R", "G"]) or indices (e.g., [4, 3, 2]).

        superpose : bool, optional
            If True, all histograms are plotted on the same figure. If False (default),
            each band gets its own separate histogram figure.

        bins : int, optional
            The number of bins for computing the histogram. Default is 100.

        xmin : float, optional
            The minimum value to plot on the x-axis. Values lower than this won't be displayed.

        xmax : float, optional
            The maximum value to plot on the x-axis. Values higher than this won't be displayed.

        title : str, optional
            The title for the histogram figure.

        histtype : str, optional
            The type of histogram to draw. Default is 'stepfilled'.
            Other options include 'bar', 'step', 'barstacked', etc.

        alpha : float, optional
            The transparency of the histogram bars (0.0 to 1.0). Default is 0.6.

        fig_size : tuple, optional
            The size of the figure in inches as (width, height). Default is DEF_FIG_SIZE.

        label : str or list of str, optional
            The labels for the histogram. If not provided, default labels will be created.

        zoom : tuple, optional
            To plot hist  only on a window of the image
                If based on pixel coordinates, you must indicate
                - the row/col coordinades of
                        the north-west corner (deb_row,deb_col)
                - the row/col coordinades of
                        the south-east corner (end_row,end_col)
                in a tuple  `zoom = ((deb_row,end_row),(deb_col,end_col))`

                If based on latitude/longitude coordinates, you must indicate
                - the lat/lon coordinades of the north-west corner (lat1,lon1)
                - the lat/lon coordinades of the south-east corner (lat2,lon2)
                `zoom = ((lon1,lon2),(lat1,lat2))`
            If not provide, plot hist of the entire image

        pixel : bool, optional
            Coordinate system flag, if zoom is given:
            - If True: Coordinates are interpreted as pixel indices
            - If False: Coordinates are interpreted as geographic coordinates
            Default is True.


        **args : dict, optional
            Additional keyword arguments passed to matplotlib's hist function.

        Returns
        -------
        None
            This method displays the histogram(s) but doesn't return any values.

        Examples
        --------
        >>> # Display histograms for all bands
        >>> image.hist(bins=100)
        >>>
        >>> # Display histogram for a single band with customization
        >>> image.hist(bands="NIR", bins=150, histtype='stepfilled',
        >>>            title="NIR Band Distribution", xmin=0, xmax=10000)
        >>>
        >>> # Superpose histograms from multiple bands
        >>> image.hist(bands=["NIR", "R", "G"], bins=100, superpose=True,
        >>>            alpha=0.7, fig_size=(10, 6))
        >>>
        >>> # Superpose histograms on a zoom from multiple bands
        >>> image.hist(bands=["NIR", "R", "G"], bins=100, superpose=True,
        >>>            alpha=0.7, fig_size=(10, 6), zoom = ((40,150),(100,300)))

        Notes
        -----
        This method is based on rasterio's show_hist function and supports most
        of matplotlib's histogram customization options. It's useful for understanding
        the distribution of values in your image and identifying potential issues like
        saturation, quantization, or outliers.
        """
        args_for_hist = {key: value for key, value in args.items() if key not in ['zoom', 'pixel']}
        if "zoom" in args:
            zoom=args['zoom']
            if "pixel" in args:
                pixel= args['pixel']
            else:
                pixel=True
            im_hist=self.crop(area=zoom,pixel=pixel)
            im_hist.__hist_complet(**args_for_hist)
        else:
            self.__hist_complet(**args_for_hist)


    def __hist_complet(self, **args):
        """
        Display histograms of the image data.

        This method provides a flexible way to visualize the distribution of pixel values
        in one or more bands of the image. It supports various customization options for
        the histogram display.

        Parameters
        ----------
        bands : str, int, list, optional
            The bands to visualize. If not specified, all bands are included.
            This can be band names (e.g., ["NIR", "R", "G"]) or indices (e.g., [4, 3, 2]).

        superpose : bool, optional
            If True, all histograms are plotted on the same figure. If False (default),
            each band gets its own separate histogram figure.

        bins : int, optional
            The number of bins for computing the histogram. Default is 100.

        xmin : float, optional
            The minimum value to plot on the x-axis. Values lower than this won't be displayed.

        xmax : float, optional
            The maximum value to plot on the x-axis. Values higher than this won't be displayed.

        title : str, optional
            The title for the histogram figure.

        histtype : str, optional
            The type of histogram to draw. Default is 'stepfilled'.
            Other options include 'bar', 'step', 'barstacked', etc.

        alpha : float, optional
            The transparency of the histogram bars (0.0 to 1.0). Default is 0.6.

        fig_size : tuple, optional
            The size of the figure in inches as (width, height). Default is DEF_FIG_SIZE.

        label : str or list of str, optional
            The labels for the histogram. If not provided, default labels will be created.

        **args : dict, optional
            Additional keyword arguments passed to matplotlib's hist function.

        Returns
        -------
        None
            This method displays the histogram(s) but doesn't return any values.

        Examples
        --------
        >>> # Display histograms for all bands
        >>> image.hist(bins=100)
        >>>
        >>> # Display histogram for a single band with customization
        >>> image.hist(bands="NIR", bins=150, histtype='stepfilled',
        >>>            title="NIR Band Distribution", xmin=0, xmax=10000)
        >>>
        >>> # Superpose histograms from multiple bands
        >>> image.hist(bands=["NIR", "R", "G"], bins=100, superpose=True,
        >>>            alpha=0.7, fig_size=(10, 6))

        Notes
        -----
        This method is based on rasterio's show_hist function and supports most
        of matplotlib's histogram customization options. It's useful for understanding
        the distribution of values in your image and identifying potential issues like
        saturation, quantization, or outliers.
        """
        # Reset matplotlib settings to avoid interference
        reset_matplotlib()

        # Make a copy of the image data for manipulation
        data = self.image.copy()

        # Handle xmin/xmax parameters to clip data range
        if "xmin" in args:
            xmin = args['xmin']
            data = np.where(data < xmin, xmin, data)
            del args['xmin']

        if "xmax" in args:
            xmax = args['xmax']
            data = np.where(data > xmax, xmax, data)
            del args['xmax']

        # Set default parameters if not provided
        if "bins" not in args:
            args["bins"] = 100

        if "histtype" not in args:
            args["histtype"] = 'stepfilled'

        if "alpha" not in args:
            args["alpha"] = 0.6

        # Extract superpose flag and figure size
        if "superpose" not in args:
            superpose = False
        else:
            superpose = args["superpose"]
            del args["superpose"]

        if "fig_size" not in args:
            fig_size = DEF_FIG_SIZE
        else:
            fig_size = args['fig_size']
            del args['fig_size']

        # Case 1: No specific bands requested (use all bands)
        if "bands" not in args:
            labels = []
            label_keys = list(self.names.keys())

            # Create labels for each band
            if "label" in args:
                if isinstance(args["label"], str):
                    labels = [f"{args['label']} - Band {label_keys[i]}" for i in range(self.__meta["count"])]
                else:
                    labels = args["label"]
            else:
                labels = [f"Band {label_keys[i]}" for i in range(self.__meta["count"])]

            args["label"] = labels

            # Either superpose all histograms or plot them separately
            if superpose is True:
                plt.figure(figsize=fig_size)
                show_hist(data, **args)
            else:
                for i in range(len(labels)):
                    args["label"] = labels[i]
                    plt.figure(figsize=fig_size)
                    show_hist(data[i, :, :], **args)

        # Case 2: Specific bands requested
        else:
            bands = numpy_to_string_list(args["bands"])

            # Validate that requested bands exist
            set1 = set(bands)
            set2 = set(self.names)
            if not(set1 <= set2):
                raise ValueError(f"Error: the requested bands ({bands}) are not all in the available bands ({self.names})")

            del args["bands"]

            # For a single band
            if len(bands) == 1:
                if "label" not in args:
                    args['label'] = ' '
                plt.figure(figsize=fig_size)
                show_hist(data[self.names[bands[0]] - 1, :, :], **args)

            # For multiple bands
            else:
                # Handle labels for multiple bands
                if "label" in args:
                    label = []
                    if isinstance(args["label"], str):
                        for i in range(len(bands)):
                            label.append(f'{args["label"]} - band {bands[i]}')
                    else:
                        for i in range(len(bands)):
                            label.append(args["label"][i])
                    del args["label"]
                else:
                    label = []
                    for i in range(len(bands)):
                        label.append(f'Band {bands[i]}')

                # Either superpose histograms or plot them separately
                if superpose is False:
                    for i in range(len(bands)):
                        args["label"] = label[i]
                        plt.figure(figsize=fig_size)
                        show_hist(data[self.names[bands[i]] - 1, :, :], **args)
                else:
                    args["label"] = []
                    for i in range(len(bands)):
                        args["label"].append(label[i])
                    plt.figure(figsize=fig_size)
                    band_indices = [self.names[band] - 1 for band in bands]
                    show_hist(data[band_indices, :, :], **args)

    def colorcomp(self, bands=None, dest_name='', percentile=2, fig_size=DEF_FIG_SIZE, title='', extent="latlon", zoom=None, pixel=True):
        """
        Create and display a color composite image from selected bands.

        This method creates an RGB color composite by assigning three bands to the red,
        green, and blue channels. It's useful for creating false color compositions,
        natural color images, or any three-band visualization.

        Parameters
        ----------
        bands : list of str, optional
            List of three band identifiers to use for the RGB composite (in order: R, G, B).
            Can be band names (e.g., ["NIR", "R", "G"]) or indices (e.g., ["4", "3", "2"]).
            If None, uses the first three bands in the image.
            Default is None.

        dest_name : str, optional
            Path to save the color composite image. If empty, the image is not saved.
            Default is ''.

        percentile : int, optional
            Percentile value for contrast stretching (e.g., 2 for a 2-98% stretch).
            This enhances the visual contrast of the image.
            Default is 2.

        fig_size : tuple, optional
            Size of the figure in inches as (width, height).
            Default is DEF_FIG_SIZE.

        title : str, optional
            Title for the visualization.
            Default is ''.

        extent : {'latlon', 'pixel', None}, optional
            Type of extent to use for the plot:
            - 'latlon': Use latitude/longitude coordinates (default)
            - 'pixel': Use pixel coordinates
            - None: Don't show coordinate axes

        zoom : tuple, optional
            To plot  only a window of the image
                If based on pixel coordinates, you must indicate
                - the row/col coordinades of
                        the north-west corner (deb_row,deb_col)
                - the row/col coordinades of
                        the south-east corner (end_row,end_col)
                in a tuple  `zoom = ((deb_row,end_row),(deb_col,end_col))`

                If based on latitude/longitude coordinates, you must indicate
                - the lat/lon coordinades of the north-west corner (lat1,lon1)
                - the lat/lon coordinades of the south-east corner (lat2,lon2)
                `zoom = ((lon1,lon2),(lat1,lat2))`
            If not provide, perform on the entire image

        pixel : bool, optional
            Coordinate system flag, if zoom is given:
            - If True: Coordinates are interpreted as pixel indices
            - If False: Coordinates are interpreted as geographic coordinates
            Default is True.

        Returns
        -------
        None
            This method displays and/or saves the color composite but doesn't return any values.

        Raises
        ------
        ValueError
            If the image has only 2 bands, which is not enough for an RGB composite.
            If an invalid extent value is provided.

        Examples
        --------
        >>> # Create a natural color composite (for Landsat/Sentinel-2 style ordering)
        >>> image.colorcomp(bands=["R", "G", "B"])
        >>>
        >>> # Create a color-infrared composite (vegetation appears red)
        >>> image.colorcomp(bands=["NIR", "R", "G"], title="Color-Infrared Composite")
        >>>
        >>> # Zoom and save a false color composite
        >>> image.colorcomp(bands=["SWIR1", "NIR", "G"], dest_name="false_color.tif",zoom=((100,300),(200,400)))
        >>>
        >>> # Change the contrast stretch
        >>> image.colorcomp(bands=["R", "G", "B"], percentile=5)  # More aggressive stretch

        Notes
        -----
        Common band combinations for satellite imagery include:
        - Natural color: R, G, B (shows the scene as human eyes would see it)
        - Color-infrared: NIR, R, G (vegetation appears red, useful for vegetation analysis)
        - Agriculture: SWIR, NIR, B (highlights crop health and soil moisture)
        - Urban: SWIR, NIR, R (emphasizes urban areas and bare soil)
        """
        # Validate extent parameter
        if extent == 'pixel':
            extent_plot = self.__extent_pixels
        elif extent == 'latlon':
            extent_plot = self.__extent_latlon
        elif extent is None:
            extent_plot = None
        else:
            raise ValueError("Invalid extent value. Use 'pixel', 'latlon', or None.")

        # Handle single-band case (grayscale)
        if self.__meta['count'] == 1:
            im = normalize(self.image.reshape((self.__meta['height'], self.__meta['width'])), percentile)

            fig, ax = plt.subplots(figsize=fig_size)
            ax.set_title(title)

            if extent_plot is None:
                ax.imshow(im, interpolation='nearest')
                plt.axis('off')
            else:
                ax.imshow(im, interpolation='nearest', extent=extent_plot)

            plt.show()

            # Save the image if requested
            if dest_name != '':
                folder = os.path.split(dest_name)[0]
                if os.path.exists(folder) is False and folder != '':
                    os.makedirs(folder)

                if os.path.exists(dest_name):
                    os.remove(dest_name)
                if os.path.exists(f'{dest_name}.aux.xml'):
                    os.remove(f'{dest_name}.aux.xml')

                with rio.open(dest_name, 'w', **self.__meta) as outds:
                    outds.write(im.reshape((1, self.__meta['height'], self.__meta['width'])))
                print(f"Image saved in {dest_name}")

        # Handle two-band case (not enough for RGB)
        elif self.__meta['count'] == 2:
            raise ValueError("Error: unable to make a color composition with 2 channels. "
                           "Need at least 3 channels for RGB composition.")

        # Handle multi-band case (create RGB composite)
        else:
            # Use first three bands if none specified
            if bands is None:
                bands = list(self.names.keys())[:3]

            bands = numpy_to_string_list(bands)

            # Validate that requested bands exist
            set1 = set(bands)
            set2 = set(self.names)
            if not(set1 <= set2):
                raise ValueError(f"Error: the requested bands ({bands}) are not all "
                               f"in the available bands ({self.names})")
            if zoom is not None:
                imcrop = self.crop(area=zoom, pixel=pixel)
                # Validate extent parameter
                if extent == 'pixel':
                    extent_plot = imcrop.__extent_pixels
                    extent_plot[0] = imcrop.__extent_pixels[0]+zoom[1][0]
                    extent_plot[1] = imcrop.__extent_pixels[1]+zoom[1][0]
                    extent_plot[2] = imcrop.__extent_pixels[2]+zoom[0][0]
                    extent_plot[3] = imcrop.__extent_pixels[3]+zoom[0][0]

                elif extent == 'latlon':
                    extent_plot = imcrop.__extent_latlon
                elif extent is None:
                    extent_plot = None
                else:
                    raise ValueError("Invalid extent value. Use 'pixel', 'latlon', or None.")
                colorcomp(imcrop.image, bands, name_save=dest_name,
                          names=imcrop.names, percentile=percentile,
                          channel_first=True, meta=imcrop.__meta,
                          fig_size=fig_size, title=title, extent=extent_plot)
            else:
                colorcomp(self.image, bands, name_save=dest_name,
                          names=self.names, percentile=percentile,
                          channel_first=True, meta=self.__meta,
                          fig_size=fig_size, title=title, extent=extent_plot)


    def convert_3bands(self, bands=None, dest_name=None, percentile=2, reformat_names=False):
        """
        Convert an image to a 3-band 8-bit RGB composite.

        This method creates a new Geoimage with exactly 3 bands in 8-bit format (0-255),
        suitable for standard RGB visualization or export to conventional image formats.

        Parameters
        ----------
        bands : list of str, optional
            List of three band identifiers to use for the RGB composite (in order: R, G, B).
            Can be band names (e.g., ["NIR", "R", "G"]) or indices (e.g., ["4", "3", "2"]).
            If None, uses the first three bands in the image.
            Default is None.

        dest_name : str, optional
            Path to save the 3-band image. If None, the image is not saved.
            Default is None.

        percentile : int, optional
            Percentile value for contrast stretching (e.g., 2 for a 2-98% stretch).
            This enhances the visual contrast of the image.
            Default is 2.

        reformat_names : bool, optional
            Whether to reset band names to a simple numeric format ("1", "2", "3").
            If False, keeps the original names of the selected bands.
            Default is False.

        Returns
        -------
        Geoimage
            A new Geoimage with 3 bands (R, G, B) in 8-bit format.

        Examples
        --------
        >>> # Create a natural color composite
        >>> rgb_image = image.convert_3bands(bands=["R", "G", "B"])
        >>> rgb_image.info()  # Should show 3 bands, uint8 data type
        >>>
        >>> # Create a false color composite with custom names
        >>> false_color = image.convert_3bands(
        >>>     bands=["SWIR", "NIR", "R"], dest_name="false_color.tif",
        >>>     reformat_names=True)

        Notes
        -----
        This method is useful for:
        - Creating standardized RGB exports
        - Preparing data for conventional image viewers that expect 3-band 8-bit data
        - Reducing file size by converting to 8-bit
        - Creating visually enhanced compositions with contrast stretching
        """
        # Use first three bands if none specified
        if bands is None:
            bands = list(self.names.keys())[:3]

        bands = numpy_to_string_list(bands)

        # Validate that requested bands exist
        set1 = set(bands)
        set2 = set(self.names)
        if not(set1 <= set2):
            raise ValueError(f"Error: the requested bands ({bands}) are not all "
                           f"in the available bands ({self.names})")

        # Validate band count
        if len(bands) != 3:
            raise ValueError(f"Error: you need to provide exactly 3 spectral bands "
                           f"(provided: {bands})")

        # Extract the requested bands
        im_3bands = self.__get_bands(bands, reformat_names=False)

        # Create the 8-bit RGB image
        im = np.zeros((3, im_3bands.image.shape[1], im_3bands.image.shape[2]), dtype=np.uint8)
        im[0, :, :] = normalize(im_3bands.image[0, :, :], percentile)
        im[1, :, :] = normalize(im_3bands.image[1, :, :], percentile)
        im[2, :, :] = normalize(im_3bands.image[2, :, :], percentile)

        # Set band names
        if reformat_names is False:
            im_3bands.upload_image(im, names=im_3bands.get_names(), inplace=True)
        else:
            im_3bands.upload_image(im, inplace=True, names=im_3bands.get_names())
        im_3bands.change_nodata(0)

        # Save if requested
        if dest_name is not None:
            im_3bands.save(dest_name)

        return im_3bands

    def plot_spectra(self, bands=None, fig_size=(15, 5), percentile=2, title='',
                     title_im="Original image (click outside to stop)",
                     title_spectra="Spectra", xlabel="Bands", ylabel="Value", zoom = None, pixel = None):
        """
        Interactive tool to explore and plot spectral values from user-selected pixels.

        This method displays the image and allows the user to click on pixels to see
        their spectral values across all bands plotted as a line graph. Multiple pixels
        can be selected to compare different spectral signatures.

        Parameters
        ----------
        bands : list of str, optional
            List of three band identifiers to use for the background image display.
            If None, uses the first three bands in the image.
            Default is None.

        fig_size : tuple, optional
            Size of the figure in inches as (width, height).
            Default is (15, 5).

        percentile : int, optional
            Percentile value for contrast stretching of the background image.
            Default is 2.

        title : str, optional
            Main title for the figure.
            Default is ''.

        title_im : str, optional
            Title for the image panel.
            Default is "Original image (click outside to stop)".

        title_spectra : str, optional
            Title for the spectral plot panel.
            Default is "Spectra".

        xlabel : str, optional
            X-axis label for the spectral plot.
            Default is "Bands".

        ylabel : str, optional
            Y-axis label for the spectral plot.
            Default is "Value".

        zoom : tuple, optional
            To visualize  only a window of the image
                If based on pixel coordinates, you must indicate
                - the row/col coordinades of
                        the north-west corner (deb_row,deb_col)
                - the row/col coordinades of
                        the south-east corner (end_row,end_col)
                in a tuple  `zoom = ((deb_row,end_row),(deb_col,end_col))`

                If based on latitude/longitude coordinates, you must indicate
                - the lat/lon coordinades of the north-west corner (lat1,lon1)
                - the lat/lon coordinades of the south-east corner (lat2,lon2)
                `zoom = ((lon1,lon2),(lat1,lat2))`
            If not provide, visualize the entire image

        pixel : bool, optional
            Coordinate system flag, if zoom is given:
            - If True: Coordinates are interpreted as pixel indices
            - If False: Coordinates are interpreted as geographic coordinates
            Default is True.



        Returns
        -------
        tuple
            A tuple containing:
            - series : list of lists - Spectral values for each selected pixel
            - pixel_i : list of int - Row coordinates of selected pixels
            - pixel_j : list of int - Column coordinates of selected pixels

        Examples
        --------
        >>> # Explore spectral signatures in the image
        >>> spectra, rows, cols = image.plot_spectra()
        >>> print(f"Selected {len(spectra)} pixels")
        >>>
        >>> # Customize the display
        >>> spectra, rows, cols = image.plot_spectra(
        >>>     bands=["NIR", "R", "G"],
        >>>     title_im="Click on different vegetation types",
        >>>     title_spectra="Vegetation Spectral Signatures")
        >>>
        >>> # Zoom of a part of the image
        >>> spectra, rows, cols = image.plot_spectra(
        >>>     bands=["NIR", "R", "G"],
        >>>     zoom=((100,200),(100,400)),
        >>>     title_im="Click on different vegetation types",
        >>>     title_spectra="Vegetation Spectral Signatures")

        Notes
        -----
        To end pixel selection, click outside the image area or on the "Finish" button.
        This tool is particularly useful for:
        - Exploring spectral differences between land cover types
        - Identifying spectral anomalies
        - Training classification algorithms
        - Building spectral libraries
        """


        if zoom is None:
            return self.__plot_spectra_entire(bands=bands,
                               fig_size=fig_size,
                               percentile=percentile,
                               title=title,
                               title_im=title_im,
                               title_spectra=title_spectra,
                               xlabel=xlabel,
                               ylabel=ylabel)
        else:
            im=self.crop(area=zoom, pixel=pixel)
            series, pixel_i, pixel_j = im.__plot_spectra_entire(bands=bands,
                                                                fig_size=fig_size,
                                                                percentile=percentile,
                                                                title=title,
                                                                title_im=title_im,
                                                                title_spectra=title_spectra,
                                                                xlabel=xlabel,
                                                                ylabel=ylabel,
                                                                offset_i=zoom[0][0],
                                                                offset_j=zoom[1][0])
            return series, pixel_i, pixel_j


    def __plot_spectra_entire(self, bands=None, fig_size=(15, 5), percentile=2, title='',
                     title_im="Original image (click outside to stop)",
                     title_spectra="Spectra", xlabel="Bands", ylabel="Value", offset_i=0,offset_j=0):
        """
        Interactive tool to explore and plot spectral values from user-selected pixels.

        This method displays the image and allows the user to click on pixels to see
        their spectral values across all bands plotted as a line graph. Multiple pixels
        can be selected to compare different spectral signatures.

        Parameters
        ----------
        bands : list of str, optional
            List of three band identifiers to use for the background image display.
            If None, uses the first three bands in the image.
            Default is None.

        fig_size : tuple, optional
            Size of the figure in inches as (width, height).
            Default is (15, 5).

        percentile : int, optional
            Percentile value for contrast stretching of the background image.
            Default is 2.

        title : str, optional
            Main title for the figure.
            Default is ''.

        title_im : str, optional
            Title for the image panel.
            Default is "Original image (click outside to stop)".

        title_spectra : str, optional
            Title for the spectral plot panel.
            Default is "Spectra".

        xlabel : str, optional
            X-axis label for the spectral plot.
            Default is "Bands".

        ylabel : str, optional
            Y-axis label for the spectral plot.
            Default is "Value".

        offset_i : int, optional
            Offset to add to i coordinates (in case of a zoom)
            Default is 0.

        offset_j : int, optional
            Offset to add to j coordinates (in case of a zoom)
            Default is 0.

        Returns
        -------
        tuple
            A tuple containing:
            - series : list of lists - Spectral values for each selected pixel
            - pixel_i : list of int - Row coordinates of selected pixels
            - pixel_j : list of int - Column coordinates of selected pixels

        Examples
        --------
        >>> # Explore spectral signatures in the image
        >>> spectra, rows, cols = image.plot_spectra()
        >>> print(f"Selected {len(spectra)} pixels")
        >>>
        >>> # Customize the display
        >>> spectra, rows, cols = image.plot_spectra(
        >>>     bands=["NIR", "R", "G"],
        >>>     title_im="Click on different vegetation types",
        >>>     title_spectra="Vegetation Spectral Signatures")

        Notes
        -----
        To end pixel selection, click outside the image area or on the "Finish" button.
        This tool is particularly useful for:
        - Exploring spectral differences between land cover types
        - Identifying spectral anomalies
        - Training classification algorithms
        - Building spectral libraries
        """
        if bands is None:
            bands = list(self.names.keys())[:3]

        bands = numpy_to_string_list(bands)

        # Get spectral values from clicked pixels
        series, pixel_i, pixel_j = Visualizer.plot_spectra(
            self, bands=bands, fig_size=fig_size,
            percentile=percentile, title=title,
            title_im=title_im, title_spectra=title_spectra,
            xlabel=xlabel, ylabel=ylabel, offset_i=offset_i,offset_j=offset_j
        )

        return series, pixel_i, pixel_j

    def __visu_entire(self, bands=None, title='', percentile=0, fig_size=DEF_FIG_SIZE, cmap=None, colorbar=False, extent='latlon', extent_plot=None):
        """
        Visualize one or more bands of the image.

        This method provides a flexible way to display individual bands or multiple bands
        as separate figures. Unlike colorcomp, which creates RGB composites, this method
        displays each band in grayscale or with a specified colormap.

        Parameters
        ----------
        bands : str, list of str, or None, optional
            The bands to visualize:
            - If None: Displays all bands separately
            - If a string: Displays a single specified band
            - If a list: Displays each specified band separately
            Default is None.

        title : str, optional
            Base title for the visualization. Band names will be appended.
            Default is ''.

        percentile : int, optional
            Percentile value for contrast stretching (e.g., 2 for a 2-98% stretch).
            Default is 2.

        fig_size : tuple, optional
            Size of the figure in inches as (width, height).
            Default is DEF_FIG_SIZE.

        cmap : str, optional
            Matplotlib colormap name to use for display.
            Examples: 'viridis', 'plasma', 'gray', 'RdYlGn'
            Default is None (uses matplotlib default).

        colorbar : bool, optional
            Whether to display a colorbar next to each image.
            Default is False.

        extent : {'latlon', 'pixel', None}, optional
            Type of extent to use for the plot:
            - 'latlon': Use latitude/longitude coordinates (default)
            - 'pixel': Use pixel coordinates
            - None: Don't show coordinate axes

        extent_plot : a man made extent (in case of a zoom)

        Examples
        --------
        >>> # Visualize all bands
        >>> image.visu()
        >>>
        >>> # Visualize a single band with a colormap and colorbar
        >>> image.visu("NIR", cmap='plasma', colorbar=True, title="Near Infrared Band")
        >>>
        >>> # Visualize selected bands
        >>> image.visu(["Red", "NIR", "NDVI"], fig_size=(10, 8))

        Notes
        -----
        This method is useful for:
        - Examining individual spectral bands in detail
        - Comparing several derived indices side by side
        - Applying different colormaps to highlight specific features
        - Visualizing single-band thematic data (e.g., elevation, classification results)
        """
        # Validate extent parameter
        if extent == 'pixel':
            if extent_plot is not None:
                extent = extent_plot
            else:
                extent = self.__extent_pixels

        elif extent == 'latlon':
            extent = self.__extent_latlon
        elif extent is None:
            extent = None
        else:
            raise ValueError("Invalid extent value. Use 'pixel', 'latlon', or None.")

        # Reset matplotlib settings
        reset_matplotlib()

        # If no bands specified, use all bands
        if bands is None:
            bands = [name for name in self.names]

        bands = numpy_to_string_list(bands)

        # Validate that requested bands exist
        set1 = set(bands)
        set2 = set(self.names)
        if not(set1 <= set2):
            raise ValueError(f"Error: the requested bands ({bands}) are not all "
                           f"in the available bands ({self.names})")

        # Display a single band
        if len(bands) == 1:
            im = get_percentile(self.image[self.names[bands[0]] - 1, :, :], percentile)

            fig, ax = plt.subplots(figsize=fig_size)
            ax.set_title(title)

            if extent is None:
                visu = ax.imshow(im, interpolation='nearest', cmap=cmap)
                if colorbar:
                    fig.colorbar(visu, ax=ax, shrink=0.8, aspect=10, pad=0.05)
                plt.axis('off')
                plt.show()
            else:
                visu = ax.imshow(im, interpolation='nearest', cmap=cmap, extent=extent)
                if colorbar:
                    fig.colorbar(visu, ax=ax, shrink=0.8, aspect=10, pad=0.05)
                plt.show()

        # Display multiple bands
        else:
            for i in range(len(bands)):
                im = get_percentile(self.image[self.names[bands[i]] - 1, :, :], percentile)

                if extent is None:
                    fig, ax = plt.subplots(figsize=fig_size)
                    ax.set_title(f'{title} band {bands[i]}')
                    visu = ax.imshow(im, interpolation='nearest', cmap=cmap)
                    if colorbar:
                        fig.colorbar(visu, ax=ax, shrink=0.8, aspect=10, pad=0.05)
                    plt.axis('off')
                    plt.show()
                else:
                    fig, ax = plt.subplots(figsize=fig_size)
                    ax.set_title(f'{title} band {bands[i]}')
                    visu = ax.imshow(im, interpolation='nearest', cmap=cmap, extent=extent)
                    if colorbar:
                        fig.colorbar(visu, ax=ax, shrink=0.8, aspect=10, pad=0.05)
                    plt.show()

    def __visu_zoom(self, bands=None, title='', percentile=0, fig_size=DEF_FIG_SIZE, cmap=None, colorbar=False, extent='latlon', zoom = None, pixel = True):
        """
        Visualize a zoom of one or more bands of the image.
        """
        im=self.crop(area=zoom, pixel=pixel)
        im.visu(bands=bands,
                title=title,
                fig_size=fig_size,
                cmap=cmap,
                colorbar=colorcar,
                extent=extent)
    def visu(self, bands=None, title='', percentile=0, fig_size=DEF_FIG_SIZE, cmap=None, colorbar=False, extent='latlon', zoom = None, pixel = True):
        """
        Visualize one or more bands of the image.

        This method provides a flexible way to display individual bands or multiple bands
        as separate figures. Unlike colorcomp, which creates RGB composites, this method
        displays each band in grayscale or with a specified colormap.

        Parameters
        ----------
        bands : str, list of str, or None, optional
            The bands to visualize:
            - If None: Displays all bands separately
            - If a string: Displays a single specified band
            - If a list: Displays each specified band separately
            Default is None.

        title : str, optional
            Base title for the visualization. Band names will be appended.
            Default is ''.

        percentile : int, optional
            Percentile value for contrast stretching (e.g., 2 for a 2-98% stretch).
            Default is 2.

        fig_size : tuple, optional
            Size of the figure in inches as (width, height).
            Default is DEF_FIG_SIZE.

        cmap : str, optional
            Matplotlib colormap name to use for display.
            Examples: 'viridis', 'plasma', 'gray', 'RdYlGn'
            Default is None (uses matplotlib default).

        colorbar : bool, optional
            Whether to display a colorbar next to each image.
            Default is False.

        extent : {'latlon', 'pixel', None}, optional
            Type of extent to use for the plot:
            - 'latlon': Use latitude/longitude coordinates (default)
            - 'pixel': Use pixel coordinates
            - None: Don't show coordinate axes

        zoom : tuple, optional
            To visualize  only a window of the image
                If based on pixel coordinates, you must indicate
                - the row/col coordinades of
                        the north-west corner (deb_row,deb_col)
                - the row/col coordinades of
                        the south-east corner (end_row,end_col)
                in a tuple  `zoom = ((deb_row,end_row),(deb_col,end_col))`

                If based on latitude/longitude coordinates, you must indicate
                - the lat/lon coordinades of the north-west corner (lat1,lon1)
                - the lat/lon coordinades of the south-east corner (lat2,lon2)
                `zoom = ((lon1,lon2),(lat1,lat2))`
            If not provide, visualize the entire image

        pixel : bool, optional
            Coordinate system flag, if zoom is given:
            - If True: Coordinates are interpreted as pixel indices
            - If False: Coordinates are interpreted as geographic coordinates
            Default is True.


        Examples
        --------
        >>> # Visualize all bands
        >>> image.visu()
        >>>
        >>> # Visualize a single band with a colormap and colorbar
        >>> image.visu("NIR", cmap='plasma', colorbar=True, title="Near Infrared Band")
        >>>
        >>> # Visualize selected bands
        >>> image.visu(["Red", "NIR", "NDVI"], fig_size=(10, 8))
        >>>
        >>> # Visualize selected bands on a zoom
        >>> image.visu(["Red", "NIR", "NDVI"], fig_size=(10, 8), zoom = ((100,200),(450,600)))

        Notes
        -----
        This method is useful for:
        - Examining individual spectral bands in detail
        - Comparing several derived indices side by side
        - Applying different colormaps to highlight specific features
        - Visualizing single-band thematic data (e.g., elevation, classification results)
        """
        if zoom is None:
            self.__visu_entire(bands=bands,
                               title=title,
                               percentile=percentile,
                               fig_size=fig_size,
                               cmap=cmap,
                               colorbar=colorbar,
                               extent=extent)
        else:
            im=self.crop(area=zoom, pixel=pixel)
            if extent=='pixel':
                extent_plot = im.__extent_pixels.copy()
                extent_plot[0] = im.__extent_pixels[0]+zoom[1][0]
                extent_plot[1] = im.__extent_pixels[1]+zoom[1][0]
                extent_plot[2] = im.__extent_pixels[2]+zoom[0][0]
                extent_plot[3] = im.__extent_pixels[3]+zoom[0][0]

                im.__visu_entire(bands=bands,
                                 title=title,
                                 fig_size=fig_size,
                                 cmap=cmap,
                                 colorbar=colorbar,
                                 extent=extent, extent_plot=extent_plot)
            else:
                im.__visu_entire(bands=bands,
                                 title=title,
                                 fig_size=fig_size,
                                 cmap=cmap,
                                 colorbar=colorbar,
                                 extent=extent)



    def numpy_channel_first(self, bands=None):
        """
        Extract image data as a NumPy array in channel-first format.

        This method returns a NumPy array representation of the image data with bands
        as the first dimension (bands, rows, cols), which is the format used by rasterio.

        Parameters
        ----------
        bands : str, list of str, or None, optional
            The bands to include in the output:
            - If None: Returns all bands
            - If a string: Returns a single specified band
            - If a list: Returns the specified bands in the given order
            Default is None.

        Returns
        -------
        numpy.ndarray
            Image data as a NumPy array with shape (bands, rows, cols)

        Examples
        --------
        >>> # Get the complete image as a NumPy array
        >>> array = image.numpy_channel_first()
        >>> print(f"Array shape: {array.shape}")
        >>> print(f"Data type: {array.dtype}")
        >>>
        >>> # Extract specific bands
        >>> rgb_array = image.numpy_channel_first(bands=["R", "G", "B"])
        >>> print(f"RGB array shape: {rgb_array.shape}")

        Notes
        -----
        This format (bands, rows, cols) is commonly used with rasterio and some other
        geospatial libraries. For libraries that expect channel-last format (like most
        image processing libraries), use numpy_channel_last() instead.
        """
        if bands is None:
            return self.image.copy()
        else:
            bands = numpy_to_string_list(bands)

            # Validate that requested bands exist
            set1 = set(bands)
            set2 = set(self.names)
            if not(set1 <= set2):
                raise ValueError(f"Error: the requested bands ({bands}) are not all "
                               f"in the available bands ({list(self.names.keys())})")

            # Get the indices of the requested bands
            band_indices = [self.names[band] - 1 for band in bands]

            # Return the selected bands
            return self.image[band_indices, :, :].copy()

    def numpy_channel_last(self, bands=None):
        """
        Extract image data as a NumPy array in channel-last format.

        This method returns a NumPy array representation of the image data with bands
        as the last dimension (rows, cols, bands), which is the format used by most
        image processing libraries and frameworks.

        Parameters
        ----------
        bands : str, list of str, or None, optional
            The bands to include in the output:
            - If None: Returns all bands
            - If a string: Returns a single specified band
            - If a list: Returns the specified bands in the given order
            Default is None.

        Returns
        -------
        numpy.ndarray
            Image data as a NumPy array with shape (rows, cols, bands)

        Examples
        --------
        >>> # Get the complete image as a NumPy array
        >>> array = image.numpy_channel_last()
        >>> print(f"Array shape: {array.shape}")
        >>>
        >>> # Extract RGB bands for use with image processing libraries
        >>> rgb = image.numpy_channel_last(bands=["R", "G", "B"])
        >>> import cv2
        >>> blurred = cv2.GaussianBlur(rgb, (5, 5), 0)

        Notes
        -----
        This format (rows, cols, bands) is commonly used with image processing libraries
        like OpenCV, scikit-image, PIL, and deep learning frameworks. For libraries that
        expect channel-first format (like rasterio), use numpy_channel_first() instead.
        """
        if bands is None:
            return rio2np(self.image.copy())
        else:
            bands = numpy_to_string_list(bands)

            # Validate that requested bands exist
            set1 = set(bands)
            set2 = set(self.names)
            if not(set1 <= set2):
                raise ValueError(f"Error: the requested bands ({bands}) are not all "
                               f"in the available bands ({list(self.names.keys())})")

            # Get the indices of the requested bands
            band_indices = [self.names[band] - 1 for band in bands]

            # Return the selected bands in channel-last format
            return rio2np(self.image[band_indices, :, :].copy())

    def numpy_table(self, bands=None):
        """
        Extract image data as a 2D table of shape (pixels, bands).

        This method reshapes the image into a 2D table where each row represents a pixel
        and each column represents a band. This format is useful for machine learning,
        statistical analysis, or any operation that treats pixels as independent samples.

        Parameters
        ----------
        bands : str, list of str, or None, optional
            The bands to include in the output:
            - If None: Returns all bands
            - If a string: Returns a single specified band
            - If a list: Returns the specified bands in the given order
            Default is None.

        Returns
        -------
        numpy.ndarray
            Image data as a 2D table with shape (rows*cols, bands)

        Examples
        --------
        >>> # Convert the entire image to a table
        >>> table = image.numpy_table()
        >>> print(f"Table shape: {table.shape}")
        >>>
        >>> # Process specific bands as a table
        >>> nir_red = image.numpy_table(bands=["NIR", "R"])
        >>> print(f"Shape: {nir_red.shape}")
        >>> ndvi = (nir_red[:, 0] - nir_red[:, 1]) / (nir_red[:, 0] + nir_red[:, 1])
        >>> print(f"Mean NDVI: {ndvi.mean()}")

        Notes
        -----
        This format is particularly useful for:
        - Machine learning where each pixel is a sample and each band is a feature
        - Clustering algorithms like K-means
        - Statistical analysis across bands
        - Vectorized operations on pixels
        """
        if bands is None:
            return image2table(self.image, channel_first=True)
        else:
            return image2table(self.numpy_channel_first(bands=bands))

    def image_from_table(self, table, names=None, dest_name=None):
        """
        Create a new Geoimage from a 2D table of shape (pixels, bands).

        This method converts a 2D table where each row represents a pixel and each column
        represents a band into a new Geoimage object. It essentially performs the inverse
        operation of numpy_table().

        Parameters
        ----------
        table : numpy.ndarray
            The 2D table to convert, with shape (rows*cols, bands) or (rows*cols,) for a single band.
        names : dict, optional
            Dictionary mapping band names to band indices. If None, bands will be named
            sequentially ("1", "2", "3", ...).
            Default is None.
        dest_name : str, optional
            Path to save the new image. If None, the image is not saved.
            Default is None.

        Returns
        -------
        Geoimage
            A new Geoimage created from the reshaped table

        Raises
        ------
        ValueError
            If the number of rows in the table doesn't match the dimensions of the original image

        Examples
        --------
        >>> # Create a modified image from a processed table
        >>> table = image.numpy_table()
        >>> normalized = (table - table.mean()) / table.std()  # Standardize
        >>> normalized_image = image.image_from_table(normalized)
        >>> normalized_image.visu()
        >>>
        >>> # Save the result
        >>> table = image.numpy_table(bands=["NIR", "R"])
        >>> ndvi = np.zeros((table.shape[0], 1))  # Create single-band output
        >>> ndvi[:, 0] = (table[:, 0] - table[:, 1]) / (table[:, 0] + table[:, 1])
        >>> ndvi_image = image.image_from_table(ndvi, names={"NDVI": 1}, dest_name="ndvi.tif")

        Notes
        -----
        The dimensions of the original image (rows, cols) are preserved, so the table must
        have exactly rows*cols rows. The number of bands can be different from the original image.
        """
        if table.shape[0] != self.__meta['width'] * self.__meta['height']:
            raise ValueError(f"Error: dimensions should match. Table has {table.shape[0]} rows but "
                           f"image size is ({self.__meta['height']}, {self.__meta['width']}) "
                           f"which corresponds to {self.__meta['width'] * self.__meta['height']} pixels")

        # Create metadata for the new image
        meta = self.__meta.copy()
        if len(table.shape) == 1:
            meta['count'] = 1
        else:
            meta['count'] = table.shape[1]
        meta['dtype'] = str(table.dtype)

        # Reshape the table back into an image
        data = table2image(table, self.get_size())

        # Save if requested
        if dest_name is not None:
            write_geoim(data, meta, dest_name, names= names)

        # Create and return the new Geoimage
        return Geoimage(data=data, meta=meta, names=names, georef=self.__georef)

    def upload_table(self, table, names=None, dest_name=None):
        """
        Update the image data with a 2D table of shape (pixels, bands).

        This method replaces the current image data with the content of a 2D table where
        each row represents a pixel and each column represents a band. The table is
        reshaped to match the image dimensions.

        Parameters
        ----------
        table : numpy.ndarray
            The 2D table to upload, with shape (rows*cols, bands) or (rows*cols,) for a single band.
        names : dict, optional
            Dictionary mapping band names to band indices. If None, bands will be named
            sequentially ("1", "2", "3", ...).
            Default is None.
        dest_name : str, optional
            Path to save the updated image. If None, the image is not saved.
            Default is None.

        Returns
        -------
        self : Geoimage
            The updated image, allowing method chaining

        Raises
        ------
        ValueError
            If the number of rows in the table doesn't match the dimensions of the original image

        Examples
        --------
        >>> # Update an image with processed data
        >>> table = image.numpy_table()
        >>> table = np.log(table + 1)  # Log transform
        >>> image.upload_table(table)
        >>> image.visu()
        >>>
        >>> # Upload a single-band result and save
        >>> ndvi_table = (nir - red) / (nir + red)  # Assuming nir and red are numpy arrays
        >>> image.upload_table(ndvi_table, names={"NDVI": 1}, dest_name="ndvi.tif")

        Notes
        -----
        Unlike image_from_table() which creates a new image, this method modifies the
        current image in place. The dimensions of the image (rows, cols) are preserved,
        but the number of bands can change if the table has a different number of columns.
        """
        if table.shape[0] != self.__meta['width'] * self.__meta['height']:
            raise ValueError(f"Error: dimensions should match. Table has {table.shape[0]} rows but "
                           f"image size is ({self.__meta['height']}, {self.__meta['width']}) "
                           f"which corresponds to {self.__meta['width'] * self.__meta['height']} pixels")

        # Update metadata for the new band count
        if len(table.shape) == 1:
            self.__meta['count'] = 1
            self.nb_bands = 1
        else:
            self.__meta['count'] = table.shape[1]
            self.nb_bands = table.shape[1]

        # Update data type
        type_str = str(table.dtype)
        self.__meta['dtype'] = type_str

        # Reshape the table into an image
        self.image = table2image(table, self.get_size())

        # Update band names
        if names is not None:
            if check_dict(names):
                self.names = names
            else:
                self.__update_names()
        else:
            self.__update_names()

        # Update derived attributes
        self.__update()

        # Save if requested
        if dest_name is not None:
            self.save(dest_name)

        if self.__history is not False:
            now = datetime.datetime.now()
            now_str = now.strftime("%Y-%m-%d %H:%M:%S")
            if len(table.shape) == 1:
                band_count = 1
            else:
                band_count = table.shape[1]
            self.__listhistory.append(f'[{now_str}] - Uploaded new image from table with {band_count} bands')
            if dest_name is not None:
                self.__listhistory.append(f'\t Image saved to: {dest_name}')

        return self

    def upload_image(self, image, names=None, dest_name=None, channel_first=True, inplace=False):
        """
        Update the image data with a new image array.

        This method replaces the current image data with a new image array. The new image
        must have compatible dimensions with the current image.

        Parameters
        ----------
        image : numpy.ndarray
            The new image data to upload, with shape:
            - (bands, rows, cols) if channel_first=True
            - (rows, cols, bands) if channel_first=False
            - (rows, cols) for a single band
        names : dict, optional
            Dictionary mapping band names to band indices. If None, bands will be named
 sequentially ("1", "2", "3", ...).
            Default is None.
        dest_name : str, optional
            Path to save the updated image. If None, the image is not saved.
            Default is None.
        channel_first : bool, optional
            Whether the input image has channels in the first dimension (True) or the last
            dimension (False).
            Default is True.
        inplace : bool, default False
            If False, return a copy. Otherwise, upload image in place and return None.

        Returns
        -------
        Geoimage
            The updated image or None if `inplace=True`

        Raises
        ------
        ValueError
            If the spatial dimensions of the new image don't match the original image

        Examples
        --------
        Examples
        --------
        >>> # Create a new filtered image without modifying the original
        >>> array = image.numpy_channel_first()
        >>> filtered = apply_some_filter(array)  # Apply some processing
        >>> filtered_image = image.upload_image(filtered)
        >>> filtered_image.visu()
        >>> image.visu()  # Original remains unchanged
        >>>
        >>> # Create a single-band image from NDVI calculation
        >>> nir = image.numpy_channel_first(bands=["NIR"])
        >>> red = image.numpy_channel_first(bands=["Red"])
        >>> ndvi = (nir - red) / (nir + red)
        >>> ndvi_image = image.upload_image(ndvi, names={"NDVI": 1},
        >>>                                dest_name="ndvi.tif")
        >>> # Update an image with processed data
        >>> array = image.numpy_channel_first()
        >>> filtered = some_filter_function(array)  # Apply some processing
        >>> image.upload_image(filtered, inplace=True)
        >>> image.visu()
        >>>
        >>> # Upload an image in channel-last format
        >>> import cv2
        >>> bgr = cv2.imread('rgb.jpg')  # OpenCV uses BGR order
        >>> rgb = bgr[:, :, ::-1]  # Convert BGR to RGB
        >>> image.upload_image(rgb, channel_first=False, dest_name="from_jpeg.tif", inplace=True))

        Notes
        -----
        The spatial dimensions (rows, cols)
        must match the original image, but the number of bands can change.
        """
        if inplace:
            # Convert to channel-first format if needed
            if (channel_first is False) or (len(image.shape) == 2):
                image = np2rio(image)

            # Validate dimensions
            if ((image.shape[1] != self.__meta['height']) or (image.shape[2] != self.__meta['width'])):
                raise ValueError(f"Error: dimensions should match. Provided image has shape "
                                f"({image.shape[1]}, {image.shape[2]}) but current image has shape "
                                f"({self.__meta['height']}, {self.__meta['width']})")

            # Update metadata
            self.__meta['count'] = image.shape[0]
            type_str = str(image.dtype)
            self.__meta['dtype'] = type_str

            # Replace image data
            self.image = image

            # Update band names
            if names is not None:
                if check_dict(names):
                    self.names = names
                else:
                    self.__update_names()
            else:
                self.__update_names()

            # Update derived attributes
            self.__update()

            # Save if requested
            if dest_name is not None:
                self.save(dest_name)

            if self.__history is not False:
                now = datetime.datetime.now()
                now_str = now.strftime("%Y-%m-%d %H:%M:%S")
                self.__listhistory.append(f'[{now_str}] - Uploaded new image with {image.shape[0]} bands')
                if dest_name is not None:
                    self.__listhistory.append(f'\t Image saved to: {dest_name}')
        else:
            return self.__apply_upload_image(image, names=names, dest_name=dest_name, channel_first=channel_first)


    def __apply_upload_image(self, image, names=None, dest_name=None, channel_first=True):
        """
        Create a new Geoimage with the provided image data.

        Unlike upload_image() which modifies the current image in place, this method
        creates a new Geoimage with the provided data, preserving the original.

        Parameters
        ----------
        image : numpy.ndarray
            The image data to use, with shape:
            - (bands, rows, cols) if channel_first=True
            - (rows, cols, bands) if channel_first=False
            - (rows, cols) for a single band
        names : dict, optional
            Dictionary mapping band names to band indices. If None, bands will be named
            sequentially ("1", "2", "3", ...).
            Default is None.
        dest_name : str, optional
            Path to save the new image. If None, the image is not saved.
            Default is None.
        channel_first : bool, optional
            Whether the input image has channels in the first dimension (True) or the last
            dimension (False).
            Default is True.

        Returns
        -------
        Geoimage
            A new Geoimage with the provided data

        Raises
        ------
        ValueError
            If the spatial dimensions of the new image don't match the original image

        Examples
        --------
        >>> # Create a new filtered image without modifying the original
        >>> array = image.numpy_channel_first()
        >>> filtered = apply_some_filter(array)  # Apply some processing
        >>> filtered_image = image.apply_upload_image(filtered)
        >>> filtered_image.visu()
        >>> image.visu()  # Original remains unchanged
        >>>
        >>> # Create a single-band image from NDVI calculation
        >>> nir = image.numpy_channel_first(bands=["NIR"])
        >>> red = image.numpy_channel_first(bands=["Red"])
        >>> ndvi = (nir - red) / (nir + red)
        >>> ndvi_image = image.apply_upload_image(ndvi, names={"NDVI": 1},
        >>>                                      dest_name="ndvi.tif")

        Notes
        -----
        This method creates a new Geoimage object, preserving metadata like CRS,
        transform, and resolution from the original image. The spatial dimensions
        (rows, cols) must match the original image, but the number of bands can change.
        """
        # Convert to channel-first format if needed
        if (channel_first is False) or (len(image.shape) == 2):
            image = np2rio(image)

        # Validate dimensions
        if ((image.shape[1] != self.__meta['height']) or (image.shape[2] != self.__meta['width'])):
            raise ValueError(f"Error: dimensions should match. Provided image has shape "
                            f"({image.shape[1]}, {image.shape[2]}) but current image has shape "
                            f"({self.__meta['height']}, {self.__meta['width']})")

        # Create metadata for the new image
        meta = self.__meta.copy()
        meta['count'] = image.shape[0]
        meta['dtype'] = str(image.dtype)


        # Create the new Geoimage
        geoim = Geoimage(data=image, meta=meta, names=names, georef=self.__georef, history=self.__history)
        if names is not None:
            if check_dict(names) is False:
                geoim.reset_names()

        # Save if requested
        if dest_name is not None:
            geoim.save(dest_name)

        if geoim.__history is not False:
            now = datetime.datetime.now()
            now_str = now.strftime("%Y-%m-%d %H:%M:%S")
            geoim.__listhistory.append(f'[{now_str}] - Created from existing image with {image.shape[0]} bands')
            if dest_name is not None:
                geoim.__listhistory.append(f'\t Saved to: {dest_name}')

        return geoim

    def astype(self, dtype, inplace=False):
        """
        Convert the image data to a specified data type.

        This method changes the data type of the image pixels (e.g., from float32 to uint8).
        This can be useful for reducing memory usage or preparing data for specific operations.

        Parameters
        ----------
        dtype : str or numpy.dtype
            The target data type (e.g., 'uint8', 'float32', 'int16')

        inplace : bool, default False
            If False, return a copy. Otherwise, do modification in place and return None.

        Returns
        -------
        self : Geoimage
            The modified image with the new data type, allowing method chaining

        Examples
        --------
        >>> # Convert to 8-bit unsigned integer
        >>> image.astype('uint8', inplace = True)
        >>> image.info()  # Should show dtype: uint8
        >>>
        >>> # Convert to 32-bit floating point
        >>> im2 = image.astype('float32')
        >>> im2.info()

        Notes
        -----
        Common data types for geospatial data:
        - uint8: 8-bit unsigned integer (0-255), useful for RGB display
        - int16: 16-bit signed integer (-32768 to 32767), common for satellite data
        - uint16: 16-bit unsigned integer (0-65535), common for satellite data
        - float32: 32-bit floating point, useful for continuous values and calculations
        - float64: 64-bit floating point, highest precision but more memory usage

        Warning: Converting to a smaller data type may result in loss of information
        or precision (for example, converting float32 to uint8).
        """

        if inplace:
            npdtype = np.dtype(dtype)
            self.image = self.image.astype(npdtype)
            self.__meta['dtype'] = dtype
    
            if self.__history is not False:
                now = datetime.datetime.now()
                now_str = now.strftime("%Y-%m-%d %H:%M:%S")
                self.__listhistory.append(f'[{now_str}] - Changed data type to {dtype}')
            return self
        else:
            im = self.copy()
            npdtype = np.dtype(dtype)
            im.upload_image(im.numpy_channel_first().astype(npdtype), inplace=True)
    
            if im.__history is not False:
                now = datetime.datetime.now()
                now_str = now.strftime("%Y-%m-%d %H:%M:%S")
                im.__listhistory.append(f'[{now_str}] - Changed data type to {dtype}')
            return im


    def __apply_resampling(self, final_resolution, dest_name=None, method='cubic_spline'):
        """
        Create a new Geoimage with a different spatial resolution.

        This method creates a resampled copy of the current image with a new spatial
        resolution. Unlike resampling() which modifies the image in place, this method
        preserves the original image.

        Parameters
        ----------
        final_resolution : float
            The desired resolution in meters or degrees
        dest_name : str, optional
            Path to save the resampled image. If None, the image is not saved.
            Default is None.
        method : str, optional
            Resampling method to use. Options include:
            - 'cubic_spline' (default): Cubic spline interpolation, best for continuous data
            - 'nearest': Nearest neighbor interpolation, best for categorical data
            - 'bilinear': Bilinear interpolation, good balance between quality and speed
            - 'cubic': Cubic interpolation, similar to cubic_spline but simpler
            - 'lanczos': Lanczos window filter, high quality for downsampling
            - 'average': Average of all contributing pixels, good for downsampling
            - 'mode': Most frequent value, good for categorical downsampling
            Default is 'cubic_spline'.

        Returns
        -------
        Geoimage
            A new Geoimage with the resampled data

        Examples
        --------
        >>> # Create a lower resolution version (e.g., from 10m to 30m)
        >>> lowres = image.apply_resampling(30)
        >>> lowres.info()
        >>> print(f"Original resolution: {image.resolution}, New resolution: {lowres.resolution}")
        >>>
        >>> # Create a higher resolution version with nearest neighbor interpolation
        >>> hires = image.apply_resampling(5, method='nearest', dest_name="hires.tif")
        >>> hires.visu()

        Notes
        -----
        - Downsampling (to a larger pixel size) reduces detail but can reduce noise
        - Upsampling (to a smaller pixel size) doesn't add new information, but can help
          with visual interpretation or alignment with other datasets
        - The resampling method matters:
          - For continuous data (elevation, reflectance), use 'cubic_spline' or 'bilinear'
          - For categorical data (land cover, classification), use 'nearest' or 'mode'
        """
        data, meta = resampling(self.image, final_resolution,
                              dest_name=dest_name, method=method,
                              channel_first=True, meta=self.__meta, names = self.names)

        # Create a new Geoimage with the resampled data
        if self.__namesgiven is False:
            names = None
        else:
            names = self.names.copy()

        result = Geoimage(data=data, meta=meta, names=names, georef=self.__georef, history=self.__history)

        if result.__history is not False:
            now = datetime.datetime.now()
            now_str = now.strftime("%Y-%m-%d %H:%M:%S")
            result.__listhistory.append(f'[{now_str}] - Created by resampling to {final_resolution} meters/degrees '
                                     f'using {method} method')
            if dest_name is not None:
                result.__listhistory.append(f'\t Saved to: {dest_name}')

        return result

    def __apply_crop(self, deb_row_lon, end_row_lon, deb_col_lat, end_col_lat, dest_name=None, pixel=True):
        """
        Create a new Geoimage containing a cropped subset of this image.

        This method extracts a rectangular region of the image, defined either by pixel
        coordinates or by geographic coordinates, and returns it as a new Geoimage.

        Parameters
        ----------
        deb_row_lon : int or float
            Start coordinate:
            - If pixel=True: Starting row (i) coordinate
            - If pixel=False: Starting longitude coordinate
        end_row_lon : int or float
            End coordinate:
            - If pixel=True: Ending row (i) coordinate
            - If pixel=False: Ending longitude coordinate
        deb_col_lat : int or float
            Start coordinate:
            - If pixel=True: Starting column (j) coordinate
            - If pixel=False: Starting latitude coordinate
        end_col_lat : int or float
            End coordinate:
            - If pixel=True: Ending column (j) coordinate
            - If pixel=False: Ending latitude coordinate
        dest_name : str, optional
            Path to save the cropped image. If None, the image is not saved.
            Default is None.
        pixel : bool, optional
            If True, coordinates are interpreted as pixel indices (i, j).
            If False, coordinates are interpreted as geographic coordinates (lon, lat).
            Default is True.

        Returns
        -------
        Geoimage
            A new Geoimage containing the cropped region

        Examples
        --------
        >>> # Crop using pixel coordinates
        >>> subset = image.apply_crop(100, 500, 200, 700)
        >>> subset.info()
        >>> print(f"Original size: {image.shape}, Cropped size: {subset.shape}")
        >>>
        >>> # Crop using geographic coordinates
        >>> lat1, lon1 = 42.5, -72.5  # Northwest corner
        >>> lat2, lon2 = 42.4, -72.4  # Southeast corner
        >>> region = image.apply_crop(lon1, lon2, lat1, lat2, pixel=False,
        >>>                          dest_name="region.tif")
        >>> region.visu()

        Notes
        -----
        - When using pixel coordinates, the format is (row_start, row_end, col_start, col_end)
        - When using geographic coordinates, the format is (lon_start, lon_end, lat_start, lat_end)
        - The cropped image inherits all metadata (projection, resolution, etc.) from the
          original, but with an updated transform to reflect the new spatial extent
        """
        # Convert geographic coordinates to pixel coordinates if needed
        if pixel is False:
            row_deb, col_deb = self.latlon2pixel(deb_col_lat, deb_row_lon)
            row_end, col_end = self.latlon2pixel(end_col_lat, end_row_lon)
            deb_row_lon_crop = row_deb
            end_row_lon_crop = row_end
            deb_col_lat_crop = col_deb
            end_col_lat_crop = col_end
        else:
            deb_row_lon_crop = deb_row_lon
            end_row_lon_crop = end_row_lon
            deb_col_lat_crop = deb_col_lat
            end_col_lat_crop = end_col_lat

        # Crop the image
        data, meta = crop_rio(self.image, deb_row_lon_crop,
                             end_row_lon_crop, deb_col_lat_crop,
                             end_col_lat_crop, dest_name=dest_name,
                             meta=self.__meta, channel_first=True,
                             names = self.names)

        # Create a new Geoimage with the cropped data
        if self.__namesgiven is False:
            names = None
        else:
            names = self.names.copy()

        result = Geoimage(data=data, meta=meta, names=names, georef=self.__georef, history=self.__history)

        if result.__history is not False:
            now = datetime.datetime.now()
            now_str = now.strftime("%Y-%m-%d %H:%M:%S")
            if pixel is True:
                result.__listhistory.append(f'[{now_str}] - Created by cropping from pixel coordinates '
                                            f'rows {deb_row_lon}-{end_row_lon}, cols {deb_col_lat}-{end_col_lat}')
            else:
                result.__listhistory.append(f'[{now_str}] - Created by cropping from geographic coordinates '
                                            f'lon {deb_row_lon}-{end_row_lon}, lat {deb_col_lat}-{end_col_lat}')
            if dest_name is not None:
                result.__listhistory.append(f'\t Saved to: {dest_name}')

        return result

    def reproject(self, projection="EPSG:3857", inplace=False, dest_name=None):
        """
        Reproject the image to a different coordinate reference system (CRS).

        This method transforms the image to a new projection system, which
        changes how the image's coordinates are interpreted. This can be useful for
        aligning data from different sources or preparing data for specific analyses.

        Parameters
        ----------
        projection : str, optional
            The target projection as an EPSG code or PROJ string.
            Examples:

            - "EPSG:4326": WGS84 geographic (lat/lon)

            - "EPSG:3857": Web Mercator (used by web maps)

            - "EPSG:32619": UTM Zone 19N
            Default is "EPSG:3857" (Web Mercator).

        inplace : bool, default False
            If False, return a copy. Otherwise, do reprojection in place and return None.

        dest_name : str, optional
            Path to save the reprojected image. If None, the image is not saved.
            Default is None.

        Returns
        -------
        Geoimage
            The reprojected image or None if `inplace=True`

        Examples
        --------
        >>> # Reproject to WGS84 (latitude/longitude)
        >>> image_reprojected = image.reproject("EPSG:4326")
        >>> image_reprojected.info()  # Shows new projection
        >>>
        >>> # Reproject to UTM Zone 17N and save
        >>> image_reprojected = image.reproject("EPSG:32617", dest_name="utm.tif")
        >>>
        >>>
        >>> # Reproject to WGS84 (latitude/longitude) and modify inplace the image
        >>> image.reproject("EPSG:4326", inplace=True)
        >>> image.info()  # Shows new projection
        >>>
        >>> # Reproject to UTM Zone 17N and save
        >>> image.reproject("EPSG:32617", dest_name="utm.tif", inplace=True)

        Notes
        -----
        - Reprojection can change the pixel values due to resampling
        - The dimensions of the image will typically change during reprojection
        - Common projection systems include:
        - EPSG:4326 - WGS84 geographic coordinates (latitude/longitude)
        - EPSG:3857 - Web Mercator (used by Google Maps, OpenStreetMap)
        - EPSG:326xx - UTM Zone xx North (projected coordinate system)
        - EPSG:327xx - UTM Zone xx South (projected coordinate system)
        """
        if inplace:
            # Get the source CRS and transform
            src_crs = self.__meta['crs']
            src_transform = self.__meta['transform']

            # Calculate the transform, width, and height for the new projection
            transform, width, height = calculate_default_transform(
                src_crs, projection, self.__meta['width'], self.__meta['height'], *self.get_bounds())

            # Update metadata with new projection info
            self.__meta.update({
                'crs': projection,
                'transform': transform,
                'width': width,
                'height': height
            })

            # Create new array for reprojected data
            reprojected_img = np.empty((self.nb_bands, height, width), dtype=self.__meta['dtype'])

            # Reproject each band
            for i in range(1, self.nb_bands + 1):
                reproject(
                    source=self.image[i-1, :, :],
                    destination=reprojected_img[i-1, :, :],
                    src_transform=src_transform,
                    src_crs=src_crs,
                    dst_transform=transform,
                    dst_crs=projection,
                    resampling=Resampling.nearest
                )

            # Update image data
            self.image = reprojected_img.copy()

            # Update derived attributes
            self.__update()

            if self.__history is not False:
                now = datetime.datetime.now()
                now_str = now.strftime("%Y-%m-%d %H:%M:%S")
                self.__listhistory.append(f'[{now_str}] - Reprojected to {projection}')

            if dest_name is not None:
                self.save(dest_name)
        else:
            return self.__apply_reproject(projection=projection, dest_name=dest_name)
    def __apply_reproject(self, projection="EPSG:3857", dest_name=None):
        """
        Create a new Geoimage with a different coordinate reference system (CRS).

        This method creates a new Geoimage with the specified projection, preserving
        the original image. Unlike reproject() which modifies the image in place,
        this method returns a new image.

        Parameters
        ----------
        projection : str, optional
            The target projection as an EPSG code or PROJ string.
            Examples:
            - "EPSG:4326": WGS84 geographic (lat/lon)
            - "EPSG:3857": Web Mercator (used by web maps)
            - "EPSG:32619": UTM Zone 19N
            Default is "EPSG:3857" (Web Mercator).

        dest_name : str, optional
            Path to save the reprojected image. If None, the image is not saved.
            Default is None.

        Returns
        -------
        Geoimage
            A new Geoimage with the reprojected data

        Examples
        --------
        >>> # Create a WGS84 version without modifying the original
        >>> wgs84 = image.apply_reproject("EPSG:4326")
        >>> wgs84.info()
        >>> image.info()  # Original remains unchanged
        >>>
        >>> # Create a UTM version and save it
        >>> utm = image.apply_reproject("EPSG:32617", dest_name="utm.tif")

        Notes
        -----
        - Reprojection can change the pixel values due to resampling
        - The dimensions of the image will typically change during reprojection
        - This method is useful when you need both the original projection and a reprojected version for different analyses
        """
        # Make a copy of metadata
        meta = self.__meta.copy()

        # Get the source CRS and transform
        src_crs = meta['crs']
        src_transform = meta['transform']

        # Calculate the transform, width, and height for the new projection
        transform, width, height = calculate_default_transform(
            src_crs, projection, meta['width'], meta['height'], *self.get_bounds())

        # Update metadata with new projection info
        meta.update({
            'crs': projection,
            'transform': transform,
            'width': width,
            'height': height
        })

        # Create new array for reprojected data
        reprojected_img = np.empty((self.nb_bands, height, width), dtype=meta['dtype'])

        # Reproject each band
        for i in range(1, self.nb_bands + 1):
            reproject(
                source=self.image[i-1, :, :],
                destination=reprojected_img[i-1, :, :],
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=transform,
                dst_crs=projection,
                resampling=Resampling.nearest
            )

        # Create a new Geoimage with the reprojected data
        result = Geoimage(data=reprojected_img, meta=meta, names=self.names.copy(), georef=self.__georef)

        if result.__history is not False:
            now = datetime.datetime.now()
            now_str = now.strftime("%Y-%m-%d %H:%M:%S")
            result.__listhistory.append(f'[{now_str}] - Created by reprojecting to {projection}')

        if dest_name is not None:
            result.save(dest_name)

        return result

    def latlon2pixel(self, coord_lat, coord_lon):
        """
        Convert geographic coordinates (latitude, longitude) to pixel coordinates.

        This method transforms a point defined by its latitude and longitude to
        the corresponding pixel location (row, col) in the image.

        Parameters
        ----------
        coord_lat : float
            Latitude of the point
        coord_lon : float
            Longitude of the point

        Returns
        -------
        tuple of int
            The pixel coordinates as (row, col) or (i, j)

        Examples
        --------
        >>> # Convert a geographic location to pixel coordinates
        >>> latitude, longitude = 42.36, -71.06  # Boston, MA
        >>> row, col = image.latlon2pixel(latitude, longitude)
        >>> print(f"This location is at pixel ({row}, {col})")
        >>>
        >>> # Check if a specific location is within the image extent
        >>> row, col = image.latlon2pixel(latitude, longitude)
        >>> in_bounds = (0 <= row < image.shape[0]) and (0 <= col < image.shape[1])
        >>> print(f"Location is within image: {in_bounds}")

        Notes
        -----
        - The image must be georeferenced (have valid CRS and transform)
        - If the point is outside the image extent, the function will still return pixel coordinates, but they may be outside the valid image dimensions
        - Row (i) corresponds to the vertical position (along latitude)
        - Column (j) corresponds to the horizontal position (along longitude)
        """
        return latlon_to_pixels(self.__meta, coord_lat, coord_lon)

    def pixel2latlon(self, i, j):
        """
        Convert pixel coordinates to geographic coordinates (latitude, longitude).

        This method transforms a pixel location (row, col) in the image to the
        corresponding point defined by its latitude and longitude.

        Parameters
        ----------
        i : int
            Row index (vertical position) in the image
        j : int
            Column index (horizontal position) in the image

        Returns
        -------
        tuple of float
            The geographic coordinates as (latitude, longitude)

        Examples
        --------
        >>> # Convert pixel coordinates to geographic location
        >>> row, col = 500, 700
        >>> latitude, longitude = image.pixel2latlon(row, col)
        >>> print(f"Pixel ({row}, {col}) is at lat/lon: ({latitude}, {longitude})")
        >>>
        >>> # Find coordinates of image corners
        >>> nw_lat, nw_lon = image.pixel2latlon(0, 0)  # Northwest corner
        >>> se_lat, se_lon = image.pixel2latlon(image.shape[0]-1, image.shape[1]-1)  # Southeast
        >>> print(f"Image covers from ({nw_lat}, {nw_lon}) to ({se_lat}, {se_lon})")

        Notes
        -----
        - The image must be georeferenced (have valid CRS and transform)
        - Pixel coordinates typically start at (0, 0) in the upper-left corner of the image
        - For most projections, latitude increases going north and longitude increases going east
        """
        return pixels_to_latlon(self.__meta, i, j)

    def save(self, dest_name):
        """
        Save the image to a GeoTIFF or JPEG2000 file.

        This method writes the image data and all its metadata (projection, transform,
        etc.) to a georeferenced file that can be read by most geospatial software.

        Parameters
        ----------
        dest_name : str
            Path to save the image. File format is determined by the extension:
            - .tif or .tiff: GeoTIFF format
            - .jp2: JPEG2000 format

        Returns
        -------
        None

        Examples
        --------
        >>> # Save as GeoTIFF
        >>> image.save("output.tif")
        >>>
        >>> # Save as JPEG2000
        >>> image.save("output.jp2")

        Notes
        -----
        - GeoTIFF (.tif) is the most widely supported format
        - JPEG2000 (.jp2) offers compression and is good for large images
        - The saved file will include all metadata (projection, transform, etc.)
        - To save a subset of bands, first use select_bands() to create a new image with only the desired bands, then save that image
        """
        # Save the image to the specified file
        write_geoim(self.image, self.__meta, dest_name, names= self.names)

        if self.__history is not False:
            now = datetime.datetime.now()
            now_str = now.strftime("%Y-%m-%d %H:%M:%S")
            self.__listhistory.append(f'[{now_str}] - Saved to {dest_name}')

    def __apply_extract_from_shapefile(self, name_shp, value, attribute='code', nodata_value=0, keep_size=False):
        """
        Create a new Geoimage containing only data from areas matching a shapefile attribute value.

        This method extracts data from the image only where the shapefile has polygons
        with the specified attribute value. All other areas are set to nodata_value.

        Parameters
        ----------
        name_shp : str
            Path to the shapefile (.shp) to use for extraction
        value : int or float
            The attribute value to extract (e.g., extract only areas with code=3)
        attribute : str, optional
            The name of the attribute field in the shapefile to use.
            Default is 'code'.
        nodata_value : int or float, optional
            Value to assign to areas outside the extracted regions.
            Default is 0.
        keep_size : bool, optional
            If True, output has the same dimensions as input.
            If False, output is cropped to the shapefile extent.
            Default is False.

        Returns
        -------
        Geoimage
            A new Geoimage containing only the extracted regions

        Examples
        --------
        >>> # Extract forest areas (assuming forest has code 3 in the shapefile)
        >>> forest = image.apply_extract_from_shapefile("landcover.shp", 3)
        >>> forest.visu()
        >>>
        >>> # Extract urban areas and keep the original image size
        >>> urban = image.apply_extract_from_shapefile(
        >>>     "landcover.shp", 1, attribute="class",
        >>>     nodata_value=-9999, keep_size=True)
        >>> urban.save("urban_areas.tif")

        Notes
        -----
        - The shapefile must be in the same CRS as the image, or reprojection may be necessary
        - Use shpfiles.get_shapefile_attributes(name_shp) to view available attributes
        - This method is useful for:
        - Extracting specific land cover types from an image
        - Creating masks for different geographic regions
        - Focusing analysis on specific areas of interest
        """
        # Read shapefile with matching resolution
        shp = shpfiles.shp2geoim(name_shp, resolution=self.resolution, attribute=attribute)

        # Ensure images have matching dimensions
        shp, imc = extract_common_areas(shp, self, resolution=None)
        shp, imc = ajust_sizes(shp, imc)

        # Create extraction mask from shapefile
        image_nodata = np.full_like(imc.image, nodata_value)
        mask_condition = (shp.image[0, :, :] == value)
        image_nodata[:, mask_condition] = imc.image[:, mask_condition]

        # Create new image with extracted data
        meta = imc.get_meta().copy()
        meta['nodata'] = nodata_value
        data = Geoimage(data=image_nodata, meta=meta, names=self.names.copy())

        # Optionally extend to original image size
        if keep_size is True:
            data, _ = extend_common_areas(data, self, nodata_value=nodata_value)

        return data

    def extract_from_shapefile(self, name_shp, value, attribute='code', nodata_value=0, inplace=False, keep_size=False):
        """
        Extract data from areas matching a shapefile attribute value.

        This method modifies the image by keeping only data where the shapefile has
        polygons with the specified attribute value. All other areas are set to nodata_value.

        Parameters
        ----------
        name_shp : str
            Path to the shapefile (.shp) to use for extraction
        value : int or float
            The attribute value to extract (e.g., extract only areas with code=3)
        attribute : str, optional
            The name of the attribute field in the shapefile to use.
            Default is 'code'.
        nodata_value : int or float, optional
            Value to assign to areas outside the extracted regions.
            Default is 0.
        inplace : bool, default False
            If False, return a copy. Otherwise, do the extraction in place
        keep_size : bool, optional
            If True, output has the same dimensions as input.
            If False, output is cropped to the shapefile extent.
            Default is False.

        Returns
        -------
        Geoimage
            The image containing only the extracted regions or None if `inplace = True`

        Examples
        --------
        >>> # Extract only forest areas (assuming forest has code 3 in the shapefile)
        >>> image_forest = image.extract_from_shapefile("landcover.shp", 3)
        >>> image.visu()
        >>>
        >>> # Keep only urban areas and preserve the original image size
        >>> image.extract_from_shapefile(
        >>>      "landcover.shp", 1, attribute="class",
        >>>      nodata_value=-9999, keep_size=True, inplace=True)
        >>> image.save("urban_areas.tif")

        Notes
        -----
        - The shapefile must be in the same CRS as the image, or reprojection may be necessary
        - Use shpfiles.get_shapefile_attributes(name_shp) to view available attributes
        """
        if inplace:
            # Read shapefile with matching resolution
            shp = shpfiles.shp2geoim(name_shp, resolution=self.resolution, attribute=attribute)

            # Ensure images have matching dimensions
            shp, imc = extract_common_areas(shp, self, resolution=None)
            shp, imc = ajust_sizes(shp, imc)

            # Create extraction mask from shapefile
            image_nodata = np.full_like(imc.image, nodata_value)
            mask_condition = (shp.image[0, :, :] == value)
            image_nodata[:, mask_condition] = imc.image[:, mask_condition]

            # Update metadata and image data
            meta = imc.get_meta().copy()
            meta['nodata'] = nodata_value
            data = Geoimage(data=image_nodata, meta=meta, names=self.names.copy())

            # Optionally extend to original image size
            if keep_size is True:
                data, _ = extend_common_areas(data, self, nodata_value=nodata_value)

            # Update current image with extracted data
            self.image = data.image
            self.names = data.names
            self.__meta = data.get_meta()
            self.nb_bands = data.nb_bands
            self.resolution = data.resolution
            self.shape = data.shape
            self.__update()

            if self.__history is not False:
                now = datetime.datetime.now()
                now_str = now.strftime("%Y-%m-%d %H:%M:%S")
                self.__listhistory.append(f'[{now_str}] - Extracted regions with {attribute}={value} '
                                        f'from shapefile {name_shp}')
        else:
            return self.__apply_extract_from_shapefile(name_shp, value, attribute=attribute, nodata_value=nodata_value, keep_size=keep_size)

    def __apply_extract_from_shapeimage(self, shp, value, attribute='code', nodata_value=0, keep_size=False):
        """
        Create a new Geoimage containing only data from areas matching a shape image value.

        This method extracts data from the image only where another Geoimage (typically
        created from a shapefile) has the specified value. All other areas are set to nodata_value.

        Parameters
        ----------
        shp : Geoimage
            A Geoimage object, typically created from a shapefile, to use for extraction
        value : int or float
            The pixel value to extract from (e.g., extract only where shp has value 3)
        attribute : str, optional
            Not used for this method, kept for API consistency with apply_extract_from_shapefile.
            Default is 'code'.
        nodata_value : int or float, optional
            Value to assign to areas outside the extracted regions.
            Default is 0.
        keep_size : bool, optional
            If True, output has the same dimensions as input.
            If False, output is cropped to the shape image extent.
            Default is False.

        Returns
        -------
        Geoimage
            A new Geoimage containing only the extracted regions

        Examples
        --------
        >>> # First create a shape image from a shapefile
        >>> landcover = shpfiles.shp2geoim("landcover.shp", attribute="class")
        >>>
        >>> # Extract forest areas (assuming forest has value 3)
        >>> forest = image.apply_extract_from_shapeimage(landcover, 3)
        >>> forest.visu()

        Notes
        -----
        - The shape image must have the same CRS as the target image, or it will be resampled to match
        """
        # Resample shape image if needed
        if shp.resolution != self.resolution:
            shp = shp.resample(self.resolution)

        # Ensure images have matching dimensions
        shp, imc = extract_common_areas(shp, self, resolution=None)
        shp, imc = ajust_sizes(shp, imc)

        # Create extraction mask from shape image
        image_nodata = np.full_like(imc.image, nodata_value)
        mask_condition = (shp.image[0, :, :] == value)
        image_nodata[:, mask_condition] = imc.image[:, mask_condition]

        # Create new image with extracted data
        meta = imc.get_meta().copy()
        meta['nodata'] = nodata_value
        data = Geoimage(data=image_nodata, meta=meta, names=self.names.copy())

        # Optionally extend to original image size
        if keep_size is True:
            data, _ = extend_common_areas(data, self, nodata_value=nodata_value)

        return data

    def extract_from_shapeimage(self, shp, value, attribute='code', inplace=False, nodata_value=0, keep_size=False):
        """
        Extract data from areas matching a shape image value.

        This method modifies the image by keeping only data where another Geoimage
        (typically created from a shapefile) has the specified value. All other areas
        are set to nodata_value.

        Parameters
        ----------
        shp : Geoimage
            A Geoimage object, typically created from a shapefile, to use for extraction
        value : int or float
            The pixel value to extract from (e.g., extract only where shp has value 3)
        attribute : str, optional
            Not used for this method, kept for API consistency with extract_from_shapefile.
            Default is 'code'.
        nodata_value : int or float, optional
            Value to assign to areas outside the extracted regions.
            Default is 0.
        inplace : bool, default False
            If False, return a copy. Otherwise, do the extraction in place
        keep_size : bool, optional
            If True, output has the same dimensions as input.
            If False, output is cropped to the shape image extent.
            Default is False.

        Returns
        -------
        Geoimage
            A new Geoimage containing only the extracted regions or None if `inplace=True`
        Examples
        --------
        >>> # First create a shape image from a shapefile
        >>> landcover = shpfiles.shp2geoim("landcover.shp", attribute="class")
        >>>
        >>> # Keep only forest areas (assuming forest has value 3)
        >>> image_forest = image.extract_from_shapeimage(landcover, 3)
        >>> image_forest.visu()

        Notes
        -----
        - The shape image must have the same CRS as the target image,or it will be resampled to match
        """
        if inplace:
            # Resample shape image if needed
            if shp.resolution != self.resolution:
                shp = shp.resampling(self.resolution)

            # Ensure images have matching dimensions
            shp, imc = extract_common_areas(shp, self, resolution=None)
            shp, imc = ajust_sizes(shp, imc)

            # Create extraction mask from shape image
            image_nodata = np.full_like(imc.image, nodata_value)
            mask_condition = (shp.image[0, :, :] == value)
            image_nodata[:, mask_condition] = imc.image[:, mask_condition]

            # Create image with extracted data
            meta = imc.get_meta().copy()
            meta['nodata'] = nodata_value
            data = Geoimage(data=image_nodata, meta=meta, names=self.names.copy())

            # Optionally extend to original image size
            if keep_size is True:
                data, _ = extend_common_areas(data, self, nodata_value=nodata_value)

            # Update current image with extracted data
            self.image = data.image
            self.names = data.names
            self.__meta = data.get_meta()
            self.nb_bands = data.nb_bands
            self.resolution = data.resolution
            self.shape = data.shape
            self.__update()

            if self.__history is not False:
                now = datetime.datetime.now()
                now_str = now.strftime("%Y-%m-%d %H:%M:%S")
                self.__listhistory.append(f'[{now_str}] - Extracted regions with value={value} '
                                        f'from shape image')
        else:
            return self.__apply_extract_from_shapeimage(shp, value, attribute=attribute, nodata_value=nodata_value, keep_size=keep_size)

    def kmeans(self, n_clusters=4, bands=None, random_state=RANDOM_STATE, dest_name=None, standardization=True, nb_points=1000):
        """
        Perform K-means clustering on the image data.

        This method performs an unsupervised classification using K-means clustering,
        which groups pixels with similar spectral characteristics into a specified
        number of clusters.

        Parameters
        ----------
        n_clusters : int, optional
            Number of clusters (classes) to create.
            Default is 4.
        bands : list of str or None, optional
            List of bands to use for clustering. If None, all bands are used.
            Default is None.
        random_state : int or None, optional
            Random seed for reproducible results. If None, results may vary between runs.
            Default is RANDOM_STATE (defined globally).
        dest_name : str, optional
            Path to save the clustered image. If None, the image is not saved.
            Default is None.
        standardization : bool, optional
            Whether to standardize bands before clustering (recommended).
            Default is True.
        nb_points : int or None, optional
            Number of random points to sample for training the model. If None,
            all valid pixels are used (may be slow for large images).
            Default is 1000.

        Returns
        -------
        Geoimage
            A new Geoimage containing the cluster IDs (0 to n_clusters-1)
        tuple
            A tuple containing (kmeans_model, scaler) for reusing the model on other images

        Examples
        --------
        >>> # Basic K-means clustering with 5 clusters
        >>> classified, model = image.kmeans(n_clusters=5)
        >>> classified.visu(colorbar=True, cmap='viridis')
        >>>
        >>> # Cluster using only specific bands and save result
        >>> classified, model = image.kmeans(
        >>>      n_clusters=3, bands=["NIR", "Red", "Green"],
        >>>      dest_name="clusters.tif")
        >>>
        >>> # Apply same model to another image
        >>> other_classified = other_image.predict(model)

        Notes
        -----
        - Standardization is recommended, especially when bands have different ranges
        - The returned model can be used with predict() on other images
        """
        # Initialize random number generator
        rng = np.random.RandomState(random_state)

        # Extract image data as table
        tab = self.numpy_table(bands=bands)

        # Remove nodata pixels
        mask = ~np.any(tab == self.nodata, axis=1)
        tab = tab[mask]

        # Sample points if requested (for speed)
        if nb_points is not None and tab.shape[0] > nb_points:
            idx = rng.randint(tab.shape[0], size=(nb_points,))
            tab = tab[idx, :]

        # Standardize data if requested
        if standardization is True:
            scaler = StandardScaler().fit(tab)
            tab = scaler.transform(tab)
        else:
            scaler = None

        # Apply K-means clustering
        kmean_model = KMeans(n_clusters=n_clusters, random_state=random_state)
        kmean_model.fit(tab)

        # Apply the model to create classified image
        im_classif = self.apply_ML_model((kmean_model, scaler), bands=bands)

        # Save if requested
        if dest_name is not None:
            im_classif.save(dest_name)

        if im_classif.__history is not False:
            now = datetime.datetime.now()
            now_str = now.strftime("%Y-%m-%d %H:%M:%S")
            im_classif.__listhistory.append(f'[{now_str}] - Created using K-means clustering 'f'with {n_clusters} clusters')
            if bands is not None:
                im_classif.__listhistory.append(f'\t Using bands: {bands}')
            if dest_name is not None:
                im_classif.__listhistory.append(f'\t Saved to: {dest_name}')

        return im_classif, (kmean_model, scaler)






    def apply_ML_model(self, model, bands=None):
        """
        Apply a pre-trained machine learning model to the image.

        NOTE: Will be obsolete in future versions, use `resample`instead

        This method applies a machine learning model (such as one created by kmeans())
        to the image data, creating a new classified or transformed image.

        Parameters
        ----------
        model : scikit model or tuple
            If tuple, it must containi (ml_model, scaler) where:
            - ml_model: A trained scikit-learn model with a predict() method
            - scaler: The scaler used for standardization (or None if not used)
        bands : list of str or None, optional
            List of bands to use as input for the model. If None, all bands are used.
            Default is None.

        Returns
        -------
        Geoimage
            A new Geoimage containing the model output

        Examples
        --------
        >>> # Train a model on one image and apply to another
        >>> classified, model = reference_image.kmeans(n_clusters=5)
        >>> new_classified = target_image.apply_ML_model(model)
        >>> new_classified.visu(colorbar=True, cmap='viridis')
        >>>
        >>> # Train on specific bands and apply to the same bands
        >>> _, model = image.kmeans(bands=["NIR", "Red"], n_clusters=3)
        >>> result = image.apply_ML_model(model, bands=["NIR", "Red"])
        >>> result.save("classified.tif")
        >>>
        >>> # Apply a RF model trained of other data to a Geoimage
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> clf = RandomForestClassifier(max_depth=2, random_state=0)
        >>> clf.fit(X, y)
        >>> result = image.apply_ML_model(clf)


        Notes
        -----
        - The model must have been trained on data with the same structure as what it's being applied to (e.g., same number of bands)
        - If a scaler was used during training, it will be applied before prediction
        - This method is useful for:
        - Applying a classification model to new images
        - Ensuring consistent classification across multiple scenes
        - Time-series analysis with consistent classification
        """
        if isinstance(model,tuple):
            # Extract model and scaler from tuple
            ml_model = model[0]
            scaler = model[1]
        else:
            ml_model = model
            scaler = None

        # Convert image data to table format
        tab_ori = self.numpy_table(bands=bands)

        # Apply scaling if a scaler was provided
        if scaler is not None:
            tab_ori = scaler.transform(tab_ori)

        # Apply the model to get predictions
        outputs = ml_model.predict(tab_ori)

        # Reshape predictions back to image format
        outputs = table2image(outputs, self.shape)

        # Create metadata for output image
        meta = self.__meta.copy()

        # Set band count based on output shape
        if len(outputs.shape) == 2:
            meta['count'] = 1
        else:
            meta['count'] = outputs.shape[0]

        # Update data type
        type_str = str(outputs.dtype)
        meta['dtype'] = type_str

        # Create and return new Geoimage with model outputs
        im_classif = Geoimage(data=outputs, meta=meta, georef=self.__georef, history=self.__history)

        if im_classif.__history is not False:
            now = datetime.datetime.now()
            now_str = now.strftime("%Y-%m-%d %H:%M:%S")
            model_type = type(ml_model).__name__
            im_classif.__listhistory.append(f'[{now_str}] - Created using ML model: {model_type}')
            if bands is not None:
                im_classif.__listhistory.append(f'\t Using bands: {bands}')

        return im_classif


    def predict(self, model, bands=None):
        """
        Apply a pre-trained machine learning model to the image.

        This method applies a machine learning model (such as one created by kmeans())
        to the image data, creating a new classified or transformed image.

        Parameters
        ----------
        model : scikit model or tuple
            If tuple, it must containi (ml_model, scaler) where:
            - ml_model: A trained scikit-learn model with a predict() method
            - scaler: The scaler used for standardization (or None if not used)
        bands : list of str or None, optional
            List of bands to use as input for the model. If None, all bands are used.
            Default is None.

        Returns
        -------
        Geoimage
            A new Geoimage containing the model output

        Examples
        --------
        >>> # Train a model on one image and apply to another
        >>> classified, model = reference_image.kmeans(n_clusters=5)
        >>> new_classified = target_image.predict(model)
        >>> new_classified.visu(colorbar=True, cmap='viridis')
        >>>
        >>> # Train on specific bands and apply to the same bands
        >>> _, model = image.kmeans(bands=["NIR", "Red"], n_clusters=3)
        >>> result = image.predict(model, bands=["NIR", "Red"])
        >>> result.save("classified.tif")
        >>>
        >>> # Apply a RF model trained of other data to a Geoimage
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> clf = RandomForestClassifier(max_depth=2, random_state=0)
        >>> clf.fit(X, y)
        >>> result = image.predict(clf)


        Notes
        -----
        - The model must have been trained on data with the same structure as what it's being applied to (e.g., same number of bands)
        - If a scaler was used during training, it will be applied before prediction
        - This method is useful for:
        - Applying a classification model to new images
        - Ensuring consistent classification across multiple scenes
        - Time-series analysis with consistent classification
        """
        if isinstance(model,tuple):
            # Extract model and scaler from tuple
            ml_model = model[0]
            scaler = model[1]
        else:
            ml_model = model
            scaler = None

        # Convert image data to table format
        tab_ori = self.numpy_table(bands=bands)

        # Apply scaling if a scaler was provided
        if scaler is not None:
            tab_ori = scaler.transform(tab_ori)

        # Apply the model to get predictions
        outputs = ml_model.predict(tab_ori)

        # Reshape predictions back to image format
        outputs = table2image(outputs, self.shape)

        # Create metadata for output image
        meta = self.__meta.copy()

        # Set band count based on output shape
        if len(outputs.shape) == 2:
            meta['count'] = 1
        else:
            meta['count'] = outputs.shape[0]

        # Update data type
        type_str = str(outputs.dtype)
        meta['dtype'] = type_str

        # Create and return new Geoimage with model outputs
        im_classif = Geoimage(data=outputs, meta=meta, georef=self.__georef, history=self.__history)

        if im_classif.__history is not False:
            now = datetime.datetime.now()
            now_str = now.strftime("%Y-%m-%d %H:%M:%S")
            model_type = type(ml_model).__name__
            im_classif.__listhistory.append(f'[{now_str}] - Created using ML model: {model_type}')
            if bands is not None:
                im_classif.__listhistory.append(f'\t Using bands: {bands}')

        return im_classif

    def transform(self, model, bands=None):
        """
        Apply a projection model  (PCA, tSNE, ...) to the image.

        This method applies a projection model (such as one created by pca())
        to the image data, creating a new   image.

        Parameters
        ----------
        model : scikit model or tuple
            If tuple, it must containi (data_model, scaler) where:
            - data_model: A trained scikit-learn model with a transform() method
            - scaler: The scaler used for standardization (or None if not used)
        bands : list of str or None, optional
            List of bands to use as input for the model. If None, all bands are used.
            Default is None.

        Returns
        -------
        Geoimage
            A new Geoimage containing the model output

        Examples
        --------
        >>> # Train a model on one image and apply to another
        >>> pca, model = reference_image.pca(n_components=5)
        >>> new_projection = target_image.transform(model)
        >>> new_projection.visu(colorbar=True, cmap='viridis')
        >>>
        >>> # Train on specific bands and apply to the same bands
        >>> _, model = image.pca(bands=["NIR", "Red"], n_components=3)
        >>> result = image.transform(model, bands=["NIR", "Red"])
        >>> result.save("pca.tif")
        >>>
        >>> # Apply a RF model trained of other data to a Geoimage
        >>> from sklearn.decomposition import PCA
        >>> clf = PCA(n_components=2, random_state=0)
        >>> clf.fit(X, y)
        >>> result = image.transform(clf)


        Notes
        -----
        - The model must have been trained on data with the same structure as what it's being applied to (e.g., same number of bands)
        - If a scaler was used during training, it will be applied before prediction
        """
        if isinstance(model,tuple):
            # Extract model and scaler from tuple
            ml_model = model[0]
            scaler = model[1]
        else:
            ml_model = model
            scaler = None

        # Convert image data to table format
        tab_ori = self.numpy_table(bands=bands)

        # Apply scaling if a scaler was provided
        if scaler is not None:
            tab_ori = scaler.transform(tab_ori)

        # Apply the model to get predictions
        outputs = ml_model.transform(tab_ori)

        # Reshape predictions back to image format
        outputs = table2image(outputs, self.shape)

        # Create metadata for output image
        meta = self.__meta.copy()

        # Set band count based on output shape
        if len(outputs.shape) == 2:
            meta['count'] = 1
        else:
            meta['count'] = outputs.shape[0]

        # Update data type
        type_str = str(outputs.dtype)
        meta['dtype'] = type_str

        # Create and return new Geoimage with model outputs
        im_classif = Geoimage(data=outputs, meta=meta, georef=self.__georef, history=self.__history)

        if im_classif.__history is not False:
            now = datetime.datetime.now()
            now_str = now.strftime("%Y-%m-%d %H:%M:%S")
            model_type = type(ml_model).__name__
            im_classif.__listhistory.append(f'[{now_str}] - Created using transformation model: {model_type}')
            if bands is not None:
                im_classif.__listhistory.append(f'\t Using bands: {bands}')

        return im_classif


    def adapt(self, imt, tab_source = None, nb=1000, mapping='gaussian', reg_e=1e-1, mu=1e0, eta=1e-2, bias=False, max_iter=20, verbose=True, sigma=1, inplace=False):
        """
        Adjust spectral characteristics to match a target image.

        This method adapts the spectral characteristics of the current image to match
        those of a target image using optimal transport methods. This is useful for
        harmonizing images from different sensors or acquisitions.

        Parameters
        ----------
        imt : Geoimage or numpy.ndarray
            Target image serving as a reference for spectral adjustment,
            or a NumPy array of shape (N, bands) containing N spectral samples.
        tab_source : numpy.ndarray, optional
            Required if `imt` is a NumPy array. Must be an array of shape (M, bands)
            containing spectral samples from the source image.
        nb : int, optional
            Number of random samples used to train the transport model.
            Default is 1000.
        mapping : str, optional
            Optimal transport method to use:
            - 'emd': Earth Mover's Distance (simplest)
            - 'sinkhorn': Sinkhorn transport with regularization (balanced)
            - 'mappingtransport': Mapping-based transport (flexible)
            - 'gaussian': Transport with Gaussian assumptions (default, robust)
            Default is 'gaussian'.
        reg_e : float, optional
            Regularization parameter for Sinkhorn transport.
            Default is 1e-1.
        mu : float, optional
            Regularization parameter for mapping-based methods.
            Default is 1e0.
        eta : float, optional
            Learning rate for mapping-based transport methods.
            Default is 1e-2.
        bias : bool, optional
            Whether to add a bias term to the transport model.
            Default is False.
        max_iter : int, optional
            Maximum number of iterations for iterative transport methods.
            Default is 20.
        verbose : bool, optional
            Whether to display progress information.
            Default is True.
        sigma : float, optional
            Standard deviation used for Gaussian transport methods.
            Default is 1.
        inplace : bool, default False
            If False, return a copy. Otherwise, do the adaptation in place and return None.

        Returns
        -------
            The image with adapted spectral characteristics or None if `inplace=True`

        Examples
        --------
        >>> # Basic spectral adaptation
        >>> image_adapt = image1.adapt(image2)
        >>> image_adapt.visu()  # Now spectrally similar to image2
        >>>
        >>> # Use specific transport method
        >>> image_adapt = image1.adapt(image2, mapping='sinkhorn', reg_e=0.01)
        >>> image_adapt.save("adapted_image.tif")
        >>>
        >>> # Adaptation using sample arrays
        >>> adapted_image = image1.adapt(tab_target, tab_source = tab_source, mapping='sinkhorn', reg_e=0.01)
        >>>
        >>> # Basic spectral adaptation and modify inplace the image
        >>> image1.adapt(image2, inplace=True)
        >>> image1.visu()  # Now spectrally similar to image2

        Notes
        -----
        - This method is useful for:
            - Harmonizing multi-sensor data
            - Matching images acquired under different conditions
            - Preparing time-series data for consistent analysis
        - Different mapping methods have different characteristics:
            - 'emd': Most accurate but slowest
            - 'sinkhorn': Good balance between accuracy and speed
            - 'mappingtransport': Flexible and can handle complex transformations
            - 'gaussian': Fastest and works well for most cases
        """
        if inplace:
            self.update_from(InferenceTools.adapt(self, imt, tab_source=tab_source, nb=nb, mapping=mapping, reg_e=reg_e, mu=mu, eta=eta, bias=bias, max_iter=max_iter,
                                    verbose=verbose, sigma=sigma))
            if self.__history is not False:
                now = datetime.datetime.now()
                now_str = now.strftime("%Y-%m-%d %H:%M:%S")
                self.__listhistory.append(f'[{now_str}] - Adapted spectral characteristics to match target image '
                                        f'using {mapping} method')
        else:
            return self.__apply_adapt(imt, tab_source = tab_source, nb=nb, mapping=mapping, reg_e=reg_e, mu=mu, eta=eta, bias=bias, max_iter=max_iter, verbose=verbose, sigma=sigma)

    def __apply_adapt(self, imt, tab_source = None, nb=1000, mapping='gaussian', reg_e=1e-1, mu=1e0, eta=1e-2, bias=False, max_iter=20, verbose=True, sigma=1):
        """
        Create a new image with spectral characteristics matching a target image.

        This method creates a new image by adapting the spectral characteristics of
        the current image to match those of a target image using optimal transport methods.
        Unlike adapt() which modifies the image in place, this preserves the original.

        Parameters
        ----------
        imt : Geoimage or numpy.ndarray
            Target image serving as a reference for spectral adjustment,
            or a NumPy array of shape (N, bands) containing N spectral samples.
        tab_source : numpy.ndarray, optional
            Required if `imt` is a NumPy array. Must be an array of shape (M, bands)
            containing spectral samples from the source image.
        nb : int, optional
            Number of random samples used to train the transport model.
            Default is 1000.
        mapping : str, optional
            Optimal transport method to use:
            - 'emd': Earth Mover's Distance (simplest)
            - 'sinkhorn': Sinkhorn transport with regularization (balanced)
            - 'mappingtransport': Mapping-based transport (flexible)
            - 'gaussian': Transport with Gaussian assumptions (default, robust)
            Default is 'gaussian'.
        reg_e : float, optional
            Regularization parameter for Sinkhorn transport.
            Default is 1e-1.
        mu : float, optional
            Regularization parameter for mapping-based methods.
            Default is 1e0.
        eta : float, optional
            Learning rate for mapping-based transport methods.
            Default is 1e-2.
        bias : bool, optional
            Whether to add a bias term to the transport model.
            Default is False.
        max_iter : int, optional
            Maximum number of iterations for iterative transport methods.
            Default is 20.
        verbose : bool, optional
            Whether to display progress information.
            Default is True.
        sigma : float, optional
            Standard deviation used for Gaussian transport methods.
            Default is 1.

        Returns
        -------
        Geoimage
            A new image with adapted spectral characteristics

        Examples
        --------
        >>> # Create spectrally adapted version of an image
        >>> adapted = image1.__apply_adapt(image2)
        >>> adapted.visu()  # Spectrally similar to image2
        >>> image1.visu()   # Original remains unchanged
        >>>
        >>> # Adaptation using sample arrays
        >>> adapted_image = image1.__apply_adapt(tab_target, tab_source = tab_source, mapping='sinkhorn', reg_e=0.01)
        >>>
        >>> # Save the result with specific transport method
        >>> adapted = image1.apply_adapt(
        >>>      image2, mapping='sinkhorn', reg_e=0.01)
        >>> adapted.save("adapted_image.tif")

        Notes
        -----
        - This method is useful for:
            - Harmonizing multi-sensor data while preserving originals
            - Testing different adaptation parameters
            - Creating multiple adaptations for comparison
        - For details on the different mapping methods, see the adapt() method
        """
        result = InferenceTools.adapt(self, imt, tab_source = tab_source, nb=nb, mapping=mapping, reg_e=reg_e,
                        mu=mu, eta=eta, bias=bias, max_iter=max_iter,
                        verbose=verbose, sigma=sigma)

        if result.__history is not False:
            now = datetime.datetime.now()
            now_str = now.strftime("%Y-%m-%d %H:%M:%S")
            result.__listhistory.append(f'[{now_str}] - Created by adapting spectral characteristics '
                                        f'to match target image using {mapping} method')
        return result



    def fuse_dempster_shafer_2(self, *images):
        """
        Fuse the 3 band image (associated with mass functions) from multiple
        sources using Dempster-Shafer theory with two hypotheses: A and B.

        Parameters
        ----------
        *images : Geoimage
            Each input is a 3-band Geoimage.

            - Band 1: mass function m(A)

            - Band 2: mass function m(B)

            - Band 3: mass function m(A ∪ B)

        Returns
        -------
        Geoimage
            A new Geoimage with 3 bands containing the fused mass functions:
            m(A), m(B), and m(A ∪ B).
        Geoimage
            A new Geoimage with 1 band containing the conflict values.

        Examples
        --------
        >>> fused, conflict = im1.fuse_dempster_shafer_2(im2)
        >>> fused, conflict = im1.fuse_dempster_shafer_2(im1, im2, im3, im4)
        """

        return InferenceTools.fuse_dempster_shafer_2hypotheses(self,*images)


    def __apply_adapt(self, imt, tab_source = None, nb=1000, mapping='gaussian', reg_e=1e-1, mu=1e0, eta=1e-2, bias=False, max_iter=20, verbose=True, sigma=1):
        """
        Create a new image with spectral characteristics matching a target image.

        This method creates a new image by adapting the spectral characteristics of
        the current image to match those of a target image using optimal transport methods.
        Unlike adapt() which modifies the image in place, this preserves the original.

        Parameters
        ----------
        imt : Geoimage or numpy.ndarray
            Target image serving as a reference for spectral adjustment,
            or a NumPy array of shape (N, bands) containing N spectral samples.
        tab_source : numpy.ndarray, optional
            Required if `imt` is a NumPy array. Must be an array of shape (M, bands)
            containing spectral samples from the source image.
        nb : int, optional
            Number of random samples used to train the transport model.
            Default is 1000.
        mapping : str, optional
            Optimal transport method to use:
            - 'emd': Earth Mover's Distance (simplest)
            - 'sinkhorn': Sinkhorn transport with regularization (balanced)
            - 'mappingtransport': Mapping-based transport (flexible)
            - 'gaussian': Transport with Gaussian assumptions (default, robust)
            Default is 'gaussian'.
        reg_e : float, optional
            Regularization parameter for Sinkhorn transport.
            Default is 1e-1.
        mu : float, optional
            Regularization parameter for mapping-based methods.
            Default is 1e0.
        eta : float, optional
            Learning rate for mapping-based transport methods.
            Default is 1e-2.
        bias : bool, optional
            Whether to add a bias term to the transport model.
            Default is False.
        max_iter : int, optional
            Maximum number of iterations for iterative transport methods.
            Default is 20.
        verbose : bool, optional
            Whether to display progress information.
            Default is True.
        sigma : float, optional
            Standard deviation used for Gaussian transport methods.
            Default is 1.

        Returns
        -------
        Geoimage
            A new image with adapted spectral characteristics

        Examples
        --------
        >>> # Create spectrally adapted version of an image
        >>> adapted = image1.__apply_adapt(image2)
        >>> adapted.visu()  # Spectrally similar to image2
        >>> image1.visu()   # Original remains unchanged
        >>>
        >>> # Adaptation using sample arrays
        >>> adapted_image = image1.__apply_adapt(tab_target, tab_source = tab_source, mapping='sinkhorn', reg_e=0.01)
        >>>
        >>> # Save the result with specific transport method
        >>> adapted = image1.apply_adapt(
        >>>      image2, mapping='sinkhorn', reg_e=0.01)
        >>> adapted.save("adapted_image.tif")

        Notes
        -----
        - This method is useful for:
            - Harmonizing multi-sensor data while preserving originals
            - Testing different adaptation parameters
            - Creating multiple adaptations for comparison
        - For details on the different mapping methods, see the adapt() method
        """
        result = InferenceTools.adapt(self, imt, tab_source = tab_source, nb=nb, mapping=mapping, reg_e=reg_e,
                        mu=mu, eta=eta, bias=bias, max_iter=max_iter,
                        verbose=verbose, sigma=sigma)

        if result.__history is not False:
            now = datetime.datetime.now()
            now_str = now.strftime("%Y-%m-%d %H:%M:%S")
            result.__listhistory.append(f'[{now_str}] - Created by adapting spectral characteristics '
                                        f'to match target image using {mapping} method')
        return result



    def __apply_standardize(self, scaler=None, dest_name=None, type='standard', dtype='float64'):
        """
        Create a new image with standardized band values.

        This method performs statistical standardization of image bands, creating a new
        image where values have been transformed to have specific statistical properties,
        such as zero mean and unit variance (for 'standard' type) or values in the 0-1
        range (for 'minmax' type). This function can also apply a scaler (option `scaler`)
        to an image

        Parameters
        ----------
        scaler : object or None, optional
            Scikit-learn scaler object to use. If None, a new scaler is created.
            Default is None.
        dest_name : str, optional
            Path to save the standardized image. If None, image is not saved.
            Default is None.
        type : {'standard', 'minmax'}, optional
            Type of standardization to apply:
            - 'standard': Standardize to zero mean and unit variance (z-scores)
            - 'minmax': Scale values to the range [0, 1]
            Default is 'standard'.
        dtype : str, optional
            Data type for the standardized image. Default is 'float64'.

        Returns
        -------
        Geoimage
            A new image with standardized values
        object
            The scaler object used, which can be used for inverse transformation
            or for standardizing other images consistently

        Examples
        --------
        >>> # Standard standardization (zero mean, unit variance)
        >>> std_image, scaler = image.__apply_standardize()
        >>> print(f"Mean: {std_image.mean()}, Std: {std_image.std()}")
        >>>
        >>> # Min-max scaling to [0, 1] range
        >>> norm_image, scaler = image.__apply_standardize(type='minmax')
        >>> print(f"Min: {norm_image.min()}, Max: {norm_image.max()}")
        >>>
        >>> # Standardize one image and apply same transformation to another
        >>> _, scaler = reference.__apply_standardize()
        >>> target_std = target.apply_standardize(scaler=scaler)

        Notes
        -----
        - Standardization is essential for:
            - Machine learning algorithms that are sensitive to input scales
            - Comparing bands with different value ranges
            - Visualizing multiple bands with consistent contrast
        - 'standard' is better for statistical analyses and most ML algorithms
        - 'minmax' is better for visualization and algorithms requiring specific input ranges (like neural networks with sigmoid activation)
        - The returned scaler can be used for inverse_standardize() to recover original values or to ensure consistent transformation across images
        """
        # Create a new scaler if none is provided
        if scaler is None:
            # Convert image to table for standardization
            tab = self.numpy_table()

            # Create a copy and remove nodata values for scaler fitting
            tab_std = tab.copy()
            mask = ~np.any(tab_std == self.nodata, axis=1)
            tab_std = tab_std[mask]

            # Create and fit the appropriate scaler
            if type == 'standard':
                scaler = StandardScaler().fit(tab_std)
            elif type == 'minmax':
                scaler = MinMaxScaler().fit(tab_std)
            else:
                raise ValueError(f"Unsupported standardization type: {type}. "
                                f"Use 'standard' or 'minmax'.")

            # Apply the scaler to all data (including nodata)
            tab = scaler.transform(tab).astype(dtype)

            # Create metadata for the new image
            meta = self.__meta.copy()
            meta['dtype'] = dtype

            # Set band names
            if self.__namesgiven is False:
                names = None
            else:
                names = self.names.copy()

            # Create new Geoimage from standardized data
            standardized = Geoimage(
                data=table2image(tab, self.shape),
                meta=meta,
                names=names,
                georef=self.__georef
            )

            if standardized.__history is not False:
                now = datetime.datetime.now()
                now_str = now.strftime("%Y-%m-%d %H:%M:%S")
                standardized.__listhistory.append(f'[{now_str}] - Created using {type} standardization')

            # Save if requested
            if dest_name is not None:
                standardized.save(dest_name)

            return standardized, scaler

        else:
            # Use the provided scaler
            tab = self.numpy_table()

            # Apply the scaler
            tab = scaler.transform(tab).astype(dtype)

            # Create metadata for the new image
            meta = self.__meta.copy()
            meta['dtype'] = dtype

            # Set band names
            if self.__namesgiven is False:
                names = None
            else:
                names = self.names.copy()

            # Create new Geoimage from standardized data
            standardized = Geoimage(
                data=table2image(tab, self.shape),
                meta=meta,
                names=names,
                georef=self.__georef
            )

            if standardized.__history is not False:
                now = datetime.datetime.now()
                now_str = now.strftime("%Y-%m-%d %H:%M:%S")
                standardized.__listhistory.append(f'[{now_str}] - Created using provided scaler')

            # Save if requested
            if dest_name is not None:
                standardized.save(dest_name)

            return standardized

    def standardize(self, scaler=None, dest_name=None, type='standard', inplace=False, dtype='float64'):
        """
        Standardize band values.

        This method performs statistical standardization of image bands, modifying
        the current image so values have specific statistical properties, such as
        zero mean and unit variance (for 'standard' type) or values in the 0-1 range
        (for 'minmax' type).

        Parameters
        ----------
        scaler : object or None, optional
            Scikit-learn scaler object to use. If None, a new scaler is created.
            Default is None.
        dest_name : str, optional
            Path to save the standardized image. If None, image is not saved.
            Default is None.
        type : {'standard', 'minmax'}, optional
            Type of standardization to apply:
            - 'standard': Standardize to zero mean and unit variance (z-scores)
            - 'minmax': Scale values to the range [0, 1]
            Default is 'standard'.
        inplace : bool, default False
            If False, return the standardization in a new image. Otherwise, do standardization
            in place and return None.
        dtype : str, optional
            Data type for the standardized image. Default is 'float64'.

        Returns
        -------
        Geoimage
            The image with standardized values and the associated scaler
            None if `inplace=True` (modify the image directly)

        Examples
        --------
        >>> # Standard standardization (zero mean, unit variance)
        >>> im_standardized,scaler  = image.standardize()
        >>> print(f"Mean: {im_standardized.mean()}, Std: {im_standardized.std()}")
        >>>
        >>> # Min-max scaling to [0, 1] range
        >>> im_standardized,scaler  = iimage.standardize(type='minmax')
        >>> print(f"Min: {im_standardized.min()}, Max: {im_standardized.max()}")
        >>>
        >>> # Standardize one image and apply same transformation to another (target)
        >>> _, scaler = image.standardize()
        >>> target_std = target.standardize(scaler=scaler)
        >>>
        >>> # Standard standardization of the image directly
        >>> # With zero mean, unit variance
        >>> image.standardize(inplace=True)
        >>> print(f"Mean: {image.mean()}, Std: {image.std()}")
        >>>
        >>> # With min-max scaling to [0, 1] range
        >>> image.standardize(type='minmax', inplace=True)
        >>> print(f"Min: {image.min()}, Max: {image.max()}")
        >>>
        >>> # Standardize one image and apply same transformation to another (target)
        >>> _, scaler = image.standardize()
        >>> target.standardize(scaler=scaler, inplace=True)

        Notes
        -----
        - When using a pre-fit scaler, make sure it was created with data having similar statistical properties.
        - Standardization is often a prerequisite for machine learning algorithms that are sensitive to data scales.
        """
        if inplace:
            # Create a new scaler if none is provided
            if scaler is None:
                # Convert image to table for standardization
                tab = self.numpy_table()

                # Create a copy and remove nodata values for scaler fitting
                tab_std = tab.copy()
                mask = ~np.any(tab_std == self.nodata, axis=1)
                tab_std = tab_std[mask]

                # Create and fit the appropriate scaler
                if type == 'standard':
                    scaler = StandardScaler().fit(tab_std)
                elif type == 'minmax':
                    scaler = MinMaxScaler().fit(tab_std)
                else:
                    raise ValueError(f"Unsupported standardization type: {type}. "
                                    f"Use 'standard' or 'minmax'.")

                # Apply the scaler to all data (including nodata)
                tab = scaler.transform(tab).astype(dtype)

                # Update metadata and image data
                self.__meta['dtype'] = dtype
                self.image = table2image(tab, self.shape)

                # Save if requested
                if dest_name is not None:
                    self.save(dest_name)

                if self.__history is not False:
                    now = datetime.datetime.now()
                    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
                    self.__listhistory.append(f'[{now_str}] - Applied {type} standardization')
                    if dest_name is not None:
                        self.__listhistory.append(f'\t Saved to: {dest_name}')

            else:
                # Use the provided scaler
                tab = self.numpy_table()

                # Apply the scaler
                tab = scaler.transform(tab).astype(dtype)

                # Update metadata and image data
                self.__meta['dtype'] = dtype
                self.image = table2image(tab, self.shape)

                # Save if requested
                if dest_name is not None:
                    self.save(dest_name)

                if self.__history is not False:
                    now = datetime.datetime.now()
                    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
                    self.__listhistory.append(f'[{now_str}] - Applied standardization using provided scaler')
                    if dest_name is not None:
                        self.__listhistory.append(f'\t Saved to: {dest_name}')

        else:
            return self.__apply_standardize(scaler=scaler, dest_name=dest_name, type=type, dtype=dtype)

    def __apply_inverse_standardize(self, scaler, dest_name=None, dtype='float64'):
        """
        Create a new image by reverting standardization.

        This method creates a new image by applying the inverse of a standardization
        transformation, converting standardized values back to their original scale.

        Parameters
        ----------
        scaler : object
            Scikit-learn scaler object that was used for the original standardization.
            This must have an inverse_transform() method (like StandardScaler or MinMaxScaler).
        dest_name : str, optional
            Path to save the restored image. If None, image is not saved.
            Default is None.
        dtype : str, optional
            Data type for the output image. Default is 'float64'.

        Returns
        -------
        Geoimage
            A new image with values transformed back to the original scale

        Examples
        --------
        >>> # Create standardized version, then restore
        >>> std_image, scaler = image.standardize()
        >>> restored = std_image.__apply_inverse_standardize(scaler)
        >>> diff = np.abs(restored.image - image.image).mean()
        >>> print(f"Mean absolute difference: {diff}")  # Should be very small
        >>>
        >>> # Apply custom processing on standardized data, then restore scale
        >>> std_img, scaler = image.standardize(type='minmax')
        >>> processed = std_img * 0.8  # Apply some processing
        >>> restored = processed.__apply_inverse_standardize(scaler)
        >>> restored.save("processed_original_scale.tif")

        Notes
        -----
        - This method is useful for:
            - Recovering original data values after analysis on standardized data
            - Applying processing in standardized space, then converting back
            - Ensuring output values are in a meaningful physical scale
        - The scaler must be the exact one used for the original standardization to ensure accurate inverse transformation
        """
        # Convert image to table for inverse standardization
        tab = self.numpy_table()

        # Apply inverse transformation
        tab = scaler.inverse_transform(tab).astype(dtype)

        # Create metadata for the new image
        meta = self.__meta.copy()
        meta['dtype'] = dtype

        # Set band names
        if self.__namesgiven is False:
            names = None
        else:
            names = self.names.copy()

        # Create new Geoimage from inverse-standardized data
        inverted = Geoimage(
            data=table2image(tab, self.shape),
            meta=meta,
            names=names,
            georef=self.__georef
        )

        if inverted.__history is not False:
            now = datetime.datetime.now()
            now_str = now.strftime("%Y-%m-%d %H:%M:%S")
            inverted.__listhistory.append(f'[{now_str}] - Created by applying inverse standardization')

        # Save if requested
        if dest_name is not None:
            inverted.save(dest_name)

        return inverted

    def inverse_standardize(self, scaler, dest_name=None, inplace=False, dtype='float64'):
        """
        Revert standardization.

        This method creates an image by applying the inverse of a standardization
        transformation, converting standardized values back to their original scale.

        Parameters
        ----------
        scaler : object
            Scikit-learn scaler object that was used for the original standardization.
            This must have an inverse_transform() method (like StandardScaler or MinMaxScaler).
        dest_name : str, optional
            Path to save the restored image. If None, image is not saved.
            Default is None.
        inplace : bool, default False
            If False, return a copy of the inverse standardization.
            Otherwise, do operation in place and return None.
        dtype : str, optional
            Data type for the output image. Default is 'float64'.

        Returns
        -------
        Geoimage
            The image with values transformed back to the original scale or None if `inplace=True`

        Examples
        --------
        >>> # Standardize and then restore original values
        >>> image_copy = image.copy()
        >>> image_copy_std, scaler = image_copy.standardize()
        >>> image_copy_back = image_copy_std.inverse_standardize(scaler)
        >>> image_copy_back.visu()  # Should look like the original
        >>>
        >>> # With inplace = True
        >>> image_copy_std, scaler = image_copy.standardize()
        >>> image_copy_std.inverse_standardize(scaler, inplace=True)
        >>> image_copy_std.visu()  # Should look like the original

        Notes
        -----
        - The scaler must be the exact one used for the original standardization
          to ensure accurate inverse transformation
        - This is often used as the final step in a processing pipeline to convert
          results back to physically meaningful units
        """
        if inplace:
            # Convert image to table for inverse standardization
            tab = self.numpy_table()

            # Apply inverse transformation
            tab = scaler.inverse_transform(tab).astype(dtype)

            # Update metadata and image data
            self.__meta['dtype'] = dtype
            self.image = table2image(tab, self.shape)

            # Save if requested
            if dest_name is not None:
                self.save(dest_name)

            if self.__history is not False:
                now = datetime.datetime.now()
                now_str = now.strftime("%Y-%m-%d %H:%M:%S")
                self.__listhistory.append(f'[{now_str}] - Applied inverse standardization')
                if dest_name is not None:
                    self.__listhistory.append(f'\t Saved to: {dest_name}')

            return self
        else:
            return self.__apply_inverse_standardize(scaler, dest_name=dest_name, dtype=dtype)

    def resampling(self, final_resolution, dest_name=None, inplace=False, method='cubic_spline', update_history=True):
        """
        Resample the image to a different resolution.

        NOTE: Will be obsolete in future versions, use `resample`instead

        This method changes the spatial resolution of the image by resampling the pixel values.
        The resampling process creates a new grid of pixels at the target resolution and
        interpolates values from the original grid.

        Parameters
        ----------
        final_resolution : float
            The target resolution in the image's coordinate system units (typically meters or degrees).
            A smaller value results in a higher-resolution (larger) image.

        dest_name : str, optional
            Path to save the resampled image. If None, the image is not saved.
            Default is None.

        inplace : bool, default False
            If False, return a copy. Otherwise, do the resampling in place and return None.


        method : str, optional
            Resampling algorithm to use. Options include:

            - 'cubic_spline' (default): High-quality interpolation, good for continuous data

            - 'nearest': Nearest neighbor interpolation, preserves original values, best for categorical data

            - 'bilinear': Linear interpolation between points, faster than cubic

            - 'cubic': Standard cubic interpolation

            - 'lanczos': High-quality downsampling

            - 'average': Takes the average of all contributing pixels, useful for downsampling

        update_history : bool, optional
            Whether to update the image processing history. Default is True.

        Returns
        -------
        Geoimage
            A copy of the resampled image or None if `inplace=True`

        Examples
        --------
        >>> # Resample to 30 meter resolution
        >>> image_resampled = image.resampling(30)
        >>> print(f"New resolution: {image.resolution}")
        >>>
        >>> # Resample using nearest neighbor (best for categorical data)
        >>> classified_image_resampled = classified_image.resampling(10, method='nearest')
        >>>
        >>> # Resample and save the result
        >>> image_resampled = image.resampling(20, dest_name='resampled_20m.tif')
        >>>
        >>>
        >>> # Resample directly the image to 30 meter resolution
        >>> image.resampling(30, inplace=True)
        >>> print(f"New resolution: {image.resolution}")
        >>>
        >>> # Resample directly the image using nearest neighbor (best for categorical data)
        >>> classified_image.resampling(10, method='nearest', inplace=True)
        >>>
        >>> # Resample and save the result
        >>> image.resampling(20, dest_name='resampled_20m.tif', inplace=True)

        Notes
        -----
        - When upsampling (to higher resolution), no new information is created;
        the function only interpolates between existing pixels
        - When downsampling (to lower resolution), information is lost
        - The choice of resampling method is important:
        - For continuous data (e.g., elevation, reflectance): 'cubic_spline', 'bilinear', or 'cubic'
        - For categorical data (e.g., land classifications): 'nearest' or 'mode'
        - This method changes the dimensions (shape) of the image
        """
        if inplace:
            try:
                original_resolution = self.resolution
                self.image, self.__meta = resampling(self.image, final_resolution,
                                                    dest_name=dest_name,
                                                    method=method, channel_first=True,
                                                    meta=self.__meta, names = self.names)
                self.__update()

                if self.__history is not False and update_history:
                    now = datetime.datetime.now()
                    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
                    self.__listhistory.append(f'[{now_str}] - Resampled from {original_resolution:.2f} to {final_resolution:.2f} using {method} method')
                    if dest_name is not None:
                        self.__listhistory.append(f'\t Resampled image saved in: {dest_name}')

            except Exception as e:
                raise RuntimeError(f"Resampling failed: {str(e)}") from e
        else:
            return(self.__apply_resampling(final_resolution, dest_name=dest_name, method=method))

    def resample(self, final_resolution, dest_name=None, inplace=False, method='cubic_spline', update_history=True):
        """
        Resample the image to a different resolution.

        This method changes the spatial resolution of the image by resampling the pixel values.
        The resampling process creates a new grid of pixels at the target resolution and
        interpolates values from the original grid.

        Parameters
        ----------
        final_resolution : float
            The target resolution in the image's coordinate system units (typically meters or degrees).
            A smaller value results in a higher-resolution (larger) image.

        dest_name : str, optional
            Path to save the resampled image. If None, the image is not saved.
            Default is None.

        inplace : bool, default False
            If False, return a copy. Otherwise, do the resampling in place and return None.


        method : str, optional
            Resampling algorithm to use. Options include:

            - 'cubic_spline' (default): High-quality interpolation, good for continuous data

            - 'nearest': Nearest neighbor interpolation, preserves original values, best for categorical data

            - 'bilinear': Linear interpolation between points, faster than cubic

            - 'cubic': Standard cubic interpolation

            - 'lanczos': High-quality downsampling

            - 'average': Takes the average of all contributing pixels, useful for downsampling

        update_history : bool, optional
            Whether to update the image processing history. Default is True.

        Returns
        -------
        Geoimage
            A copy of the resampled image or None if `inplace=True`

        Examples
        --------
        >>> # Resample to 30 meter resolution
        >>> image_resampled = image.resample(30)
        >>> print(f"New resolution: {image.resolution}")
        >>>
        >>> # Resample using nearest neighbor (best for categorical data)
        >>> classified_image_resampled = classified_image.resample(10, method='nearest')
        >>>
        >>> # Resample and save the result
        >>> image_resampled = image.resample(20, dest_name='resampled_20m.tif')
        >>>
        >>>
        >>> # Resample directly the image to 30 meter resolution
        >>> image.resample(30, inplace=True)
        >>> print(f"New resolution: {image.resolution}")
        >>>
        >>> # Resample directly the image using nearest neighbor (best for categorical data)
        >>> classified_image.resample(10, method='nearest', inplace=True)
        >>>
        >>> # Resample and save the result
        >>> image.resample(20, dest_name='resampled_20m.tif', inplace=True)

        Notes
        -----
        - Same function as `resampling` but rather prefer this one
        - When upsampling (to higher resolution), no new information is created;
        the function only interpolates between existing pixels
        - When downsampling (to lower resolution), information is lost
        - The choice of resampling method is important:
        - For continuous data (e.g., elevation, reflectance): 'cubic_spline', 'bilinear', or 'cubic'
        - For categorical data (e.g., land classifications): 'nearest' or 'mode'
        - This method changes the dimensions (shape) of the image
        """
        if inplace:
            try:
                original_resolution = self.resolution
                self.image, self.__meta = resampling(self.image, final_resolution,
                                                    dest_name=dest_name,
                                                    method=method, channel_first=True,
                                                    meta=self.__meta, names = self.names)
                self.__update()

                if self.__history is not False and update_history:
                    now = datetime.datetime.now()
                    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
                    self.__listhistory.append(f'[{now_str}] - Resampled from {original_resolution:.2f} to {final_resolution:.2f} using {method} method')
                    if dest_name is not None:
                        self.__listhistory.append(f'\t Resampled image saved in: {dest_name}')

            except Exception as e:
                raise RuntimeError(f"Resampling failed: {str(e)}") from e
        else:
            return(self.__apply_resampling(final_resolution, dest_name=dest_name, method=method))


    def crop(self, *args,
             area = None, dest_name=None, pixel=True, inplace=False):
        """
        Crop the image to a specified extent.

        This method extracts a rectangular subset of the image, defined either by pixel
        coordinates or by geographic coordinates, and updates the current image to contain
        only the cropped region.

        Parameters
        ----------
        area : tuple
            Area to crop
                If based on pixel coordinates, you must indicate
                - the row/col coordinades of
                        the north-west corner (deb_row,deb_col)
                - the row/col coordinades of
                        the south-east corner (end_row,end_col)
                in a tuple  `area = ((deb_row,end_row),(deb_col,end_col))`

                If based on latitude/longitude coordinates, you must indicate
                - the lat/lon coordinades of the north-west corner (lat1,lon1)
                - the lat/lon coordinades of the south-east corner (lat2,lon2)
                `area = ((lon1,lon2),(lat1,lat2))`

        inplace : bool, default False
            If False, return a copy. Otherwise, do cropping in place and return None.

        pixel : bool, optional
            Coordinate system flag:
            - If True: Coordinates are interpreted as pixel indices (row, col)
            - If False: Coordinates are interpreted as geographic coordinates (lon, lat)
            Default is True.

        Returns
        -------
        Geoimage
            A copy of the cropped image or None if `inplace=True`

        Examples
        --------
        >>> # Crop using pixel coordinates
        >>> original_shape = image.shape
        >>> image_crop = image.crop(area=((100, 500), (200, 600)))
        >>> print(f"Original shape: {original_shape}, New shape: {image_crop.shape}")
        >>>
        >>> # Crop using geographic coordinates
        >>> image_crop = image.crop(area=((-122.5, -122.3), (37.8, 37.7)), pixel=False)
        >>> image.visu()
        >>>
        >>> # Crop and save the result
        >>> image_crop = image.crop(area=((100, 500), (200, 600)), dest_name='cropped_area.tif')
        >>>
        >>>
        >>> # Crop using pixel coordinates
        >>> original_shape = image.shape
        >>> image.crop(area=((100, 500), (200, 600)), inplace=True) # inplace = True : modify directly the image
        >>> print(f"Original shape: {original_shape}, New shape: {image.shape}")
        >>>
        >>> # Crop using geographic coordinates
        >>> image.crop(area=((-122.5, -122.3), (37.8, 37.7)), pixel=False, inplace=True)
        >>> image.visu()
        >>>
        >>> # Crop and save the result
        >>> image.crop(area=((100, 500), (200, 600)), dest_name='cropped_area.tif', inplace=True)

        Notes
        -----
        - For consistency with older versions, a use with 4 parameters (deb_row_lon, end_row_lon, deb_col_lat, end_col_lat)
          instead of the `area` tuple is possible
        deb_row_lon : int or float
            Starting position (north):
            - If pixel=True: Starting row (y) coordinate
            - If pixel=False: Starting longitude coordinate

        end_row_lon : int or float
            Ending position (south):
            - If pixel=True: Ending row (y) coordinate
            - If pixel=False: Ending longitude coordinate

        deb_col_lat : int or float
            Starting position (west):
            - If pixel=True: Starting column (x) coordinate
            - If pixel=False: Starting latitude coordinate

        end_col_lat : int or float
            Ending position (east):
            - If pixel=True: Ending column (x) coordinate
            - If pixel=False: Ending latitude coordinate

        dest_name : str, optional
            Path to save the cropped image. If None, the image is not saved.
            Default is None.

        - The cropping operation changes the spatial extent of the image but preserves
        the resolution and projection.
        - When using pixel coordinates, the format is (row_start, row_end, col_start, col_end).
        - When using geographic coordinates, the format is (lon_start, lon_end, lat_start, lat_end).
        """
        if args:
            if area is not None:
                raise TypeError("Cannot specify both positional arguments and the 'area' keyword.")

            if len(args) == 4:
                # Avertir l'utilisateur que cette méthode est obsolète
                warnings.warn(
                    "Calling crop() with 4 positional arguments is deprecated. "
                    "Use the 'area=((start_row, end_row), (start_col, end_col))' keyword instead.",
                    DeprecationWarning,
                    stacklevel=2
                )
                deb_row_lon, end_row_lon, deb_col_lat, end_col_lat = args
                # On reconstruit le paramètre 'area' à partir des anciens arguments
                area = ((deb_row_lon, end_row_lon), (deb_col_lat, end_col_lat))
            else:
                raise TypeError(f"crop() takes 4 positional arguments for legacy calls, but {len(args)} were given.")




        if area is not None:
            deb_row_lon = area[0][0]
            end_row_lon = area[0][1]
            deb_col_lat = area[1][0]
            end_col_lat = area[1][1]

        if inplace:
            original_shape = self.shape

            # Store original coordinates for history
            orig_deb_row_lon = deb_row_lon
            orig_end_row_lon = end_row_lon
            orig_deb_col_lat = deb_col_lat
            orig_end_col_lat = end_col_lat

            # Convert geographic coordinates to pixel coordinates if needed
            if pixel is False:
                try:
                    row_deb, col_deb = self.latlon2pixel(deb_col_lat, deb_row_lon)
                    row_end, col_end = self.latlon2pixel(end_col_lat, end_row_lon)
                    deb_row_lon_crop = row_deb
                    end_row_lon_crop = row_end
                    deb_col_lat_crop = col_deb
                    end_col_lat_crop = col_end
                except Exception as e:
                    raise ValueError(f"Failed to convert geographic coordinates to pixel coordinates: {str(e)}")
            else:
                deb_row_lon_crop = deb_row_lon
                end_row_lon_crop = end_row_lon
                deb_col_lat_crop = deb_col_lat
                end_col_lat_crop = end_col_lat

            # Ensure coordinates are within image bounds
            image_height, image_width = self.shape
            if pixel and (deb_row_lon_crop < 0 or deb_col_lat_crop < 0 or
                        end_row_lon_crop > image_height or end_col_lat_crop > image_width):
                warnings.warn(f"Crop coordinates exceed image bounds ({image_height}x{image_width}). "
                            f"Results will be clipped to image extent.")

            # Ensure end coordinates are greater than start coordinates
            if deb_row_lon_crop > end_row_lon_crop:
                deb_row_lon_crop, end_row_lon_crop = end_row_lon_crop, deb_row_lon_crop
                warnings.warn("Start row/longitude greater than end row/longitude, swapping values.")

            if deb_col_lat_crop > end_col_lat_crop:
                deb_col_lat_crop, end_col_lat_crop = end_col_lat_crop, deb_col_lat_crop
                warnings.warn("Start col/latitude greater than end col/latitude, swapping values.")

            # Perform the crop operation
            try:
                self.image, self.__meta = crop_rio(self.image,
                                                deb_row_lon_crop, end_row_lon_crop,
                                                deb_col_lat_crop, end_col_lat_crop,
                                                dest_name=dest_name,
                                                meta=self.__meta,
                                                channel_first=True,
                                                names = self.names)
                self.__update()

                # Add to history if enabled
                if self.__history is not False:
                    now = datetime.datetime.now()
                    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
                    if pixel is True:
                        self.__listhistory.append(
                            f'[{now_str}] - Cropped from shape {original_shape} to {self.shape} '
                            f'using pixel coordinates rows {orig_deb_row_lon}:{orig_end_row_lon}, '
                            f'cols {orig_deb_col_lat}:{orig_end_col_lat}'
                        )
                    else:
                        self.__listhistory.append(
                            f'[{now_str}] - Cropped from shape {original_shape} to {self.shape} '
                            f'using geographic coordinates lon {orig_deb_row_lon:.6f}:{orig_end_row_lon:.6f}, '
                            f'lat {orig_deb_col_lat:.6f}:{orig_end_col_lat:.6f}'
                        )
                    if dest_name is not None:
                        self.__listhistory.append(f'\t Cropped image saved to: {dest_name}')

            except Exception as e:
                raise RuntimeError(f"Cropping failed: {str(e)}") from e

        else:
            return self.__apply_crop(deb_row_lon, end_row_lon, deb_col_lat, end_col_lat, dest_name=dest_name, pixel=pixel)

    def select_bands(self, bands=None, dest_name=None, inplace = False, reformat_names=False):
        """
        Select only specified bands in the image

        This method modifies the image to contain only the specified bands, discarding
        all other bands. Band naming can be preserved or updated based on parameters.

        Parameters
        ----------
        bands : str, list, int, or None, optional
            The bands to keep in the image. Format depends on band naming:
            - If using named bands: band name(s) as string(s) (e.g., 'NIR', ['R', 'G', 'B'])
            - If using indexed bands: band index/indices as int(s) or string(s) (e.g., 3, ['1', '4', '7'])
            If None, no bands are selected (invalid operation).

        dest_name : str, optional
            Path to save the modified image. If None, the image is not saved.
            Default is None.

        inplace : bool, default False
            If False, return a copy. Otherwise, modify the image by keeping only selected bands

        reformat_names : bool, optional
            Band naming behavior:
            - If True: Rename bands sequentially as "1", "2", "3", etc.
            - If False: Preserve original band names when possible
            Default is False.

        Returns
        -------
        Geoimage
            The modified image with only selected bands or None if `inplace=True`.

        Raises
        ------
        ValueError
            If no bands are specified, or if any specified band doesn't exist in the image.

        Examples
        --------
        >>> # Extract only 3 specific bands
        >>> original_bands = list(image.names.keys())
        >>> image_selected = image.select_bands(['NIR', 'Red', 'Green'])
        >>> print(f"Original bands: {original_bands}, New bands: {list(image_selected.names.keys())}")
        >>>
        >>> # Keep bands and renumber them sequentially
        >>> image.select_bands([4, 3, 2], reformat_names=True, inplace=True)
        >>> print(f"Band names after reordering: {list(image.names.keys())}")
        >>>
        >>> # Select a single band
        >>> nir = image.select_bands('NIR', dest_name='nir_only.tif')

        Notes
        -----
        - If band names contain duplicates, they will be automatically reformatted.
        - The band order in the result matches the order in the 'bands' parameter.
        """
        if inplace:
            # Check if bands parameter is provided
            if bands is None:
                raise ValueError("No bands specified for selection. Please provide at least one band name or index.")

            # Keep track of original bands for history
            original_bands = list(self.names.keys())
            original_count = self.nb_bands

            # Convert bands to list of strings
            bands = numpy_to_string_list(bands)

            # Validate band existence
            set1 = set(bands)
            set2 = set(self.names)
            if not(set1 <= set2):
                missing = set1 - set2
                available = ", ".join(sorted(list(set2)))
                raise ValueError(f"The following bands do not exist: {missing}. "
                                f"Available bands are: {available}")

            try:
                # Get indices of selected bands
                band_indices = [self.names[band] - 1 for band in bands]

                # Keep only selected bands
                self.image = self.image[band_indices, :, :]

                # Update metadata
                self.__meta['count'] = len(bands)
                self.nb_bands = self.__meta['count']

                # Handle band naming
                if has_duplicates(bands):
                    # Force rename if duplicates are present
                    self.__update_names()
                    name_handling = "renamed due to duplicates"
                elif reformat_names:
                    # Rename if explicitly requested
                    self.__update_names()
                    name_handling = "renamed sequentially"
                elif self.__namesgiven or not reformat_names:
                    # Preserve original names but update indices
                    self.names = reindex_dictionary_keep_order(self.names, bands)
                    name_handling = "preserved with updated indices"
                else:
                    # Default fallback
                    self.__update_names()
                    name_handling = "reset to defaults"

                # Update derived properties
                self.__update()

                # Save if requested
                if dest_name is not None:
                    write_geoim(self.image, self.__meta, dest_name, names= self.names)

                # Update history if enabled
                if self.__history is not False:
                    now = datetime.datetime.now()
                    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
                    self.__listhistory.append(
                        f'[{now_str}] - Selected {len(bands)}/{original_count} bands: {bands}. '
                        f'Band names were {name_handling}.'
                    )
                    if dest_name is not None:
                        self.__listhistory.append(f'\t Image with selected bands saved to: {dest_name}')

            except Exception as e:
                raise RuntimeError(f"Band selection failed: {str(e)}") from e

        else:
            return self.__get_bands(bands=bands, dest_name=dest_name, reformat_names=reformat_names)
    def __get_bands(self, bands=None, dest_name=None, reformat_names=False):
        """
        Extract specified bands into a new Geoimage without modifying the original.

        This method creates a new Geoimage containing only the specified bands,
        while preserving the original image. Unlike select_bands(), which modifies
        the original image, this method returns a new image with the selected bands.

        Parameters
        ----------
        bands : str, list, int, or None, optional
            The bands to include in the new image. Format depends on band naming:
            - If using named bands: band name(s) as string(s) (e.g., 'NIR', ['R', 'G', 'B'])
            - If using indexed bands: band index/indices as int(s) or string(s) (e.g., 3, ['1', '4', '7'])
            If None, returns a copy of the entire image.

        dest_name : str, optional
            Path to save the new image. If None, the image is not saved.
            Default is None.

        reformat_names : bool, optional
            Band naming behavior:
            - If True: Rename bands sequentially as "1", "2", "3", etc.
            - If False: Preserve original band names when possible
            Default is False.

        Returns
        -------
        Geoimage
            A new Geoimage containing only the specified bands.

        Raises
        ------
        ValueError
            If any specified band doesn't exist in the image.

        Examples
        --------
        >>> # Extract RGB bands
        >>> rgb = image.__get_bands(['R', 'G', 'B'])
        >>> rgb.visu()
        >>>
        >>> # Extract a single band with its original name
        >>> nir = image.__get_bands('NIR', dest_name='nir_band.tif')
        >>> print(f"NIR band names: {nir.names}")
        >>>
        >>> # Extract multiple bands and rename them sequentially
        >>> subset = image.__get_bands([7, 4, 2], reformat_names=True)
        >>> print(f"Band names after extraction: {subset.names}")

        Notes
        -----
        - If the original image has custom band names, they can be preserved in the new image.
        - This is useful for creating band subsets without modifying the original data.
        - For permanent modifications to the original image, use select_bands() instead.
        """
        # If no bands are specified, return a copy of the entire image
        if bands is None:
            return self.copy()

        # Convert bands to list of strings
        bands = numpy_to_string_list(bands)

        # Validate band existence
        set1 = set(bands)
        set2 = set(self.names)
        if not(set1 <= set2):
            missing = set1 - set2
            available = ", ".join(sorted(list(set2)))
            raise ValueError(f"The following bands do not exist: {missing}. "
                            f"Available bands are: {available}")

        try:
            # Get indices of selected bands
            band_indices = [self.names[band] - 1 for band in bands]

            # Extract selected bands
            data = self.image[band_indices, :, :].copy()  # Explicit copy to avoid reference issues

            # Copy and update metadata
            meta = self.__meta.copy()
            meta['count'] = len(bands)

            # Handle band naming based on parameters and conditions
            name_handling = ""
            if has_duplicates(bands):
                # Initialize new names if duplicates are present
                names = initialize_dict(data.shape[0])
                name_handling = "renamed sequentially due to duplicates"
            elif reformat_names:
                # Initialize new names if reformatting is requested
                names = initialize_dict(data.shape[0])
                name_handling = "renamed sequentially as requested"
            elif self.__namesgiven or not reformat_names:
                # Preserve original names but update indices
                names = reindex_dictionary_keep_order(self.names, bands)
                name_handling = "preserved with updated indices"
            else:
                # Default fallback
                names = initialize_dict(data.shape[0])
                name_handling = "set to default sequential names"

            # Create new Geoimage with selected bands
            result = Geoimage(data=data, meta=meta, names=names, georef=self.__georef)

            # Copy history if present in original
            if hasattr(self, '__history') and self.__history:
                result.activate_history()
                if hasattr(self, '__listhistory') and self.__listhistory:
                    result.__listhistory = self.__listhistory.copy()

                # Add extraction entry to history
                now = datetime.datetime.now()
                now_str = now.strftime("%Y-%m-%d %H:%M:%S")
                result.__listhistory.append(
                    f'[{now_str}] - Created by extracting {len(bands)}/{self.nb_bands} bands: {bands}. '
                    f'Band names were {name_handling}.'
                )

            # Save if requested
            if dest_name is not None:
                result.save(dest_name)
                if result.__history:
                    result.__listhistory.append(f'\t Saved to: {dest_name}')

            return result

        except Exception as e:
            raise RuntimeError(f"Band extraction failed: {str(e)}") from e

    def stack(self, im_to_stack, dtype=None, dest_name=None, inplace=False, reformat_names=False):
        """
        Stack bands from another image onto this image.

        This method combines the bands from another image with the current image,
        modifying the current image to include all bands from both sources.

        Parameters
        ----------
        im_to_stack : Geoimage
            The image whose bands will be stacked onto this image.
            Should have the same spatial dimensions (rows, cols).

        dtype : str or None, optional
            The data type for the stacked image. If None, an appropriate type is
            determined based on the types of both input images.
            Common values: 'float64', 'float32', 'int32', 'uint16', 'uint8'.
            Default is None.

        dest_name : str, optional
            Path to save the stacked image. If None, the image is not saved.
            Default is None.

        inplace : bool, default False
            If False, return a copy of the stacked image. Otherwise, do stacking in place and return None.

        reformat_names : bool, optional
            If True, band names will be reset to a simple numeric format ("1", "2", "3", ...).
            If False, the function will preserve original band names where possible,
            adding suffixes if needed to resolve conflicts.
            Default is False.

        Returns
        -------
        Geoimage
            The image with additional bands or None if `inplace=True`

        Raises
        ------
        ValueError
            If the spatial dimensions of the images don't match or an unknown dtype is specified.

        Examples
        --------
        >>> # Stack two images with different spectral bands
        >>> optical = Geoimage("optical.tif", names={'R': 1, 'G': 2, 'B': 3})
        >>> thermal = Geoimage("thermal.tif", names={'T': 1})
        >>> combined = optical.stack(thermal)
        >>> print(f"Combined bands: {list(combined.names.keys())}")
        >>>
        >>> # Stack and rename bands sequentially
        >>> combined = optical.stack(thermal, reformat_names=True)
        >>> print(f"After renaming: {list(combined.names.keys())}")
        >>>
        >>> # Stack with explicit data type
        >>> combined = optical.stack(thermal, dtype='float32', dest_name='combined.tif')
        >>>
        >>> # Stack in the image directly
        >>> optical.stack(thermal, reformat_names=True, inplace=True)
        >>> print(f"After renaming: {list(combined.names.keys())}")

        Notes
        -----
        - The bands from both images are combined along the band dimension (axis 0).
        - Band naming conflicts are resolved automatically, adding suffixes if needed.
        - The spatial dimensions (rows, cols) of both images must match.
        """
        if inplace:
            # Validate spatial dimensions match
            if self.shape != im_to_stack.shape:
                raise ValueError(f"Images have different spatial dimensions: "
                                f"{self.shape} vs {im_to_stack.shape}. "
                                f"Images must have the same dimensions to stack.")

            # Track original band count
            original_band_count = self.nb_bands
            stacked_band_count = im_to_stack.nb_bands

            # Determine the output data type
            original_dtype = self.__meta['dtype']
            stacked_dtype = im_to_stack.get_meta()['dtype']

            if dtype is not None:
                # Use the explicitly specified dtype
                self.__meta['dtype'] = dtype
            else:
                # Auto-determine best dtype based on input dtypes
                dtype_priority = {
                    'float64': 1, 'float32': 2, 'int32': 3,
                    'uint16': 4, 'int16': 5, 'uint8': 6, 'int8': 7
                }

                # Define dtype promotion rules for pairs of dtypes
                dtype_promotion = {
                    ('float64', 'any'): 'float64',
                    ('float32', 'float64'): 'float64',
                    ('float32', 'any'): 'float32',
                    ('int32', 'float64'): 'float64',
                    ('int32', 'float32'): 'float32',
                    ('int32', 'any'): 'int32',
                    ('uint16', 'float64'): 'float64',
                    ('uint16', 'float32'): 'float32',
                    ('uint16', 'int32'): 'int32',
                    ('uint16', 'int16'): 'float32',
                    ('uint16', 'uint16'): 'uint16',
                    ('uint16', 'any'): 'uint16',
                    ('int16', 'float64'): 'float64',
                    ('int16', 'float32'): 'float32',
                    ('int16', 'int32'): 'int32',
                    ('int16', 'uint16'): 'float32',
                    ('int16', 'int16'): 'int16',
                    ('int16', 'any'): 'int16',
                    ('uint8', 'float64'): 'float64',
                    ('uint8', 'float32'): 'float32',
                    ('uint8', 'int32'): 'int32',
                    ('uint8', 'uint16'): 'uint16',
                    ('uint8', 'int16'): 'int16',
                    ('uint8', 'uint8'): 'uint8',
                    ('int8', 'any'): 'int8'
                }

                # Sort dtypes by priority (higher priority first)
                dtype1, dtype2 = sorted([original_dtype, stacked_dtype],
                                    key=lambda x: dtype_priority.get(x, 999))

                # Look up the promotion rule
                key = (dtype1, dtype2) if (dtype1, dtype2) in dtype_promotion else (dtype1, 'any')
                self.__meta['dtype'] = dtype_promotion.get(key, 'float32')

            # Handle band naming
            if has_common_key(self.names, im_to_stack.names):
                # If there are common band names, add suffixes
                self.names = concat_dicts_with_keys(self.names, im_to_stack.names)
                name_handling = "disambiguated with suffixes due to name conflicts"
            else:
                # If no conflicts, combine directly
                self.names = concat_dicts_with_keys_unmodified(self.names, im_to_stack.names)
                name_handling = "combined without modification (no conflicts)"

            try:
                # Convert and concatenate bands
                target_dtype = np.dtype(self.__meta['dtype'])
                self.image = np.concatenate(
                    (self.image.astype(target_dtype),
                    im_to_stack.image.astype(target_dtype)),
                    axis=0
                )

                # Update metadata
                self.__meta['count'] = len(self.names)
                self.nb_bands = len(self.names)

                # Reset names if requested
                if reformat_names:
                    self.reset_names()
                    name_handling = "reset to sequential numbering"

                # Update other metadata and derived properties
                self.__update()

                # Save if requested
                if dest_name is not None:
                    write_geoim(self.image, self.__meta, dest_name, names= self.names)

                # Update history if enabled
                if self.__history is not False:
                    now = datetime.datetime.now()
                    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
                    band_list = ', '.join(list(im_to_stack.names.keys()))
                    self.__listhistory.append(
                        f'[{now_str}] - Stacked {stacked_band_count} bands ({band_list}) onto existing {original_band_count} bands. '
                        f'Output type: {self.__meta["dtype"]}. Band names were {name_handling}.'
                    )
                    if dest_name is not None:
                        self.__listhistory.append(f'\t Stacked image saved to: {dest_name}')

            except Exception as e:
                raise RuntimeError(f"Stacking failed: {str(e)}") from e
        else:
            return self.__apply_stack(im_to_stack, dtype=dtype, dest_name=dest_name, reformat_names=reformat_names)

    def __apply_stack(self, im_to_stack, dtype=None, dest_name=None, reformat_names=False):
        """
        Create a new image by stacking bands from another image.

        This method creates a new Geoimage by combining the bands from the current image
        with those from another image. Unlike stack(), this method doesn't modify
        the original images but returns a new one.

        Parameters
        ----------
        im_to_stack : Geoimage
            The image whose bands will be stacked onto this image.
            Should have the same spatial dimensions (rows, cols).

        dtype : str or None, optional
            The data type for the stacked image. If None, an appropriate type is
            determined based on the types of both input images.
            Common values: 'float64', 'float32', 'int32', 'uint16', 'uint8'.
            Default is None.

        dest_name : str, optional
            Path to save the stacked image. If None, the image is not saved.
            Default is None.

        reformat_names : bool, optional
            If True, band names will be reset to a simple numeric format ("1", "2", "3", ...).
            If False, the function will preserve original band names where possible,
            adding suffixes if needed to resolve conflicts.
            Default is False.

        Returns
        -------
        Geoimage
            A new Geoimage containing all bands from both input images.

        Raises
        ------
        ValueError
            If the spatial dimensions of the images don't match or an unknown dtype is specified.

        Examples
        --------
        >>> # Stack two images while preserving the originals
        >>> optical = Geoimage("optical.tif", names={'R': 1, 'G': 2, 'B': 3})
        >>> thermal = Geoimage("thermal.tif", names={'T': 1})
        >>> combined = optical.apply_stack(thermal)
        >>> print(f"Combined bands: {list(combined.names.keys())}")
        >>> print(f"Original optical bands: {list(optical.names.keys())}")  # Unchanged
        >>>
        >>> # Stack with simplified band naming
        >>> combined = optical.apply_stack(thermal, reformat_names=True)
        >>> print(f"Sequential band names: {list(combined.names.keys())}")
        >>>
        >>> # Stack with explicit data type and save
        >>> combined = optical.apply_stack(thermal, dtype='float32', dest_name='combined.tif')

        Notes
        -----
        - This method creates a new image without modifying the inputs.
        - The bands from both images are combined along the band dimension (axis 0).
        - Band naming conflicts are resolved automatically, adding suffixes if needed.
        - The spatial dimensions (rows, cols) of both images must match.
        """
        # Validate spatial dimensions match
        if self.shape != im_to_stack.shape:
            raise ValueError(f"Images have different spatial dimensions: "
                            f"{self.shape} vs {im_to_stack.shape}. "
                            f"Images must have the same dimensions to stack.")

        # Track band counts for documentation
        orig_band_count = self.nb_bands
        stack_band_count = im_to_stack.nb_bands

        # Make a copy of metadata
        meta = self.__meta.copy()

        # Determine the output data type
        orig_dtype = self.__meta['dtype']
        stack_dtype = im_to_stack.get_meta()['dtype']

        if dtype is not None:
            # Use the explicitly specified dtype
            meta['dtype'] = dtype
        else:
            # Auto-determine best dtype based on input dtypes
            dtype_priority = {
                'float64': 1, 'float32': 2, 'int32': 3,
                'uint16': 4, 'int16': 5, 'uint8': 6, 'int8': 7
            }

            # Define dtype promotion rules for pairs of dtypes
            dtype_promotion = {
                ('float64', 'any'): 'float64',
                ('float32', 'float64'): 'float64',
                ('float32', 'any'): 'float32',
                ('int32', 'float64'): 'float64',
                ('int32', 'float32'): 'float32',
                ('int32', 'any'): 'int32',
                ('uint16', 'float64'): 'float64',
                ('uint16', 'float32'): 'float32',
                ('uint16', 'int32'): 'int32',
                ('uint16', 'int16'): 'float32',
                ('uint16', 'uint16'): 'uint16',
                ('uint16', 'any'): 'uint16',
                ('int16', 'float64'): 'float64',
                ('int16', 'float32'): 'float32',
                ('int16', 'int32'): 'int32',
                ('int16', 'uint16'): 'float32',
                ('int16', 'int16'): 'int16',
                ('int16', 'any'): 'int16',
                ('uint8', 'float64'): 'float64',
                ('uint8', 'float32'): 'float32',
                ('uint8', 'int32'): 'int32',
                ('uint8', 'uint16'): 'uint16',
                ('uint8', 'int16'): 'int16',
                ('uint8', 'uint8'): 'uint8',
                ('int8', 'any'): 'int8'
            }

            # Sort dtypes by priority (higher priority first)
            dtype1, dtype2 = sorted([orig_dtype, stack_dtype],
                                    key=lambda x: dtype_priority.get(x, 999))

            # Look up the promotion rule
            key = (dtype1, dtype2) if (dtype1, dtype2) in dtype_promotion else (dtype1, 'any')
            meta['dtype'] = dtype_promotion.get(key, 'float32')

        # Handle band naming
        name_handling = ""
        if has_common_key(self.names, im_to_stack.names):
            # If there are common band names, add suffixes
            names = concat_dicts_with_keys(self.names, im_to_stack.names)
            name_handling = "disambiguated with suffixes due to name conflicts"
        else:
            # If no conflicts, combine directly
            names = concat_dicts_with_keys_unmodified(self.names, im_to_stack.names)
            name_handling = "combined without modification (no conflicts)"

        # Apply band renaming if requested
        if reformat_names:
            names = initialize_dict(len(names))
            name_handling = "reset to sequential numbering"

        try:
            # Convert and concatenate bands
            target_dtype = np.dtype(meta['dtype'])
            stacked_image = np.concatenate(
                (self.image.astype(target_dtype),
                im_to_stack.image.astype(target_dtype)),
                axis=0
            )

            # Update metadata count
            meta['count'] = len(names)

            # Create new Geoimage with stacked data
            result = Geoimage(data=stacked_image, meta=meta, names=names, georef=self.__georef)

            # Initialize history if needed
            if hasattr(self, '__history') and self.__history:
                result.activate_history()

                # Copy existing history if available
                if hasattr(self, '__listhistory') and self.__listhistory:
                    result.__listhistory = self.__listhistory.copy()

                # Add stacking entry to history
                now = datetime.datetime.now()
                now_str = now.strftime("%Y-%m-%d %H:%M:%S")

                stack_bands = ", ".join(list(im_to_stack.names.keys()))
                result.__listhistory.append(
                    f'[{now_str}] - Created by stacking {orig_band_count} bands from first image with '
                    f'{stack_band_count} bands ({stack_bands}) from second image. '
                    f'Output type: {meta["dtype"]}. Band names were {name_handling}.'
                )

            # Save if requested
            if dest_name is not None:
                result.save(dest_name)
                if result.__history:
                    result.__listhistory.append(f'\t Stacked image saved to: {dest_name}')

            return result

        except Exception as e:
            raise RuntimeError(f"Stacking failed: {str(e)}") from e

    def remove_bands(self, bands, inplace=False, reformat_names=False, dest_name=None):
        """
        Remove specified bands from the image.

        This method modifies the current image by removing the specified bands.
        The remaining bands can be renamed sequentially or retain their original names.

        Parameters
        ----------
        bands : str, list, int, or array-like
            The bands to remove from the image. Format depends on band naming:
            - If using named bands: band name(s) as string(s) (e.g., 'NIR', ['R', 'G', 'B'])
            - If using indexed bands: band index/indices as int(s) or string(s) (e.g., 3, ['1', '4', '7'])

        inplace : bool, default False
            If False, return a copy. Otherwise, do removing in place and return None.


        reformat_names : bool, optional
            Band naming behavior after removal:
            - If True: Rename remaining bands sequentially as "1", "2", "3", etc.
            - If False: Preserve original band names with their indices updated
            Default is False.

        dest_name : str, optional
            Path to save the modified image. If None, the image is not saved.
            Default is None.

        Returns
        -------
        Geoimage
            The image with specified bands removed or None if `inplace=True`

        Raises
        ------
        ValueError
            If any specified band doesn't exist in the image, or if removing all bands.

        Examples
        --------
        >>> # Remove a single band
        >>> original_bands = list(image.names.keys())
        >>> image_removed = image.remove_bands('B4')
        >>> print(f"Original: {original_bands}, After removal: {list(image_removed.names.keys())}")
        >>>
        >>> # Remove multiple bands and rename sequentially
        >>> image_removed = image.remove_bands(['B1', 'B2'], reformat_names=True)
        >>> print(f"After renaming: {list(image_removed = .names.keys())}")
        >>>
        >>> # Remove bands and save the result
        >>> image_removed = image.remove_bands(['SWIR1', 'SWIR2'], dest_name='visible_only.tif')
        >>>
        >>> # Remove a single band
        >>> original_bands = list(image.names.keys())
        >>> image.remove_bands('B4', inplace=True)
        >>> print(f"Original: {original_bands}, After removal: {list(image.names.keys())}")
        >>>
        >>> # Remove multiple bands and rename sequentially
        >>> image.remove_bands(['B1', 'B2'], reformat_names=True, inplace=True)
        >>> print(f"After renaming: {list(image.names.keys())}")
        >>>
        >>> # Remove bands and save the result
        >>> image.remove_bands(['SWIR1', 'SWIR2'], dest_name='visible_only.tif', inplace=True)

        Notes
        -----
        - If reformat_names=False (default), band names are preserved but indices are updated.
        - If reformat_names=True, bands are renamed sequentially (1, 2, 3, ...).
        """
        if inplace:
            # Convert bands to list of strings
            bands = numpy_to_string_list(bands)

            # Track original state for history
            original_band_count = self.nb_bands
            original_bands = list(self.names.keys())

            # Validate band existence
            set1 = set(bands)
            set2 = set(self.names)
            if not(set1 <= set2):
                missing = set1 - set2
                available = ", ".join(sorted(list(set2)))
                raise ValueError(f"The following bands do not exist: {missing}. "
                                f"Available bands are: {available}")

            # Check if trying to remove all bands
            if len(bands) >= self.nb_bands:
                raise ValueError(f"Cannot remove all bands. You're trying to remove {len(bands)} "
                                f"bands from an image with {self.nb_bands} bands.")

            try:
                # Get indices of bands to remove
                band_indices = [self.names[band] - 1 for band in bands]

                # Create a mask for bands to keep
                mask = np.ones(self.get_nb_bands(), dtype=bool)
                mask[band_indices] = False

                # Keep only non-masked bands
                self.image = self.image[mask, :, :]

                # Update metadata count
                self.__meta['count'] = self.image.shape[0]
                self.nb_bands = self.__meta['count']

                # Record naming approach for history
                if reformat_names is True:
                    self.__update_names()
                    name_handling = "reset to sequential numbering"
                elif self.__namesgiven or not reformat_names:
                    # Preserve original names but update indices
                    self.names = reindex_dictionary(self.names, bands)
                    name_handling = "preserved with updated indices"
                else:
                    # Default fallback
                    self.__update_names()
                    name_handling = "reset to defaults"

                # Update derived properties
                self.__update()

                # Save if requested
                if dest_name is not None:
                    write_geoim(self.image, self.__meta, dest_name, names= self.names)

                # Update history if enabled
                if self.__history is not False:
                    now = datetime.datetime.now()
                    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
                    remaining = original_band_count - len(bands)
                    removed_bands = ", ".join(bands)
                    self.__listhistory.append(
                        f'[{now_str}] - Removed {len(bands)}/{original_band_count} bands: {removed_bands}. '
                        f'{remaining} bands remain. Band names were {name_handling}.'
                    )
                    if dest_name is not None:
                        self.__listhistory.append(f'\t Image with removed bands saved to: {dest_name}')

            except Exception as e:
                raise RuntimeError(f"Band removal failed: {str(e)}") from e

        else:
            return self.__apply_remove_bands(bands, reformat_names=reformat_names, dest_name=dest_name)

    def __apply_remove_bands(self, bands, reformat_names=False, dest_name=None):
        """
        Create a new image with specified bands removed.

        This method creates a new Geoimage without the specified bands, while preserving
        the original image. Unlike remove_bands() which modifies the original image,
        this method returns a new image with the specified bands removed.

        Parameters
        ----------
        bands : str, list, int, or array-like
            The bands to remove from the image. Format depends on band naming:
            - If using named bands: band name(s) as string(s) (e.g., 'NIR', ['R', 'G', 'B'])
            - If using indexed bands: band index/indices as int(s) or string(s) (e.g., 3, ['1', '4', '7'])

        reformat_names : bool, optional
            Band naming behavior in the new image:
            - If True: Rename remaining bands sequentially as "1", "2", "3", etc.
            - If False: Preserve original band names with their indices updated
            Default is False.

        dest_name : str, optional
            Path to save the new image. If None, the image is not saved.
            Default is None.

        Returns
        -------
        Geoimage
            A new Geoimage with the specified bands removed.

        Raises
        ------
        ValueError
            If any specified band doesn't exist in the image, or if trying to remove all bands.

        Examples
        --------
        >>> # Create a new image without certain bands
        >>> original = Geoimage("multispectral.tif", names={'R': 1, 'G': 2, 'B': 3, 'NIR': 4})
        >>> rgb_only = original.apply_remove_bands('NIR')
        >>> print(f"Original bands: {list(original.names.keys())}")  # Unchanged
        >>> print(f"New bands: {list(rgb_only.names.keys())}")
        >>>
        >>> # Remove multiple bands and rename sequentially
        >>> visible = original.apply_remove_bands(['NIR', 'SWIR'], reformat_names=True)
        >>> print(f"New sequential names: {list(visible.names.keys())}")
        >>>
        >>> # Create a subset and save it
        >>> subset = original.apply_remove_bands(['B1', 'B7'], dest_name='subset.tif')

        Notes
        -----
        - This method creates a new image without modifying the original.
        - For permanent modifications to the original image, use remove_bands() instead.
        - If trying to select specific bands to keep (rather than remove), __get_bands()
        might be more straightforward.
        """
        # Convert bands to list of strings
        bands = numpy_to_string_list(bands)

        # Validate band existence
        set1 = set(bands)
        set2 = set(self.names)
        if not(set1 <= set2):
            missing = set1 - set2
            available = ", ".join(sorted(list(set2)))
            raise ValueError(f"The following bands do not exist: {missing}. "
                            f"Available bands are: {available}")

        # Check if trying to remove all bands
        if len(bands) >= self.nb_bands:
            raise ValueError(f"Cannot remove all bands. You're trying to remove {len(bands)} "
                            f"bands from an image with {self.nb_bands} bands.")

        try:
            # Get indices of bands to remove
            band_indices = [self.names[band] - 1 for band in bands]

            # Create a mask for bands to keep
            mask = np.ones(self.get_nb_bands(), dtype=bool)
            mask[band_indices] = False

            # Extract only non-masked bands
            data = self.image[mask, :, :].copy()  # Explicit copy to avoid reference issues

            # Copy and update metadata
            meta = self.__meta.copy()
            meta['count'] = data.shape[0]

            # Handle band naming based on parameters
            name_handling = ""
            if reformat_names:
                # Initialize new names if reformatting is requested
                names = initialize_dict(data.shape[0])
                name_handling = "reset to sequential numbering"
            elif self.__namesgiven or not reformat_names:
                # Preserve original names but update indices
                names = reindex_dictionary(self.names, bands)
                name_handling = "preserved with updated indices"
            else:
                # Default fallback
                names = initialize_dict(data.shape[0])
                name_handling = "set to default sequential names"

            # Create new Geoimage with remaining bands
            result = Geoimage(data=data, meta=meta, names=names, georef=self.__georef)

            # Copy history if present in original
            if hasattr(self, '__history') and self.__history:
                result.activate_history()
                if hasattr(self, '__listhistory') and self.__listhistory:
                    result.__listhistory = self.__listhistory.copy()

                # Add removal entry to history
                now = datetime.datetime.now()
                now_str = now.strftime("%Y-%m-%d %H:%M:%S")
                removed_bands = ", ".join(bands)
                remaining = self.nb_bands - len(bands)
                result.__listhistory.append(
                    f'[{now_str}] - Created by removing {len(bands)}/{self.nb_bands} bands: {removed_bands}. '
                    f'{remaining} bands remain. Band names were {name_handling}.'
                )

            # Save if requested
            if dest_name is not None:
                result.save(dest_name)
                if result.__history:
                    result.__listhistory.append(f'\t Saved to: {dest_name}')

            return result

        except Exception as e:
            raise RuntimeError(f"Band removal failed: {str(e)}") from e

    def reorder_bands(self, band_order, inplace=False):
        """
        Reorder the image bands according to the specified order.

        This method changes the order of bands in the image based on the specified
        band_order parameter. The current image is modified in-place or in a new Geoimage.

        Parameters
        ----------
        band_order : list or dict
            The desired band order specification:
            - If list: A list of band names in the desired order
            Example: ['NIR', 'Red', 'Green', 'Blue']
            - If dict: A dictionary mapping band names to their desired positions (1-based)
            Example: {'NIR': 1, 'Red': 2, 'Green': 3, 'Blue': 4}

        inplace : bool, default False
            If False, return a copy. Otherwise, do reorder bands in place and return None.

        Returns
        -------
        Geoimage
            A copy of the image with reordered bands or None if `inplace=True`

        Raises
        ------
        ValueError
            If band_order is not a list or dictionary, or if it contains bands
            that don't exist in the image.

        Examples
        --------
        >>> # Reorder bands using a list (most common usage)
        >>> image.info()  # Shows original band order
        >>> image_reorder = image.reorder_bands(['B6', 'B5', 'B4'])
        >>> image_reorder.info()  # Shows new band order
        >>>
        >>> # Directly reorder bands using a dictionary with explicit positions
        >>> image.reorder_bands({'NIR': 1, 'Red': 2, 'Green': 3}, inplace=True)
        >>>
        >>> # Reorder bands and save
        >>> image.reorder_bands(['R', 'G', 'B']).save('rgb_order.tif')

        Notes
        -----
        - All bands in the image must be included in band_order if using a list.
        - If using a dictionary, bands not specified will be excluded.
        - The band indices in the result will be updated to match the new order.
        """
        if inplace:
            original_bands = list(self.names.keys())

            try:
                # Handle list-based ordering
                if isinstance(band_order, list):
                    # Convert all elements to strings
                    band_order = numpy_to_string_list(band_order)

                    # Validate that all specified bands exist
                    set1 = set(band_order)
                    set2 = set(self.names)
                    if not(set1 <= set2):
                        missing = set1 - set2
                        available = ", ".join(sorted(list(set2)))
                        raise ValueError(f"The following bands do not exist: {missing}. "
                                        f"Available bands are: {available}")

                    # Validate that all bands are accounted for in list mode
                    if len(band_order) != self.nb_bands:
                        if len(band_order) < self.nb_bands:
                            missing_bands = set(self.names.keys()) - set(band_order)
                            message = (f"Not all bands specified in reordering list. "
                                    f"Missing bands: {missing_bands}. "
                                    f"Use a dictionary if you want to exclude some bands.")
                        else:
                            message = (f"Too many bands specified in reordering list. "
                                    f"Image has {self.nb_bands} bands, but {len(band_order)} were provided.")
                        raise ValueError(message)

                    # Get the reordered bands
                    im2 = self.__get_bands(band_order)

                # Handle dictionary-based ordering
                elif isinstance(band_order, dict):
                    # Validate dictionary values
                    if not all(isinstance(pos, (int, float)) for pos in band_order.values()):
                        raise ValueError("Dictionary values must be numbers representing band positions.")

                    # Convert to string keys if needed
                    band_order = {str(k): v for k, v in band_order.items()}

                    # Validate that all specified bands exist
                    set1 = set(band_order.keys())
                    set2 = set(self.names)
                    if not(set1 <= set2):
                        missing = set1 - set2
                        available = ", ".join(sorted(list(set2)))
                        raise ValueError(f"The following bands do not exist: {missing}. "
                                        f"Available bands are: {available}")

                    # Reorder bands based on the dictionary
                    names_to_keep = list(map(str, reorder_dict_by_values(band_order).keys()))
                    im2 = self.__get_bands(names_to_keep)

                else:
                    raise ValueError("band_order must be a list or a dictionary.")

                # Update the current image with the reordered bands
                self.update_from(im2)

                # Add to history if enabled
                if hasattr(self, '__history') and self.__history:
                    now = datetime.datetime.now()
                    now_str = now.strftime("%Y-%m-%d %H:%M:%S")

                    if isinstance(band_order, list):
                        ordering_type = "list-based"
                        new_order = ", ".join(band_order)
                    else:
                        ordering_type = "dictionary-based"
                        ordered_bands = reorder_dict_by_values(band_order)
                        new_order = ", ".join([f"{k}:{v}" for k, v in ordered_bands.items()])

                    self.__listhistory.append(
                        f'[{now_str}] - Reordered bands using {ordering_type} ordering. '
                        f'Original order: {", ".join(original_bands)}. '
                        f'New order: {new_order}.'
                    )

            except Exception as e:
                raise RuntimeError(f"Band reordering failed: {str(e)}") from e

        else:
            return self.__apply_reorder_bands(band_order)

    def __apply_reorder_bands(self, band_order):
        """
        Create a new image with bands reordered according to the specified order.

        This method creates a new Geoimage with bands rearranged according to the
        specified band_order parameter. Unlike reorder_bands(), this method doesn't
        modify the original image but returns a new one.

        Parameters
        ----------
        band_order : list or dict
            The desired band order specification:
            - If list: A list of band names in the desired order
            Example: ['NIR', 'Red', 'Green', 'Blue']
            - If dict: A dictionary mapping band names to their desired positions (1-based)
            Example: {'NIR': 1, 'Red': 2, 'Green': 3, 'Blue': 4}

        Returns
        -------
        Geoimage
            A new Geoimage with reordered bands.

        Raises
        ------
        ValueError
            If band_order is not a list or dictionary, or if it contains bands
            that don't exist in the image.

        Examples
        --------
        >>> # Create a new image with reordered bands
        >>> original = Geoimage("multispectral.tif")
        >>> rgb_order = original.__apply_reorder_bands(['R', 'G', 'B'])
        >>> print(f"Original bands: {list(original.names.keys())}")  # Unchanged
        >>> print(f"Reordered bands: {list(rgb_order.names.keys())}")
        >>>
        >>> # Reorder using a dictionary with explicit positions
        >>> nir_rgb = original.__apply_reorder_bands({'NIR': 1, 'R': 2, 'G': 3, 'B': 4})
        >>> nir_rgb.save('nir_rgb.tif')

        Notes
        -----
        - All bands in the image must be included in band_order if using a list.
        - If using a dictionary, bands not specified will be excluded from the result.
        """
        try:
            # Create a copy of the original image
            result = self.copy()

            # Apply reordering to the copy
            result.reorder_bands(band_order,inplace=True)

            # Update history if available
            if hasattr(result, '__history') and result.__history:
                now = datetime.datetime.now()
                now_str = now.strftime("%Y-%m-%d %H:%M:%S")

                if isinstance(band_order, list):
                    ordering_type = "list-based"
                    new_order = ", ".join(numpy_to_string_list(band_order))
                else:
                    ordering_type = "dictionary-based"
                    ordered_bands = reorder_dict_by_values(band_order)
                    new_order = ", ".join([f"{k}:{v}" for k, v in ordered_bands.items()])

                result.__listhistory.append(
                    f'[{now_str}] - Created as a copy with reordered bands using {ordering_type} ordering. '
                    f'New order: {new_order}.'
                )

            return result

        except Exception as e:
            raise RuntimeError(f"Creating image with reordered bands failed: {str(e)}") from e

    def switch_band(self, band_to_switch, band_to_position=None, inplace=False):
        """
        Change the position of a specified band in the image.

        This method modifies the current image by moving a specified band to a new
        position, either at the beginning of the band sequence or after a specific band.

        Parameters
        ----------
        band_to_switch : str, int, or list
            The band(s) to move to a new position. Can be specified as a band name,
            band index, or a list containing a single band identifier.

        band_to_position : str, int, or None, optional
            The target position specification:
            - If None: Move the band to the first position (beginning of the sequence)
            - If specified: Move the band to the position immediately after this band
            Default is None.

        inplace : bool, default False
            If False, return a copy. Otherwise, do switch in place and return None.

        Returns
        -------
        Geoimage
            The image with the reordered bands or None if `inplace=True`

        Raises
        ------
        ValueError
            If specified bands don't exist or if input parameters are invalid.

        Examples
        --------
        >>> # Move NIR band to the first position
        >>> image.info()  # Check original band order
        >>> im_switch = image.switch_band('NIR')
        >>> im_switch.info()  # NIR is now the first band
        >>>
        >>> # Move SWIR band to position after Red band
        >>> im_switch = image.switch_band('SWIR', 'Red')
        >>> im_switch.info()  # SWIR now follows Red
        >>>
        >>> # Using band indices instead of names
        >>> iim_switch = mage.switch_band(5, 2)  # Move band 5 to after band 2
        >>>
        >>> # Move NIR band to the first position and change image directly
        >>> image.info()  # Check original band order
        >>> image.switch_band('NIR', inplace=True)
        >>> image.info()  # NIR is now the first band
        >>>
        >>> # Move SWIR band to position after Red band
        >>> image.switch_band('SWIR', 'Red', inplace=True)
        >>> image.info()  # SWIR now follows Red
        >>>
        >>> # Using band indices instead of names
        >>> image.switch_band(5, 2, inplace=True)  # Move band 5 to after band 2

        Notes
        -----
        - When multiple bands should be moved as a unit, provide them in a list
        as the band_to_switch parameter.
        - The band indices in the result will be updated to reflect the new order.
        """
        if inplace:
            original_order = list(self.names.keys())

            try:
                # Ensure band_to_switch is a list of strings
                band_to_switch = numpy_to_string_list(band_to_switch)

                # Validate band_to_switch existence
                set1 = set(band_to_switch)
                set2 = set(self.names)
                if not(set1 <= set2):
                    missing = set1 - set2
                    available = ", ".join(sorted(list(set2)))
                    raise ValueError(f"The following bands to switch do not exist: {missing}. "
                                f"Available bands are: {available}")

                # Case 1: Move to the beginning (no target position)
                if band_to_position is None:
                    # Extract the band(s) to move
                    ima = self.__get_bands(band_to_switch)

                    # Remove those band(s) from the original
                    imb = self.__apply_remove_bands(band_to_switch)

                    # Place the extracted band(s) at the beginning
                    self.update_from(ima.__apply_stack(imb))

                    position_desc = "the beginning"

                # Case 2: Move after a specific band
                else:
                    # Convert target position to string
                    band_to_position = numpy_to_string_list(band_to_position)[0]

                    # Validate band_to_position existence
                    if band_to_position not in self.names:
                        available = ", ".join(sorted(list(self.names.keys())))
                        raise ValueError(f"Target position band '{band_to_position}' does not exist. "
                                    f"Available bands are: {available}")

                    # Extract the band(s) to move
                    im_toswitch = self.__get_bands(band_to_switch)

                    # Remove those band(s) from the original
                    im_remove = self.__apply_remove_bands(band_to_switch)

                    # Split the remaining bands into "before target" and "after target"
                    before, after = split_keys(im_remove.names, band_to_position)

                    # Extract "before target" and "after target" bands
                    imb = im_remove.__get_bands(before)
                    ima = im_remove.__get_bands(after)

                    # Reconstruct image in the new order: before + target + switched + after
                    self.update_from(imb.stack(im_toswitch))
                    self.update_from(self.stack(ima))

                    position_desc = f"after band '{band_to_position}'"

                # Add to history if enabled
                if hasattr(self, '__history') and self.__history:
                    now = datetime.datetime.now()
                    now_str = now.strftime("%Y-%m-%d %H:%M:%S")

                    bands_str = ", ".join(band_to_switch)
                    new_order = ", ".join(list(self.names.keys()))

                    self.__listhistory.append(
                        f'[{now_str}] - Moved band(s) [{bands_str}] to {position_desc}. '
                        f'Original order: {", ".join(original_order)}. '
                        f'New order: {new_order}.'
                    )

            except Exception as e:
                raise RuntimeError(f"Band switching failed: {str(e)}") from e
        else:
            return self.__apply_switch_band(band_to_switch, band_to_position=band_to_position)
    def __apply_switch_band(self, band_to_switch, band_to_position=None):
        """
        Create a new image with a specified band moved to a new position.

        This method creates a new Geoimage by moving a specified band to a new
        position, either at the beginning of the band sequence or after a specific band.
        Unlike switch_band(), this method doesn't modify the original image but
        returns a new one.

        Parameters
        ----------
        band_to_switch : str, int, or list
            The band(s) to move to a new position. Can be specified as a band name,
            band index, or a list containing a single band identifier.

        band_to_position : str, int, or None, optional
            The target position specification:
            - If None: Move the band to the first position (beginning of the sequence)
            - If specified: Move the band to the position immediately after this band
            Default is None.

        Returns
        -------
        Geoimage
            A new Geoimage with the reordered bands.

        Raises
        ------
        ValueError
            If specified bands don't exist or if input parameters are invalid.

        Examples
        --------
        >>> # Create new image with NIR band moved to the beginning
        >>> original = Geoimage("multispectral.tif")
        >>> nir_first = original.apply_switch_band('NIR')
        >>> print(f"Original bands: {list(original.names.keys())}")  # Unchanged
        >>> print(f"New bands: {list(nir_first.names.keys())}")
        >>>
        >>> # Create new image with SWIR band after Red band
        >>> swir_after_red = original.apply_switch_band('SWIR', 'Red')
        >>> swir_after_red.save('reordered.tif')

        Notes
        -----
        - When multiple bands should be moved as a unit, provide them in a list
        as the band_to_switch parameter.
        """
        try:
            # Create a copy of the original image
            result = self.copy()

            # Apply band switching to the copy
            result.switch_band(band_to_switch, band_to_position, inplace = True)

            # Modify history to clarify this was a non-destructive operation
            if hasattr(result, '__history') and result.__history:
                # Find and modify the last history entry (added by switch_band)
                if hasattr(result, '__listhistory') and result.__listhistory:
                    last_entry = result.__listhistory[-1]
                    if "Moved band(s)" in last_entry:
                        now = datetime.datetime.now()
                        now_str = now.strftime("%Y-%m-%d %H:%M:%S")

                        # Parse the band information from the existing entry
                        band_info = last_entry.split("Moved band(s)")[1].split(".")[0]

                        # Replace with a new entry that clarifies this was a copy operation
                        result.__listhistory[-1] = (
                            f'[{now_str}] - Created as a copy with band(s){band_info}. '
                            f'Original image was not modified.'
                        )

            return result

        except Exception as e:
            raise RuntimeError(f"Creating image with switched bands failed: {str(e)}") from e

    def add_band(self, spectral_band, name_band=None, after_band=None, dtype=None, inplace = False, dest_name=None):
        """
        Add a new spectral band to the image.

        This method adds a new spectral band to the current image. The new band can
        be placed at the end of the band stack (default) or after a specified band.

        Parameters
        ----------
        spectral_band : numpy.ndarray
            The spectral band data to add. Can be in any of the following formats:
            - 2D array with shape (rows, cols)
            - 3D array with shape (1, rows, cols)
            - 3D array with shape (rows, cols, 1)
            The spatial dimensions must match the current image.

        name_band : str, optional
            Name to assign to the new band. If None, a sequential name will be used.
            Default is None.

        after_band : str, int, or None, optional
            Specify where to insert the new band:
            - If None: Add to the end of the band stack (default)
            - If str or int: Insert after the specified band name or index
            Default is None.

        dtype : str or None, optional
            Data type for the new band and resulting image. If None, preserves the
            highest precision type between the current image and the new band.
            Common values: 'float64', 'float32', 'int32', 'uint16', 'uint8'.
            Default is None.

        inplace : bool, default False
            If False, return a copy of the image with added band
            Otherwise, adding band in place and return None.

        dest_name : str, optional
            Path to save the updated image. If None, the image is not saved.
            Default is None.

        Returns
        -------
        Geoimage
            The modified image with the new band added or None if `inplace=True`.

        Raises
        ------
        ValueError
            If dimensions don't match, if the dtype is unknown, or if the after_band
            doesn't exist in the image.

        Examples
        --------
        >>> # Add a NDVI band to the end
        >>> ndvi = (image.select_band('NIR') - image.select_band('Red') / (image.select_band('NIR') + image.select_band('Red')
        >>> image_and_ndvi = image.add_band(ndvi, name_band='NDVI')
        >>> image_and_ndvi.info()  # Shows NDVI as the last band
        >>>
        >>> # Add a band after a specific position
        >>> image_and_ndvi = image.add_band(thermal_data, name_band='TIR', after_band='NIR')
        >>>
        >>> # Add with explicit data type and save
        >>> image.add_band(elevation, name_band='DEM', dtype='float32',inplace = True,
        >>>                dest_name='with_dem.tif')

        Notes
        -----
        - This method modifies the current image by adding a new band.
        - The spatial dimensions (rows, cols) of the new band must match the current image.
        """
        if inplace:
            # Track original state for history
            original_band_count = self.nb_bands
            original_bands = list(self.names.keys())

            try:
                # Reshape input to match expected format (bands, rows, cols)
                if len(spectral_band.shape) == 2:
                    # Convert 2D (rows, cols) to 3D (1, rows, cols)
                    spectral_band = spectral_band.reshape((1, spectral_band.shape[0], spectral_band.shape[1]))
                elif len(spectral_band.shape) == 3 and spectral_band.shape[2] == 1:
                    # Convert 3D (rows, cols, 1) to 3D (1, rows, cols)
                    spectral_band = np2rio(spectral_band)

                # Validate dimensions match
                if (spectral_band.shape[1] != self.shape[0] or
                    spectral_band.shape[2] != self.shape[1]):
                    raise ValueError(f"Band dimensions ({spectral_band.shape[1]}, {spectral_band.shape[2]}) "
                                f"don't match image dimensions ({self.shape[0]}, {self.shape[1]})")

                # Add the new band name to the dictionary
                old_names = self.names.copy()
                self.names = add_ordered_key(self.names, key_name=name_band)

                # If name_band was auto-generated, get the actual name used
                if name_band is None:
                    # Find the new key that wasn't in the old names
                    new_keys = set(self.names.keys()) - set(old_names.keys())
                    if new_keys:
                        name_band = list(new_keys)[0]

                # Determine insertion position
                if after_band is None:
                    # Add to the end
                    position = self.nb_bands
                    position_desc = "the end"
                else:
                    # Convert after_band to string if needed
                    after_band = str(after_band)

                    # Validate after_band exists
                    if after_band not in self.names:
                        available = ", ".join(sorted(list(self.names.keys())))
                        raise ValueError(f"Band '{after_band}' not found for 'after_band' parameter. "
                                    f"Available bands are: {available}")

                    # Insert after the specified band
                    position = self.names[after_band]
                    position_desc = f"after band '{after_band}'"

                # Determine best dtype if not specified
                if dtype is None:
                    # Auto-determine dtype based on input data and current image
                    input_dtype = str(spectral_band.dtype)
                    current_dtype = self.__meta['dtype']

                    # Simple dtype precedence (higher precision wins)
                    dtype_precedence = {
                        'float64': 1, 'float32': 2, 'int64': 3, 'int32': 4,
                        'uint32': 5, 'int16': 6, 'uint16': 7, 'int8': 8, 'uint8': 9
                    }

                    # Choose the dtype with higher precision
                    if dtype_precedence.get(input_dtype, 999) < dtype_precedence.get(current_dtype, 999):
                        dtype = input_dtype
                    else:
                        dtype = current_dtype

                # Convert to numpy dtype for consistent handling
                np_dtype = np.dtype(dtype)

                # Insert the new band at the specified position
                self.image = np.insert(
                    self.image.astype(np_dtype),
                    position,
                    spectral_band.astype(np_dtype),
                    axis=0
                )

                # Update metadata
                self.__meta['count'] += 1
                self.__meta['dtype'] = str(np_dtype)
                self.__update()

                # Save if requested
                if dest_name is not None:
                    write_geoim(self.image, self.__meta, dest_name, names= self.names)

                # Update history if enabled
                if self.__history is not False:
                    now = datetime.datetime.now()
                    now_str = now.strftime("%Y-%m-%d %H:%M:%S")

                    self.__listhistory.append(
                        f'[{now_str}] - Added band "{name_band}" at {position_desc}. '
                        f'Original bands: {original_band_count}, New total: {self.nb_bands}. '
                        f'Data type: {dtype}.'
                    )

                    if dest_name is not None:
                        self.__listhistory.append(f'\t Image with added band saved to: {dest_name}')

            except Exception as e:
                raise RuntimeError(f"Adding band failed: {str(e)}") from e

        else:
            return self.__apply_add_band(spectral_band, name_band=name_band, after_band=after_band, dtype=dtype, dest_name=dest_name)

    def __apply_add_band(self, spectral_band, name_band=None, after_band=None, dtype=None, dest_name=None):
        """
        Create a new image with an additional spectral band.

        This method creates a new Geoimage with an additional band, without
        modifying the original image. The new band can be placed at the end
        of the band stack (default) or after a specified band.

        Parameters
        ----------
        spectral_band : numpy.ndarray
            The spectral band data to add. Can be in any of the following formats:
            - 2D array with shape (rows, cols)
            - 3D array with shape (1, rows, cols)
            - 3D array with shape (rows, cols, 1)
            The spatial dimensions must match the current image.

        name_band : str, optional
            Name to assign to the new band. If None, a sequential name will be used.
            Default is None.

        after_band : str, int, or None, optional
            Specify where to insert the new band:
            - If None: Add to the end of the band stack (default)
            - If str or int: Insert after the specified band name or index
            Default is None.

        dtype : str or None, optional
            Data type for the new band and resulting image. If None, preserves the
            highest precision type between the current image and the new band.
            Common values: 'float64', 'float32', 'int32', 'uint16', 'uint8'.
            Default is None.

        dest_name : str, optional
            Path to save the new image. If None, the image is not saved.
            Default is None.

        Returns
        -------
        Geoimage
            A new Geoimage with the additional band.

        Raises
        ------
        ValueError
            If dimensions don't match, if the dtype is unknown, or if the after_band
            doesn't exist in the image.

        Examples
        --------
        >>> # Create a new image with an additional NDVI band
        >>> with_ndvi = image.__apply_add_band(ndvi, name_band='NDVI')
        >>> print(f"Original bands: {list(image.names.keys())}")  # Unchanged
        >>> print(f"New bands: {list(with_ndvi.names.keys())}")  # Has NDVI band
        >>>
        >>> # Add a band after a specific position
        >>> with_tir = image.__apply_add_band(thermal_data, name_band='TIR', after_band='NIR')
        >>>
        >>> # Add with explicit data type and save
        >>> with_dem = image.__apply_add_band(elevation, name_band='DEM', dtype='float32',
        >>>                                 dest_name='with_dem.tif')

        Notes
        -----
        - The spatial dimensions (rows, cols) of the new band must match the current image.
        """
        # Track original state for documentation
        original_band_count = self.nb_bands

        try:
            # Reshape input to match expected format (bands, rows, cols)
            if len(spectral_band.shape) == 2:
                # Convert 2D (rows, cols) to 3D (1, rows, cols)
                spectral_band = spectral_band.reshape((1, spectral_band.shape[0], spectral_band.shape[1]))
            elif len(spectral_band.shape) == 3 and spectral_band.shape[2] == 1:
                # Convert 3D (rows, cols, 1) to 3D (1, rows, cols)
                spectral_band = np2rio(spectral_band)

            # Validate dimensions match
            if (spectral_band.shape[1] != self.shape[0] or
                spectral_band.shape[2] != self.shape[1]):
                raise ValueError(f"Band dimensions ({spectral_band.shape[1]}, {spectral_band.shape[2]}) "
                            f"don't match image dimensions ({self.shape[0]}, {self.shape[1]})")

            # Create copy of names to avoid modifying the original
            names = self.names.copy()

            # Add the new band name to the dictionary
            old_names = names.copy()
            names = add_ordered_key(names, key_name=name_band)

            # If name_band was auto-generated, get the actual name used
            if name_band is None:
                # Find the new key that wasn't in the old names
                new_keys = set(names.keys()) - set(old_names.keys())
                if new_keys:
                    name_band = list(new_keys)[0]

            # Copy metadata
            meta = self.__meta.copy()

            # Create a copy of the original image to avoid modifying it
            image_copy = self.image.copy()

            # Determine insertion position
            if after_band is None:
                # Add to the end (concatenate)
                position_desc = "the end"

                # Determine best dtype if not specified
                if dtype is None:
                    # Auto-determine dtype based on input data and current image
                    input_dtype = str(spectral_band.dtype)
                    current_dtype = self.__meta['dtype']

                    # Simple dtype precedence (higher precision wins)
                    dtype_precedence = {
                        'float64': 1, 'float32': 2, 'int64': 3, 'int32': 4,
                        'uint32': 5, 'int16': 6, 'uint16': 7, 'int8': 8, 'uint8': 9
                    }

                    # Choose the dtype with higher precision
                    if dtype_precedence.get(input_dtype, 999) < dtype_precedence.get(current_dtype, 999):
                        dtype = input_dtype
                    else:
                        dtype = current_dtype

                # Convert to numpy dtype for consistent handling
                np_dtype = np.dtype(dtype)

                # Concatenate the new band at the end
                data = np.concatenate(
                    (image_copy.astype(np_dtype),
                    spectral_band.astype(np_dtype)),
                    axis=0
                )

            else:
                # Convert after_band to string if needed
                after_band = str(after_band)

                # Validate after_band exists
                if after_band not in self.names:
                    available = ", ".join(sorted(list(self.names.keys())))
                    raise ValueError(f"Band '{after_band}' not found for 'after_band' parameter. "
                                f"Available bands are: {available}")

                # Determine position for insertion
                position = self.names[after_band]
                position_desc = f"after band '{after_band}'"

                # Determine best dtype if not specified
                if dtype is None:
                    # Auto-determine dtype based on input data and current image
                    input_dtype = str(spectral_band.dtype)
                    current_dtype = self.__meta['dtype']

                    # Simple dtype precedence (higher precision wins)
                    dtype_precedence = {
                        'float64': 1, 'float32': 2, 'int64': 3, 'int32': 4,
                        'uint32': 5, 'int16': 6, 'uint16': 7, 'int8': 8, 'uint8': 9
                    }

                    # Choose the dtype with higher precision
                    if dtype_precedence.get(input_dtype, 999) < dtype_precedence.get(current_dtype, 999):
                        dtype = input_dtype
                    else:
                        dtype = current_dtype

                # Convert to numpy dtype for consistent handling
                np_dtype = np.dtype(dtype)

                # Insert the new band at the specified position
                data = np.insert(
                    image_copy.astype(np_dtype),
                    position,
                    spectral_band.astype(np_dtype),
                    axis=0
                )

            # Update metadata
            meta['count'] = data.shape[0]
            meta['dtype'] = str(np_dtype)

            # Create new Geoimage with added band
            result = Geoimage(data=data, meta=meta, names=names, georef=self.__georef)

            # Initialize history if enabled
            if hasattr(self, '__history') and self.__history:
                result.activate_history()

                # Copy existing history if available
                if hasattr(self, '__listhistory') and self.__listhistory:
                    result.__listhistory = self.__listhistory.copy()

                # Add band addition entry to history
                now = datetime.datetime.now()
                now_str = now.strftime("%Y-%m-%d %H:%M:%S")

                result.__listhistory.append(
                    f'[{now_str}] - Created as a copy with additional band "{name_band}" at {position_desc}. '
                    f'Original bands: {original_band_count}, New total: {result.nb_bands}. '
                    f'Data type: {dtype}. Original image was not modified.'
                )

            # Save if requested
            if dest_name is not None:
                result.save(dest_name)
                if result.__history:
                    result.__listhistory.append(f'\t New image saved to: {dest_name}')

            return result

        except Exception as e:
            raise RuntimeError(f"Creating image with additional band failed: {str(e)}") from e

    def __generic_filter(self, kernel, inplace = False, dest_name=None):
        """
        Apply a generic 2D convolution filter to the spectral bands of the image.

        This method applies a user-defined kernel (2D array) to each spectral band
        of the image using convolution. It can be used, for example, to perform
        smoothing (mean filter), edge detection, or other custom spatial filtering
        operations.

        Parameters
        ----------
        kernel : numpy.ndarray
            A 2D convolution kernel. Must be a square or rectangular matrix
            (e.g., a Gaussian blur kernel, Sobel operator, etc.).

        inplace : bool, default False
            If False, returns a new Geoimage instance with the filtered data.
            If True, modifies the current image in place.

        dest_name : str, optional
            Path to save the filtered image. If None, the image is not saved.
            Default is None.

        Returns
        -------
        Geoimage or None
            A new Geoimage containing the filtered image if `inplace=False`.
            Returns None if `inplace=True`.

        Raises
        ------
        ValueError
            If the kernel dimensions are invalid or if the image data type
            is not supported.

        Examples
        --------
        >>> # Create an average filter of size 5
        >>> blur_kernel = np.ones((5, 5)) / 25
        >>> # Apply the filter and return a new Geoimage
        >>> Image_filtered = Image.generic_filter(blur_kernel)
        >>> # Apply the filter in place and save the result
        >>> Image.generic_filter(blur_kernel, inplace=True, dest_name="im_filtered.tif")

        Notes
        -----
        - The kernel is applied independently to each spectral band.
        - This function uses convolution; kernel normalization (e.g., sum to 1)
          is the responsibility of the user.
        - For large images, filtering may require significant memory.
        """

        blurred_image = apply_filter(self.numpy_channel_last().astype(np.float64), kernel)
        if inplace:
            self.upload_image(blurred_image,channel_first=False, inplace = True, names=self.get_names())
            if dest_name is not None:
                self.save(dest_name)

            if self.__history is not False:
                now = datetime.datetime.now()
                now_str = now.strftime("%Y-%m-%d %H:%M:%S")

                self.__listhistory.append(
                    f'[{now_str}] - Filtered image with generic kernel.'
                )




        else:
            imf = self.upload_image(blurred_image,channel_first=False, inplace = False, names=self.get_names())
            if dest_name is not None:
                imf.save(dest_name)
            if imf.__history:
                now = datetime.datetime.now()
                now_str = now.strftime("%Y-%m-%d %H:%M:%S")
                imf.__listhistory.append(
                    f'[{now_str}] - Filtered image with generic kernel.'
                )

            return imf

    def filter(self, method="generic", kernel=None,
               sigma=1, size=3, axis=-1, pre_smooth_sigma=None,
               inplace=False, dest_name=None):
        """
        Apply a spatial filter to the Geoimage.

        Parameters
        ----------
        method : str, default="generic"
            Type of filter. Options:
            - "generic" : Generic convolution with a kernel.
            - "gaussian" : Gaussian filter.
            - "median"   : Median filter.
            - "sobel"    : Sobel edge detection (discrete operator).
            - "laplace"  : Laplacian operator (discrete operator).

        kernel : numpy.ndarray, optional
            Convolution kernel (required if mode="generic").

        sigma : float, default=1
            Standard deviation for Gaussian filter (if mode="gaussian").

        size : int, default=3
            Size of the filter window (for median).

        axis : int, default=-1
            Axis along which to compute the Sobel filter (if mode="sobel").
            It is 0 for x, 1 for y. If None, computes gradient magnitude.

        pre_smooth_sigma : float or None, default=None
            If set (e.g., 1.0 or 2.0), a Gaussian filter is applied before Sobel or Laplace,
            useful to reduce noise and simulate larger kernels.

        inplace : bool, default False
            If False, returns a new Geoimage instance with the filtered data.
            If True, modifies the current image in place.

        dest_name : str, optional
            Path to save the filtered image. If None, the image is not saved.
            Default is None.

        Returns
        -------
        Geoimage
            A new filtered Geoimage if inplace=False, otherwise self.

        Raises
        ------
        ValueError
            If `method` is unknown.

        Examples
        --------
        >>> # Create a gaussian with sigma = 8
        >>> imf = image.filter("gaussian", sigma=8)
        >>> # Create a median with size = 7
        >>> imf = image.filter("median", size=7)
        >>> # Create a sobel in x-axis
        >>> imf = image.filter("sobel", axis=0)
        >>> # Create a sobel in y-axis
        >>> imf = image.filter("sobel", axis=1)
        >>> # Create the norm of sobel
        >>> imf = image.filter("sobel")
        >>> # Create a sobel in x-axis with pre_smooth_sigma = 2
        >>> imf = image.filter("sobel", axis=0, pre_smooth_sigma=2)
        >>> # Create a sobel in y-axis with pre_smooth_sigma = 2
        >>> imf = image.filter("sobel", axis=1, pre_smooth_sigma=2)
        >>> # Create the norm of sobel with pre_smooth_sigma = 2
        >>> imf = image.filter("sobel", pre_smooth_sigma=2))
        >>> # Create a laplacian filter
        >>> imf = image.filter("laplace")
        >>> # Create a laplacian filter pre_smooth_sigma = 2
        >>> imf = image.filter("laplace", pre_smooth_sigma=2)

        """
        if method=='generic':
            if inplace:
                self.__generic_filter(kernel, inplace = True, dest_name=dest_name)
            else:
                return  self.__generic_filter(kernel, inplace = False, dest_name=dest_name)
        else:
            def _filter_band(band):
                if method == "gaussian":
                    return gaussian_filter(band.astype(np.float64), sigma=sigma)

                elif method == "median":
                    return median_filter(band, size=size)

                elif method == "laplace":
                    if pre_smooth_sigma is not None:
                        band=gaussian_filter(band.astype(np.float64), sigma=pre_smooth_sigma)
                    return laplace(band.astype(np.float64))

                elif method == "sobel":
                    if pre_smooth_sigma is not None:
                        band=gaussian_filter(band.astype(np.float64), sigma=pre_smooth_sigma)
                    if axis is None:
                        dx = sobel(band.astype(np.float64), axis=0)
                        dy = sobel(band.astype(np.float64), axis=1)
                        return np.hypot(dx, dy)
                    else:
                        return sobel(band.astype(np.float64), axis=axis)

                else:
                    raise ValueError(f"Unknown filter method: {method}")

            arr = self.numpy_channel_last()


            if arr.ndim == 2:
                filtered = _filter_band(arr)
            else:
                filtered = np.stack([_filter_band(arr[:, :, b]) for b in range(arr.shape[2])], axis=2)

            if inplace:
                self.upload_image(filtered, channel_first=False, inplace=True, names=self.get_names())
                if dest_name:
                    self.save(dest_name)
                if self.__history is not False:
                    now = datetime.datetime.now()
                    now_str = now.strftime("%Y-%m-%d %H:%M:%S")

                    self.__listhistory.append(
                        f'[{now_str}] - Filtered image with %s kernel.'%method
                    )
            else:
                new_im = self.upload_image(filtered, channel_first=False, inplace=False, names=self.get_names())
                if dest_name:
                    new_im.save(dest_name)
                if new_im.__history:
                    now = datetime.datetime.now()
                    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
                    new_im.__listhistory.append(
                        f'[{now_str}] - Filtered image with %s kernel.'%method
                    )
                return new_im


    def pca(self, n_components=4, bands=None, random_state=RANDOM_STATE, dest_name=None, standardization=True, nb_points=1000):
        """
        Perform PCA on the image data.

        This method computes a Principal Component Analysis (PCA) on selected image bands.

        Parameters
        ----------
        n_components : int, optional
            Number of components to keep (if None, all components are kept).
            Default is 4.
        bands : list of str or None, optional
            List of bands to use. If None, all bands are used.
            Default is None.
        random_state : int or None, optional
            Random seed for reproducible results. If None, results may vary between runs.
            Default is RANDOM_STATE (defined globally).
        dest_name : str, optional
            Path to save the decomposition. If None, the image is not saved.
            Default is None.
        standardization : bool, optional
            Whether to standardize bands before PCA (recommended).
            Default is True.
        nb_points : int or None, optional
            Number of random points to sample for PCA computation. If None,
            all valid pixels are used (may be slow for large images).
            Default is 1000.

        Returns
        -------
        Geoimage
            A new Geoimage containing the PCA bands.
        tuple
            A tuple (pca_model, scaler) to reuse the transformation on other images.

        Examples
        --------
        >>> # Basic PCA with 5 components
        >>> pca, (pca_model, scaler) = image.pca(n_components=5)
        >>> pca.visu(colorbar=True, cmap='viridis')

        >>> # PCA only on specific bands and save result
        >>> pca, (pca_model, scaler) = image.pca(
        ...     n_components=3, bands=["NIR", "Red", "Green"],
        ...     dest_name="pca.tif")

        >>> # Apply the same model to another image
        >>> other_pca = other_image.transform((pca_model, scaler))

        Notes
        -----
        - Standardization is recommended, especially when bands have different ranges.
        - The returned (pca_model, scaler) can be reused to project other images into the same PCA space.
        """
        from sklearn.decomposition import PCA
        # Initialize random number generator
        rng = np.random.RandomState(random_state)

        # Extract image data as table
        tab = self.numpy_table(bands=bands)

        # Remove nodata pixels
        mask = ~np.any(tab == self.nodata, axis=1)
        tab = tab[mask]

        # Sample points if requested (for speed)
        if nb_points is not None and tab.shape[0] > nb_points:
            idx = rng.randint(tab.shape[0], size=(nb_points,))
            tab = tab[idx, :]

        # Standardize data if requested
        if standardization is True:
            scaler = StandardScaler().fit(tab)
            tab = scaler.transform(tab)
        else:
            scaler = None

        # Apply PCA
        pca_model = PCA(n_components=n_components, random_state=random_state)
        pca_model.fit(tab)

        # Apply the model to create PCA image
        pca = self.transform((pca_model, scaler), bands=bands)

        # Change the names
        new_names={f'PCA_{i}': i for i in range(1, n_components+1)}
        pca.change_names(new_names)

        # Save if requested
        if dest_name is not None:
            pca.save(dest_name)

        if pca.__history is not False:
            now = datetime.datetime.now()
            now_str = now.strftime("%Y-%m-%d %H:%M:%S")
            pca.__listhistory.append(f'[{now_str}] - Created using PCA 'f'with {n_components} components')
            if bands is not None:
                pca.__listhistory.append(f'\t Using bands: {bands}')
            if dest_name is not None:
                pca.__listhistory.append(f'\t Saved to: {dest_name}')

        return pca, (pca_model, scaler)




    def tsne(self, n_components=4, perplexity = 5, bands=None, random_state=RANDOM_STATE, dest_name=None, standardization=True):
        """
        Perform TSNE on the image data.

        This method computes a t-distributed Stochastic Neighbor Embeddings (tSNE) on selected image bands.

        Parameters
        ----------
        n_components : int, optional
            Number of components to keep (if None, all components are kept).
            Default is 4.
        perplexity : int, optional
            Perplexity in TSNE. It is related to the number of nearest neighbors
               that is used in other manifold learning algorithms.
            Default is 4.
        bands : list of str or None, optional
            List of bands to use. If None, all bands are used.
            Default is None.
        random_state : int or None, optional
            Random seed for reproducible results. If None, results may vary between runs.
            Default is RANDOM_STATE (defined globally).
        dest_name : str, optional
            Path to save the decomposition. If None, the image is not saved.
            Default is None.
        standardization : bool, optional
            Whether to standardize bands before PCA (recommended).
            Default is True.

        Returns
        -------
        Geoimage
            A new Geoimage containing the TSNE bands.
        tuple
            A tuple (tsne_model, scaler) to reuse the transformation on other images.

        Examples
        --------
        >>> # Basic TSNE with 5 components
        >>> tsne = image.tsne(n_components=5, perplexity = 5)
        >>> tsne.visu(colorbar=True, cmap='viridis')

        >>> # TSNE only on specific bands and save result
        >>> tsne = image.tsne(
        ...     n_components=3, , perplexity = 3, bands=["NIR", "Red", "Green"],
        ...     dest_name="tsne.tif")


        Notes
        -----
        - Standardization is recommended, especially when bands have different ranges.
        - The returned (tsne_model, scaler) can be reused to project other images into the same PCA space.
        - Unlike PCA, here we apply TSNE to the entire image. The model can not be applied to other ones
        """
        from sklearn.manifold import TSNE
        # Initialize random number generator
        rng = np.random.RandomState(random_state)

        # Extract image data as table
        tab = self.numpy_table(bands=bands)

        # Standardize data if requested
        if standardization is True:
            scaler = StandardScaler().fit(tab)
            tab = scaler.transform(tab)
        else:
            scaler = None

        # Apply TSNE
        tsne_model = TSNE(n_components=n_components, random_state=random_state, perplexity=perplexity)
        tsne = tsne_model.fit_transform(tab)

        outputs = table2image(tsne, self.shape)

        # Create metadata for output image
        meta = self.__meta.copy()

        # Set band count based on output shape
        if len(outputs.shape) == 2:
            meta['count'] = 1
        else:
            meta['count'] = outputs.shape[0]

        # Update data type
        type_str = str(outputs.dtype)
        meta['dtype'] = type_str

        # Create and return new Geoimage with model outputs
        tsne = Geoimage(data=outputs, meta=meta, georef=self.__georef, history=self.__history)

        # Change the names
        new_names={f'TSNE_{i}': i for i in range(1, n_components+1)}
        tsne.change_names(new_names)

        # Save if requested
        if dest_name is not None:
            tsne.save(dest_name)

        if tsne.__history is not False:
            now = datetime.datetime.now()
            now_str = now.strftime("%Y-%m-%d %H:%M:%S")
            tsne.__listhistory.append(f'[{now_str}] - Created using TSNE 'f'with {n_components} components')
            if bands is not None:
                tsne.__listhistory.append(f'\t Using bands: {bands}')
            if dest_name is not None:
                tsne.__listhistory.append(f'\t Saved to: {dest_name}')

        return tsne


    def lle(self, n_components=2, n_neighbors=8, bands=None, nb_points=5000, standardization=True, dest_name=None, random_state=RANDOM_STATE, **kwargs):
        """
        Perform Locally Linear Embedding (LLE) on the image data.

        This method computes a Locally Linear Embedding reduction to unfold the
        manifold on which the pixel values lie. It's particularly useful for
        data with an intrinsic low-dimensional structure that is non-linear.

        Parameters
        ----------
        n_components : int, optional
            The number of coordinates for the manifold (target dimension).
            Default is 2.
        n_neighbors : int, optional
            Number of neighbors to consider for each point. This is a crucial
            parameter for LLE that significantly impacts the result.
            Default is 8.
        bands : list of str or None, optional
            List of bands to use for the computation. If None, all bands are used.
            Default is None.
        nb_points : int or None, optional
            Number of random pixels to sample for the LLE computation. Since LLE
            is computationally intensive, using a sample is highly recommended for
            large images. If None, all valid pixels are used.
            Default is 5000.
        standardization : bool, optional
            Whether to standardize bands before applying LLE (highly recommended).
            Default is True.
        dest_name : str or None, optional
            Path to save the resulting LLE image. If None, the image is not saved.
            Default is None.
        random_state : int or None, optional
            Random seed for pixel sampling and for the ARPACK solver, ensuring
            reproducible results.
            Default is RANDOM_STATE.
        **kwargs : dict, optional
            Additional keyword arguments to pass to the scikit-learn
            `LocallyLinearEmbedding` function, such as `method` ('standard',
            'modified', 'hessian', 'ltsa'), `reg`, or `eigen_solver`.

        Returns
        -------
        Geoimage
            A new Geoimage instance containing the LLE components as bands.
        tuple
            A tuple (lle_model, scaler) containing the fitted LLE model and the
            scaler, which can be used to transform other images.

        Examples
        --------
        >>> # Basic LLE with 2 components
        >>> lle_img, (lle_model, scaler) = image.lle(n_components=2)
        >>> lle_img.visu(cmap='viridis')

        >>> # LLE with more neighbors on specific bands and save the result
        >>> lle_img, _ = image.lle(
        ...     n_components=3,
        ...     n_neighbors=20,
        ...     bands=["NIR", "Red", "Green"],
        ...     dest_name="lle_result.tif"
        ... )

        >>> # Apply the same LLE model to another image
        >>> other_image_lle = other_image.transform((lle_model, scaler))

        Notes
        -----
        - LLE is computationally more expensive than PCA. Using a subset of pixels
          via `nb_points` is strongly advised for large rasters.
        - The choice of `n_neighbors` is critical. A value too small may fail to
          capture the underlying manifold, while a value too large may over-smooth it.
        - The returned (lle_model, scaler) tuple can be used to project other images
          into the same embedding space, assuming they lie on the same manifold.
        """
        from sklearn.manifold import LocallyLinearEmbedding
        # Initialize random number generator
        rng = np.random.RandomState(random_state)

        # Extract image data as table
        tab = self.numpy_table(bands=bands)

        # Remove nodata pixels
        mask = ~np.any(tab == self.nodata, axis=1)
        tab = tab[mask]

        # Sample points if requested (for speed)
        if nb_points is not None and tab.shape[0] > nb_points:
            idx = rng.randint(tab.shape[0], size=(nb_points,))
            tab = tab[idx, :]

        # Standardize data if requested
        if standardization is True:
            scaler = StandardScaler().fit(tab)
            tab = scaler.transform(tab)
        else:
            scaler = None

        # Apply LLE
        lle_model = LocallyLinearEmbedding(n_components=n_components, n_neighbors=n_neighbors, random_state=random_state, **kwargs)
        lle_model.fit(tab)
        # Apply the model to create PCA image
        lle = self.transform((lle_model, scaler), bands=bands)

        # Change the names
        new_names={f'LLE_{i}': i for i in range(1, n_components+1)}
        lle.change_names(new_names)

        # Save if requested
        if dest_name is not None:
            lle.save(dest_name)

        if lle.__history is not False:
            now = datetime.datetime.now()
            now_str = now.strftime("%Y-%m-%d %H:%M:%S")
            lle.__listhistory.append(f'[{now_str}] - Created using LLE 'f'with {n_components} components')
            if bands is not None:
                lle.__listhistory.append(f'\t Using bands: {bands}')
            if dest_name is not None:
                lle.__listhistory.append(f'\t Saved to: {dest_name}')

        return lle, (lle_model, scaler)
