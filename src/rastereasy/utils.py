import os
from itertools import product
import numpy as np
from affine import Affine
import matplotlib.pyplot as plt

import rasterio as rio
from rasterio import windows
from rasterio.enums import Resampling
from rasterio.transform import xy
from rasterio.warp import transform
from rasterio.transform import rowcol
import math


from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from ipywidgets import Output
from IPython.display import display, clear_output
import IPython
from IPython import get_ipython
import matplotlib
from matplotlib.widgets import Button
import ipywidgets as widgets
import warnings
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('agg')

from scipy import ndimage, signal
import json




def parse_rdc_meta(rdc_path):
    """Reads the .rdc file and returns a dictionary of metadata."""
    with open(rdc_path, 'r') as file:
        data = file.readlines()
    meta = {}
    for line in data:
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta

def get_crs_from_rdc(rdc_meta):
    """
    Creates a CRS object based on the information extracted from the RDC file.

    Args:
        rdc_meta (dict): A dictionary containing metadata from the RDC file.

    Returns:
        CRS: A CRS object defined by rasterio.
    """
    ref_system = rdc_meta.get("ref. system", "").lower()
    if ref_system == "plane":
        # If "plane" system, define a simple local CRS (e.g., UTM zone 33N as an example)
        # You can modify the zone depending on your specific region
        crs = CRS.from_proj4("+proj=utm +zone=33 +datum=WGS84 +units=m +no_defs")
    else:
        # If another CRS is provided, you can directly map it
        # For example, if it is "WGS84", we use EPSG:4326
        if ref_system == "wgs84":
            crs = CRS.from_epsg(4326)
        else:
            # Default to a generic CRS if not recognized
            crs = CRS.from_epsg(4326)  # You can replace this with your preferred CRS
    return crs



def isfile_casesensitive(path):
    if not os.path.isfile(path): return False   # exit early
    directory, filename = os.path.split(path)
    return filename in os.listdir(directory)


def find_rst_and_rdc(base_name):
    """
    Finds the correct file names for .rst and .rdc files, handling case sensitivity.

    Args:
        base_name (str): The base name of the files (without extension).

    Returns:
        tuple: A tuple containing:
            - rst_file (str): The full path to the .rst file, or None if not found.
            - rdc_file (str): The full path to the .rdc file, or None if not found.
    """
    extensions = [".rst", ".Rst", ".RST", ".rdc", ".Rdc", ".RDC"]
    rst_file, rdc_file = None, None

    for ext in extensions:
        full_path = f"{base_name}{ext}"
        if isfile_casesensitive(full_path):
            print(' path is file : ',full_path)
            if ext.lower() == ".rst":
                rst_file = full_path
            elif ext.lower() == ".rdc":
                rdc_file = full_path

    return rst_file, rdc_file

def read_rst_with_rdc(rst_file, rdc_file, target_crs="EPSG:4326"):
    """
    Reads an .rst file with its associated .rdc metadata file, reprojects the data
    if the coordinate system is not standard, and returns the data array and metadata.

    Args:
        rst_file (str): Path to the .rst raster file.
        rdc_file (str): Path to the associated .rdc metadata file.
        target_crs (str): Target CRS for reprojection (default is "EPSG:4326").

    Returns:
        tuple: A tuple containing:
            - data (numpy.ndarray): The raster data array.
            - metadata (dict): Metadata including the CRS, transform, and other info.
    """
    # Step 1: Parse the .rdc file to extract metadata
    def parse_rdc(rdc_path):
        """Reads the .rdc file and returns a dictionary of metadata."""
        with open(rdc_path, 'r') as file:
            data = file.readlines()
        meta = {}
        for line in data:
            if ":" in line:
                key, value = line.split(":", 1)
                meta[key.strip()] = value.strip()
        return meta

    rdc_meta = parse_rdc(rdc_file)

    # Step 2: Extract relevant metadata from the .rdc file
    ref_system = rdc_meta.get("ref. system", "").lower()
    ref_units = rdc_meta.get("ref. units", "").lower()
    min_x = float(rdc_meta.get("min. X", 0))
    max_x = float(rdc_meta.get("max. X", 0))
    min_y = float(rdc_meta.get("min. Y", 0))
    max_y = float(rdc_meta.get("max. Y", 0))
    resolution = float(rdc_meta.get("resolution", 1.0))
    print('ref_system',ref_system)
    print('ref_units',ref_units)

    # Default CRS setup: Assuming "plane" means a local Cartesian system
#    if ref_system == "plane":
    if True:
        rdc_meta = parse_rdc_meta(rdc_file)
        crs = get_crs_from_rdc(rdc_meta)
    else:
        raise ValueError(f"Unsupported or unknown reference system: {ref_system}")

    # Step 3: Read the .rst file
    with rio.open(rst_file) as src:
        # Update metadata to include the inferred CRS and transform
        transform = rio.transform.from_bounds(
            min_x, min_y, max_x, max_y, src.width, src.height
        )
        metadata = src.meta.copy()
        metadata.update({
            "crs": crs,
            "transform": transform
        })
        data = src.read()  # Read the first band

        # Step 4: Reproject if the CRS is not the target CRS
        if crs.to_string() != target_crs:
            transform, width, height = calculate_default_transform(
                crs, target_crs, src.width, src.height, *src.bounds
            )
            reprojected_data = np.empty((height, width), dtype=data.dtype)
            reproject(
                source=data,
                destination=reprojected_data,
                src_transform=transform,
                src_crs=crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=Resampling.nearest
            )
            metadata.update({
                "crs": target_crs,
                "transform": transform,
                "width": width,
                "height": height
            })
            return reprojected_data, metadata

    # If no reprojection was needed, return the original data and metadata
    return data, metadata





def list_fles(path,ext):
    """
    returns a list with sorted files in the folder "path" with extension "ext"
    """

    listfiles = os.listdir(path)

    # Filtrer ceux qui se terminent par l'ext
    files_ext = [f for f in listfiles if f.endswith(ext)]

    # Trie par ordre alphabétique
    files_ext.sort()
    files_ext = [os.path.join(path, f) for f in files_ext]

    return files_ext

def normalize(band,percentile=2):
    if band.dtype !='bool':
        if percentile==0:
            min=np.nanmin(band)
            max=np.nanmax(band)
            return (255.*np.clip((band-min)/(max-min),0.,1.)).astype(np.uint8)
        else:
            min=np.nanpercentile(band,percentile)
            max=np.nanpercentile(band,100.-percentile)
            return (255.*np.clip((band.astype(np.float64)-min)/(max-min).astype(np.float64),0.,1.)).astype(np.uint8)
    else:
        return band


def get_percentile(band,percentile=2):
    if percentile==0:
        return band
    else:
        if band.dtype !='bool':
            min=np.nanpercentile(band,percentile)
            max=np.nanpercentile(band,100.-percentile)
            return (np.clip(band,min,max))
        else:
            return band

def get_tiles_recovery(ds, width=128, height=128,overlap=28):
    """
    get window and transformation information for the
    split of an image ds in snippets of size width x height with an overlap
    """
    nols, nrows = ds.meta['width'], ds.meta['height']
    offsets = product(range(-overlap, nols, width-overlap), range(-overlap, nrows, height-overlap))
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
    for col_off, row_off in  offsets:
        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        transform = windows.transform(window, ds.transform)
        yield window, transform

def split_image_to_tiles(source_name, dest_name, size_row, size_col, overlap=0, name='sequence', verbose=0, name_tile=None):
    """
    Split a georeferenced image into smaller tiles with optional overlap and save the resulting tiles and associated georeferencing information.

    Parameters
    ----------
    source_name : str
        The file path to the input georeferenced image (e.g., a GeoTIFF) that needs to be split.

    dest_name : str
        The destination folder where the tiles and their associated georeferencing information will be saved.

    size_row : int
        The number of rows (row) for each tile. This defines the vertical size of each tile.

    size_col : int
        The number of columns (col) for each tile. This defines the horizontal size of each tile.

    overlap : int, optional, default=0
        The number of pixels to overlap between adjacent tiles. If set to 0, there will be no overlap between tiles.

    name : str, optional, default='sequence'
        The base name for the saved tiles. The tiles will be named sequentially (e.g., 'sequence_1.tif', 'sequence_2.tif', etc.) unless `name_tile` is provided.

    verbose : int, optional, default=0
        If set to 1 or higher, additional information about the process will be printed (e.g., progress messages).

    name_tile : str, optional, default=None
        If specified, the base name for the tiles will be overridden. Each tile will be named according to this value (e.g., 'tile_1.tif', 'tile_2.tif', etc.).

    Returns
    -------
    None
        The function saves the resulting tiles and georeferencing information into the specified destination folder. It does not return any value.

    Examples
    --------
    >>> split_image_to_tiles('input_image.tif', 'tiles_folder', 256, 256)
    >>> # Split an image into 256x256 tiles with no overlap:
    >>>
    >>> split_image_to_tiles('input_image.tif', 'tiles_folder', 256, 256, overlap=50)
    >>> # Split an image into 256x256 tiles with 50 pixels overlap:
    >>>
    >>> split_image_to_tiles('input_image.tif', 'tiles_folder', 128, 128, name='part')
    >>> # Split an image into 128x128 tiles and name the tiles 'part_1', 'part_2', etc.:

    Notes
    -----
    - The function reads a georeferenced image, splits it into smaller tiles of size `size_row` x `size_col`, and saves each tile with its associated georeferencing information.
    - If an overlap is specified, the tiles will include the specified number of pixels from adjacent tiles, ensuring that there is continuity across the tiles when reconstructed.
    """


    if name_tile is None:
        generic_name=os.path.splitext(os.path.split(source_name)[1])[0]
    else:
        generic_name = name_tile
    if name=='sequence':
        output_filename='%s/%s_tiles_{:05d}.tif'%(dest_name,generic_name)
    else:
        output_filename='%s/%s_tiles_{:05d}-{:05d}.tif'%(dest_name,generic_name)
    i=0
    #out_path=os.path.dirname(dest_name)
    with rio.open(source_name) as inds:
        tile_width, tile_height = size_row, size_col
        meta = inds.meta.copy()
        if verbose ==1:
            print('size of the tile : %d x %d'%(meta['width'], meta['height']))
        meta['driver']='GTiff'
        for window, transform in get_tiles_recovery(inds, size_row, size_col,overlap):
            meta['transform'] = transform
            meta['width'], meta['height'] = window.width, window.height
#            outpath = os.path.join(out_path, output_filename.format(int(window.col_off), int(window.row_off)))
#            outpath = output_filename.format(int(window.col_off), int(window.row_off))
            if name == 'sequence':
                outpath = output_filename.format(int(i))
            else:
                outpath = output_filename.format(int(window.col_off), int(window.row_off))

            if ((meta['width']==size_row) and (meta['height']==size_col)):
                i=i+1
                with rio.open(outpath, 'w', **meta) as outds:
                    outds.write(inds.read(window=window))
    return 1

def resampling_source(source_name, final_resolution, dest_name=None, method='cubic_spline', channel_first=True):
    """
    Resamples a GeoTIFF image to a specified resolution.

    Args:
        source_name (str): Name of the input GeoTIFF image.
        final_resolution (float): Desired resolution in meters.
        dest_name (str, optional): Name of the output resampled image. Defaults to None.
        method (str, optional): Resampling method. Defaults to "cubic_spline".
            See details here:
            https://rasterio.readthedocs.io/en/stable/api/rasterio.enums.html#rasterio.enums.Resampling.
        channel_first (bool, optional): If True, outputs an image with shape
            (bands x rows x cols). If False, outputs an image with shape
            (rows x cols x bands). Defaults to True.

    Returns:
            - image (numpy.ndarray): The resampled image.
            - meta (dict): Metadata of the resampled image.
    """
    if method == 'cubic_spline':
        resample_method = Resampling.cubic_spline
    elif method =='nearest':
        resample_method = Resampling.nearest
    elif method =='bilinear':
        resample_method = Resampling.bilinear
    elif method =='cubic':
        resample_method = Resampling.cubic
    elif method =='lanczos':
        resample_method = Resampling.lanczos
    elif method =='average':
        resample_method = Resampling.average
    elif method =='mode':
        resample_method = Resampling.mode
    elif method =='max':
        resample_method = Resampling.max
    elif method =='min':
        resample_method = Resampling.min
    elif method =='med':
        resample_method = Resampling.med
    elif method =='sum':
        resample_method = Resampling.sum
    elif method =='q1':
        resample_method = Resampling.q1
    elif method =='q3':
        resample_method = Resampling.q3
    else:
        raise ValueError("Error : unknown interpolation method %s"%method)


    with rio.open(source_name) as dataset:
        resolution_originale=dataset.meta['transform'][0]
        data_bool=False
        if dataset.meta['dtype'] == 'bool':
            data_bool=True
            resample_method = Resampling.nearest

        #print('Image  : ',source_name,'initial resolution  : ',resolution_originale,' meters')
        coef_multiplication = np.float64(resolution_originale/final_resolution)
        #print('The coef to apply to reach %f meters is %f'%(final_resolution,coef_multiplication))

        data = dataset.read(
            out_shape=(
                dataset.count,
                int(dataset.height * coef_multiplication),
                int(dataset.width * coef_multiplication)
            ),
            resampling=resample_method
        )

        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]),
            (dataset.height / data.shape[-2])
        )
        meta=dataset.meta.copy()
        meta['width']=int(dataset.width * coef_multiplication)
        meta['height']=int(dataset.height * coef_multiplication)
        meta['transform']=transform

    if dest_name is not None:
        folder=os.path.split(dest_name)[0]
        if os.path.exists(folder) is False and folder != '':
            os.makedirs(folder)
        with rio.open(dest_name,'w',**meta) as dst:
            dst.write(data)
    if channel_first is True:
        return data,meta
    else:
        return np.rollaxis(data,0,3),meta






def resample_image_with_resolution(
        src_filename, resolution, method='cubic_spline',
        dest_name=None, channel_first=True):
    """
    Resamples an image to a specified resolution.

    Args:
        src_filename (str): Name of the input GeoTIFF image.
        resolution (float): Desired resolution in meters.
        method (str, optional): Resampling method. Defaults to "cubic_spline".
            See details here:
            https://rasterio.readthedocs.io/en/stable/api/rasterio.enums.html#rasterio.enums.Resampling.
        dest_name (str, optional): Name of the output resampled image. Defaults to None.
        channel_first (bool, optional): If True, outputs an image with shape
            (bands x rows x cols). If False, outputs an image with shape
            (rows x cols x bands). Defaults to True.

    Returns:
        tuple: A tuple containing:
            - image (numpy.ndarray): The resampled image.
            - meta (dict): Metadata of the resampled image.
    """
    # Define resampling method based on input string
    if method == 'cubic_spline':
        resample_method = Resampling.cubic_spline
    elif method == 'nearest':
        resample_method = Resampling.nearest
    elif method == 'bilinear':
        resample_method = Resampling.bilinear
    elif method == 'cubic':
        resample_method = Resampling.cubic
    elif method == 'lanczos':
        resample_method = Resampling.lanczos
    elif method == 'average':
        resample_method = Resampling.average
    elif method == 'mode':
        resample_method = Resampling.mode
    elif method == 'max':
        resample_method = Resampling.max
    elif method == 'min':
        resample_method = Resampling.min
    elif method == 'med':
        resample_method = Resampling.med
    elif method == 'sum':
        resample_method = Resampling.sum
    elif method == 'q1':
        resample_method = Resampling.q1
    elif method == 'q3':
        resample_method = Resampling.q3
    else:
        raise ValueError("Error: unknown interpolation method %s" % method)

    # Ouvrir le fichier source
    with rio.open(src_filename) as src:
        src_array = src.read()
        src_meta = src.meta
        src_transform = src.transform

    new_width = int((src.bounds.right - src.bounds.left) / resolution)
    new_height = int((src.bounds.top - src.bounds.bottom) / resolution)

    new_transform = from_origin(src.bounds.left, src.bounds.top, resolution, resolution)

    # Mettre à jour les métadonnées de destination
    dst_meta = src_meta.copy()
    dst_meta.update({
        'height': new_height,
        'width': new_width,
        'transform': new_transform
    })
    data_bool=False
    if src_array.dtype == 'bool':
        data_bool=True
        print("Traitement d'une image booléenne")

        # Convertir les booléens en entiers pour le rééchantillonnage (0 et 1)
        src_array = src_array.astype(np.uint8)

        # Utiliser le rééchantillonnage 'nearest' pour les booléens
        resample_method = Resampling.nearest

    # Créer un tableau de destination (rééchantillonné)
    dst_array = np.empty((src_meta['count'], new_height, new_width), dtype=src_array.dtype)

    # Rééchantillonnage
    reproject(
        source=src_array,
        destination=dst_array,
        src_transform=src_transform,
        dst_transform=new_transform,
        src_crs=src_meta['crs'],
        dst_crs=src_meta['crs'],
        resampling=resample_method
    )

    # Si les données originales étaient booléennes, reconvertir en booléen
    if data_bool:
        dst_array = dst_array.astype(bool)

    if dest_name is not None:
        folder=os.path.split(dest_name)[0]
        if os.path.exists(folder) is False and folder != '':
            os.makedirs(folder)

        with rio.open(dest_name,'w',**dst_meta) as dst:
            dst.write(dst_array)
    if channel_first is True:
        return dst_array, dst_meta
    else:
        return np.rollaxis(dst_array,0,3),dst_meta




def resampling_image(image, meta, final_resolution, dest_name=None, method='cubic_spline', channel_first=True, names=None):
    """
    Resampling of an image

    - image: the input image as a numpy array
    - meta: metadata of the image (dictionary)
    - final_resolution: the desired resolution (in meters)
    - dest_name: (optional) name of the resampled image file
    - method: resampling method (default: cubic_spline)
    - name: (optional) name of the spectral bands in a dict
    - channel_first (optional, default True): output an image of shape (bands x height x width)
                                              otherwise: (height x width x bands)
    - return: resampled image, updated meta
    """

    # Define resampling method based on input string
    if method == 'cubic_spline':
        resample_method = Resampling.cubic_spline
    elif method == 'nearest':
        resample_method = Resampling.nearest
    elif method == 'bilinear':
        resample_method = Resampling.bilinear
    elif method == 'cubic':
        resample_method = Resampling.cubic
    elif method == 'lanczos':
        resample_method = Resampling.lanczos
    elif method == 'average':
        resample_method = Resampling.average
    elif method == 'mode':
        resample_method = Resampling.mode
    elif method == 'gauss':
        resample_method = Resampling.gauss
    elif method == 'max':
        resample_method = Resampling.max
    elif method == 'min':
        resample_method = Resampling.min
    elif method == 'med':
        resample_method = Resampling.med
    elif method == 'sum':
        resample_method = Resampling.sum
    elif method == 'q1':
        resample_method = Resampling.q1
    elif method == 'q3':
        resample_method = Resampling.q3
    else:
        raise ValueError("Error: unknown interpolation method %s" % method)
    data_bool=False
    if image.dtype == 'bool':
        data_bool=True
        resample_method = Resampling.nearest
        image = image.astype(np.uint8)
    # Get original resolution from metadata
    original_resolution = meta['transform'][0]
    #print('Initial resolution: ', original_resolution, ' meters')

    # Calculate the scaling factor for resampling
    scale_factor = np.float64(original_resolution / final_resolution)
    #print(f'The scaling factor to reach {final_resolution} meters is {scale_factor}')

    # Perform resampling
    if channel_first is False:
        image=np.rollaxis(image,2,0)
    bands, height, width = image.shape
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)

    # Use rasterio's resampling to resize the image
    data = np.empty((bands, new_height, new_width), dtype=image.dtype)
    for i in range(bands):
        data[i] = rio.warp.reproject(
            source=image[i],
            destination=np.empty((new_height, new_width), dtype=image.dtype),
            src_transform=meta['transform'],
            dst_transform=meta['transform'] * meta['transform'].scale(
                (width / new_width),
                (height / new_height)
            ),
            src_crs=meta['crs'],  # Pass CRS from metadata
            dst_crs=meta['crs'],  # Use the same CRS for the destination
            resampling=resample_method
        )[0]

    # Update metadata for the resampled image
    new_transform = meta['transform'] * meta['transform'].scale(
        (width / new_width),
        (height / new_height)
    )

    new_meta = meta.copy()
    new_meta.update({
        'width': new_width,
        'height': new_height,
        'transform': new_transform
    })
    if data_bool:
        data = data.astype(bool)
    # Optionally save the resampled image
    if dest_name is not None:
        folder=os.path.split(dest_name)[0]
        if os.path.exists(folder) is False and folder != '':
            os.makedirs(folder)
        with rio.open(dest_name, 'w', **new_meta) as dst:
            dst.write(data)
            if names is not None:
                dst.update_tags(EXTRA_TAGS=json.dumps(names))

    # Adjust output shape based on channel_first flag
    if channel_first:
        return data, new_meta
    else:
        return np.rollaxis(data, 0, 3), new_meta

def numpy_to_string_list(a, delimiter=', '):
    """Converts a NumPy object into a list of strings, with an optional delimiter.

    Args:
        a: A NumPy object (integer, array of integers or strings).
        delimiter: The delimiter to use between elements of the list (default: ', ').

    Returns:
        A list of strings.
    """
    if isinstance(a, str):
        return [a]
    if isinstance(a, int):
        return [str(a)]

    # On convertit en tableau NumPy si ce n'est pas déjà le cas
    a = np.array(a)

    # On convertit chaque élément en chaîne de caractères et on l'ajoute à une liste
    return [str(item) for item in a]



def latlon_to_pixels(meta, lat, lon):
    """
    Converts latitude/longitude coordinates to pixel coordinates (i, j).

    Arguments:
    - meta: metadata dictionary (from src.meta)
    - lat: latitude
    - lon: longitude

    Returns:
    - pix_i, pix_j: pixel indices (row, column) corresponding to the input coordinates
    """
    try:
        transform_affine = meta['transform']
        crs = meta['crs']

        x, y = transform({'init': 'EPSG:4326'}, crs, [lon], [lat])

        pix_j, pix_i = rowcol(transform_affine, x[0], y[0])
    except Exception as e:
        warnings.warn("Unknow projection system. Can not deal anymore with reprojection. Some function may not work ")
        pix_j, pix_i = 0,0

    return pix_j, pix_i
def pixels_to_latlon(meta, pix_i, pix_j):
    """
    Converts pixel coordinates to latitude/longitude.

    Arguments:
    - meta: metadata dictionary (from src.meta)
    - pix_i: row (i) coordinate of the pixel
    - pix_j: column (j) coordinate of the pixel

    Returns:
    - lat, lon: corresponding latitude and longitude
    """
    try:
        transform_affine = meta['transform']
        crs = meta['crs']

        x, y = xy(transform_affine, pix_i, pix_j)

        lon, lat = transform(crs, {'init': 'EPSG:4326'}, [x], [y])
        lon=lon[0]
        lat=lat[0]

    except Exception as e:
        warnings.warn("Unknow projection system. Can not deal anymore with reprojection. Some function may not work ")

        lon, lat = 0.,0.


    return lat, lon

def reindex_dictionary(dictionary, keys_to_remove):
    """
    Reindexes a dictionary after removing specified keys.

    Args:
        dictionary: The dictionary to modify.
        keys_to_remove: A list of keys to remove.

    Returns:
        A new dictionary with the removed keys and reindexed values.
    """

    new_dict = {}
    index = 1
    for key, value in dictionary.items():
        if key not in keys_to_remove:
            new_dict[key] = index
            index += 1
    return new_dict

def reindex_dictionary_keep_order(dictionary, keys_to_keep):
    """
    Reindexes a dictionary keeping only specified keys and preserving their order.

    Args:
        dictionary: The dictionary to modify.
        keys_to_keep: A list of keys to keep.

    Returns:
        A new dictionary with the specified keys and reindexed values.
    """

    new_dict = {}
    index = 1
    for key in keys_to_keep:
        if key in dictionary:
            new_dict[key] = index
            index += 1
    return new_dict

def reorder_dict_by_values(dictionary):
    """
    Reorders a dictionary by its values in ascending order.

    Args:
        dictionary: The dictionary to reorder.

    Returns:
        A new dictionary with items ordered by their values.
    """

    return dict(sorted(dictionary.items(), key=lambda item: item[1]))

def lowest_value_dict(dictionary):
    """
    returns the lowest in a dictionary
    """
    if len(dictionary)==0:
        return 1
    else:
        return dict(sorted(dictionary.items(), key=lambda item: item[1]))[list(dict(sorted(dictionary.items(), key=lambda item: item[1])).keys())[0]]


def has_duplicates(tab):
  """Verify if a table has similar values

  Args:
    tab: the table

  Returns:
    True if it contains similar values, False else
  """

  return len(set(tab)) != len(tab)

def split_keys(names_dict, key):
    """
    Returns two lists of keys: one with keys before a given key and one with keys after it.

    Parameters
    ----------
    names_dict : dict
        Dictionary containing keys and their values.
    key : str
        The key used to split the lists of keys before and after.

    Returns
    -------
    list, list
        Two lists: one with keys before the given key and one with keys after.
    """
    keys = list(names_dict.keys())  # Get the list of keys
    if key not in keys:
        raise ValueError("The provided key is not in the dictionary.")

    index = keys.index(key)  # Find the index of the given key
    before_keys = keys[:index+1]  # Keys before the given key
    after_keys = keys[index + 1:]  # Keys after the given key

    return before_keys, after_keys

def initialize_dict(n):
    """
    create a dictionnary such as
    {"1":1,"2":2,...,"n":n}
    """

    names={}
    for i in range(n):
        names[str(i+1)]=i+1
    return names


def add_ordered_key(dictionary_input, key_name=None):
    """
    Adds a key to the end of an ordered dictionary, following specific rules.

    Args:
        dictionary: The ordered dictionary to modify.
        key_name: The name of the new key (optional).

    Returns:
        The modified dictionary with the new key.
    """
    dictionary = dictionary_input.copy()
    # If no key name is provided, generate one automatically
    if key_name is None:
        # Check if at least one key contains a number
        contains_numbers = any(any(char.isdigit() for char in key) for key in dictionary)

        if contains_numbers:
            # Extract numeric parts from keys and find the max
            numeric_parts = [int(key) for key in dictionary if all(char.isdigit() for char in key)]
            max_numeric_key = max(numeric_parts) if numeric_parts else 0  # Handle case where no numeric part
            if max_numeric_key==0:
                key_name = str(len(dictionary) + 1)
            else:
                key_name = str(max_numeric_key + 1)  # Convert to string directly
        else:
            # No key contains a number, simply use the index
            key_name = str(len(dictionary) + 1)  # Convert to string directly

    # Add the new key with the next value
    new_value = max(dictionary.values(), default=0) + 1
    dictionary[key_name] = new_value

    return dictionary

def check_dict(d):
    """
    Check whether a dictionary satisfies the following conditions:

    1. All keys are unique.
    2. The set of values is exactly {1, 2, ..., N},
       where N is the number of keys.

    Parameters
    ----------
    d : dict
        Dictionary with string keys and integer values.

    Returns
    -------
    bool
        True if the dictionary is valid, False otherwise.

    Examples
    --------
    >>> d1 = {'name1': 1, 'name2': 3, 'name3': 2}
    >>> check_dict(d1)
    True

    >>> d2 = {'name1': 1, 'name2': 3, 'name3': 6}
    >>> check_dict(d2)
    False
    """
    n = len(d)
    errors = []

    # Vérif unicité des clés
    if len(d) != len(set(d.keys())):
        errors.append("Les clés ne sont pas toutes uniques.")

    # Vérif plage des valeurs
    values = set(d.values())
    expected = set(range(1, n+1))

    missing = expected - values
    extra = values - expected

    if missing:
        errors.append(f"Valeurs manquantes : {sorted(missing)}")
    if extra:
        errors.append(f"Valeurs en trop : {sorted(extra)}")

    if errors:
#        return False, errors
        return False
#    return True, ["Dictionnaire valide."]
    return True


def concat_dicts_with_keys(dict1, dict2):
    """
    Concatenates two dictionaries, renaming duplicate keys with a suffix and assigning sequential values.

    Args:
        dict1: The first dictionary.
        dict2: The second dictionary.

    Returns:
        A new dictionary with renamed keys and sequential values.
    """

    # Create an empty dictionary to store the result
    new_dict = {}

    # Iterate over the first dictionary and add keys with '_1' suffix
    for key in dict1:
        new_dict[f"{key}_1"] = 0

    # Iterate over the second dictionary and add keys with '_2' suffix
    for key in dict2:
        new_dict[f"{key}_2"] = 0

    # Assign sequential values to the keys, starting from 1
    for i, key in enumerate(new_dict, start=1):
        new_dict[key] = i

    return new_dict

def has_common_key(dict1, dict2):
    """
    Checks if there is at least one common key between two dictionaries.

    Args:
        dict1 (dict): The first dictionary.
        dict2 (dict): The second dictionary.

    Returns:
        bool: True if there is at least one common key, False otherwise.
    """
    # Convert the keys of both dictionaries to sets and find the intersection.
    # If the intersection is non-empty, it means there is a common key.
    return bool(set(dict1.keys()) & set(dict2.keys()))


def concat_dicts_with_keys_unmodified(dict1, dict2):
    """
    Concatenates two dictionaries, renaming duplicate keys with a suffix and assigning sequential values.

    Args:
        dict1: The first dictionary.
        dict2: The second dictionary.

    Returns:
        A new dictionary with renamed keys and sequential values.
    """

    # Create an empty dictionary to store the result
    new_dict = {}

    # Iterate over the first dictionary and add keys with '_1' suffix
    for key in dict1:
        new_dict[f"{key}"] = 0

    # Iterate over the second dictionary and add keys with '_2' suffix
    for key in dict2:
        new_dict[f"{key}"] = 0

    # Assign sequential values to the keys, starting from 1
    for i, key in enumerate(new_dict, start=1):
        new_dict[key] = i

    return new_dict


def latlon2meters(lon1, lat1, lon2, lat2):
    # Earth's radius
    R = 6378.137

    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)

    #  Haversine
    a = math.sin(dLon / 2) ** 2 + math.cos(math.radians(lon1)) * math.cos(math.radians(lon2)) * math.sin(dLat / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    d = R * c
    return d * 1000

def calculate_bounds(meta):
    """
    Compute boundaries from meta_informations
    Input : meta informations
    Output : bounding bos
    """

    # Extraire la transformation affine et les dimensions de l'image
    transform = meta['transform']
    width = meta['width']
    height = meta['height']

    # Calculer les coordonnées géographiques des quatre coins de l'image
    # Les coins en (0, 0) pour le coin supérieur gauche et (width, height) pour le coin inférieur droit
    left, top = transform * (0, 0)
    right, bottom = transform * (width, height)

    # Créer un objet bounding box avec ces coordonnées
    bounds = rio.coords.BoundingBox(left, bottom, right, top)

    return bounds



def ajust_sizes(im1,im2):
    im1common=im1.copy()
    im2common=im2.copy()
    row1=im1common.shape[0]
    row2=im2common.shape[0]
    col1=im1common.shape[1]
    col2=im2common.shape[1]
    offset_col=col2-col1
    offset_row=row2-row1

    if offset_col != 0 or offset_row !=0:
        lat1,lon1=im2common.pixel2latlon(im2common.shape[0],im2common.shape[1])
        lat2,lon2=im1common.pixel2latlon(im1common.shape[0],im1common.shape[1])
        diff_end=latlon2meters(lat1, lon1, lat2, lon2)
        lat1,lon1=im2common.pixel2latlon(0,0)
        lat2,lon2=im1common.pixel2latlon(0,0)
        diff_deb=latlon2meters(lat1, lon1, lat2, lon2)
        row1=im1common.shape[0]
        row2=im2common.shape[0]
        col1=im1common.shape[1]
        col2=im2common.shape[1]
        if ((row1!= row2) or (col1!= col2)):
            if diff_deb < diff_end:
                im1common.crop(0,np.min((row1,row2)),0,np.min((col1,col2)),inplace=True)
                im2common.crop(0,np.min((row1,row2)),0,np.min((col1,col2)),inplace=True)
            else:
                if row1<=row2:
                    deb_row1=0
                    end_row1=im1common.shape[0]
                    deb_row2=(row2-row1)
                    end_row2=im2common.shape[0]
                else:
                    deb_row2=0
                    end_row2=im2common.shape[0]
                    deb_row1=(row1-row2)
                    end_row1=im1common.shape[0]
                if col1 <= col2:
                    deb_col1=0
                    end_col1=im1common.shape[1]
                    deb_col2=(col2-col1)
                    end_col2=im2common.shape[1]
                else:
                    deb_col2=0
                    end_col2=im2common.shape[1]
                    deb_col1=(col1-col2)
                    end_col1=im1common.shape[1]
                im1common.crop(deb_row1,end_row1,deb_col1,end_col1,inplace=True)
                im2common.crop(deb_row2,end_row2,deb_col2,end_col2,inplace=True)
    return im1common,im2common

def is_value_compatible(value, dtype_str):
    """
    Checks if the given value is compatible with the specified data type (provided as a string).

    Parameters:
    - value: the value to be checked
    - dtype_str: the data type as a string (e.g., 'bool', 'uint8', 'int32', etc.)

    Returns:
    - True if the value is compatible with the data type, False otherwise
    """
    # Convert the string to a numpy data type
    try:
        dtype = np.dtype(dtype_str)
    except TypeError:
        raise ValueError(f"The type '{dtype_str}' is not recognized.")

    # Check limits based on the data type
    if np.issubdtype(dtype, np.integer):
        # For integer types, check if the value is an integer and within the type's limits
        if isinstance(value, (int, np.integer)):
            dtype_info = np.iinfo(dtype)
            return dtype_info.min <= value <= dtype_info.max
        else:
            return False  # Value is not an integer and thus incompatible
    elif np.issubdtype(dtype, np.floating):
        # For floating-point types, check if the value is a float and within the type's limits
        if isinstance(value, (float, np.floating)):
            dtype_info = np.finfo(dtype)
            return dtype_info.min <= value <= dtype_info.max
        else:
            return False  # Value is not a float and thus incompatible
    elif np.issubdtype(dtype, np.bool_):
        # Boolean values are limited to True/False or 1/0
        return value in [0, 1, False, True]
    else:
        raise ValueError(f"The type '{dtype_str}' is not supported.")
def adapt_nodata(dtype_str, nodata):
    """
    Adjusts the nodata value if it's not compatible with the specified data type.

    Parameters:
    - dtype_str: the target data type as a string (e.g., 'bool', 'uint8', 'int32', etc.)
    - nodata: the current nodata value

    Returns:
    - The adjusted nodata value, compatible with dtype_str
    """
    # Check if the current nodata value is compatible with the specified data type
    if nodata is None:
        return nodata
    else:
        if is_value_compatible(nodata, dtype_str):
            return nodata  # If compatible, return the existing nodata value as is

        # Define a new nodata value based on the target data type
        dtype = np.dtype(dtype_str)

        if np.issubdtype(dtype, np.integer):
            # Set a safe nodata value for integer types (use min value for signed integers to avoid overflow issues)
            if np.issubdtype(dtype, np.signedinteger):
                new_nodata = np.iinfo(dtype).min
            else:  # For unsigned integer types
                new_nodata = np.iinfo(dtype).max
        elif np.issubdtype(dtype, np.floating):
            # For float types, use NaN (Not a Number) as the default nodata
            new_nodata = np.nan
        elif np.issubdtype(dtype, np.bool_):
            # For boolean, set nodata to 0 (False)
            new_nodata = 0
        else:
            raise ValueError(f"The type '{dtype_str}' is not supported for nodata adaptation.")

        return new_nodata

def extract_colorcomp(im_input, bands=None, percentile=2):
    """Extracts a color composition from an image.

    Args:
        im_input (geoimage): The input image.
        bands (numpy.ndarray, optional): A NumPy array specifying the bands to extract.
            - Can be a list of integer indices (e.g., `[4, 2, 3]`).
            - Can be a list of band names (e.g., `["R", "G", "B"]`). If band names are used, the `names` argument must also be provided.
        percentile (int, optional): The percentile to use for stretching the intensity values (default: 2).

    Returns:
        geoimage: A new geoimage containing the extracted color composition.

    Raises:
        ValueError: If invalid band indices or names are provided.

    **Note:**
    - If band names are used, a corresponding dictionary must be provided in the `names` argument to map band names to indices.
    """
    image = np.rollaxis(im_input.image,0,3)
    if bands is None:
        bands = list(im_input.names.keys())[:3]

    im=np.zeros((image.shape[0],image.shape[1],3)).astype(np.uint8)
    im[:,:,0]=normalize(image[:,:,im_input.names[bands[0]]-1],percentile)
    im[:,:,1]=normalize(image[:,:,im_input.names[bands[1]]-1],percentile)
    im[:,:,2]=normalize(image[:,:,im_input.names[bands[2]]-1],percentile)
    return im



def reset_matplotlib(mode='inline'):
    plt.close('all')
    #plt.clf()

    ipython = get_ipython()
    if ipython is not None:
        try:
            ipython.run_line_magic('matplotlib', mode)
        except Exception as e:
            print(f"Warning: Impossible to run run_line_magic : {e}")


def is_notebook():
    try:
        from IPython import get_ipython
        return 'IPKernelApp' in get_ipython().config
    except:
        return False

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import sys

# You will need to install a library for the Qt backend.
# In your terminal: pip install pyqt5

def is_notebook():
    """Checks if the code is running in a Jupyter-like environment."""
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably a standard Python interpreter

class SpectraCollector:
    """
    A class to manage an interactive Matplotlib window for collecting
    spectra by clicking on an image.
    """
    def __init__(self, im, imc, figsize=(15, 5), names=None, title_im="Click on the image", 
                 title_spectra="Spectra", xlabel="Bands", ylabel="Value", plot_legend=False,
                 offset_i=0, offset_j=0):
        
        # --- Backend Configuration ---
        # Force an interactive backend if not in a notebook.
        # Jupyter's 'widget' backend must be enabled beforehand with %matplotlib widget.
        if is_notebook():
            ipython = get_ipython()
            ipython.run_line_magic('matplotlib', 'inline')
            ipython.run_line_magic('matplotlib', 'widget')
        else:
            plt.ion()  # Interactive mode for the standard shell

        plt.close('all')

        if not is_notebook():
            try:
                matplotlib.use('Qt5Agg')
            except ImportError:
                print("Warning: PyQt5 is not installed. Interactivity might not work.")
                print("Please install it with: pip install pyqt5")

        # --- Initialize variables ---
        self.im = im
        self.imc = imc
        self.names = names
        self.plot_legend = plot_legend
        self.offset_i = offset_i
        self.offset_j = offset_j
        self.end_collect = False
        self.series_spectrales = []
        self.val_i = []
        self.val_j = []
        
        # --- Create the figure ---
        plt.close('all')
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=figsize)
        
        self.ax1.imshow(self.imc, extent=[offset_j, self.imc.shape[1] + offset_j, self.imc.shape[0] + offset_i, offset_i])
        self.ax1.set_title(title_im)
        self.ax2.set_title(title_spectra)
        self.ax2.set_xlabel(xlabel)
        self.ax2.set_ylabel(ylabel)

        # --- Add widgets ---
        self.close_button = Button(plt.axes([0.04, 0.92, 0.1, 0.05]), 'Finish')
        self.close_button.on_clicked(self.close_figure)

        # --- Connect the click event ---
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        # --- Message for the user (replaces ipywidgets) ---
        print("➡️ Interactive window is open. Please click on the image to collect data.")
        
        # Show the figure and block execution until it is closed.
        plt.show()

    def onclick(self, event):
        # If the click is outside the image axes, close the figure.
        if event.inaxes != self.ax1:
            self.close_figure()
            return
        else:
            print('inside')

        i, j = int(event.ydata - self.offset_i), int(event.xdata - self.offset_j)
        self.val_i.append(i + self.offset_i)
        self.val_j.append(j + self.offset_j)

        serie_spectrale = self.im[i, j, :]
        self.series_spectrales.append(serie_spectrale)

        self.update_spectra_plot()

    def update_spectra_plot(self):
        self.ax2.clear() # Clear axes to redraw
        for idx, spectre in enumerate(self.series_spectrales):
            label = f'Series {idx} ({self.val_i[idx]},{self.val_j[idx]})'
            if self.plot_legend:
                self.ax2.plot(spectre, label=label)
                self.ax2.legend()
            else:
                self.ax2.plot(spectre)
        
        if self.names is not None:
            self.ax2.set_xticks(range(len(self.names)))
            self.ax2.set_xticklabels(self.names, rotation=45)
            
        self.fig.canvas.draw_idle()

    def close_figure(self, event=None):
        # Check if the window still exists before trying to close it
        if self.fig.canvas.manager and self.fig.canvas.manager.window:
            self.fig.canvas.mpl_disconnect(self.cid)
            plt.close(self.fig)
            print("✅ Collection finished. Data is now available.")
            self.end_collect = True

    
    
    def get_data(self):
        """Returns the collected data."""
        return self.series_spectrales, self.val_i, self.val_j, self.end_collect
        
def plot_clic_spectra2(im, imc, figsize=(15, 5),
                      plot_legend=False,
                      names=None,
                      title_im="Original image (click outside or finish button to stop)",
                      title_spectra="Spectra",
                      xlabel="Bands",
                      ylabel="Value",
                      callback=None,
                      offset_i=0,
                      offset_j = 0):
    # 'im' and 'imc' are your numpy image arrays
    collector = SpectraCollector(im, imc,
                                names=names,
                                title_im=title_im,
                                title_spectra=title_spectra,
                                xlabel=xlabel,
                                ylabel=ylabel,
                                plot_legend=plot_legend,
                                offset_i=offset_i,
                                offset_j=offset_j)

    spectra, i_coords, j_coords, end_collect = collector.get_data()

    print(f"Collected {len(spectra)} spectra.")
    return spectra, i_coords, j_coords, end_collect

def plot_clic_spectra(im, imc, figsize=(15, 5),
                      plot_legend=False,
                      names=None,
                      title_im="Original image (click outside or finish button to stop)",
                      title_spectra="Spectra",
                      xlabel="Bands",
                      ylabel="Value",
                      callback=None,
                      offset_i=0,
                      offset_j = 0,
                      colab=False):
    if is_notebook():
        print('nootebook')
        ipython = get_ipython()
        ipython.run_line_magic('matplotlib', 'inline')
        ipython.run_line_magic('matplotlib', 'widget')
    else:
        plt.ion()  # Interactive mode for the standard shell
        if colab is False:
            matplotlib.use('Qt5Agg')
        else:
            ipython = get_ipython()
            ipython.run_line_magic('matplotlib', 'widget')
            
    plt.close('all')

    series_spectrales = []
    val_i = []
    val_j = []

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle("")
    if colab is False:
        ax1.imshow(imc, extent=[offset_j,imc.shape[1]+offset_j,imc.shape[0]+offset_i,offset_i])
    else:
        ax1.imshow(imc)
    ax1.set_title(title_im)

    close_button = Button(plt.axes([0.04, 0.92, 0.1, 0.05]), 'Finish')  # Adjusted position for top-left
    global end_collect
    end_collect = False

    def close_figure(event=None):
        fig.canvas.mpl_disconnect(cid)
        plt.close(fig)
        ready_label.value = "Click finished. Data is ready."
        end_collect = True
        # Call the callback if provided
        if callback:
            callback(series_spectrales, val_i, val_j)
        reset_matplotlib()
    close_button.on_clicked(close_figure)

    def onclick(event):
        # If the click is outside the image, stop collecting data
        if event.inaxes != ax1:
            close_figure()
            return

        i, j = int(event.ydata-offset_i), int(event.xdata-offset_j)
        val_i.append(i+offset_i)
        val_j.append(j+offset_j)

        # Extract the spectral values at the clicked point
        serie_spectrale = im[i, j, :]  # Extract spectral data
        series_spectrales.append(serie_spectrale)

        # Update the spectrum display
        ax2.clear()
        for indice in range(len(series_spectrales)):
            label = f'Series {indice} ({val_i[indice]},{val_j[indice]})'
            if plot_legend:
                ax2.plot(series_spectrales[indice], label=label)
                ax2.legend()
            else:
                ax2.plot(series_spectrales[indice])
        if names is not None:
            ax2.set_xticks(range(len(names)))
            ax2.set_xticklabels([key for key in names], rotation=45)

        ax2.set_title(title_spectra)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel)
        fig.canvas.draw()

    # Connect the click event
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    # Add a widget to indicate that a click is expected
    ready_label = widgets.Label(value="Click on the image to retrieve spectra.")
    display(ready_label)

    # Display the figure and allow the user to click
    plt.show()
    # Return the data after the figure is closed
    return series_spectrales, val_i, val_j,end_collect

def dict_keys_to_list(d):
    """
    Convert dictionary keys to a list of strings.

    Parameters
    ----------
    d : dict
        The input dictionary.

    Returns
    -------
    list of str
        A list containing the keys of the dictionary as strings.

    Example
    -------
    >>> dict_keys_to_list({'a': 1, 'b': 2, 'c': 3})
    ['a', 'b', 'c']
    """
    return list(map(str, d.keys()))
def match_dimensions(tab1, tab2):
    if tab1.shape == tab2.shape:
        return tab1  # Les dimensions sont identiques, on ne fait rien
    elif tab1.shape[0] == 1 and tab1.shape[1:] == tab2.shape[1:]:
        # Dupliquer la bande unique de tab1 pour correspondre à la dimension N de tab2
        return np.repeat(tab1, tab2.shape[0], axis=0)
    else:
        raise ValueError("Les dimensions des tableaux ne sont pas compatibles.")

def resize_array(array, new_shape):
    old_rows, old_cols = array.shape
    new_rows, new_cols = new_shape

    row_idx = np.linspace(0, old_rows - 1, new_rows)
    col_idx = np.linspace(0, old_cols - 1, new_cols)

    interpolated_rows = np.array([np.interp(col_idx, np.arange(old_cols), row) for row in array])

    resized_array = np.array([np.interp(row_idx, np.arange(old_rows), col) for col in interpolated_rows.T]).T

    return resized_array


def fusion_2classes(m1,m2):
    """
    Fuse two mass function tables using a specified rule.

    Parameters
    ----------
    m1 : numpy.ndarray
        A N x 3 array representing source 1.
        Columns correspond to m(class 1), m(class 2), and m(class 1 ∪ class 2).

    m2 : numpy.ndarray
        A N x 3 array representing source 2.
        Columns correspond to m(class 1), m(class 2), and m(class 1 ∪ class 2).

    Returns
    -------
    numpy.ndarray
        A N x 3 array containing the fused mass functions:
        m(class 1), m(class 2), and m(class 1 ∪ class 2).

    numpy.ndarray
        A 1D array of length N containing the conflict values for each row.

    Examples
    --------
    >>> fused, conflict = fuse_mass_functions(m1, m2)
    """
    fusion = np.zeros(m1.shape)
    # Hypothesis1, source 1
    H1_1 = m1[:,0]
    # Hypothesis1, source 2
    H1_2 = m2[:, 0]
    # Hypothesis2, source 1
    H2_1 = m1[:,1]
    # Hypothesis2, source 2
    H2_2 = m2[:, 1]
    # 1u2, source 1
    H1U2_1= m1[:,2]
    # 1u2, source 2
    H1U2_2= m2[:,2]
    # Conflict

    conflit =  H1_1*H2_2 + H2_1*H1_2

    fusion[:,0] = (H1_1*H1_2 + H1_1*H1U2_2+H1_2*H1U2_1) / (1 -conflit)
    fusion[:,1] = (H2_1*H2_2 + H2_1*H1U2_2+H2_2*H1U2_1) / (1 -conflit)
    fusion[:,2] = (H1U2_2*H1U2_1) / (1-conflit)
    return fusion, conflit


def fusion_multisource(*args):
    """
    Fuse multiple mass function tables from different sources.

    Parameters
    ----------
    *args : tuple of numpy.ndarray
        Each argument is an N x 3 array representing a source.
        Columns correspond to m(class 1), m(class 2), and m(class 1 ∪ class 2).

    Returns
    -------
    numpy.ndarray
        A N x 3 array containing the fused mass functions:
        m(class 1), m(class 2), and m(class 1 ∪ class 2).

    numpy.ndarray
        A 1D array of length N containing the conflict values for each row.

    Examples
    --------
    >>> fused, conflict = fusion_multisource(m1, m2, m3)
    """

    fusion = np.zeros(args[0].shape)

    fusion,conflit = fusion_2classes(args[0],args[1])

    for i in range(2,len(args)):

        fusion, conflit = fusion_2classes(fusion, args[i])

    return fusion, conflit

def apply_filter2(image, kernel):
    if len(image.shape) == 3:
        result = np.zeros_like(image)
        for i in range(image.shape[2]):
            result[:, :, i] = ndimage.convolve(image[:, :, i], kernel)
        return result
    else:
        # Si l'image est en niveaux de gris
        return ndimage.convolve(image, kernel)

def apply_filter(image, kernel, method="auto"):
    """
    Apply a 2D filter to an image (multi-band or single-band).

    Parameters
    ----------
    image : numpy.ndarray
        Input image (2D or 3D with channels last).
    kernel : numpy.ndarray
        2D filter kernel.
    method : str, optional
        'auto' (default): use direct convolution for small kernels, FFT for large.
        'direct': always use ndimage.convolve.
        'fft': always use signal.fftconvolve.

    Returns
    -------
    numpy.ndarray
        Filtered image, same shape as input.
    """

    def _filter_band(band, kernel, method):
        if method == "direct":
#            return ndimage.convolve(band, kernel, mode="reflect")
            return ndimage.convolve(band, kernel)
        elif method == "fft":
#            return signal.fftconvolve(band, kernel, mode="same")
            return signal.fftconvolve(band, kernel)
        else:  # auto
            if kernel.shape[0] * kernel.shape[1] <= 225:  # e.g. 15x15
#                return ndimage.convolve(band, kernel, mode="reflect")
                return ndimage.convolve(band, kernel)
            else:
#                return signal.fftconvolve(band, kernel, mode="same")
                return signal.fftconvolve(band, kernel)

    if image.ndim == 3:
        result = np.zeros_like(image)
        for i in range(image.shape[2]):
            result[:, :, i] = _filter_band(image[:, :, i], kernel, method)
        return result
    else:
        return _filter_band(image, kernel, method)
