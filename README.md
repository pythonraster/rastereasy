



Introduction
============

**rastereasy** is a Python library for simple manipulation of geospatial raster and vector data (e.g., *.tif, *.jp2, *.shp). The goal is to simplify geospatial workflows by offering tools for reading and processing raster and vector files, resampling, cropping, reprojecting, stacking, filtering, etc of raster images, easy visualizations such as color composites and spectral plots, use (train / apply) some classical Machine Learning algorithms on images, provide some tools for late fusion of classifications (Dempster-Shafer), ...

The main class, `Geoimage`, enables to process raster similarly than numpy arrays while keeping and adapting all meta data.

**Documentation**

A complete documentation can be found [here](https://rastereasy.github.io/)


**Example Usage**

Here's a quick example of what you can do with rastereasy:

```python
 import rastereasy

 # Load a georeferenced image
 image = rastereasy.Geoimage("example.tif")

 # Get image information
 image.info()

 # Print value of pixel [100,200]
 print(image[100,200])

 # Create a color composite
 image.colorcomp(['4', '3', '2'])

 # Resample and reproject
 image_resampled = image.resampling(2)
 image_reproject = image.reproject("EPSG:4326")

 # Save the processed image
 image.save("processed_image.tif")


```
# Installation

####  With pip

Install **rastereasy** via `pip` (the easiest method):

```shell
 $ pip install rastereasy
```

#### From source

To install rastereasy from source:

1. Clone the repository from GitHub:
```shell
$ git clone https://github.com/pythonraster/rastereasy.git
```
2. Navigate to the repository's root directory:
```shell
$ cd rastereasy
```
3. Install the package using pip:
```shell
$ pip install .
```

# Interactive Visualization Notes
 ![Spectra visualization](./illus/spectra.png "Spectra visualization")

## Jupyter notebooks

As illustrated, **rastereasy** supports interactive plotting of spectral bands for individual pixels. To enable this functionality in Jupyter Notebook, rastereasy installs some Jupyter extensions in your environment. If it doesn't work, you may need to rebuild jupyter by the command:

```
jupyter lab build
```

## Google Colab

To use the interactive plotting features in Google Colab, a special two-step setup is required.Follow these steps in the exact order. Separating the commands into different cells and restarting the session is **essential**.

### Step 1: Install Libraries

Run the following cell to install rastereasy and the necessary dependencies for interactive widgets.

```
!pip install rastereasy ipympl
from google.colab import output
output.enable_custom_widget_manager()
```

### Step 2: Restart the Runtime

After the installation is complete, you must restart the runtime.

Go to the menu: `Runtime > Restart` runtime (or use the shortcut Ctrl+M).

### Step 3: Run Your Code

After restarting, you can now enable the interactive mode and use the library in a new cell.

```
%matplotlib widget
import rastereasy
```


 <!--
1. Install the required Jupyter extensions: --
 ```
pip install ipympl
```

2. Rebuild JupyterLab:
```
jupyter lab build
``` -->
------

# To do

Check conda installation

Authors
=======
- [Thomas Corpetti](https://tcorpetti.github.io/)
- Pierrick Matelot
- Augustin de la Brosse
- [Candide Lissak](https://clissak.github.io/)

Citation
=======
If you use rastereasy, please cite:

Thomas Corpetti, Pierrick Matelot, Augustin de la Brosse, Candide Lissak
Rastereasy: A Python package for an easy manipulation of remote sensing images
Journal of Open Source Software, submitted, 2025.


## License
This project is licensed under the MIT License – see the [LICENCE](https://github.com/pythonraster/rastereasy/blob/main/LICENCE) file for details.

## Releases

0.2.1
-----
- Added a new boolean test: `image.isnan()` to check for NaN values in an image.

- Renamed `resampling()` to `resample()`.

  - Both functions remain available in this version, but `resampling()` is deprecated and will be removed in a future release.

- Renamed `apply_ML_model()` to `predict()`.

  - Both functions remain available in this version, but apply_ML_model() is deprecated and will be removed in a future release.


0.2.0
-----
This release introduces several new features (custom band names persistence, metadata-only loading, partial image reading, improved lat/lon visualization, and warnings for multi-band stacks) while remaining fully backward compatible.
The version has therefore been bumped from 0.1.4 to 0.2.0.

Here are the main changes

- User-defined band names

  - Band names set by the user via im.change_names are now automatically saved with im.save and reloaded with rastereasy.Geoimage.

- Metadata-only loading

  - You can now load only the metadata without reading the full image using `meta_only=True`: `im = rastereasy.Geoimage('myimage.tif', meta_only=True)`

- Partial image reading (window or area)

  - It is now possible to read a specific part of the image with the area parameter:

  - By indices: `area=((start_row, end_row), (start_col, end_col))`

  - By geographic coordinates: `area=((lon1, lon2), (lat1, lat2))` with `extent='latlon'`


- Warning for multi-band images in files2stack

  - When using rastereasy.files2stack with images containing multiple bands, a warning is displayed to inform the user.

- Minor bugs in visualization for latitude/longitude coordinates has also been fixed.




0.1.4
-----

September 2025.
Add useful functions for ML

0.1.3
-----

September 2025.
Minor corrections in help of functions

0.1.2
-----

September 2025.
Add filters (gaussian, median, laplace, sobel and generic) functions.

0.1.1
-----

June 2025.
Minor bugs related to interactive visualization fixed (works in console and notebooks).


0.1.0
-----

June 2025.
First release, version 0.1.0
