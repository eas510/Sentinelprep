# Sentinelprep
Sentinelprep is a Python library for Sentinel-2 L2A imagery designed to simplify raw-data pre-processing. It provides convenient TIFF image reading and writing, converts DN values to reflectance, and performs PCA dimensionality reduction on user-selected bands.

# Dependencies and installaton
Sentinelprep relies on the following libraries:
numpy – Handles the number caculations and keeps the pixel data in neat arrays.
rasterio: Gives the Pythonic access to GDAL to read, warp, and write Sentinel‑2 imagery.
scikit‑learn: Provides the PCA implementation used for dimensionality reduction.
tqdm – Adds progress bars to track the progeress.
To install dependencies required by sentinelprep,the fastest way is to use the environment.yml
<code>`conda env create -f environment.yaml`</code>

# Features and Use Cases
1. Band File Management (rename_bands, find_band_files)
These two functions locate the original JP2 files. rename_bands shortens the long ESA filenames to simple band or quality-layer codes such as 02.jp2 or AOT.jp2. find_band_files then searches for the requested bands at the chosen spatial resolution. If any file is missing, it raises an error.
```python
from s2utils import rename_bands
from s2utils import find_band_files
n = rename_bands("S2_L2A_SAMPLE")
print(f"Renamed {n} JP2 files")'''
paths = find_band_files(
    "S2_L2A_SAMPLE",             # product root
    bands=("02", "03", "04", "08"),  # Blue, Green, Red, NIR
    resolution=10                 # 10‑metre grid
)

for bid, p in paths.items():
    print(bid, "→", p.name)'''
```       
2. High-performance Reading Reprojection (read_sentinel2)
This function reads several JP2 band files together, reprojects them to the same CRS, and resamples them to the target resolution. It returns a three-dimensional numpy.ndarray with aligned bands and Rasterio profile.
```python
from s2utils import read_sentinel2

stack, profile = read_sentinel2(
    "S2_L2A_SAMPLE",
    bands=("02", "03", "04", "08"),
    resolution=10,
    dst_crs="EPSG:3857"          # optional reprojection (Web‑Mercator)
)
```     
3. Reflectance Scaling (scale_reflectance)
With the official factor 0.0001, this function converts raw DN values to surface reflectance.
```python
from s2utils import scale_reflectance

reflectance = scale_reflectance(stack)  # values ∈ [0, 1]
```
4. PCA Dimensionality Reduction (pca_reduce)
This function applies Principal Component Analysis to any set of bands, producing the first n principal components and returning the fitted sklearn.PCA model.
```python
from s2utils import pca_reduce
pca_img, pca_model = pca_reduce(reflectance, n_components=3, whiten=True)
print(pca_img.shape)  # (3, rows, cols)
```
5. GeoTIFF Writer (save_geotiff)
save_geotiff writes a 2-D or 3-D array to a compressed GeoTIFF.
```python 
from s2utils import save_geotiff

out = save_geotiff(pca_img, profile, "pca_rgb.tif", out_dir="outputs")
print("Saved:", out)
```
# Future Improvements
To simplify the post-processing of Sentinel-2 data, future releases could add more dimensionality-reduction techniques and export formats (e.g., JPG) to accommodate different tasks.
