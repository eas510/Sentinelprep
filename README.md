# Sentinelprep
Sentinelprep is a Python library for Sentinel-2 L2A imagery designed to simplify raw-data pre-processing. It provides convenient TIFF image reading and writing, converts DN values to reflectance, and performs PCA dimensionality reduction on user-selected bands.

# Install dependencies
Use the environment.yml
<code>`conda env create -f environment.yaml`</code>

# Features and use cases
1. Band File Management (rename_bands, find_band_files)
These two functions locate the original JP2 files. rename_bands shortens the long ESA filenames to simple band or quality-layer codes such as 02.jp2 or AOT.jp2. find_band_files then searches for the requested bands at the chosen spatial resolution. If any file is missing, it raises an error.
'''python
from s2utils import rename_bands

n = rename_bands("S2_L2A_SAMPLE")
print(f"Renamed {n} JP2 files")'''
'''python
from s2utils import find_band_files
paths = find_band_files(
    "S2_L2A_SAMPLE",             # product root
    bands=("02", "03", "04", "08"),  # Blue, Green, Red, NIR
    resolution=10                 # 10‑metre grid
)

for bid, p in paths.items():
    print(bid, "→", p.name)'''


3. High-performance Reading Reprojection (read_sentinel2)
This function reads several JP2 band files together, reprojects them to the same CRS, and resamples them to the target resolution. It returns a three-dimensional numpy.ndarray with aligned bands and Rasterio profile.

4. Reflectance Scaling (scale_reflectance)
With the official factor 0.0001, this function converts raw DN values to surface reflectance. 

5. PCA Dimensionality Reduction (pca_reduce)
This function applies Principal Component Analysis to any set of bands, producing the first n principal components and returning the fitted sklearn.PCA model.

6. GeoTIFF Writer (save_geotiff)
save_geotiff writes a 2-D or 3-D array to a compressed GeoTIFF. 
