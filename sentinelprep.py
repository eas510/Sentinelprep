import os
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from sklearn.decomposition import PCA
from tqdm import tqdm
import shutil

_LONGNAME_RE = re.compile(
    r"_((?:B\d{2})|AOT|SCL|WVP|TCI)(?:_\d{2}m)?\.jp2$",
    re.IGNORECASE,
)

def _win_unc(p: str | Path) -> str:
    p = Path(p).resolve()
    return f"\\\\?\\{p}" if os.name == "nt" else str(p)


__all__: List[str] = [
    "rename_bands",
    "find_band_files",
    "read_sentinel2",
    "scale_reflectance",
    "pca_reduce",
    "save_geotiff",
]

_BAND_RE = re.compile(r"_B(\d{2})_\d{2}m\.jp2$", flags=re.IGNORECASE)

def rename_bands(root_dir: str | os.PathLike = "data", overwrite: bool = False) -> int:
    """Rename Sentinel‑2 JP2 filenames to short numeric names."""

    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"{root} is not a directory")

    renamed = 0
    for jp2 in root.rglob("*.jp2"):
        if "_" not in jp2.stem:
            continue

        m = _LONGNAME_RE.search(jp2.name)
        if not m:  # skip the unexpected pattern
            continue

        token = m.group(1).upper()          # B02 / AOT
        short = token[1:] if token.startswith("B") else token
        target = jp2.with_name(f"{short}.jp2")

        if target.exists():
            if not overwrite:
                continue
            target.unlink()

        shutil.move(str(jp2), str(target))
        renamed += 1

    return renamed

def _iter_jp2(root: Path, resolution: int):
    """Yield *.jp2 under ``root/data`` (any depth). Resolution filter via suffix."""
    data_dir = root / "data"
    search_dir = data_dir if data_dir.is_dir() else root  # fallback to root
    yield from search_dir.rglob("*.jp2")


def find_band_files(
    root_dir: str | os.PathLike,
    bands: Sequence[str] = ("02", "03", "04", "08"),
    resolution: int = 10,
) -> Dict[str, Path]:
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"{root} is not a directory")

    located: Dict[str, Path] = {}
    suffix = f"_{resolution}m.jp2"

    for jp2 in _iter_jp2(root, resolution):

        short = jp2.stem.upper()  # '02' / 'AOT'
        if short in [b.upper() for b in bands]:
            located[short[-2:]] = jp2  # save '02'
            continue
        if not jp2.name.endswith(f"_{resolution}m.jp2"):
            continue
        m = _BAND_RE.search(jp2.name)
        if m and (bid := m.group(1)) in bands:
            located[bid] = jp2

    missing = set(bands) - located.keys()
    if missing:
        raise RuntimeError(f"Missing bands {sorted(missing)} at {resolution} m in {root}")
    return located


def read_sentinel2(
    source: str | os.PathLike | Dict[str, os.PathLike] | Sequence[str | os.PathLike],
    bands: Sequence[str] | None = None,
    resolution: int = 10,
    resampling: Resampling = Resampling.nearest,
    dst_crs: str | CRS | None = None,
    show_progress: bool = True,
):
    """Read Sentinel-2 bands into a stacked array + Rasterio profile."""
    # 1) resolve band → path mapping
    if isinstance(source, (str, os.PathLike)):
        if bands is None:
            raise ValueError("'bands' must be given when source is a directory")
        path_map = find_band_files(source, bands=bands, resolution=resolution)
        ordered = list(bands)
    elif isinstance(source, dict):
        path_map = {str(k): Path(v) for k, v in source.items()}
        ordered = list(path_map)
    else:  # sequence of paths
        paths = [Path(p) for p in source]
        ordered = [re.search(r"B?(\\d{2})", p.stem).group(1) for p in paths]
        path_map = dict(zip(ordered, paths))

    # 2) build reference grid
    ref_path = path_map[ordered[0]]
    with rasterio.open(_win_unc(ref_path)) as ref:
        dst_crs = CRS.from_user_input(dst_crs) if dst_crs else ref.crs
        dst_transform, width, height = calculate_default_transform(
            ref.crs, dst_crs, ref.width, ref.height, *ref.bounds, resolution=resolution
        )

    profile = {
        "driver": "GTiff",
        "dtype": np.float32,
        "count": len(ordered),
        "height": height,
        "width": width,
        "crs": dst_crs,
        "transform": dst_transform,
        "compress": "lzw",
    }

    # 3) reproject & stack
    stack = np.empty((len(ordered), height, width), dtype=np.float32)
    iterator = tqdm(ordered, desc="Reading bands", unit="band") if show_progress else ordered
    for i, bid in enumerate(iterator):
        with rasterio.open(_win_unc(path_map[bid])) as src:
            reproject(
                rasterio.band(src, 1),
                stack[i],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=resampling,
            )

    return stack, profile

def scale_reflectance(
    dn_array: np.ndarray,
    scale_factor: float = 1 / 10000.0,
    clip: Tuple[float, float] | None = (0.0, 1.0),
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    refl = dn_array.astype(np.float32) * scale_factor
    return np.clip(refl, *clip).astype(dtype, copy=False) if clip else refl.astype(dtype, copy=False)


# use PCA to reduce the dimensionality
def pca_reduce(
    data: np.ndarray,
    n_components: int = 3,
    whiten: bool = False,
    model: PCA | None = None,
    feature_axis: int = 0,
):
    """
    Apply PCA on an image cube or a 2-D table.

    Returns the reduced np.ndarray
        If the input data is 3D, shape (n_components, H, W)
        If the input data is 2D, shape (samples, n_components)
    """
    # reshape to (pixels, bands)
    if data.ndim == 3:
        if feature_axis != 0:
            data = np.moveaxis(data, feature_axis, 0)
        bands, h, w = data.shape
        flat = data.reshape(bands, -1).T          # (pixels, bands)
    else:               # already 2-D
        flat = data
        h = w = None

    pca = model or PCA(n_components=n_components, whiten=whiten, svd_solver="full")
    reduced_flat = pca.fit_transform(flat) if model is None else pca.transform(flat)

    #  reshape back
    if data.ndim == 3:
        reduced = reduced_flat.T.reshape(n_components, h, w)  # (C, H, W)
    else:
        reduced = reduced_flat                                # (samples, C)

    return reduced.astype(np.float32), pca

def save_geotiff(
    array: np.ndarray,
    profile: dict,
    out_name: str,
    out_dir: str | os.PathLike = "result",
) -> Path:
    """
    save array as GeoTIFF and return its Path.
    The input data can be both 2D and 3D
    2D: (rows, cols),written as single‑band TIFF
    3D: (bands, rows, cols)
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    dst = out_path / out_name

    arr = array if array.ndim == 3 else array[np.newaxis, ...]

    prof = profile.copy()
    prof.update(count=arr.shape[0], dtype=arr.dtype)

    if "crs" not in prof or prof["crs"] is None:
        raise ValueError("profile['crs'] must be set before saving GeoTIFF")
    if isinstance(prof["crs"], (str, int)):
        prof["crs"] = CRS.from_user_input(prof["crs"])

    with rasterio.open(dst, "w", **prof) as ds:
        ds.write(arr)

    return dst