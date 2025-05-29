import os
import sys
from pathlib import Path
import unittest

plugin_dir = Path(sys.prefix) / "Library" / "lib" / "gdalplugins"
os.environ["GDAL_DRIVER_PATH"] = str(plugin_dir)


bin_dir = Path(sys.prefix) / "Library" / "bin"
os.add_dll_directory(str(bin_dir))
os.environ["PATH"] = f"{bin_dir};{os.environ['PATH']}"

import numpy as np
import rasterio
import sentinelprep as sp


DATA_DIR = Path(__file__).parent / "data"   # the path of test .jp2 data
BANDS    = ("02", "03", "04", "08")


class TestSentinelPrepRealData(unittest.TestCase):
    """test targeting on real sentinel-2 L2A image product"""

    @classmethod
    def setUpClass(cls):
        if not DATA_DIR.is_dir():
            raise unittest.SkipTest(f"{DATA_DIR} 不存在，请把 JP2 放入 data/")
        print("─ rename_bands →", sp.rename_bands(DATA_DIR, overwrite=True))
        print("─ data/ contains:", [p.name for p in DATA_DIR.glob('*.jp2')])

        # rename long-named files to prevent issues caused by long file paths
        renamed = sp.rename_bands(DATA_DIR, overwrite=False)
        print(f"[rename_bands] {renamed} file(s) renamed.")

        cls.stack, cls.profile = sp.read_sentinel2(
            source=DATA_DIR,
            bands=BANDS,
            resolution=10,
            show_progress=False,
        )

    def test_stack_shape(self):
        bands, rows, cols = self.stack.shape
        self.assertEqual(bands, len(BANDS))
        self.assertGreater(rows, 0)
        self.assertGreater(cols, 0)

    # DN
    def test_scale_reflectance(self):
        refl = sp.scale_reflectance(self.stack)
        self.assertTrue(np.nanmin(refl) >= 0)
        self.assertTrue(np.nanmax(refl) <= 1.2)  # 允许少量 >1 像元

    def test_pca_and_save(self):
        reduced, _ = sp.pca_reduce(self.stack, n_components=3)
        self.assertEqual(reduced.shape[0], 3)

        out_tif = sp.save_geotiff(reduced, self.profile, "pca_real.tif")
        self.assertTrue(out_tif.exists())

        with rasterio.open(out_tif) as ds:
            self.assertEqual(ds.count, 3)
            self.assertEqual(ds.width, self.profile["width"])
            self.assertEqual(ds.height, self.profile["height"])


if __name__ == "__main__":
    unittest.main()
