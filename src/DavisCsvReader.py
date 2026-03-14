import os
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm


class DavisCsvReader:
    """
    Reads DAVIS event-camera image CSV files.

    On first use the raw byte-string data is decoded and written to
    ``<csv_dir>/images_cache/`` as individual ``.npy`` files plus a
    lightweight ``metadata.csv`` (no data column).  Subsequent runs skip
    the large CSV entirely and load directly from the cache.
    """

    _CACHE_SUBDIR = "images_cache"
    _META_FILE    = "metadata.csv"

    def __init__(self, csv_file: str):
        self.csv_file  = csv_file
        csv_dir        = os.path.dirname(os.path.abspath(csv_file))
        self._cache_dir = os.path.join(csv_dir, self._CACHE_SUBDIR)
        meta_path       = os.path.join(self._cache_dir, self._META_FILE)

        if self._cache_valid(meta_path):
            self.df = pd.read_csv(meta_path)
            print(f"[DavisCsvReader] Loaded {len(self.df)} cached images from {self._cache_dir}")
        else:
            self._build_cache(meta_path)

        self._iter = None

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_valid(self, meta_path: str) -> bool:
        if not os.path.isfile(meta_path):
            return False
        meta = pd.read_csv(meta_path)
        n = len(meta)
        if n == 0:
            return False
        # Spot-check: first and last npy files must exist
        return (os.path.isfile(os.path.join(self._cache_dir, "0.npy"))
                and os.path.isfile(os.path.join(self._cache_dir, f"{n - 1}.npy")))

    def _build_cache(self, meta_path: str) -> None:
        os.makedirs(self._cache_dir, exist_ok=True)
        print(f"[DavisCsvReader] Building image cache — reading {self.csv_file} ...")
        full_df = pd.read_csv(self.csv_file)

        for idx, row in tqdm(full_df.iterrows(), total=len(full_df), desc="Caching images"):
            img = self._decode_data(row)
            np.save(os.path.join(self._cache_dir, f"{idx}.npy"), img)

        # Lightweight metadata: drop raw data, add stable cache index column
        meta_df = full_df.drop(columns=["data"]).copy()
        meta_df["_cache_idx"] = range(len(meta_df))
        meta_df.to_csv(meta_path, index=False)

        self.df = meta_df
        print(f"[DavisCsvReader] Cached {len(meta_df)} images to {self._cache_dir}")

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def _decode_data(self, row) -> np.ndarray:
        """Decode the raw byte-string in the 'data' column → 2-D uint8 array."""
        height    = int(row["height"])
        width     = int(row["width"])
        encoding  = str(row["encoding"]).lower()
        data_str  = str(row["data"]).strip()

        if encoding not in ("mono8", "8uc1"):
            raise ValueError(f"Unexpected encoding: {encoding}")

        if data_str.startswith("b'") or data_str.startswith('b"'):
            img_bytes = ast.literal_eval(data_str)
            img_flat  = np.frombuffer(img_bytes, dtype=np.uint8)
        else:
            raise ValueError("Data field does not contain a valid byte string")

        if img_flat.size != height * width:
            raise ValueError(f"Size mismatch: got {img_flat.size}, expected {height * width}")

        return img_flat.reshape((height, width))

    def _parse_row(self, row) -> dict:
        """Load one image — from cache if available."""
        cache_idx = int(row["_cache_idx"])
        img = np.load(os.path.join(self._cache_dir, f"{cache_idx}.npy"))
        return {
            "time":    row["Time"],
            "image":   img,
            "height":  int(row["height"]),
            "width":   int(row["width"]),
        }

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self):
        self._iter = self.df.iterrows()
        return self

    def __next__(self):
        _, row = next(self._iter)
        return self._parse_row(row)

    def iter_frames(self):
        for _, row in self.df.iterrows():
            yield self._parse_row(row)
