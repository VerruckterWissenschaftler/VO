import pandas as pd
import numpy as np
import ast


class DavisCsvReader:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.df = pd.read_csv(csv_file)
        self._iter = None

    def __iter__(self):
        self._iter = self.df.iterrows()
        return self

    def __next__(self):
        _, row = next(self._iter)
        return self._parse_row(row)

    def iter_frames(self):
        for _, row in self.df.iterrows():
            yield self._parse_row(row)

    def _parse_row(self, row):
        time = row["Time"]
        height = int(row["height"])
        width = int(row["width"])
        encoding = str(row["encoding"]).lower()
        data_str = str(row["data"]).strip()

        if encoding not in ("mono8", "8uc1"):
            raise ValueError(f"Unexpected encoding: {encoding}")

        if data_str.startswith("b'") or data_str.startswith('b"'):
            img_bytes = ast.literal_eval(data_str)
            img_flat = np.frombuffer(img_bytes, dtype=np.uint8)
        else:
            raise ValueError("Data field does not contain a valid byte string")
        if img_flat.size != height * width:
            raise ValueError(
                f"Size mismatch: got {img_flat.size}, expected {height * width}"
            )

        img = img_flat.reshape((height, width))
        return {"time": time, "image": img, "height": height, "width": width}

