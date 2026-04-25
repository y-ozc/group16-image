import cv2
import numpy as np
import pandas as pd
import os

import os

p = "data/train/train/8"

rows = []

for e in os.scandir(p):
    if e.is_file():
        img = cv2.imread(e.path, cv2.IMREAD_GRAYSCALE)
        flat_img = img.astype(np.uint8).flatten()

        row = {
            "id": e.name,
            "label": 0,
            **{f"px_{i}": v for i, v in enumerate(flat_img)}
        }

        rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("8_processed.csv", index=False)


p = "data/train/train/9"

rows = []

for e in os.scandir(p):
    if e.is_file():
        img = cv2.imread(e.path, cv2.IMREAD_GRAYSCALE)
        flat_img = img.astype(np.uint8).flatten()

        row = {
            "id": e.name,
            "label": 3,
            **{f"px_{i}": v for i, v in enumerate(flat_img)}
        }

        rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("9_processed.csv", index=False)
