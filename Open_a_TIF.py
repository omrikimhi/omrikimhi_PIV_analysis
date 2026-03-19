
# For only open and display a single .tif image
###############################################
from pathlib import Path
import os
from skimage import io
import matplotlib.pyplot as plt

from dotenv import find_dotenv, load_dotenv

# Base folder of the run (the folder that contains RawData/)
BASE_DIR = Path(__file__).resolve().parent   # התיקייה של data_analist_PIV.py


def resolve_raw_dir(base_dir):
    load_dotenv(find_dotenv(usecwd=True))

    raw_dir_value = os.getenv("PIV_RAW_DIR")
    if raw_dir_value:
        raw_dir = Path(raw_dir_value).expanduser()
        if not raw_dir.is_absolute():
            raw_dir = base_dir / raw_dir
        return raw_dir.resolve()

    return (base_dir / "RawData").resolve()


RAW_DIR = resolve_raw_dir(BASE_DIR)

if not RAW_DIR.exists():
	raise FileNotFoundError(f"RAW_DIR does not exist: {RAW_DIR}")

image_candidates = sorted(RAW_DIR.glob("*.tif")) + sorted(RAW_DIR.glob("*.tiff"))
if not image_candidates:
	raise RuntimeError(f"No .tif or .tiff files found in {RAW_DIR}")

# Pick the first available image from RawData
img = io.imread(image_candidates[0])

plt.imshow(img, cmap="gray")
plt.axis("off")
plt.show()