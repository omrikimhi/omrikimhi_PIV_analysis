
# For only open and display a single .tif image
###############################################
from pathlib import Path
from skimage import io
import matplotlib.pyplot as plt

# Base folder of the run (the folder that contains RawData/)
BASE_DIR = Path(__file__).resolve().parent   # התיקייה של data_analist_PIV.py
RAW_DIR  = BASE_DIR / "RawData"

# Pick any .tif from RawData
img = io.imread(RAW_DIR / "test_run_1_001247.T000.D000.P000.H000.LA.tif")

plt.imshow(img, cmap="gray")
plt.axis("off")
plt.show()