from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
VEC_DIR = BASE_DIR / "Analysis"

def read_insight_vec(file_path):
    df = pd.read_csv(
        file_path,
        sep=r"[,\s]+",
        skiprows=1,
        header=None,
        engine="python"
    )
    df.columns = ["x", "y", "u", "v", "chc", "unc_low", "unc_high"]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

vec_files = sorted(VEC_DIR.glob("*.vec"))

all_U = []
all_V = []

X = None
Y = None

for file_path in vec_files:
    df = read_insight_vec(file_path)
    df.loc[df["chc"] == 0, ["u", "v"]] = np.nan

    x_unique = np.sort(df["x"].dropna().unique())
    y_unique = np.sort(df["y"].dropna().unique())

    if X is None:
        X, Y = np.meshgrid(x_unique, y_unique)

    U = df.pivot(index="y", columns="x", values="u").sort_index().to_numpy()
    V = df.pivot(index="y", columns="x", values="v").sort_index().to_numpy()

    all_U.append(U)
    all_V.append(V)

all_U = np.stack(all_U, axis=0)   # shape = (nt, ny, nx)
all_V = np.stack(all_V, axis=0)

print("all_U shape:", all_U.shape)
print("all_V shape:", all_V.shape)

t = 0

plt.quiver(X, Y, all_U[t], all_V[t])
plt.title(f"Frame {t}")
plt.axis("equal")
plt.show()
