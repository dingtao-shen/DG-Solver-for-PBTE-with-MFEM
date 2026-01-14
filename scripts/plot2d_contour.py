import argparse
import re
from pathlib import Path

import numpy as np


def read_slice(path: Path):
    lines = path.read_text().splitlines()
    if len(lines) < 3:
        raise ValueError("File too short")

    nx = ny = None
    m = re.match(r"^\s*#\s*nx\s+(\d+)\s+ny\s+(\d+)\s*$", lines[0])
    if m:
        nx = int(m.group(1))
        ny = int(m.group(2))
        data = np.loadtxt(path, skiprows=2)
    else:
        data = np.loadtxt(path, skiprows=1)

    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError("Expected columns: x y T")

    x = data[:, 0]
    y = data[:, 1]
    t = data[:, 2]

    if nx is None or ny is None:
        raise ValueError("Missing nx/ny header. Re-run with the C++ sampler header enabled.")

    if x.size != nx * ny:
        raise ValueError(f"Data size mismatch: got {x.size}, expected nx*ny={nx*ny}")

    # The writer loops j (y) outer, i (x) inner -> row-major on y.
    X = x.reshape(ny, nx)
    Y = y.reshape(ny, nx)
    T = t.reshape(ny, nx)
    return X, Y, T


def main():
    ap = argparse.ArgumentParser(description="Plot contour from sampled (x,y,T) slice output.")
    ap.add_argument("input", type=Path, help="Path to sampled text file (e.g. output/2D/results/T_slice.txt)")
    ap.add_argument("--levels", type=int, default=21, help="Number of contour levels")
    ap.add_argument("--cmap", type=str, default="plasma", help="Matplotlib colormap")
    ap.add_argument("--output", type=Path, default=None, help="Optional output image path (png/pdf)")
    ap.add_argument("--show", action="store_true", help="Show the plot window")
    ap.add_argument("--stats", action="store_true", help="Print min/max stats and exit (no plotting required)")
    args = ap.parse_args()

    X, Y, T = read_slice(args.input)

    if args.stats:
        print(f"nx={X.shape[1]} ny={X.shape[0]}")
        print(f"T_min={np.nanmin(T)} T_max={np.nanmax(T)}")
        return

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as ex:
        raise SystemExit(
            "matplotlib is required for plotting. Install it (e.g. `pip install matplotlib`) "
            "or run with `--stats` to verify parsing without plotting."
        ) from ex

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")

    vmin = np.nanmin(T)
    vmax = np.nanmax(T)
    levels = np.linspace(vmin, vmax, args.levels)
    c = ax.contourf(X, Y, T, levels=levels, cmap=args.cmap)
    fig.colorbar(c, ax=ax, shrink=0.85)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.output)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()

