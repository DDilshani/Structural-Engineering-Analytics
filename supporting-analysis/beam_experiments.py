import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


def load_beams(path):
    df = pd.read_excel(path, header=None)
    clean = df[~df.iloc[:, 0].astype(str).str.contains("Load", na=False)].reset_index(drop=True)

    names = []
    for i in range(0, clean.shape[1], 3):
        if pd.notna(clean.iloc[0, i]):
            names.append(str(clean.iloc[0, i]))

    data = clean.iloc[1:].reset_index(drop=True)
    result = {}

    for idx, name in enumerate(names):
        load = pd.to_numeric(data.iloc[:, idx*3], errors="coerce")
        disp = pd.to_numeric(data.iloc[:, idx*3+1], errors="coerce")
        mask = load.notna() & disp.notna()

        result[name] = {
            "load": load[mask].to_numpy(),
            "disp": disp[mask].to_numpy(),
        }

    return result


def plot_all(beams):
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, values in beams.items():
        ax.plot(values["disp"], values["load"], label=name)
        i = int(np.argmax(values["load"]))
        print(
            f"{name}: Max Load={values['load'][i]:.4f}, "
            f"Displacement={values['disp'][i]:.4f}"
        )

    ax.set_xlabel("Displacement")
    ax.set_ylabel("Load")
    ax.set_title("Load vs Displacement for Different Beams")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig


def monotonic_segments(beams):
    filtered = {}

    for name, values in beams.items():
        order = np.argsort(values["disp"])
        x = values["disp"][order]
        y = values["load"][order]

        x_keep = [x[0]]
        y_keep = [y[0]]

        for i in range(1, len(x)):
            if y[i] >= y_keep[-1] and x[i] > x_keep[-1]:
                x_keep.append(x[i])
                y_keep.append(y[i])

        filtered[name] = {
            "disp": np.asarray(x_keep),
            "load": np.asarray(y_keep),
        }

    return filtered


def plot_reference_spline(beams):
    points = np.array([
        [0.48, 45.64],
        [1.265, 101.549],
        [1.56, 124.043],
        [4.685, 327.63],
        [5.8076, 375.1062],
    ])

    spline = CubicSpline(points[:, 0], points[:, 1], bc_type="natural")
    xfine = np.linspace(points[:, 0].min(), points[:, 0].max(), 200)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xfine, spline(xfine), label="Interpolated reference curve")

    for name, values in beams.items():
        ax.plot(values["disp"], values["load"], label=name)

    ax.set_xlabel("Displacement (mm)")
    ax.set_ylabel("Load (kN)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("excel")
    args = parser.parse_args()

    beams = load_beams(args.excel)
    plot_all(beams)
    plot_reference_spline(beams)
    plot_all(monotonic_segments(beams))
    plt.show()
