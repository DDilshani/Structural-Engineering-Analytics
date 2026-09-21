import pandas as pd
import matplotlib.pyplot as plt


df = pd.DataFrame({
    "Jacket thickness (mm)": [0, 10, 20, 30, 40],
    "Flexural capacity (kN)": [273, 300, 326, 351, 376],
    "Shear capacity (kN)": [163, 220, 276, 333, 390],
})

fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(
    df["Jacket thickness (mm)"],
    df["Flexural capacity (kN)"],
    label="Flexural capacity",
    marker="o",
)
ax.scatter(
    df["Jacket thickness (mm)"],
    df["Shear capacity (kN)"],
    label="Shear capacity",
    marker="s",
)
ax.set_xlabel("Jacket thickness / (mm)")
ax.set_ylabel("Capacity / (kN)")
ax.set_ylim(0, 450)
ax.grid(True, linestyle="--", linewidth=0.6)
ax.legend()
fig.tight_layout()
plt.show()
