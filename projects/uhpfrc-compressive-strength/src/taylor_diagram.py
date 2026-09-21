import joblib
import numpy as np
import matplotlib.pyplot as plt

from common import load_dataset, split_dataset, PROJECT_ROOT


class TaylorDiagram:
    def __init__(self, refstd, fig=None, rect=111, label="Reference", srange=(0, 1.5)):
        from matplotlib.projections import PolarAxes
        import mpl_toolkits.axisartist.floating_axes as FA
        import mpl_toolkits.axisartist.grid_finder as GF

        self.refstd = refstd
        transform = PolarAxes.PolarTransform()

        rlocs = np.array([0, .2, .4, .6, .7, .8, .9, .95, .99, 1])
        tlocs = np.arccos(rlocs)

        helper = FA.GridHelperCurveLinear(
            transform,
            extremes=(0, np.pi/2, srange[0]*refstd, srange[1]*refstd),
            grid_locator1=GF.FixedLocator(tlocs),
            tick_formatter1=GF.DictFormatter(dict(zip(tlocs, map(str, rlocs)))),
        )

        fig = fig or plt.figure()
        ax = FA.FloatingSubplot(fig, rect, grid_helper=helper)
        fig.add_subplot(ax)

        ax.axis["top"].set_axis_direction("bottom")
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].label.set_text("Correlation")
        ax.axis["left"].set_axis_direction("bottom")
        ax.axis["left"].label.set_text("Standard deviation")
        ax.axis["right"].set_axis_direction("top")
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].label.set_text("Standard deviation")
        ax.axis["bottom"].set_visible(False)

        self._ax = ax
        self.ax = ax.get_aux_axes(transform)
        ref, = self.ax.plot([0], refstd, "k*", ms=10, label=label)
        self.points = [ref]
        self.smin = srange[0]*refstd
        self.smax = srange[1]*refstd

    def add_sample(self, stddev, corrcoef, **kwargs):
        p, = self.ax.plot(np.arccos(corrcoef), stddev, **kwargs)
        self.points.append(p)

    def add_contours(self, levels=5, **kwargs):
        rs, ts = np.meshgrid(
            np.linspace(self.smin, self.smax),
            np.linspace(0, np.pi/2),
        )
        rms = np.sqrt(
            self.refstd**2 + rs**2 - 2*self.refstd*rs*np.cos(ts)
        )
        return self.ax.contour(ts, rs, rms, levels, **kwargs)

    def add_grid(self):
        self._ax.grid(True)


MODELS = [
    "svr", "random_forest", "gradient_boosting",
    "decision_tree", "xgboost", "lightgbm", "mlp",
]


def main():
    df = load_dataset()
    _, X_test, _, y_test = split_dataset(df)

    ref_std = y_test.std(ddof=1)
    fig = plt.figure(figsize=(7, 6))
    dia = TaylorDiagram(ref_std, fig=fig)

    for name in MODELS:
        model = joblib.load(PROJECT_ROOT / "models" / f"{name}.joblib")
        pred = np.asarray(model.predict(X_test))
        std = pred.std(ddof=1)
        corr = np.corrcoef(y_test, pred)[0, 1]
        dia.add_sample(std, corr, label=name, marker="o", ms=7)

    contours = dia.add_contours(levels=5, colors="0.5")
    plt.clabel(contours, inline=True, fontsize=8)
    dia.add_grid()

    fig.legend(
        dia.points,
        [p.get_label() for p in dia.points],
        loc="upper right",
        bbox_to_anchor=(1.25, 0.95),
        frameon=False,
    )

    fig.savefig(
        PROJECT_ROOT / "results" / "taylor_diagram.png",
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()
