import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
import numpy as np

TEXT_SCALE = 2.0
FIG_HEIGHT_SCALE = 1.5


def _apply_global_plot_style(text_scale: float = TEXT_SCALE):
    base = 10.0
    s = float(base) * float(text_scale)
    plt.rcParams.update(
        {
            "font.size": s,
            "axes.titlesize": s,
            "axes.labelsize": s,
            "xtick.labelsize": s,
            "ytick.labelsize": s,
            "legend.fontsize": s,
            "legend.title_fontsize": s,
        }
    )


def _scaled_figsize(w: float, h: float):
    return (float(w), float(h) * float(FIG_HEIGHT_SCALE))


_apply_global_plot_style()

@dataclass(frozen=True)
class EvalRecord:
    path: Path
    machine: int
    part: int
    thresholds: dict
    results: dict
    confusion: dict
    shapes: dict
    model_params: dict
    timestamp_utc: str | None = None

    def models(self):
        return list(self.results.keys())

    def _cm(self, model):
        if model in self.confusion:
            return self.confusion[model]

        r = self.results[model]
        if all(k in r for k in ("tp", "tn", "fp", "fn")):
            return {
                "tp": int(r["tp"]),
                "tn": int(r["tn"]),
                "fp": int(r["fp"]),
                "fn": int(r["fn"]),
            }
        raise KeyError(
            f"No confusion matrix for model='{model}' in {self.path}")

    def anomaly_share(self):
        models = self.models()
        preferred = "bgme_default" if "bgme_default" in models else models[0]
        cm = self._cm(preferred)
        n = cm["tp"] + cm["tn"] + cm["fp"] + cm["fn"]
        if n == 0:
            return float("nan")
        return (cm["tp"] + cm["fn"]) / n


class EvaluationCache:
    def __init__(self, eval_dir, pattern="evaluation_*.json"):
        self.eval_dir = Path(eval_dir)
        self.pattern = pattern
        self._records = []
        self._loaded = False

    def load(self):
        paths = sorted(self.eval_dir.glob(self.pattern))
        if not paths:
            raise FileNotFoundError(
                f"No files matching '{self.pattern}' in {self.eval_dir}")

        records = []
        for p in paths:
            with p.open("r", encoding="utf-8") as f:
                d = json.load(f)

            confusion = d.get("confusion_matrices") or d.get("confusion") or {}

            records.append(
                EvalRecord(
                    path=p,
                    machine=int(d.get("machine", -1)),
                    part=int(d.get("part", -1)),
                    thresholds=d.get("thresholds", {}),
                    results=d.get("results", {}),
                    confusion=confusion,
                    shapes=d.get("shapes", {}),
                    model_params=d.get("model_params", {}),
                    timestamp_utc=d.get("timestamp_utc"),
                )
            )

        records.sort(key=lambda r: (r.machine, r.part))
        self._records = records
        self._loaded = True
        return self

    @property
    def records(self):
        if not self._loaded:
            raise RuntimeError("Call .load() first")
        return self._records

    def machines(self):
        return sorted({r.machine for r in self.records})

    def models_union(self):
        seen = []
        for r in self.records:
            for m in r.models():
                if m not in seen:
                    seen.append(m)
        return seen


def _ensure_out_dir(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _save_fig(out_dir, filename, dpi=220):
    out_dir = _ensure_out_dir(out_dir)
    path = out_dir / filename
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return path


def _colors_for_models(models):
    fixed = {
        "baseline_gmm": "#4D4D4D",  # dark gray
        "bgme_default": "#1b78db",  # light blue
        "bgme_correct": "#fa59d2",  # dark pink
    }

    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not cycle:
        cycle = ["C0", "C1", "C2", "C3", "C4", "C5"]

    out = {}
    j = 0
    for m in models:
        if m in fixed:
            out[m] = fixed[m]
        else:
            out[m] = cycle[j % len(cycle)]
            j += 1
    return out



# Model names in legend
MODEL_DISPLAY_NAMES = {
    "baseline_gmm": "Baseline-GMM",
    "bgme_default": "BGME + Fehler",
    "bgme_correct": "BGME + alle Daten",
}


def _model_label(model: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model, model)


def _convex_hull(points):
    """Monotonic chain convex hull. Returns hull points in CCW order."""
    pts = sorted({(float(x), float(y)) for (x, y) in points})
    if len(pts) <= 1:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def _lowess_grid(x, y, x_grid, frac=0.35, it=2):
    """Simple LOWESS (robust local linear regression) evaluated on x_grid."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        return np.full_like(x_grid, np.nan, dtype=float)

    # Collapse duplicate x by averaging y
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    uniq_x, inv = np.unique(x, return_inverse=True)
    if uniq_x.size != x.size:
        y_mean = np.zeros_like(uniq_x)
        counts = np.zeros_like(uniq_x)
        np.add.at(y_mean, inv, y)
        np.add.at(counts, inv, 1)
        y = y_mean / np.maximum(1, counts)
        x = uniq_x

    n = x.size
    if n == 1:
        return np.full_like(x_grid, y[0], dtype=float)
    if n == 2:
        return np.interp(x_grid, x, y)

    k = max(3, int(math.ceil(frac * n)))
    k = min(k, n)

    def tricube(u):
        u = np.clip(u, 0.0, 1.0)
        return (1.0 - u**3) ** 3

    def predict_with_robust(robust_w, x_eval):
        out = np.empty_like(x_eval, dtype=float)
        for j, x0 in enumerate(x_eval):
            d = np.abs(x - x0)
            idx = np.argpartition(d, k - 1)[:k]
            h = d[idx].max()
            if not np.isfinite(h) or h <= 0:
                out[j] = float(np.nanmean(y[idx]))
                continue

            w = tricube(d[idx] / h) * robust_w[idx]
            wsum = w.sum()
            if wsum <= 0:
                out[j] = float(np.nanmean(y[idx]))
                continue

            x_d = x[idx] - x0
            xbar = np.sum(w * x_d) / wsum
            ybar = np.sum(w * y[idx]) / wsum
            denom = np.sum(w * (x_d - xbar) ** 2)
            if denom <= 1e-12:
                out[j] = float(ybar)
                continue
            b = np.sum(w * (x_d - xbar) * (y[idx] - ybar)) / denom
            a = ybar - b * xbar
            out[j] = float(a)
        return out

    robust = np.ones(n, dtype=float)
    for _ in range(max(0, int(it))):
        y_fit = predict_with_robust(robust, x)
        resid = y - y_fit
        s = np.nanmedian(np.abs(resid))
        if not np.isfinite(s) or s <= 1e-12:
            break
        u = resid / (6.0 * s)
        robust = (1.0 - u**2) ** 2
        robust[np.abs(u) >= 1.0] = 0.0

    return predict_with_robust(robust, x_grid)


def _bootstrap_lowess_band(x, y, x_grid, frac=0.35, it=2, n_boot=500, seed=0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = x.size
    if n < 6:
        return None

    rng = np.random.default_rng(seed)
    preds = np.empty((int(n_boot), x_grid.size), dtype=float)
    for b in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        preds[b] = _lowess_grid(x[idx], y[idx], x_grid, frac=frac, it=max(1, int(it) - 1))

    lo = np.nanpercentile(preds, 2.5, axis=0)
    hi = np.nanpercentile(preds, 97.5, axis=0)
    return lo, hi


def _machine_marker_map(machines):
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    return {m: markers[i % len(markers)] for i, m in enumerate(machines)}


def _metric(rec, model, key):
    if model not in rec.results:
        return None
    v = rec.results[model].get(key)
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _x_positions_labels(cache):
    recs = cache.records
    xs = list(range(len(recs)))
    labels = [f"{r.machine}-{r.part}" for r in recs]
    return xs, labels


def _scatter_kwargs_small():
    return {"s": 18, "alpha": 0.85}


def _style_legend_frame(leg, lw=1.2):
    if leg is None:
        return
    frame = leg.get_frame()
    frame.set_edgecolor("black")
    frame.set_linewidth(lw)
    frame.set_alpha(1.0)


def _legend_bottom_compact(fig, handles, title=None, y=0.01, ncol=1):
    """Bottom legend, compact (not stretched), centered."""
    if not handles:
        return None

    leg = fig.legend(
        handles=handles,
        title=title,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=max(1, int(ncol)),
        frameon=True,
        fancybox=True,
        borderpad=0.8,
        columnspacing=1.2,
        handletextpad=0.7,
    )
    _style_legend_frame(leg)
    return leg


def _legend_two_blocks_right_center(ax, model_handles, machine_handles, model_title="Model (color)", machine_title="Machine (marker)"):
    """Place BOTH legends under the plot (wide), stacked: models then machines."""
    fig = ax.figure

    # Reserve room for two stacked legends below the axes.
    fig.subplots_adjust(bottom=0.44)

    # Models (upper legend band) - vertical list
    leg1 = _legend_bottom_compact(
        fig,
        model_handles,
        title=model_title,
        y=0.22,
        ncol=1,
    )
    if leg1 is not None:
        fig.add_artist(leg1)

    # Machines (lower legend band) - vertical list
    _legend_bottom_compact(
        fig,
        machine_handles,
        title=machine_title,
        y=0.03,
        ncol=1,
    )

def plot_precision_recall_scatter_all(cache, out_dir, filename="precision_recall_scatter_all.png"):
    models = cache.models_union()
    machines = cache.machines()
    colors = _colors_for_models(models)
    markers = _machine_marker_map(machines)

    fig, ax = plt.subplots(figsize=_scaled_figsize(13.5, 6.5))
    fig.subplots_adjust(bottom=0.44)

    scatter_kw = _scatter_kwargs_small()

    for r in cache.records:
        mach = r.machine
        for model in models:
            prec = _metric(r, model, "precision")
            rec = _metric(r, model, "recall")
            if prec is None or rec is None:
                continue
            ax.scatter(
                rec, prec, color=colors[model], marker=markers[mach], **scatter_kw)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Precision–Recall Scatter (all machines, per part)")

    ax.grid(False)

    model_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            color=colors[m],
            label=_model_label(m),
        )
        for m in models
    ]
    machine_handles = [plt.Line2D([0], [0], marker=markers[x], linestyle="None",
                                  color="black", label=f"machine_{x}") for x in machines]

    _legend_two_blocks_right_center(ax, model_handles, machine_handles)

    return _save_fig(out_dir, filename)


def plot_fp_fn_tradeoff_all(cache, out_dir, log_scale=True, filename="fp_fn_tradeoff_all.png"):
    models = cache.models_union()
    machines = cache.machines()
    colors = _colors_for_models(models)

    fig, ax = plt.subplots(figsize=_scaled_figsize(13.5, 6.5))
    fig.subplots_adjust(bottom=0.30)

    scatter_kw = {"s": 40, "alpha": 0.85, "marker": "o"}

    # scatter visualization
    points_by_model = {m: [] for m in models}

    for r in cache.records:
        for model in models:
            try:
                cm = r._cm(model)
            except Exception:
                continue
            tp = float(cm.get("tp", 0.0))
            tn = float(cm.get("tn", 0.0))
            fp = float(cm.get("fp", 0.0))
            fn = float(cm.get("fn", 0.0))
            n = tp + tn + fp + fn
            if n <= 0:
                continue

            fp = (fp / n) * 100.0
            fn = (fn / n) * 100.0
            points_by_model[model].append((fp, fn))
            ax.scatter(fp, fn, color=colors[model], **scatter_kw)

    ax.set_xlabel("Prozentualer Anteil False Positives")
    ax.set_ylabel("Prozentualer Anteil False Negatives")
    ax.set_title("FP-FN-Trade-off auf allen Maschinen")
    if log_scale:
        ax.set_xscale("symlog", linthresh=0.1)
        ax.set_yscale("symlog", linthresh=0.1)

    ax.grid(False)

    # Draw a low-alpha envelope around each model's point cloud.
    for model in models:
        pts = points_by_model.get(model) or []
        if not pts:
            continue

        # Separate alpha for fill vs outline; tuned per model for readability.
        face_alpha = {
            "baseline_gmm": 0.06,
            "bgme_correct": 0.20,
            "bgme_default": 0.25,
        }.get(model, 0.10)
        edge_alpha = {
            "baseline_gmm": 0.24,
            "bgme_correct": 0.38,
            "bgme_default": 0.50,
        }.get(model, 0.35)

        face_rgba = mcolors.to_rgba(colors[model], face_alpha)
        edge_rgba = mcolors.to_rgba(colors[model], edge_alpha)

        unique = sorted({(float(x), float(y)) for (x, y) in pts})
        if len(unique) >= 2:
            disp_pts = [tuple(ax.transData.transform([(x, y)])[0]) for (x, y) in unique]
        else:
            disp_pts = []

        def add_disp_polygon(disp_vertices):
            data_vertices = ax.transData.inverted().transform(disp_vertices)
            ax.add_patch(
                Polygon(
                    data_vertices,
                    closed=True,
                    facecolor=face_rgba,
                    edgecolor=edge_rgba,
                    linewidth=1.4,
                    zorder=1,
                )
            )

        if len(unique) >= 3:
            hull_disp = _convex_hull(disp_pts)
            if len(hull_disp) >= 3:
                add_disp_polygon(hull_disp)
            else:
                # Degenerate hull (often collinear): fall back to a tight bbox in display space.
                xs_d = [p[0] for p in disp_pts]
                ys_d = [p[1] for p in disp_pts]
                minx, maxx = min(xs_d), max(xs_d)
                miny, maxy = min(ys_d), max(ys_d)
                pad = 8.0  # pixels
                box_disp = [
                    (minx - pad, miny - pad),
                    (maxx + pad, miny - pad),
                    (maxx + pad, maxy + pad),
                    (minx - pad, maxy + pad),
                ]
                add_disp_polygon(box_disp)
        elif len(unique) == 2:
            xs_d = [p[0] for p in disp_pts]
            ys_d = [p[1] for p in disp_pts]
            minx, maxx = min(xs_d), max(xs_d)
            miny, maxy = min(ys_d), max(ys_d)
            pad = 10.0  # pixels
            box_disp = [
                (minx - pad, miny - pad),
                (maxx + pad, miny - pad),
                (maxx + pad, maxy + pad),
                (minx - pad, maxy + pad),
            ]
            add_disp_polygon(box_disp)
        else:
            (x0, y0) = unique[0]
            ax.scatter(
                [x0],
                [y0],
                s=260,
                facecolors=face_rgba,
                edgecolors=edge_rgba,
                linewidths=1.4,
                zorder=1,
            )

    # Single legend (models only)
    model_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="-",
            color=colors[m],
            label=_model_label(m),
        )
        for m in models
    ]
    _legend_bottom_compact(
        fig,
        model_handles,
        title="Model",
        y=0.01,
        ncol=1,
    )

    return _save_fig(out_dir, filename)


def plot_f1_over_sequence_split_by_machine(cache, out_dir, filename="f1_over_sequence_by_machine.png"):
    models = cache.models_union()
    colors = _colors_for_models(models)

    # Emphasize specific machine-part segments in this plot by shading the
    # corresponding x-position background (low-alpha light red)
    highlight_segments = {(1, 6), (2, 2), (2, 7), (3, 4)}

    machines = cache.machines()  # expected [1,2,3]
    nrows = len(machines)

    # compute max number of segments among machines to size width reasonably
    max_parts = 1
    for m in machines:
        cnt = sum(1 for r in cache.records if r.machine == m)
        max_parts = max(max_parts, cnt)

    base_w = 14.0
    extra_w = min(0.45 * max_parts, 20.0)
    fig_w = base_w + extra_w
    row_h = 4.0
    fig_h = (row_h * nrows + 1.0) * FIG_HEIGHT_SCALE

    fig, axes = plt.subplots(nrows=nrows, ncols=1,
                             figsize=(fig_w, fig_h), sharey=True)
    if nrows == 1:
        axes = [axes]

    # Slightly less spacing between machine subplots, and a bit more space down to the legend.
    fig.subplots_adjust(hspace=0.48, bottom=0.25)

    for ax, mach in zip(axes, machines):
        recs = [r for r in cache.records if r.machine == mach]
        xs = list(range(len(recs)))
        labels = [f"{r.machine}-{r.part}" for r in recs]  # Maschine-Part

        # Background highlights (behind the points)
        for i, r in enumerate(recs):
            if (r.machine, r.part) in highlight_segments:
                ax.axvspan(
                    i - 0.25,
                    i + 0.25,
                    color="#ff8080",
                    alpha=0.10,
                    zorder=0,
                )

        for model in models:
            y_series = []
            for r in recs:
                f1 = _metric(r, model, "f1")
                y_series.append(float("nan") if f1 is None else f1)

            if any(math.isfinite(y) for y in y_series):
                ax.plot(
                    xs,
                    y_series,
                    color=colors[model],
                    marker="o",
                    linestyle="-",
                    linewidth=1.6,
                    markersize=5,
                    alpha=0.9,
                    label=_model_label(model),
                    zorder=2,
                )

        ax.set_title(f"Maschinen-Gruppe {mach}")
        ax.set_ylim(0, 1)
        ax.set_ylabel("F1-Score")
        ax.grid(False)

        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=45, ha="right")

    axes[-1].set_xlabel("Segment (einzelne Maschine)")

    # Single model legend on the right (no machine/marker legend here)
    model_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="-",
                   color=colors[m], label=_model_label(m))
        for m in models
    ]
    _legend_bottom_compact(
        fig,
        model_handles,
        title="Model",
        y=0.05,
        ncol=1,
    )

    return _save_fig(out_dir, filename)


def plot_f1_over_sequence_all(cache, out_dir, filename="f1_over_sequence_all.png"):
    models = cache.models_union()
    colors = _colors_for_models(models)
    xs, labels = _x_positions_labels(cache)
    recs = cache.records

    base = 14.0
    extra = min(0.35 * len(xs), 18.0)
    fig_w = base + extra

    fig, ax = plt.subplots(figsize=_scaled_figsize(fig_w, 5.8))
    fig.subplots_adjust(bottom=0.44)

    for model in models:
        x_used, y_used = [], []
        for i, r in enumerate(recs):
            f1 = _metric(r, model, "f1")
            if f1 is None:
                continue
            x_used.append(xs[i])
            y_used.append(f1)

        if x_used:
            ax.scatter(
                x_used, y_used,
                color=colors[model],
                s=36,
                alpha=0.9,
                label=model
            )

    ax.set_xlabel("Segment (Maschine-Part)")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1)
    ax.set_title("F1-Score aller Maschinen-Parts")
    ax.grid(False)

    ax.set_xticks(xs)
    # 45 degree labels
    ax.set_xticklabels(labels, rotation=45, ha="right")

    machines = cache.machines()
    markers = _machine_marker_map(machines)

    model_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            color=colors[m],
            label=_model_label(m),
        )
        for m in models
    ]
    machine_handles = [plt.Line2D([0], [0], marker=markers[x], linestyle="None",
                                  color="black", label=f"machine_{x}") for x in machines]
    _legend_two_blocks_right_center(ax, model_handles, machine_handles)

    return _save_fig(out_dir, filename)


def plot_f1_vs_anomaly_share_all(cache, out_dir, x_in_percent=True, filename="f1_vs_anomaly_share_all.png"):
    models = cache.models_union()
    colors = _colors_for_models(models)

    fig, ax = plt.subplots(figsize=_scaled_figsize(13.5, 6.5))
    fig.subplots_adjust(bottom=0.30)

    scatter_kw = {"s": 40, "alpha": 0.85, "marker": "o"}

    points_by_model = {m: ([], []) for m in models}
    all_x = []

    for r in cache.records:
        share = r.anomaly_share()
        if share is None or not math.isfinite(share):
            continue
        x = share * 100.0 if x_in_percent else share

        for model in models:
            f1 = _metric(r, model, "f1")
            if f1 is None:
                continue
            points_by_model[model][0].append(x)
            points_by_model[model][1].append(f1)
            all_x.append(x)
            ax.scatter(x, f1, color=colors[model], **scatter_kw)

    ax.set_xlabel(
        "Anomalieanteil (%)" if x_in_percent else "Anomalieanteil (0–1)")
    ax.set_ylabel("F1-Score")
    ax.set_ylim(0, 1)
    # x-axis should end at the maximum observed anomaly share (not forced to 100%)
    if all_x:
        x_min_data = float(np.nanmin(all_x))
        x_max_data = float(np.nanmax(all_x))
        if math.isfinite(x_min_data) and math.isfinite(x_max_data) and x_max_data > x_min_data:
            pad = 0.02 * (x_max_data - x_min_data)
            x_left = 0.0 if x_in_percent else 0.0
            x_right = x_max_data + pad
            ax.set_xlim(x_left, x_right)
    ax.set_title("F1-Score in Abhängigkeit vom Anomalieanteil")
    ax.grid(False)

    # Robust trendline + bootstrapped band per model
    for model in models:
        xs = np.asarray(points_by_model[model][0], dtype=float)
        ys = np.asarray(points_by_model[model][1], dtype=float)
        mask = np.isfinite(xs) & np.isfinite(ys)
        xs = xs[mask]
        ys = ys[mask]
        if xs.size < 3:
            continue

        # Evaluate on the full axis range so the line/band ends flush at the plot boundary.
        x_left, x_right = ax.get_xlim()
        x_grid = np.linspace(float(x_left), float(x_right), 200)
        y_line = _lowess_grid(xs, ys, x_grid, frac=0.55, it=4) # NOTE: frac hoch glatter, it hoch robuster
        y_line = np.clip(y_line, 0.0, 1.0)

        band = _bootstrap_lowess_band(xs, ys, x_grid, frac=0.55, it=4, n_boot=500, seed=7) # NOTE: n_boot hoch für stabilere Bänder
        if band is not None:
            lo, hi = band
            lo = np.clip(lo, 0.0, 1.0)
            hi = np.clip(hi, 0.0, 1.0)

            band_alpha = 0.12
            if model == "baseline_gmm":
                band_alpha = 0.06
            ax.fill_between(
                x_grid,
                lo,
                hi,
                color=colors[model],
                alpha=band_alpha,
                linewidth=0,
                zorder=1,
            )

        ax.plot(
            x_grid,
            y_line,
            color=colors[model],
            linewidth=2.0,
            alpha=0.95,
            zorder=2,
        )

    model_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="-",
            color=colors[m],
            label=_model_label(m),
        )
        for m in models
    ]
    _legend_bottom_compact(
        fig,
        model_handles,
        title="Model",
        y=0.01,
        ncol=1,
    )

    return _save_fig(out_dir, filename)


def save_all_plots(cache, out_dir):
    out_dir = _ensure_out_dir(out_dir)
    paths = []
    # paths.append(plot_precision_recall_scatter_all(cache, out_dir))
    paths.append(plot_fp_fn_tradeoff_all(cache, out_dir, log_scale=True))
    paths.append(plot_f1_vs_anomaly_share_all(
        cache, out_dir, x_in_percent=True))
    paths.append(plot_f1_over_sequence_split_by_machine(cache, out_dir))
    return paths


if __name__ == "__main__":
    cache = EvaluationCache("./evaluation/500retrain", pattern="evaluation_*.json").load()
    paths = save_all_plots(cache, "./plots/new_test_v4_titel_fix")

    print("Saved:")
    for p in paths:
        print(" -", p)
