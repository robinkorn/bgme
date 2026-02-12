from dataclasses import dataclass
from pathlib import Path

import json
import numpy as np
import sys

try:
	from src.data.server_machine_dataset import ServerMachineDataset, ServerMachineDatasetConfig
except ImportError:
	repo_root = Path(__file__).resolve().parents[2]
	sys.path.insert(0, str(repo_root))
	from src.data.server_machine_dataset import ServerMachineDataset, ServerMachineDatasetConfig

class FeatureDrift:
	feature_idx: int
	train_mean: float
	train_std: float
	test_mean: float
	test_std: float
	mean_shift_z: float
	std_ratio: float


class TrainLikeSummary:
	train_like_threshold: float
	test_train_like_fraction: float
	prefix_end_index: int | None
	block_train_like_fractions: tuple[float, ...]


FeatureDrift = dataclass(frozen=True, slots=True)(FeatureDrift)
TrainLikeSummary = dataclass(frozen=True, slots=True)(TrainLikeSummary)


class _TeeWriter:
	def __init__(self, *streams):
		self._streams = [s for s in streams if s is not None]

	def write(self, s):
		for st in self._streams:
			st.write(s)
		return len(s)

	def flush(self):
		for st in self._streams:
			try:
				st.flush()
			except Exception:
				pass


def _run_with_tee_to_txt(txt_path, fn, *args, **kwargs):
	"""Run fn(*args, **kwargs) while teeing stdout+stderr into txt_path."""
	txt_path = Path(txt_path)
	txt_path.parent.mkdir(parents=True, exist_ok=True)

	old_stdout = sys.stdout
	old_stderr = sys.stderr
	with txt_path.open("w", encoding="utf-8", newline="\n") as f:
		sys.stdout = _TeeWriter(old_stdout, f)
		sys.stderr = _TeeWriter(old_stderr, f)
		try:
			return fn(*args, **kwargs)
		finally:
			sys.stdout = old_stdout
			sys.stderr = old_stderr


def _load_evaluation_json(machine, part, evaluation_root, evaluation_subfolder):
	"""Load evaluation JSON for (machine, part) from evaluation/<subfolder>.

	Tries filename first: evaluation_{machine}_{part}.json
	Falls back to scanning all *.json in the folder and matching the JSON fields.
	"""
	if evaluation_subfolder is None:
		return None

	root = Path(evaluation_root) if evaluation_root is not None else Path("evaluation")
	folder = root / str(evaluation_subfolder)
	if not folder.exists() or not folder.is_dir():
		return None

	machine_i = int(machine)
	part_i = int(part)

	# Fast path: expected naming
	direct = folder / f"evaluation_{machine_i}_{part_i}.json"
	if direct.exists():
		try:
			return json.loads(direct.read_text(encoding="utf-8"))
		except Exception:
			# fall back to scan below
			pass

	# Fallback: scan JSONs and match machine/part inside.
	for p in sorted(folder.glob("*.json")):
		try:
			obj = json.loads(p.read_text(encoding="utf-8"))
		except Exception:
			continue
		if isinstance(obj, dict) and int(obj.get("machine", -1)) == machine_i and int(obj.get("part", -1)) == part_i:
			return obj

	return None


def _print_evaluation_results(eval_obj):
	if not isinstance(eval_obj, dict):
		return
	results = eval_obj.get("results")
	if not isinstance(results, dict) or not results:
		return

	print("-" * 80)
	print("Evaluation (Model Results aus JSON):")
	# Print in stable order
	for model_name in sorted(results.keys()):
		m = results.get(model_name)
		if not isinstance(m, dict):
			continue

		# Common metrics (may be missing)
		acc = m.get("accuracy")
		prec = m.get("precision")
		rec = m.get("recall")
		f1 = m.get("f1")
		tp = m.get("tp")
		tn = m.get("tn")
		fp = m.get("fp")
		fn = m.get("fn")

		line = f"- {model_name}:"
		parts = []
		if acc is not None:
			parts.append(f"acc={float(acc):.4f}")
		if prec is not None:
			parts.append(f"prec={float(prec):.4f}")
		if rec is not None:
			parts.append(f"rec={float(rec):.4f}")
		if f1 is not None:
			parts.append(f"f1={float(f1):.4f}")
		if tp is not None and tn is not None and fp is not None and fn is not None:
			parts.append(f"tp={int(tp)} tn={int(tn)} fp={int(fp)} fn={int(fn)}")

		if parts:
			print(line + " " + " | ".join(parts))
		else:
			print(line + " (keine Standard-Metriken gefunden)")


def _safe_std(x, axis=0, eps=1e-8):
	s = np.std(x, axis=axis)
	return np.maximum(s, eps)


def _diag_mahalanobis_sq(X, mean, std):
	z = (X - mean) / std
	return np.mean(z * z, axis=1)


def _rolling_mean_bool(x, window):
	if window <= 1:
		return x.astype(np.float32)
	kernel = np.ones(int(window), dtype=np.float32)
	return np.convolve(x.astype(np.float32), kernel, mode="same") / float(window)


def _anomaly_runs(y):
	"""Return list of inclusive ranges (start, end) where y==1."""
	y = np.asarray(y).astype(np.int8).reshape(-1)
	runs = []
	in_run = False
	start = 0
	for i, v in enumerate(y):
		if v == 1 and not in_run:
			in_run = True
			start = i
		elif v == 0 and in_run:
			in_run = False
			runs.append((start, i - 1))
	if in_run:
		runs.append((start, len(y) - 1))
	return runs


def _compute_feature_drift(X_train, X_test, top_k=10, eps=1e-8):
	mu_tr = np.mean(X_train, axis=0)
	sd_tr = _safe_std(X_train, axis=0, eps=eps)
	mu_te = np.mean(X_test, axis=0)
	sd_te = _safe_std(X_test, axis=0, eps=eps)

	mean_shift_z = np.abs(mu_te - mu_tr) / sd_tr
	std_ratio = sd_te / sd_tr

	order = np.argsort(-mean_shift_z)
	top = order[: min(int(top_k), len(order))]

	out = []
	for idx in top:
		out.append(
			FeatureDrift(
				feature_idx=int(idx),
				train_mean=float(mu_tr[idx]),
				train_std=float(sd_tr[idx]),
				test_mean=float(mu_te[idx]),
				test_std=float(sd_te[idx]),
				mean_shift_z=float(mean_shift_z[idx]),
				std_ratio=float(std_ratio[idx]),
			)
		)
	return out


def _train_like_summary(
	X_train,
	X_test,
	quantile=0.95,
	blocks=10,
	window=200,
	prefix_rate_threshold=0.8,
	eps=1e-8,
):
	mu_tr = np.mean(X_train, axis=0)
	sd_tr = _safe_std(X_train, axis=0, eps=eps)

	d_tr = _diag_mahalanobis_sq(X_train, mu_tr, sd_tr)
	thr = float(np.quantile(d_tr, float(quantile)))

	d_te = _diag_mahalanobis_sq(X_test, mu_tr, sd_tr)
	train_like = d_te <= thr

	test_train_like_fraction = float(np.mean(train_like))

	# Estimate: where does the test stop being predominantly "train-like"?
	roll = _rolling_mean_bool(train_like, window=max(1, int(window)))
	below = np.where(roll < float(prefix_rate_threshold))[0]
	prefix_end = int(below[0]) if below.size > 0 else None

	# Block-wise summary for trend detection.
	blocks = max(1, int(blocks))
	n = len(train_like)
	edges = np.linspace(0, n, num=blocks + 1, dtype=int)
	fracs = []
	for i in range(blocks):
		a, b = int(edges[i]), int(edges[i + 1])
		if b <= a:
			continue
		fracs.append(float(np.mean(train_like[a:b])))

	return TrainLikeSummary(
		train_like_threshold=thr,
		test_train_like_fraction=test_train_like_fraction,
		prefix_end_index=prefix_end,
		block_train_like_fractions=tuple(fracs),
	)


def analyze_machine_part(
	machine,
	part,
	dataset_root=None,
	*,
	evaluation_subfolder=None,
	evaluation_root=None,
	top_k_features=10,
	train_like_quantile=0.95,
	train_like_blocks=10,
	train_like_window=200,
	prefix_rate_threshold=0.8,
	high_mean_shift_z=3.0,
	high_std_ratio=2.0,
):
	"""Analyze one (machine, part) pair and print key diagnostics.

	What it reports (console):
	- Basic shapes / number of datapoints
	- Anomaly fraction in the test set (from test_label)
	- Strong train-vs-test distribution drift (per feature)
	- Whether the start of the test looks "train-like" and if that decreases over time
	"""

	config = ServerMachineDatasetConfig(
		root=Path(dataset_root) if dataset_root is not None else Path("data/ServerMachineDataset")
	)
	ds = ServerMachineDataset(config)

	X_train = ds.load_train(machines=[int(machine)], parts=[int(part)], concat=True)
	X_test = ds.load_test(int(machine), int(part))
	y_test = ds.load_test_labels(int(machine), int(part))

	if len(y_test) != len(X_test):
		raise ValueError(
			f"Length mismatch: X_test has {len(X_test)} rows, y_test has {len(y_test)} labels"
		)

	n_tr, d_tr = X_train.shape
	n_te, d_te = X_test.shape
	if d_tr != d_te:
		raise ValueError(f"Feature mismatch: train has D={d_tr}, test has D={d_te}")

	anomaly_count = int(np.sum(y_test.astype(np.int64)))
	anomaly_frac = float(anomaly_count / max(1, len(y_test)))
	runs = _anomaly_runs(y_test)

	drift_top = _compute_feature_drift(X_train, X_test, top_k=top_k_features)
	train_like = _train_like_summary(
		X_train,
		X_test,
		quantile=train_like_quantile,
		blocks=train_like_blocks,
		window=train_like_window,
		prefix_rate_threshold=prefix_rate_threshold,
	)

	eval_obj = _load_evaluation_json(
		machine=int(machine),
		part=int(part),
		evaluation_root=evaluation_root,
		evaluation_subfolder=evaluation_subfolder,
	)

	print("=" * 80)
	print(f"Analyse: ServerMachineDataset | machine={int(machine)} part={int(part)}")
	print(f"Root: {Path(config.root).resolve()}")
	print("-" * 80)
	print(f"Train: N={n_tr:,}  D={d_tr}")
	print(f"Test : T={n_te:,}  D={d_te}")
	print("-" * 80)

	print("Labels (Test):")
	print(f"- Anomalieanteil: {anomaly_frac:.4f} ({anomaly_count:,} / {n_te:,})")
	if runs:
		lengths = [b - a + 1 for a, b in runs]
		print(f"- Anomalie-Segmente: {len(runs)}")
		print(f"- Erstes Segment: [{runs[0][0]}, {runs[0][1]}] (len={lengths[0]})")
		print(f"- Längstes Segment: len={max(lengths)}")
		print(f"- Letztes Segment: [{runs[-1][0]}, {runs[-1][1]}] (len={lengths[-1]})")
	else:
		print("- Keine Anomalie-Segmente (y_test enthält nur 0)")

	print("-" * 80)
	print("Train vs Test Drift (Top Features nach Mittelwert-Shift in Std-Einheiten):")
	if not drift_top:
		print("- (keine Features gefunden)")
	else:
		flagged = 0
		for d in drift_top:
			warn = (d.mean_shift_z >= high_mean_shift_z) or (d.std_ratio >= high_std_ratio) or (
				d.std_ratio <= 1.0 / max(high_std_ratio, 1e-6)
			)
			if warn:
				flagged += 1
			prefix = "!" if warn else " "
			print(
				f"{prefix} f[{d.feature_idx:02d}] "
				f"mean_tr={d.train_mean:.4g} std_tr={d.train_std:.4g} | "
				f"mean_te={d.test_mean:.4g} std_te={d.test_std:.4g} | "
				f"|Δmean|/std_tr={d.mean_shift_z:.3f} std_ratio={d.std_ratio:.3f}"
			)
		if flagged == 0:
			print(
				f"- Keine auffälligen Features in den Top-{len(drift_top)} nach Schwellwerten "
				f"(|Δmean|/std_tr>={high_mean_shift_z} oder std_ratio>={high_std_ratio})."
			)

	print("-" * 80)
	print("Train-like Analyse (Test ähnelt Trainingsverteilung?):")
	print(
		f"- Schwelle: {train_like_quantile:.2f}-Quantil der Train-Distanz (diag-Mahalanobis^2) = "
		f"{train_like.train_like_threshold:.4f}"
	)
	print(f"- Anteil train-like im Test gesamt: {train_like.test_train_like_fraction:.3f}")
	if train_like.prefix_end_index is None:
		print(
			f"- Prefix: Rolling-Rate blieb >= {prefix_rate_threshold:.2f} (window={train_like_window}) "
			f"über den gesamten Test"
		)
	else:
		print(
			f"- Prefix-Ende (ungefähr): t≈{train_like.prefix_end_index} "
			f"(rolling train-like < {prefix_rate_threshold:.2f}, window={train_like_window})"
		)

	if train_like.block_train_like_fractions:
		print(f"- Verlauf (Blöcke={len(train_like.block_train_like_fractions)}):")
		for i, frac in enumerate(train_like.block_train_like_fractions, start=1):
			print(f"  Block {i:02d}: train-like={frac:.3f}")

	# Optional: anomaly rate per block for quick co-variation hints.
	blocks = len(train_like.block_train_like_fractions)
	if blocks > 0:
		edges = np.linspace(0, n_te, num=blocks + 1, dtype=int)
		print("-" * 80)
		print("Test-Labels Verlauf (Anomalieanteil je Block):")
		for i in range(blocks):
			a, b = int(edges[i]), int(edges[i + 1])
			if b <= a:
				continue
			frac = float(np.mean(y_test[a:b]))
			print(f"  Block {i+1:02d}: anomaly_frac={frac:.3f} (n={b-a})")

	_print_evaluation_results(eval_obj)

	print("=" * 80)

	return {
		"machine": int(machine),
		"part": int(part),
		"n_train": int(n_tr),
		"n_test": int(n_te),
		"n_features": int(d_tr),
		"anomaly_count": int(anomaly_count),
		"anomaly_fraction": float(anomaly_frac),
		"anomaly_runs": runs,
		"feature_drift_top": drift_top,
		"train_like": train_like,
		"evaluation": eval_obj,
	}


if __name__ == "__main__":
	MACHINE = 2
	PART = 2
	DATASET_ROOT = None  # r"D:\\BachelorArbeit\\bachelor-arbeit\\data\\ServerMachineDataset"

	# Evaluation integration (set to None to disable)
	EVALUATION_SUBFOLDER = "500retrain"  # "500retrain", "500retrain_old"
	EVALUATION_ROOT = "evaluation"  # usually "evaluation"

	# Save this analysis as txt under evaluation/<subfolder> (set to None to disable)
	EVALUATION_SUBFOLDER_TXT = "analysis_txt"  # "analysis_txt", "diagnostics"

	TOP_K_FEATURES = 10
	TRAIN_LIKE_QUANTILE = 0.95
	TRAIN_LIKE_BLOCKS = 10
	TRAIN_LIKE_WINDOW = 200
	PREFIX_RATE_THRESHOLD = 0.8

	HIGH_MEAN_SHIFT_Z = 3.0
	HIGH_STD_RATIO = 2.0

	kwargs = dict(
		machine=MACHINE,
		part=PART,
		dataset_root=DATASET_ROOT,
		evaluation_subfolder=EVALUATION_SUBFOLDER,
		evaluation_root=EVALUATION_ROOT,
		top_k_features=TOP_K_FEATURES,
		train_like_quantile=TRAIN_LIKE_QUANTILE,
		train_like_blocks=TRAIN_LIKE_BLOCKS,
		train_like_window=TRAIN_LIKE_WINDOW,
		prefix_rate_threshold=PREFIX_RATE_THRESHOLD,
		high_mean_shift_z=HIGH_MEAN_SHIFT_Z,
		high_std_ratio=HIGH_STD_RATIO,
	)

	if EVALUATION_SUBFOLDER_TXT is not None:
		out_dir = Path(EVALUATION_ROOT) / str(EVALUATION_SUBFOLDER_TXT)
		out_path = out_dir / f"analysis_{int(MACHINE)}_{int(PART)}.txt"
		_run_with_tee_to_txt(out_path, analyze_machine_part, **kwargs)
		print(f"Saved analysis to: {out_path.resolve()}")
	else:
		analyze_machine_part(**kwargs)
