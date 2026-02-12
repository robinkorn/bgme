import json
import math
import sys
from pathlib import Path


DEFAULT_FOLDER = Path("evaluation") / "500retrain"


def _iter_json_files(folder: Path):
	return sorted(p for p in folder.glob("*.json") if p.is_file())


def _extract_f1_scores(payload):
	"""Extract model -> f1 from an evaluation JSON payload.

	Expected schema (from your evaluation files):
		payload["results"][model_name]["f1"]
	"""
	results = payload.get("results") if isinstance(payload, dict) else None
	if not isinstance(results, dict):
		return {}

	f1_by_model = {}
	for model_name, metrics in results.items():
		if not isinstance(metrics, dict):
			continue
		f1 = metrics.get("f1")
		if isinstance(f1, (int, float)) and math.isfinite(float(f1)):
			f1_by_model[str(model_name)] = float(f1)
	return f1_by_model


def _mean_variance(values):
	if len(values) == 0:
		return float("nan"), float("nan")
	mean = sum(values) / len(values)
	if len(values) < 2:
		return mean, float("nan")
	var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
	return mean, var


def compute_f1_stats(folder: Path):
	"""Load all JSONs, return (models, rows)."""
	files = _iter_json_files(folder)
	if not files:
		raise FileNotFoundError(f"No .json files found in: {folder}")

	per_file = []
	for file in files:
		try:
			payload = json.loads(file.read_text(encoding="utf-8"))
		except UnicodeDecodeError:
			payload = json.loads(file.read_text(encoding="utf-8-sig"))
		except Exception as exc:  # noqa: BLE001
			raise RuntimeError(f"Failed to read/parse JSON: {file}: {exc}") from exc

		f1_by_model = _extract_f1_scores(payload)
		if f1_by_model:
			per_file.append(f1_by_model)

	if not per_file:
		raise RuntimeError(
			f"Found {len(files)} JSON files but none contained any 'results.*.f1' values."
		)

	model_sets = [set(d.keys()) for d in per_file]
	models = sorted(set.intersection(*model_sets))
	if not models:
		raise RuntimeError("No common model keys with f1 were found across files.")

	rows = []
	for d in per_file:
		row = [d[m] for m in models]
		if all(math.isfinite(x) for x in row):
			rows.append(row)

	if not rows:
		raise RuntimeError("No files had a full, finite f1 vector for all common models.")

	return models, rows


def main():
	folder = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FOLDER
	if not folder.exists() or not folder.is_dir():
		print(f"Folder does not exist or is not a directory: {folder}", file=sys.stderr)
		return 2

	models, rows = compute_f1_stats(folder)

	pooled = [x for row in rows for x in row]
	pooled_mean, pooled_var = _mean_variance(pooled)

	print(f"Folder: {folder}")
	print(f"JSON files used: {len(rows)}")
	print(f"Models (intersection across files): {', '.join(models)}")
	print()

	print("Per-model average F1 and variability (sample):")
	for j, model in enumerate(models):
		col = [row[j] for row in rows]
		mu, var = _mean_variance(col)
		std = math.sqrt(var) if math.isfinite(var) else float("nan")
		# Coefficient of variation can be useful for comparing variability across models
		cv = (std / mu) if (math.isfinite(std) and mu != 0.0) else float("nan")
		print(f"  {model}: mean={mu:.6f}  var={var:.6f}  std={std:.6f}  cv={cv:.6f}")

	print()
	print("All F1 scores combined (all models x all files):")
	print(f"  mean={pooled_mean:.6f}")
	if math.isfinite(pooled_var):
		print(f"  var={pooled_var:.6f}")
		print(f"  std={math.sqrt(pooled_var):.6f}")
	else:
		print("  var/std=nan (need >= 2 values)")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
