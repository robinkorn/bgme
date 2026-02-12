from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

from sklearn.mixture import GaussianMixture

try:
	from src.sim import DataSim, DataSimConfig, AnomalyConfig
	from src.models.BGME import OnlineBGMEAnomalyDetector
	from src.models.BGMEConfig import BGMEConfig
	from src.models.OnlineAnomalyConfig import OnlineAnomalyConfig
except ImportError:
	# Allows running via: 'python src/eval/eval.py' from the repo root
	repo_root = Path(__file__).resolve().parents[2]
	sys.path.insert(0, str(repo_root))
	from src.sim import DataSim, DataSimConfig, AnomalyConfig
	from src.models.BGME import OnlineBGMEAnomalyDetector
	from src.models.BGMEConfig import BGMEConfig
	from src.models.OnlineAnomalyConfig import OnlineAnomalyConfig

# Where to write per-run JSON summaries (if None, defaults to repo_root/evaluation)
EVAL_JSON_OUTPUT_DIR: Path | None = "./evaluation/longterm_sim_simple/"

# Console progress printing while streaming test points
PROGRESS_EVERY = 1000

# BGME hyperparams
BGME_N_ESTIMATORS = 8
BGME_N_COMPONENTS = 5
BGME_COVARIANCE_TYPE = "full"  # 'full', 'tied', 'diag', or 'spherical'
BGME_RANDOM_STATE = 42
BGME_MAX_ITER = 300
BGME_TOL = 1e-3
BGME_REG_COVAR = 3e-4
BGME_COMBINE = "logmeanexp"  # "mean_loglik" | "logmeanexp"
HARD_QUANTILE = 0.9  # 0.9 default
LEARNING_RATE = 0.5  # 0.7 default
MIN_WEIGHT = 1e-6
MAX_WEIGHT = 200
ALPHA_DECAY_RATE = None 
MAX_MODELS_PER_ENSEMBLE = 30

# Online detection settings
THRESHOLD_QUANTILE = 5e-4
SCORING_MODE = "separation"  # "normal_only" | "separation"
SEPARATION_THRESHOLD = 0.55

REFIT_WINDOW_SIZE = 5000
INCLUDE_CORRECT_PREDICTIONS = False
MIN_SAMPLES_FOR_REFIT = 500

# Additional BGME variant: always include correct predictions during internal refit
EVAL_INCLUDE_CORRECT_PREDICTIONS_VARIANT = True

# Baseline (single GMM) settings
# Keep current behavior: use more components than BGME to be competitive.
BASELINE_GMM_N_COMPONENTS = int((BGME_N_COMPONENTS * BGME_N_ESTIMATORS) / 2)


def _json_safe(value):
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if is_dataclass(value):
        return _json_safe(asdict(value))
    # numpy scalars
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    # avoid dumping big arrays; keep it explicit
    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
    return str(value)


def save_evaluation_run_json(
    *,
    output_dir: Path | str | None,
    payload: dict,
) -> Path:
    if output_dir is None:
        repo_root = Path(__file__).resolve().parents[2]
        output_dir = repo_root / "evaluation"

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"longterm_sim_evaluation.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, ensure_ascii=False,
                  indent=2, sort_keys=False)
    print(f"Wrote evaluation JSON: {path}")
    return path


def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
	y_true = np.asarray(y_true).astype(np.int8).reshape(-1)
	y_pred = np.asarray(y_pred).astype(np.int8).reshape(-1)
	if y_true.shape != y_pred.shape:
		raise ValueError(
			f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")

	tp = int(np.sum((y_true == 1) & (y_pred == 1)))
	tn = int(np.sum((y_true == 0) & (y_pred == 0)))
	fp = int(np.sum((y_true == 0) & (y_pred == 1)))
	fn = int(np.sum((y_true == 1) & (y_pred == 0)))
	return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _safe_div(n: float, d: float) -> float:
	return float(n / d) if d != 0 else float("nan")


def _metrics_from_counts(counts: dict[str, int]) -> dict[str, float]:
	tp = int(counts["tp"])
	tn = int(counts["tn"])
	fp = int(counts["fp"])
	fn = int(counts["fn"])
	total = tp + tn + fp + fn

	precision = _safe_div(tp, tp + fp)
	recall = _safe_div(tp, tp + fn)
	f1 = _safe_div(2 * precision * recall, precision + recall)
	accuracy = _safe_div(tp + tn, total)

	return {
		"tp": float(tp),
		"tn": float(tn),
		"fp": float(fp),
		"fn": float(fn),
		"accuracy": float(accuracy),
		"precision": float(precision),
		"recall": float(recall),
		"f1": float(f1),
	}


def initialize_dataset(sim_config, anomaly_config):
    ds = DataSim(sim_config=sim_config, anomaly_config=anomaly_config)
    return ds


def evaluate_machine_part(ds):
    X_train, _, comp = ds.simulate_data()
    stream = ds.simulate_continuous_data(components=comp)
    X_test = []
    y_test = []
    for _ in range(ds.sim_config.iterations_for_generator * ds.sim_config.samples_per_iteration):
        sample, _, is_anomaly = next(stream)
        X_test.append(sample)
        y_test.append(1 if is_anomaly else 0)

    return X_train, X_test, y_test


def full_eval_pipeline(output_dir: Path | str | None = None):
    """
    maybe one more harder eval after this one?
    """
    sim_config = DataSimConfig(
        random_seed=42,
        mean_range=None, var_range=None,
        means_list=[
            (1, 1, 1, 1, 1),
            (1, 5, 10, 5, 10),
            (10, 10, 10, 10, 10),
            (6, 2, 12, 2, 12),
            (12, 6, 3, 6, 3),
        ],
        vars_list=[
            (0.5, 0.2, 0.7, 0.2, 0.4),
            (0.1, 0.3, 0.5, 0.3, 0.5),
            (0.2, 0.1, 0.5, 0.1, 0.5),
            (0.4, 0.4, 0.2, 0.4, 0.2),
            (0.15, 0.25, 0.35, 0.25, 0.35),
        ],
        num_dimensions=5,
        num_modes=5,
        # 5 Modi => 10 Gewichte (Mode + Transition pro Mode-Paar).
        # Übergänge klein halten, sonst “verschmiert” alles.
        mode_weights=[0.22, 0.02, 0.20, 0.02,
                      0.18, 0.02, 0.16, 0.02, 0.14, 0.02],

        samples_per_iteration=2000,
        iterations_during_training=20,
        iterations_for_generator=400,

        noise_student_t_df=4.0,
        noise_student_t_scale=0.30,
    )


    anomaly_config = AnomalyConfig(
        modes=["mean_shift", "var_shift", "new_mode"],
        stream_anomaly_prob=0.10,
        stream_use_seen_template_prob=0.75,
        max_stream_anomaly_templates=10,

        mean_shift_prob=0.45,
        var_shift_prob=0.45,
        new_mode_prob=0.10,

        mean_shift_scale_min=0.8,
        mean_shift_scale_max=2.2,
        var_scale_min=0.7,
        var_scale_max=2.0,
    )
    """
    sim_config = DataSimConfig(random_seed=BGME_RANDOM_STATE,
                               mean_range=None, var_range=None,
                               means_list=[
                                   (1, 1, 1), (1, 5, 10), (10, 10, 10)],
                               vars_list=[(0.5, 0.2, 0.7),
                                          (0.1, 0.3, 0.5), (0.2, 0.1, 0.5)],
                               num_dimensions=3,
                               num_modes=3,
                               mode_weights=[0.5, 0.05, 0.3, 0.03, 0.11, 0.01],
                               noise_student_t_df=5.0, noise_student_t_scale=0.2,
                               samples_per_iteration=1000,
                               iterations_during_training=20,
                               iterations_for_generator=500
        )

    anomaly_config = AnomalyConfig(stream_anomaly_prob=0.08,
                                   stream_use_seen_template_prob=0.85,
                                   modes=["mean_shift", "var_shift"],
                                   max_stream_anomaly_templates=8,
                                   mean_shift_prob=0.5,
                                   var_shift_prob=0.5,
                                   new_mode_prob=0,
                                   mean_shift_scale_min=0.75,
                                   mean_shift_scale_max=1.5,
                                   var_scale_min=0.75,
                                   var_scale_max=1.5
        )
    """
    ds = initialize_dataset(sim_config, anomaly_config)
    print(f"\nLong Term Evaluation on Simulator started")
    X_train, X_test, y_test = evaluate_machine_part(ds)
    bgme_cfg = BGMEConfig(
        n_estimators=int(BGME_N_ESTIMATORS),
        n_components=int(BGME_N_COMPONENTS),
        covariance_type=str(BGME_COVARIANCE_TYPE),
        reg_covar=float(BGME_REG_COVAR),
        max_iter=int(BGME_MAX_ITER),
        tol=float(BGME_TOL),
        random_state=int(BGME_RANDOM_STATE),
        combine=str(BGME_COMBINE),
        hard_quantile=float(HARD_QUANTILE),
        learning_rate=float(LEARNING_RATE),
        min_weight=float(MIN_WEIGHT),
        max_weight=int(MAX_WEIGHT),
        alpha_decay_rate=ALPHA_DECAY_RATE,
        max_models_per_ensemble=int(MAX_MODELS_PER_ENSEMBLE)
    )

    online_cfg = OnlineAnomalyConfig(
        ll_threshold=None,
        threshold_quantile=float(THRESHOLD_QUANTILE),
        refit_window_size=int(REFIT_WINDOW_SIZE),
        include_correct_predictions=bool(INCLUDE_CORRECT_PREDICTIONS),
        min_samples_for_refit=int(MIN_SAMPLES_FOR_REFIT),
        scoring_mode=str(SCORING_MODE),
        separation_threshold=float(SEPARATION_THRESHOLD)
    )

    online_cfg_correct = OnlineAnomalyConfig(
        ll_threshold=None,
        threshold_quantile=float(THRESHOLD_QUANTILE),
        refit_window_size=int(REFIT_WINDOW_SIZE),
        include_correct_predictions=bool(
            EVAL_INCLUDE_CORRECT_PREDICTIONS_VARIANT),
        min_samples_for_refit=int(MIN_SAMPLES_FOR_REFIT),
        scoring_mode=str(SCORING_MODE),
        separation_threshold=float(SEPARATION_THRESHOLD)
    )

    detector = OnlineBGMEAnomalyDetector(
        normal_config=bgme_cfg,
        online_config=online_cfg,
    ).fit_initial_normal(X_train)

    detector_correct = OnlineBGMEAnomalyDetector(
        normal_config=bgme_cfg,
        online_config=online_cfg_correct,
    ).fit_initial_normal(X_train)

    # Stream test points and evaluate
    y_pred_bgme = np.zeros_like(y_test, dtype=np.int8)
    y_pred_bgme_correct = np.zeros_like(y_test, dtype=np.int8)
    T = len(X_test)
    last_diag = None
    last_diag_correct = None

    for i in range(T):
        if PROGRESS_EVERY and (i > 0) and (i % int(PROGRESS_EVERY) == 0):
            print(f"Processed {i} / {T} test samples...")
        res = detector.predict_one(X_test[i], int(y_test[i]))
        res_correct = detector_correct.predict_one(
            X_test[i], int(y_test[i]))
        last_diag = res
        last_diag_correct = res_correct
        y_pred_bgme[i] = int(res["y_pred"])
        y_pred_bgme_correct[i] = int(res_correct["y_pred"])

        if res.get("refit_happened") and res.get("refit_stats") is not None and PROGRESS_EVERY:
            refit_stats = res["refit_stats"]
            print(
                f"  [BGME default] Refit at sample {i}: TP={refit_stats['TP']}, TN={refit_stats['TN']}, "
                f"FP={refit_stats['FP']}, FN={refit_stats['FN']}, "
                f"Normal ensemble size: {refit_stats['normal_ensemble_size']}, "
                f"Anomaly ensemble size: {refit_stats['anomaly_ensemble_size']}"
            )

        if res_correct.get("refit_happened") and res_correct.get("refit_stats") is not None and PROGRESS_EVERY:
            refit_stats = res_correct["refit_stats"]
            print(
                f"  [BGME +correct] Refit at sample {i}: TP={refit_stats['TP']}, TN={refit_stats['TN']}, "
                f"FP={refit_stats['FP']}, FN={refit_stats['FN']}, "
                f"Normal ensemble size: {refit_stats['normal_ensemble_size']}, "
                f"Anomaly ensemble size: {refit_stats['anomaly_ensemble_size']}"
            )

    # Optionally refit on any remaining samples in the final incomplete window
    final_stats = detector.flush_window_and_refit()
    if final_stats is not None and PROGRESS_EVERY:
        print(
            f"  [BGME default] Final refit: TP={final_stats['TP']}, TN={final_stats['TN']}, "
            f"FP={final_stats['FP']}, FN={final_stats['FN']}"
        )

    final_stats_correct = detector_correct.flush_window_and_refit()
    if final_stats_correct is not None and PROGRESS_EVERY:
        print(
            f"  [BGME +correct] Final refit: TP={final_stats_correct['TP']}, TN={final_stats_correct['TN']}, "
            f"FP={final_stats_correct['FP']}, FN={final_stats_correct['FN']}"
        )

    counts_bgme = _confusion_counts(y_test, y_pred_bgme)
    metrics_bgme = _metrics_from_counts(counts_bgme)

    counts_bgme_correct = _confusion_counts(y_test, y_pred_bgme_correct)
    metrics_bgme_correct = _metrics_from_counts(counts_bgme_correct)

    # Baseline: single normal GMM fitted on the same training data
    gmm = GaussianMixture(
        n_components=int(BASELINE_GMM_N_COMPONENTS),
        covariance_type=str(bgme_cfg.covariance_type),
        reg_covar=float(bgme_cfg.reg_covar),
        max_iter=int(bgme_cfg.max_iter),
        tol=float(bgme_cfg.tol),
        random_state=bgme_cfg.random_state,
    ).fit(X_train)

    ll_train_gmm = gmm.score_samples(X_train)
    ll_threshold_gmm = float(np.quantile(
        ll_train_gmm, float(THRESHOLD_QUANTILE)))
    ll_test_gmm = gmm.score_samples(X_test)
    y_pred_gmm = (ll_test_gmm < ll_threshold_gmm).astype(np.int8)

    counts_gmm = _confusion_counts(y_test, y_pred_gmm)
    metrics_gmm = _metrics_from_counts(counts_gmm)

    print("ServerMachineDataset evaluation")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test  shape: {len(X_test)}")
    print(f"  threshold_quantile: {THRESHOLD_QUANTILE}")
    if last_diag is not None:
        if last_diag.get("ll_threshold") is not None:
            print(
                f"  ll_threshold (online BGME default): {float(last_diag['ll_threshold']):.6f}")
    if last_diag_correct is not None:
        if last_diag_correct.get("ll_threshold") is not None:
            print(
                f"  ll_threshold (online BGME +correct): {float(last_diag_correct['ll_threshold']):.6f}")
    print(f"  ll_threshold (baseline GMM): {ll_threshold_gmm:.6f}")
    print(f"  scoring_mode: {SCORING_MODE}")

    print("\nConfusion Matrix (positive=anomaly=1)")
    print("\nOnline BGME (default)")
    print(f"  True Positives  (TP): {int(metrics_bgme['tp']):7d}")
    print(f"  True Negatives  (TN): {int(metrics_bgme['tn']):7d}")
    print(f"  False Positives (FP): {int(metrics_bgme['fp']):7d}")
    print(f"  False Negatives (FN): {int(metrics_bgme['fn']):7d}")
    print(f"  accuracy : {metrics_bgme['accuracy']:.6f}")
    print(f"  precision: {metrics_bgme['precision']:.6f}")
    print(f"  recall   : {metrics_bgme['recall']:.6f}")
    print(f"  f1       : {metrics_bgme['f1']:.6f}")

    print("\nOnline BGME (+correct)")
    print(f"  True Positives  (TP): {int(metrics_bgme_correct['tp']):7d}")
    print(f"  True Negatives  (TN): {int(metrics_bgme_correct['tn']):7d}")
    print(f"  False Positives (FP): {int(metrics_bgme_correct['fp']):7d}")
    print(f"  False Negatives (FN): {int(metrics_bgme_correct['fn']):7d}")
    print(f"  accuracy : {metrics_bgme_correct['accuracy']:.6f}")
    print(f"  precision: {metrics_bgme_correct['precision']:.6f}")
    print(f"  recall   : {metrics_bgme_correct['recall']:.6f}")
    print(f"  f1       : {metrics_bgme_correct['f1']:.6f}")

    print("\nBaseline single GMM")
    print(f"  True Positives  (TP): {int(metrics_gmm['tp']):7d}")
    print(f"  True Negatives  (TN): {int(metrics_gmm['tn']):7d}")
    print(f"  False Positives (FP): {int(metrics_gmm['fp']):7d}")
    print(f"  False Negatives (FN): {int(metrics_gmm['fn']):7d}")
    print(f"  accuracy : {metrics_gmm['accuracy']:.6f}")
    print(f"  precision: {metrics_gmm['precision']:.6f}")
    print(f"  recall   : {metrics_gmm['recall']:.6f}")
    print(f"  f1       : {metrics_gmm['f1']:.6f}")

    # JSON
    # Includes: (1) machine/part, (2) all-uppercase config values in this module,
    # (3) model configs, and (4) confusion matrices/metrics per model.
    run_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "name": "longterm_sim_evaluation",
        "shapes": {
            "X_train": list(X_train.shape),
            "X_test": len(X_test),
            "y_test": len(y_test),
        },
        "sim_config": asdict(sim_config),
        "anomaly_config": asdict(anomaly_config),

        "model_params": {
            "bgme_cfg": asdict(bgme_cfg),
            "online_cfg_default": asdict(online_cfg),
            "online_cfg_correct": asdict(online_cfg_correct),
            "baseline_gmm": {
                "n_components": int(BASELINE_GMM_N_COMPONENTS),
                "covariance_type": str(bgme_cfg.covariance_type),
                "reg_covar": float(bgme_cfg.reg_covar),
                "max_iter": int(bgme_cfg.max_iter),
                "tol": float(bgme_cfg.tol),
                "random_state": bgme_cfg.random_state,
            },
        },
        "thresholds": {
            "online_bgme_default_ll_threshold": None if last_diag is None else _json_safe(last_diag.get("ll_threshold")),
            "online_bgme_correct_ll_threshold": None if last_diag_correct is None else _json_safe(last_diag_correct.get("ll_threshold")),
            "baseline_gmm_ll_threshold": float(ll_threshold_gmm),
        },
        "results": {
            "bgme_default": metrics_bgme,
            "bgme_correct": metrics_bgme_correct,
            "baseline_gmm": metrics_gmm,
        },
        "confusion_matrices": {
            "bgme_default": counts_bgme,
            "bgme_correct": counts_bgme_correct,
            "baseline_gmm": counts_gmm,
        },
    }

    resolved_output_dir = output_dir if output_dir is not None else EVAL_JSON_OUTPUT_DIR
    save_evaluation_run_json(output_dir=resolved_output_dir, payload=run_payload)


if __name__ == "__main__":
    full_eval_pipeline()
