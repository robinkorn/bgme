from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OnlineAnomalyConfig:
    # Thresholding
    # If you set ll_threshold=None, a threshold will be estimated from the training data
    # using threshold_quantile (lower tail of ll => anomalous).
    ll_threshold: float | None = None
    threshold_quantile: float = 0.01

    # Periodic retraining based on prediction outcomes
    # After every refit_window_size predictions, true labels are provided and models are updated
    refit_window_size: int = 1000
    
    # Whether to include correct predictions (TP, TN) in retraining, or only errors (FP, FN)
    # If False: only FP added to normal ensemble, only FN added to anomaly ensemble
    # If True: FP+TN added to normal ensemble, FN+TP added to anomaly ensemble
    include_correct_predictions: bool = False
    
    # Minimum samples needed to train a new GMM for each category
    min_samples_for_refit: int = 20

    # What to do when anomaly model exists:
    # - "normal_only": ignore anomaly model, just keep it around
    # - "separation": use both models; anomaly_score = ll_anom - ll_norm (bigger => more anomalous)
    scoring_mode: str = "separation"  # "normal_only" | "separation"

    # Used when scoring_mode == "separation":
    # predict anomaly if (ll_anomaly - ll_normal) > separation_threshold
    separation_threshold: float = 0.0
