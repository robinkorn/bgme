from sklearn.mixture import GaussianMixture
import numpy as np

from src.models.BGMEConfig import BGMEConfig
from src.models.OnlineAnomalyConfig import OnlineAnomalyConfig


# BGME
class BoostedGaussianMixtureEnsemble:
    def __init__(self, config: BGMEConfig):
        self.config = config
        self.models = []
        self.alphas = None

        self._validate_config()

    def _validate_config(self) -> None:
        if self.config.alpha_decay_rate is not None:
            r = float(self.config.alpha_decay_rate)
            if not (0.0 < r <= 1.0):
                raise ValueError(f"alpha_decay_rate must be in (0, 1], got {self.config.alpha_decay_rate}")

        if self.config.max_models_per_ensemble is not None:
            m = int(self.config.max_models_per_ensemble)
            if m <= 0:
                raise ValueError(f"max_models_per_ensemble must be >= 1, got {self.config.max_models_per_ensemble}")

    def fit(self, X):
        X = self._check_X_fit(X)
        n = X.shape[0]

        w = np.full(n, 1.0 / n, dtype=np.float64)
        self.models = []

        rng = np.random.default_rng(self.config.random_state)

        for k in range(self.config.n_estimators):
            gmm = GaussianMixture(
                n_components=self.config.n_components,
                covariance_type=self.config.covariance_type,
                reg_covar=self.config.reg_covar,
                max_iter=self.config.max_iter,
                tol=self.config.tol,
                random_state=None if self.config.random_state is None else int(self.config.random_state) + k,
            )

            idx = rng.choice(np.arange(n), size=n, replace=True, p=w)
            X_resampled = X[idx]
            gmm.fit(X_resampled)

            self.models.append(gmm)

            ll = gmm.score_samples(X)
            q = np.quantile(ll, 1.0 - self.config.hard_quantile)
            hard = ll <= q

            w = w * np.exp(self.config.learning_rate * hard.astype(np.float64))
            w = np.clip(w, self.config.min_weight, self.config.max_weight)
            w = w / np.sum(w)

        self.alphas = np.full(len(self.models), 1.0 / len(self.models), dtype=np.float64)
        self._prune_to_max_models()
        self._renormalize_alphas()
        return self

    def score_samples(self, X):
        """
        Ensemble log-likelihood per sample. Higher => more like training distribution.
        """
        self._require_fitted()
        X = self._check_X_score(X)

        ll_stack = np.vstack([m.score_samples(X) for m in self.models])  # (K, n)

        if self.config.combine == "mean_loglik":
            return np.average(ll_stack, axis=0, weights=self.alphas)

        if self.config.combine == "logmeanexp":
            a = self.alphas
            m = np.max(ll_stack, axis=0)
            return m + np.log(np.sum(a[:, None] * np.exp(ll_stack - m[None, :]), axis=0))

        raise ValueError(f"Unknown combine mode: {self.config.combine}")

    def anomaly_score(self, X):
        return -self.score_samples(X)

    def add_model(self, X):
        """
        Train a new single GMM on provided data and add it to the ensemble.
                - If config.alpha_decay_rate is None: rebalances alphas to equal weights (legacy behavior).
                - Else: applies alpha decay to existing models, adds the new model with weight 1.0,
                    prunes to config.max_models_per_ensemble (if set), then renormalizes.
        """
        X = self._check_X_fit(X)

        new_gmm = GaussianMixture(
            n_components=self.config.n_components,
            covariance_type=self.config.covariance_type,
            reg_covar=self.config.reg_covar,
            max_iter=self.config.max_iter,
            tol=self.config.tol,
            random_state=None if self.config.random_state is None else int(self.config.random_state) + len(self.models),
        )
        new_gmm.fit(X)

        # Add with either legacy equal-weighting, or decayed-weighting.
        if self.alphas is None:
            # Ensemble may be empty (e.g. anomaly model before first refit)
            self.models.append(new_gmm)
            self.alphas = np.array([1.0], dtype=np.float64)
            self._prune_to_max_models()
            self._renormalize_alphas()
            return self

        if self.config.alpha_decay_rate is None:
            self.models.append(new_gmm)
            self.alphas = np.full(len(self.models), 1.0 / len(self.models), dtype=np.float64)
            self._prune_to_max_models()
            self._renormalize_alphas()
            return self

        decay = float(self.config.alpha_decay_rate)
        self.alphas = self.alphas * decay
        self.models.append(new_gmm)
        self.alphas = np.append(self.alphas, 1.0).astype(np.float64, copy=False)
        self._prune_to_max_models()
        self._renormalize_alphas()
        return self

    def _prune_to_max_models(self) -> None:
        max_models = self.config.max_models_per_ensemble
        if max_models is None:
            return

        max_models = int(max_models)
        if len(self.models) <= max_models:
            return

        n_remove = int(len(self.models) - max_models)

        if self.alphas is None or len(self.alphas) != len(self.models):
            # Fallback: remove oldest models
            self.models = list(self.models[n_remove:])
            if self.models:
                self.alphas = np.full(len(self.models), 1.0 / len(self.models), dtype=np.float64)
            else:
                self.alphas = None
            return

        # Remove lowest-alpha models; stable => if ties, drop older ones first.
        order = np.argsort(self.alphas, kind="stable")
        remove_idx = set(int(i) for i in order[:n_remove])
        self.models = [m for i, m in enumerate(self.models) if i not in remove_idx]
        self.alphas = np.asarray([a for i, a in enumerate(self.alphas) if i not in remove_idx], dtype=np.float64)

    def _renormalize_alphas(self) -> None:
        if not self.models:
            self.alphas = None
            return

        if self.alphas is None or len(self.alphas) != len(self.models):
            self.alphas = np.full(len(self.models), 1.0 / len(self.models), dtype=np.float64)
            return

        s = float(np.sum(self.alphas))
        if not np.isfinite(s) or s <= 0.0:
            self.alphas = np.full(len(self.models), 1.0 / len(self.models), dtype=np.float64)
            return

        self.alphas = (self.alphas / s).astype(np.float64, copy=False)

    def _require_fitted(self):
        if not self.models:
            raise RuntimeError("Model is not fitted yet. Call fit(X) first.")

    @staticmethod
    def _check_X_fit(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"Expected X to be 2D array (n_samples, n_features), got {X.shape}")
        if X.shape[0] < 2:
            raise ValueError("Need at least 2 samples to fit a GMM.")
        return X

    @staticmethod
    def _check_X_score(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"Expected X to be 2D array (n_samples, n_features), got {X.shape}")
        if X.shape[0] < 1:
            raise ValueError("Need at least 1 sample to score.")
        return X


# Online detector with INTERNAL auto-refit
class OnlineBGMEAnomalyDetector:
    """
    NORMAL BGME + ANOMALY BGME with periodic INTERNAL updates.

    API:
      - predict_one(x, y_true): predicts, stores (x, y_pred, y_true),
        and when the window reaches refit_window_size it auto-refits and clears the window.
    """

    def __init__(
        self,
        normal_config: BGMEConfig,
        online_config: OnlineAnomalyConfig,
        anomaly_config: BGMEConfig | None = None,
    ):
        self.normal_bgme = BoostedGaussianMixtureEnsemble(normal_config)
        self.anomaly_bgme = BoostedGaussianMixtureEnsemble(anomaly_config or normal_config)
        self.online = online_config

        self._seen_dim = None
        self._threshold_ll = self.online.ll_threshold

        # anomaly model starts unfitted/empty
        self._anomaly_model_fitted = False

        # current block memory
        self._win_X = []
        self._win_pred = []
        self._win_true = []
        self._win_outcome = []

        # counters
        self._total_predictions = 0
        self._total_refits = 0

        # cumulative confusion
        self._tp_count = 0
        self._tn_count = 0
        self._fp_count = 0
        self._fn_count = 0

        if int(self.online.refit_window_size) <= 0:
            raise ValueError("refit_window_size must be > 0")
        if int(self.online.min_samples_for_refit) < 2:
            raise ValueError("min_samples_for_refit must be >= 2 to fit a GMM")

    def fit_initial_normal(self, X_normal):
        """
        Fit normal BGME on initial normal data.
        If ll_threshold is None, estimate from X_normal using threshold_quantile.
        """
        X_normal = np.asarray(X_normal, dtype=np.float64)
        if X_normal.ndim != 2 or X_normal.shape[0] < 2:
            raise ValueError("X_normal must be (n_samples, n_features) with n_samples>=2.")

        self.normal_bgme.fit(X_normal)
        self._seen_dim = int(X_normal.shape[1])

        if self._threshold_ll is None:
            ll = self.normal_bgme.score_samples(X_normal)
            self._threshold_ll = float(np.quantile(ll, float(self.online.threshold_quantile)))

        return self

    def predict_one(self, x, y_true: int):
        """
        Predict a single point, store it with its label, and auto-refit when window is full.

        y_true convention:
          - 0 = normal
          - 1 = anomaly

        Returns:
          dict with prediction info, plus:
            - refit_happened: bool
            - refit_stats: dict | None
        """
        self._require_normal_fitted()

        x = self._check_x(x)
        y_true = int(y_true)
        if y_true not in (0, 1):
            raise ValueError(f"y_true must be 0 or 1, got {y_true}")

        ll_norm = float(self.normal_bgme.score_samples(x[None, :])[0])
        is_anom_by_threshold = (ll_norm < float(self._threshold_ll))

        ll_anom = None
        separation_score = None

        if self._anomaly_model_fitted:
            ll_anom = float(self.anomaly_bgme.score_samples(x[None, :])[0])
            separation_score = float(ll_anom - ll_norm)

        # decision + output score
        if self.online.scoring_mode == "normal_only":
            y_pred = int(is_anom_by_threshold)
            anomaly_score = -ll_norm

        elif self.online.scoring_mode == "separation":
            if separation_score is None:
                y_pred = int(is_anom_by_threshold)
                anomaly_score = -ll_norm
            else:
                y_pred = int(separation_score > float(self.online.separation_threshold))
                anomaly_score = float(separation_score)

        else:
            raise ValueError(f"Unknown scoring_mode: {self.online.scoring_mode}")

        # store into internal window (includes y_true)
        self._win_X.append(x.copy())
        self._win_pred.append(int(y_pred))
        self._win_true.append(int(y_true))

        # store per-sample confusion outcome
        if int(y_pred) == 1 and int(y_true) == 1:
            outcome = "TP"
            self._tp_count += 1
        elif int(y_pred) == 0 and int(y_true) == 0:
            outcome = "TN"
            self._tn_count += 1
        elif int(y_pred) == 1 and int(y_true) == 0:
            outcome = "FP"
            self._fp_count += 1
        else:
            outcome = "FN"
            self._fn_count += 1

        self._win_outcome.append(outcome)
        self._total_predictions += 1

        window_len_after_store = int(len(self._win_X))

        refit_happened = False
        refit_stats = None

        # auto-refit when window full
        if len(self._win_X) >= int(self.online.refit_window_size):
            refit_stats = self._end_window_and_refit_internal()
            refit_happened = True

        return {
            "y_pred": int(y_pred),
            "y_true": int(y_true),
            "is_anomaly_pred": bool(y_pred == 1),
            "is_anomaly": bool(y_pred == 1),
            "confusion": outcome,
            "ll_normal": ll_norm,
            "ll_threshold": float(self._threshold_ll),
            "anomaly_score": float(anomaly_score),
            "ll_anomaly": ll_anom,
            "separation_score": separation_score,
            "window_len": int(window_len_after_store),
            "refit_window_size": int(self.online.refit_window_size),
            "anomaly_model_fitted": bool(self._anomaly_model_fitted),
            "refit_happened": bool(refit_happened),
            "block_full": bool(refit_happened),
            "refit_stats": refit_stats,
        }

    def flush_window_and_refit(self):
        """Force a refit on the currently stored (possibly incomplete) window.

        Returns:
          dict with the same structure as the internal refit stats, or None if the window is empty.
        """
        if len(self._win_X) == 0:
            return None
        return self._end_window_and_refit_internal()

    # refit
    def _end_window_and_refit_internal(self):
        """
        Uses stored (X, y_pred, y_true) of current window.
        Updates ensembles and clears the window.
        """
        if len(self._win_X) == 0:
            raise RuntimeError("Internal error: window empty at refit time.")

        X = np.vstack(self._win_X)
        y_pred = np.asarray(self._win_pred, dtype=int)
        y_true = np.asarray(self._win_true, dtype=int)

        if len(self._win_outcome) != int(X.shape[0]):
            raise RuntimeError("Internal error: window outcome length mismatch.")

        tp_mask = (y_pred == 1) & (y_true == 1)
        tn_mask = (y_pred == 0) & (y_true == 0)
        fp_mask = (y_pred == 1) & (y_true == 0)
        fn_mask = (y_pred == 0) & (y_true == 1)

        # window-only counts (cumulative counters are updated per prediction in predict_one)
        TP = int(np.sum(tp_mask))
        TN = int(np.sum(tn_mask))
        FP = int(np.sum(fp_mask))
        FN = int(np.sum(fn_mask))

        # build training sets according to your rules
        if bool(self.online.include_correct_predictions):
            normal_train = X[fp_mask | tn_mask]
            anomaly_train = X[fn_mask | tp_mask]
        else:
            normal_train = X[fp_mask]
            anomaly_train = X[fn_mask]

        # add models
        if normal_train.shape[0] >= int(self.online.min_samples_for_refit):
            self.normal_bgme.add_model(normal_train)

        if anomaly_train.shape[0] >= int(self.online.min_samples_for_refit):
            if not self._anomaly_model_fitted:
                self.anomaly_bgme.add_model(anomaly_train)
                self._anomaly_model_fitted = True
            else:
                self.anomaly_bgme.add_model(anomaly_train)

        self._total_refits += 1

        # clear window
        self._win_X.clear()
        self._win_pred.clear()
        self._win_true.clear()
        self._win_outcome.clear()

        return {
            "window_size": int(X.shape[0]),
            "TP": TP,
            "TN": TN,
            "FP": FP,
            "FN": FN,
            "cumulative_TP": int(self._tp_count),
            "cumulative_TN": int(self._tn_count),
            "cumulative_FP": int(self._fp_count),
            "cumulative_FN": int(self._fn_count),
            "normal_ensemble_size": int(len(self.normal_bgme.models)),
            "anomaly_ensemble_size": int(len(self.anomaly_bgme.models)) if self._anomaly_model_fitted else 0,
            "anomaly_model_fitted": bool(self._anomaly_model_fitted),
            "total_refits": int(self._total_refits),
        }

    # helpers
    def _check_x(self, x):
        x = np.asarray(x, dtype=np.float64)

        if x.ndim == 2:
            if x.shape[0] == 1:
                x = x.reshape(-1)
            else:
                raise ValueError(f"Expected x to be 1D (n_features,) or (1, n_features), got {x.shape}")
        elif x.ndim != 1:
            raise ValueError(f"Expected x to be 1D (n_features,) or (1, n_features), got {x.shape}")

        if x.size < 1:
            raise ValueError("x must have at least 1 feature.")

        if self._seen_dim is not None and int(x.shape[0]) != int(self._seen_dim):
            raise ValueError(f"x has wrong feature dimension: expected {self._seen_dim}, got {x.shape[0]}")

        return x

    def _require_normal_fitted(self):
        self.normal_bgme._require_fitted()
        if self._threshold_ll is None:
            raise RuntimeError(
                "Threshold not initialized. Call fit_initial_normal(...) first or set ll_threshold."
            )


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    Xn = rng.normal(0, 1, size=(5000, 4))

    normal_cfg = BGMEConfig(n_estimators=5, n_components=3, random_state=42)
    online_cfg = OnlineAnomalyConfig(
        ll_threshold=None,
        threshold_quantile=0.01,
        refit_window_size=1000,
        include_correct_predictions=False,
        min_samples_for_refit=20,
        scoring_mode="separation",
        separation_threshold=0.0,
    )

    det = OnlineBGMEAnomalyDetector(normal_cfg, online_cfg).fit_initial_normal(Xn)

    # stream: 
    X_stream = rng.normal(0, 1, size=(3000, 4))
    y_stream = np.zeros(3000, dtype=int)
    y_stream[rng.choice(np.arange(3000), size=120, replace=False)] = 1

    for i in range(3000):
        out = det.predict_one(X_stream[i], y_stream[i])
        if out["refit_happened"]:
            print("AUTO-REFIT:", out["refit_stats"])
