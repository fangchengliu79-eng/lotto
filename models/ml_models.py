"""
Machine learning models for lottery prediction - fully parameterized by cfg.

Implements LSTM, XGBoost, and RandomForest models.
All input/output dimensions are parameterized based on cfg.main_count + cfg.sub_count.
Works for any lottery type (DLT, SSQ, etc.) via cfg parameterization.
"""
import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from utils.helpers import validate_numbers, get_logger


# ---------------------------------------------------------------------------
# Feature / Sequence helpers
# ---------------------------------------------------------------------------

def _build_sequences(
    main_nums: np.ndarray,
    sub_nums: np.ndarray,
    cfg,
    seq_length: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build feature/label sequences for supervised learning.

    For each position i, the feature is a window of seq_length preceding draws.
    Label is the next draw's numbers (concatenated: main + sub).

    Returns
    -------
    (X, y) where:
        X : shape (n_sequences, seq_length, main_count + sub_count)
        y : shape (n_sequences, main_count + sub_count)
    """
    all_nums = np.concatenate([main_nums, sub_nums], axis=1)  # (n_draws, total_count)
    n_draws = len(all_nums)
    total_count = cfg.main_count + cfg.sub_count

    X, y = [], []
    for i in range(seq_length, n_draws):
        X.append(all_nums[i - seq_length:i])  # window of seq_length
        y.append(all_nums[i])                 # next draw

    if not X:
        return np.empty((0, seq_length, total_count)), np.empty((0, total_count))

    return np.array(X), np.array(y)


def _build_features_from_nums(
    main_nums: np.ndarray,
    sub_nums: np.ndarray,
    cfg,
) -> np.ndarray:
    """Build flat feature vectors for tree-based models.

    For each draw, features are: the draw numbers (flattened).
    For tree models like XGBoost/RF, we use rolling windows of size 3.
    """
    all_nums = np.concatenate([main_nums, sub_nums], axis=1)
    total_count = cfg.main_count + cfg.sub_count
    n_draws = len(all_nums)

    features = []
    labels = []
    window_size = 3
    for i in range(window_size, n_draws):
        # Feature: last window_size draws flattened
        feat = all_nums[i - window_size:i].flatten()  # shape: (window_size * total_count,)
        features.append(feat)
        labels.append(all_nums[i])

    if not features:
        return np.empty((0, 0)), np.empty((0, 0))

    return np.array(features), np.array(labels)


# ---------------------------------------------------------------------------
# Base ML Model
# ---------------------------------------------------------------------------

class BaseMLModel(ABC):
    """Abstract base for all ML prediction models."""

    def __init__(self, cfg, name: str = "ml_base"):
        self.cfg = cfg
        self.name = name
        self.logger = get_logger(cfg)
        self._fitted = False
        self._total_count = cfg.main_count + cfg.sub_count

    @abstractmethod
    def fit(self, main_nums: np.ndarray, sub_nums: np.ndarray) -> None:
        ...

    @abstractmethod
    def predict_main(self, n_predictions: int = 1) -> List[List[int]]:
        ...

    @abstractmethod
    def predict_sub(self, n_predictions: int = 1) -> List[List[int]]:
        ...

    def predict(self, n_predictions: int = 1) -> List[Dict[str, Any]]:
        mains = self.predict_main(n_predictions)
        subs = self.predict_sub(n_predictions)
        return [
            {"main": mains[i], "sub": subs[i], "model": self.name}
            for i in range(n_predictions)
        ]

    def _clip_predictions(
        self,
        raw: np.ndarray,
        min_val: int,
        max_val: int,
    ) -> np.ndarray:
        """Clip raw predictions to valid range and round to integers."""
        clipped = np.clip(np.round(raw).astype(int), min_val, max_val)
        return clipped

    def _ensure_unique_sorted(
        self,
        arr: np.ndarray,
        count: int,
        min_val: int,
        max_val: int,
    ) -> List[int]:
        """Ensure a prediction has exactly `count` unique numbers in range.

        If there are duplicates or insufficient numbers, fills with the
        most commonly predicted numbers that aren't already included.
        """
        unique = sorted(set(arr))
        unique = [n for n in unique if min_val <= n <= max_val]

        if len(unique) >= count:
            return unique[:count]

        # Fill missing slots
        missing_count = count - len(unique)
        # Use median of valid range as defaults
        filler = [
            n for n in range(min_val, max_val + 1)
            if n not in unique
        ]
        # Pick spread-out numbers
        step = max(1, len(filler) // max(missing_count, 1))
        extras = [filler[min(i * step, len(filler) - 1)] for i in range(missing_count)]
        return sorted(unique + extras)


# ---------------------------------------------------------------------------
# LSTM Model
# ---------------------------------------------------------------------------

class LSTM(BaseMLModel):
    """LSTM neural network for sequence prediction.

    Uses a simple recurrent architecture: Embedding -> LSTM -> Dense.
    Input dimension: seq_length x (main_count + sub_count)
    Output dimension: main_count + sub_count (multi-output regression)

    Falls back to a heuristic time-series approach if tensorflow/keras
    is not available.
    """

    def __init__(self, cfg, seq_length: int = 10, epochs: int = None, batch_size: int = None):
        super().__init__(cfg, name="lstm")
        self.seq_length = seq_length
        self.epochs = epochs or cfg.lstm_epochs
        self.batch_size = batch_size or cfg.lstm_batch_size
        self._model = None
        self._has_keras = False
        self._last_window: Optional[np.ndarray] = None

        # Check for keras availability
        try:
            import tensorflow as tf  # noqa: F401
            self._has_keras = True
        except ImportError:
            try:
                import keras  # noqa: F401
                self._has_keras = True
            except ImportError:
                self._has_keras = False

    def fit(self, main_nums: np.ndarray, sub_nums: np.ndarray) -> None:
        """Fit LSTM on historical sequences.

        Falls back to simple frequency analysis if keras unavailable.
        """
        self._total_count = self.cfg.main_count + self.cfg.sub_count

        if self._has_keras:
            self._fit_keras(main_nums, sub_nums)
        else:
            self._fit_fallback(main_nums, sub_nums)

    def _fit_keras(self, main_nums: np.ndarray, sub_nums: np.ndarray) -> None:
        """Fit using Keras/TensorFlow LSTM."""
        try:
            import tensorflow as tf
        except ImportError:
            import keras as tf  # type: ignore

        from keras.models import Sequential
        from keras.layers import LSTM as KERAS_LSTM, Dense, Embedding, Flatten, Dropout
        from keras.callbacks import EarlyStopping

        X, y = _build_sequences(main_nums, sub_nums, self.cfg, self.seq_length)

        if len(X) < 5:
            self.logger.warning("Too few sequences (%d) for LSTM, using fallback", len(X))
            self._fit_fallback(main_nums, sub_nums)
            return

        # Normalize targets
        self._y_mean = np.mean(y, axis=0)
        self._y_std = np.std(y, axis=0) + 1e-8
        y_norm = (y - self._y_mean) / self._y_std

        total_count = self._total_count

        model = Sequential()
        model.add(KERAS_LSTM(
            64, input_shape=(self.seq_length, total_count), return_sequences=True
        ))
        model.add(Dropout(0.2))
        model.add(KERAS_LSTM(32, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(total_count))

        model.compile(optimizer="adam", loss="mse")
        es = EarlyStopping(patience=5, restore_best_weights=True)

        model.fit(
            X, y_norm,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            callbacks=[es],
            verbose=0,
        )

        self._model = model
        self._last_window = X[-1:] if len(X) > 0 else None
        self._fitted = True
        self.logger.info("LSTM fitted on %d sequences for %s", len(X), self.cfg.name)

    def _fit_fallback(self, main_nums: np.ndarray, sub_nums: np.ndarray) -> None:
        """Fallback when Keras is unavailable: use weighted recent frequency."""
        from collections import Counter

        total_draws = len(main_nums)
        # Weight recent draws more heavily
        self._main_weights = {}
        self._sub_weights = {}

        for idx in range(total_draws):
            weight = (idx + 1) / total_draws  # recent draws weighted higher
            for n in main_nums[idx]:
                self._main_weights[n] = self._main_weights.get(n, 0) + weight
            for n in sub_nums[idx]:
                self._sub_weights[n] = self._sub_weights.get(n, 0) + weight

        # Normalize
        total_main = sum(self._main_weights.values()) or 1
        total_sub = sum(self._sub_weights.values()) or 1
        self._main_probs = {
            n: w / total_main
            for n, w in self._main_weights.items()
        }
        self._sub_probs = {
            n: w / total_sub
            for n, w in self._sub_weights.items()
        }

        self._fitted = True
        self.logger.info(
            "LSTM (fallback frequency mode) fitted on %d draws for %s",
            total_draws, self.cfg.name,
        )

    def predict_main(self, n_predictions: int = 1) -> List[List[int]]:
        if not self._fitted:
            raise RuntimeError("LSTM not fitted. Call fit() first.")

        if self._has_keras and self._model is not None and self._last_window is not None:
            return self._predict_keras_main(n_predictions)
        else:
            return self._predict_fallback_main(n_predictions)

    def _predict_keras_main(self, n_predictions: int) -> List[List[int]]:
        """Make predictions using trained Keras model."""
        predictions = []
        for _ in range(n_predictions):
            raw = self._model.predict(self._last_window, verbose=0)[0]
            # Denormalize
            denorm = raw * self._y_std + self._y_mean
            main_raw = denorm[:self.cfg.main_count]
            main_clipped = self._clip_predictions(
                main_raw, self.cfg.main_min, self.cfg.main_max
            )
            pred = self._ensure_unique_sorted(
                main_clipped, self.cfg.main_count,
                self.cfg.main_min, self.cfg.main_max,
            )
            predictions.append(pred)
        return predictions

    def _predict_fallback_main(self, n_predictions: int) -> List[List[int]]:
        """Weighted sampling using fitted probabilities."""
        numbers = list(range(self.cfg.main_min, self.cfg.main_max + 1))
        probs = np.array([
            self._main_probs.get(n, 0) for n in numbers
        ])
        probs = probs / probs.sum()  # normalize

        predictions = []
        for _ in range(n_predictions):
            pred = list(np.random.choice(
                numbers, size=self.cfg.main_count, replace=False, p=probs,
            ))
            predictions.append(sorted(pred))
        return predictions

    def predict_sub(self, n_predictions: int = 1) -> List[List[int]]:
        if not self._fitted:
            raise RuntimeError("LSTM not fitted. Call fit() first.")

        if self._has_keras and self._model is not None and self._last_window is not None:
            return self._predict_keras_sub(n_predictions)
        else:
            return self._predict_fallback_sub(n_predictions)

    def _predict_keras_sub(self, n_predictions: int) -> List[List[int]]:
        predictions = []
        for _ in range(n_predictions):
            raw = self._model.predict(self._last_window, verbose=0)[0]
            denorm = raw * self._y_std + self._y_mean
            sub_raw = denorm[self.cfg.main_count:self._total_count]
            sub_clipped = self._clip_predictions(
                sub_raw, self.cfg.sub_min, self.cfg.sub_max
            )
            pred = self._ensure_unique_sorted(
                sub_clipped, self.cfg.sub_count,
                self.cfg.sub_min, self.cfg.sub_max,
            )
            predictions.append(pred)
        return predictions

    def _predict_fallback_sub(self, n_predictions: int) -> List[List[int]]:
        numbers = list(range(self.cfg.sub_min, self.cfg.sub_max + 1))
        probs = np.array([
            self._sub_probs.get(n, 0) for n in numbers
        ])
        probs = probs / probs.sum()

        predictions = []
        for _ in range(n_predictions):
            pred = list(np.random.choice(
                numbers, size=self.cfg.sub_count, replace=False, p=probs,
            ))
            predictions.append(sorted(pred))
        return predictions


# ---------------------------------------------------------------------------
# XGBoost Model
# ---------------------------------------------------------------------------

class XGBoost(BaseMLModel):
    """XGBoost gradient boosting for lottery number prediction.

    Uses multi-output regression (one target per number position).
    Falls back to RandomForestRegressor if xgboost not installed.
    Input dimension: window_size * (main_count + sub_count)
    Output dimension: main_count + sub_count
    """

    def __init__(self, cfg, n_estimators: int = None, max_depth: int = None):
        super().__init__(cfg, name="xgboost")
        self.n_estimators = n_estimators or cfg.xgb_params.get("n_estimators", 200)
        self.max_depth = max_depth or cfg.xgb_params.get("max_depth", 6)
        self._model = None
        self._has_xgb = False

        try:
            import xgboost  # noqa: F401
            self._has_xgb = True
        except ImportError:
            self._has_xgb = False

    def fit(self, main_nums: np.ndarray, sub_nums: np.ndarray) -> None:
        self._total_count = self.cfg.main_count + self.cfg.sub_count

        if self._has_xgb:
            self._fit_xgb(main_nums, sub_nums)
        else:
            self._fit_sklearn(main_nums, sub_nums)

    def _fit_xgb(self, main_nums: np.ndarray, sub_nums: np.ndarray) -> None:
        """Fit using XGBoost regressor (multi-output via wrapping)."""
        import xgboost as xgb

        X, y = _build_features_from_nums(main_nums, sub_nums, self.cfg)

        if len(X) < 5:
            self.logger.warning("Too few samples (%d) for XGBoost, using fallback", len(X))
            self._fit_sklearn(main_nums, sub_nums)
            return

        # XGBoost supports multi-target with xgboost>=1.7
        try:
            self._model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=0.1,
                random_state=42,
                verbosity=0,
                objective="reg:squarederror",
            )
            self._model.fit(X, y)
        except Exception as exc:
            self.logger.warning("XGBoost multi-target failed: %s, using sklearn fallback", exc)
            self._fit_sklearn(main_nums, sub_nums)
            return

        self._last_X = X[-1:] if len(X) > 0 else None
        self._fitted = True
        self.logger.info("XGBoost fitted on %d samples for %s", len(X), self.cfg.name)

    def _fit_sklearn(self, main_nums: np.ndarray, sub_nums: np.ndarray) -> None:
        """Fallback: use sklearn RandomForestRegressor."""
        from sklearn.ensemble import RandomForestRegressor

        X, y = _build_features_from_nums(main_nums, sub_nums, self.cfg)

        if len(X) < 5:
            self.logger.warning("Too few samples for sklearn fallback, using frequency mode")
            self._fit_fallback(main_nums, sub_nums)
            return

        self._model = RandomForestRegressor(
            n_estimators=min(self.n_estimators, 100),
            max_depth=min(self.max_depth, 10),
            random_state=42,
        )
        self._model.fit(X, y)
        self._last_X = X[-1:] if len(X) > 0 else None
        self._fitted = True
        self.logger.info("XGBoost (sklearn fallback) fitted on %d samples for %s", len(X), self.cfg.name)

    def _fit_fallback(self, main_nums: np.ndarray, sub_nums: np.ndarray) -> None:
        """Last resort: use recent frequency distribution."""
        from collections import Counter

        total_draws = len(main_nums)
        main_flat = main_nums.flatten()
        sub_flat = sub_nums.flatten()
        mc = Counter(main_flat.tolist())
        sc = Counter(sub_flat.tolist())

        self._main_probs = np.array([
            mc.get(n, 0) + 1 for n in range(self.cfg.main_min, self.cfg.main_max + 1)
        ], dtype=float)
        self._main_probs /= self._main_probs.sum()

        self._sub_probs = np.array([
            sc.get(n, 0) + 1 for n in range(self.cfg.sub_min, self.cfg.sub_max + 1)
        ], dtype=float)
        self._sub_probs /= self._sub_probs.sum()

        self._fallback_mode = True
        self._fitted = True
        self.logger.info("XGBoost (frequency fallback) fitted on %d draws for %s", total_draws, self.cfg.name)

    def _predict_using_model(self, n_predictions: int) -> np.ndarray:
        """Use the fitted sklearn/xgb model to predict."""
        predictions = []
        for _ in range(n_predictions):
            raw = self._model.predict(self._last_X)[0]
            # Add small noise for diversity
            raw += np.random.normal(0, 0.5, size=len(raw))
            predictions.append(raw)
        return np.array(predictions)

    def predict_main(self, n_predictions: int = 1) -> List[List[int]]:
        if not self._fitted:
            raise RuntimeError("XGBoost not fitted. Call fit() first.")

        if getattr(self, "_fallback_mode", False):
            return self._predict_fallback_main(n_predictions)

        raw_preds = self._predict_using_model(n_predictions)
        predictions = []
        for raw in raw_preds:
            main_raw = raw[:self.cfg.main_count]
            main_clipped = self._clip_predictions(
                main_raw, self.cfg.main_min, self.cfg.main_max
            )
            pred = self._ensure_unique_sorted(
                main_clipped, self.cfg.main_count,
                self.cfg.main_min, self.cfg.main_max,
            )
            predictions.append(pred)
        return predictions

    def _predict_fallback_main(self, n_predictions: int) -> List[List[int]]:
        numbers = list(range(self.cfg.main_min, self.cfg.main_max + 1))
        predictions = []
        for _ in range(n_predictions):
            pred = list(np.random.choice(
                numbers, size=self.cfg.main_count, replace=False, p=self._main_probs,
            ))
            predictions.append(sorted(pred))
        return predictions

    def predict_sub(self, n_predictions: int = 1) -> List[List[int]]:
        if not self._fitted:
            raise RuntimeError("XGBoost not fitted. Call fit() first.")

        if getattr(self, "_fallback_mode", False):
            return self._predict_fallback_sub(n_predictions)

        raw_preds = self._predict_using_model(n_predictions)
        predictions = []
        for raw in raw_preds:
            sub_raw = raw[self.cfg.main_count:self._total_count]
            sub_clipped = self._clip_predictions(
                sub_raw, self.cfg.sub_min, self.cfg.sub_max
            )
            pred = self._ensure_unique_sorted(
                sub_clipped, self.cfg.sub_count,
                self.cfg.sub_min, self.cfg.sub_max,
            )
            predictions.append(pred)
        return predictions

    def _predict_fallback_sub(self, n_predictions: int) -> List[List[int]]:
        numbers = list(range(self.cfg.sub_min, self.cfg.sub_max + 1))
        predictions = []
        for _ in range(n_predictions):
            pred = list(np.random.choice(
                numbers, size=self.cfg.sub_count, replace=False, p=self._sub_probs,
            ))
            predictions.append(sorted(pred))
        return predictions


# ---------------------------------------------------------------------------
# RandomForest Model
# ---------------------------------------------------------------------------

class RandomForest(BaseMLModel):
    """Random Forest regressor for lottery number prediction.

    Multi-output regression, one target per number position.
    Input dimension: window_size * (main_count + sub_count)
    Output dimension: main_count + sub_count
    """

    def __init__(self, cfg, n_estimators: int = 100, max_depth: int = 10):
        super().__init__(cfg, name="random_forest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self._model = None

    def fit(self, main_nums: np.ndarray, sub_nums: np.ndarray) -> None:
        """Fit Random Forest regressor."""
        from sklearn.ensemble import RandomForestRegressor

        self._total_count = self.cfg.main_count + self.cfg.sub_count
        X, y = _build_features_from_nums(main_nums, sub_nums, self.cfg)

        if len(X) < 5:
            self.logger.warning("Too few samples (%d) for RandomForest, using frequency mode", len(X))
            self._fit_fallback(main_nums, sub_nums)
            return

        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=42,
        )
        self._model.fit(X, y)
        self._last_X = X[-1:] if len(X) > 0 else None
        self._fitted = True
        self.logger.info("RandomForest fitted on %d samples for %s", len(X), self.cfg.name)

    def _fit_fallback(self, main_nums: np.ndarray, sub_nums: np.ndarray) -> None:
        """Fallback: use frequency distribution."""
        from collections import Counter

        main_flat = main_nums.flatten()
        sub_flat = sub_nums.flatten()
        mc = Counter(main_flat.tolist())
        sc = Counter(sub_flat.tolist())

        self._main_probs = np.array([
            mc.get(n, 0) + 1 for n in range(self.cfg.main_min, self.cfg.main_max + 1)
        ], dtype=float)
        self._main_probs /= self._main_probs.sum()

        self._sub_probs = np.array([
            sc.get(n, 0) + 1 for n in range(self.cfg.sub_min, self.cfg.sub_max + 1)
        ], dtype=float)
        self._sub_probs /= self._sub_probs.sum()

        self._fallback_mode = True
        self._fitted = True
        self.logger.info("RandomForest (frequency fallback) fitted on %d draws for %s", len(main_nums), self.cfg.name)

    def predict_main(self, n_predictions: int = 1) -> List[List[int]]:
        if not self._fitted:
            raise RuntimeError("RandomForest not fitted. Call fit() first.")

        if getattr(self, "_fallback_mode", False):
            return self._predict_fallback_main(n_predictions)

        predictions = []
        for _ in range(n_predictions):
            raw = self._model.predict(self._last_X)[0]
            raw += np.random.normal(0, 0.5, size=len(raw))
            main_raw = raw[:self.cfg.main_count]
            main_clipped = self._clip_predictions(
                main_raw, self.cfg.main_min, self.cfg.main_max
            )
            pred = self._ensure_unique_sorted(
                main_clipped, self.cfg.main_count,
                self.cfg.main_min, self.cfg.main_max,
            )
            predictions.append(pred)
        return predictions

    def _predict_fallback_main(self, n_predictions: int) -> List[List[int]]:
        numbers = list(range(self.cfg.main_min, self.cfg.main_max + 1))
        predictions = []
        for _ in range(n_predictions):
            pred = list(np.random.choice(
                numbers, size=self.cfg.main_count, replace=False, p=self._main_probs,
            ))
            predictions.append(sorted(pred))
        return predictions

    def predict_sub(self, n_predictions: int = 1) -> List[List[int]]:
        if not self._fitted:
            raise RuntimeError("RandomForest not fitted. Call fit() first.")

        if getattr(self, "_fallback_mode", False):
            return self._predict_fallback_sub(n_predictions)

        predictions = []
        for _ in range(n_predictions):
            raw = self._model.predict(self._last_X)[0]
            raw += np.random.normal(0, 0.5, size=len(raw))
            sub_raw = raw[self.cfg.main_count:self._total_count]
            sub_clipped = self._clip_predictions(
                sub_raw, self.cfg.sub_min, self.cfg.sub_max
            )
            pred = self._ensure_unique_sorted(
                sub_clipped, self.cfg.sub_count,
                self.cfg.sub_min, self.cfg.sub_max,
            )
            predictions.append(pred)
        return predictions

    def _predict_fallback_sub(self, n_predictions: int) -> List[List[int]]:
        numbers = list(range(self.cfg.sub_min, self.cfg.sub_max + 1))
        predictions = []
        for _ in range(n_predictions):
            pred = list(np.random.choice(
                numbers, size=self.cfg.sub_count, replace=False, p=self._sub_probs,
            ))
            predictions.append(sorted(pred))
        return predictions


# Backward compatibility aliases
LSTMSequenceModel = LSTM
XGBoostModel = XGBoost
RandomForestModel = RandomForest
