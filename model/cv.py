"""Cross-validation split helpers."""

import numpy as np


def create_expanding_window_folds(
    n_samples: int, initial_window: int = 60, horizon: int = 1
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create expanding-window CV folds.

    Each fold uses samples ``[0, train_end)`` for training and the next
    ``horizon`` sample(s) for testing.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0.")
    if initial_window <= 0:
        raise ValueError("initial_window must be > 0.")
    if horizon <= 0:
        raise ValueError("horizon must be > 0.")
    if initial_window + horizon > n_samples:
        raise ValueError("initial_window + horizon must be <= n_samples.")

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for train_end in range(initial_window, n_samples - horizon + 1):
        # Why: expanding the train window mirrors real forecasting, where new history accumulates over time.
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(train_end, train_end + horizon)
        folds.append((train_idx, test_idx))
    return folds
