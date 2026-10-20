import numpy as np
import scipy
import tqdm


def _euclidean_alignment(X):
    n_trials, n_channels, n_times = X.shape

    covariances = np.zeros((n_trials, n_channels, n_channels))
    for n in range(n_trials):
        covariances[n, :, :] = X[n, :, :] @ X[n, :, :].T / n_times

    R = np.mean(covariances, axis=0)

    R_inv_sqrt = scipy.linalg.fractional_matrix_power(R, -0.5)

    X_aligned = np.zeros(X.shape)
    for n in range(n_trials):
        X_aligned[n, :, :] = R_inv_sqrt @ X[n, :, :]

    return X_aligned


def euclidean_alignment(X, online=False, enable_tqdm=True, min_trials=1):
    """
    https://doi.org/10.1109/TBME.2019.2913914

    """
    if not online:
        return _euclidean_alignment(X=X)

    n_trials, n_channels, n_times = X.shape
    X_aligned = np.empty_like(X)

    R_sum = np.zeros((n_channels, n_channels), dtype=X.dtype)

    iterator = tqdm.tqdm(range(n_trials), disable=not enable_tqdm)

    for n in iterator:
        x = X[n]
        R_sum += x @ x.T / n_times

        if n + 1 < min_trials:
            X_aligned[n] = x
        else:
            R = R_sum / (n + 1)
            eigenvalues, eigenvectors = scipy.linalg.eigh(R)
            R_inv_sqrt = (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.T
            X_aligned[n] = R_inv_sqrt @ x

    return X_aligned


def channel_normalization(X, online=False, eps=1e-12):
    X = np.asarray(X, dtype=np.float64)

    if not online:
        mean_power = np.mean(X**2, axis=(0, 2))
        return X / np.sqrt(mean_power[None, :, None] + eps)

    aligned = np.empty_like(X)
    power_sum = np.zeros(X.shape[1], dtype=np.float64)

    for n, x in enumerate(X):
        power_sum += np.mean(x**2, axis=-1)
        mean_power = power_sum / (n + 1)
        aligned[n] = x / np.sqrt(mean_power[:, None] + eps)

    return aligned


"""
import numpy as np


class Covariance:
    def __init__(self):
        self.data = {}
        self.cov = None
        self.ch_names = None

    def __call__(self, X, ch_names):
        self.cov = X @ X.T / X.shape[1]
        self.ch_names = list(ch_names)
        self.data = {}

        for i, n1 in enumerate(ch_names):
            self.data[n1] = {}
            for j, n2 in enumerate(ch_names):
                self.data[n1][n2] = self.cov[i, j]

        self.observed_data = {ch: values.copy() for ch, values in self.data.items()}
        return self

    def check_missing(self, ch_names):
        return [name for name in ch_names if name not in self.ch_names]

    def is_ch_exist(self, channel):
        return channel in self.ch_names

    def is_pair_exist(self, ch1, ch2):
        return ch1 in self.ch_names and ch2 in self.ch_names

    def to_numpy(self, ch_names):
        return np.array(
            [[self.data[ch1][ch2] for ch2 in ch_names] for ch1 in ch_names], dtype=float
        )


def fill_missing(cov, cov_list, all_names):
    for ch in all_names:
        cov.data.setdefault(ch, {})

    for i, ch1 in enumerate(all_names):
        for ch2 in all_names[i:]:
            if ch2 in cov.observed_data.get(ch1, {}):
                continue

            observed_values = []

            for candidate in cov_list:
                original_row = candidate.observed_data.get(ch1, {})
                if ch2 in original_row:
                    observed_values.append(original_row[ch2])

            if not observed_values:
                raise ValueError(f"No observations available for pair ({ch1}, {ch2})")

            value = float(np.mean(observed_values))
            cov.data[ch1][ch2] = value
            cov.data[ch2][ch1] = value

    return cov


def flexible_channel_euclidean_alignment(X, ch_names, eps=1e-6):
    X = [np.asarray(x, dtype=float) for x in X]
    ch_names = [list(names) for names in ch_names]

    if not X or len(X) != len(ch_names):
        raise ValueError("X and ch_names must be non-empty lists of equal length")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")

    all_names = list(dict.fromkeys(name for names in ch_names for name in names))

    cov_list = []

    for x, names in zip(X, ch_names):
        if x.ndim != 2 or x.shape[0] != len(names):
            raise ValueError(
                "x must have shape (channels, samples), "
                "with one channel name per channel"
            )
        if not names or x.shape[1] == 0:
            raise ValueError("Each trial must contain channels and samples")
        if len(set(names)) != len(names):
            raise ValueError("Channel names must be unique within each trial")
        if not np.isfinite(x).all():
            raise ValueError("X must not contain NaN or infinite values")

        cov_list.append(Covariance()(x, names))

    for cov in cov_list:
        fill_missing(cov, cov_list, all_names)

    filled_covs = []

    for cov in cov_list:
        filled = np.array(
            [[cov.data[ch1][ch2] for ch2 in all_names] for ch1 in all_names]
        )
        filled_covs.append(filled)

    R = np.mean(filled_covs, axis=0)

    values, vectors = np.linalg.eigh(R)

    print(values)

    values, vectors = np.linalg.eigh((R + R.T) / 2)
    min_eigenvalue = eps * max(values.max(), np.finfo(float).tiny)
    values = np.maximum(values, min_eigenvalue)

    R = (vectors * values) @ vectors.T

    X_aligned = []

    for x, names in zip(X, ch_names):
        indices = [all_names.index(name) for name in names]
        R_sub = R[np.ix_(indices, indices)]

        values, vectors = np.linalg.eigh(R_sub)
        values = np.maximum(values, min_eigenvalue)
        R_inv_sqrt = (vectors * (1 / np.sqrt(values))) @ vectors.T

        X_aligned.append(R_inv_sqrt @ x)

    return X_aligned, R, all_names, filled_covs
"""
