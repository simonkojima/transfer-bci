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