def _riemannian_alignment(covariances, scaling=False):
    import pyriemann

    mean_cov = pyriemann.utils.mean.mean_covariance(covariances)
    n_covs, _, _ = covariances.shape

    if scaling:
        d_tilda_sqrt = 0
        for m in range(n_covs):
            d_tilda_sqrt += pyriemann.utils.distance.distance(
                mean_cov, covariances[m, :, :], squared=True
            )
        d_tilda = np.sqrt(d_tilda_sqrt)

    mean_cov_inv_sqrt = scipy.linalg.fractional_matrix_power(mean_cov, -0.5)

    aligned_covs = np.zeros(covariances.shape)
    for m in range(n_covs):
        aligned_covs[m, :, :] = (
            mean_cov_inv_sqrt @ covariances[m, :, :] @ mean_cov_inv_sqrt
        )
        if scaling:
            aligned_covs[m, :, :] = scipy.linalg.fractional_matrix_power(
                aligned_covs[m, :, :], 1 / d_tilda
            )

    return aligned_covs


def riemannian_alignment(covariances, scaling=False, online=False, enable_tqdm=True):
    if online is False:
        return _riemannian_alignment(covariances=covariances, scaling=scaling)
    else:
        new_covariances = []
        for n in tqdm.tqdm(range(covariances.shape[0]), disable=not enable_tqdm):
            if n == 0:
                _scaling = False
            else:
                _scaling = scaling
            new_covariances.append(
                _riemannian_alignment(covariances[0 : (n + 1), :, :], scaling=_scaling)[
                    -1, :, :
                ]
            )
        return np.stack(new_covariances, axis=0)