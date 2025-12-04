import numpy as np

def make_stable_LDS_matrix(n, stable_dims, stability_strength=0.8, random_axes=False):
    """
    Create an n×n discrete-time linear dynamical system matrix A
    with eigenvalues < 1 in magnitude for the dimensions listed in stable_dims.

    Parameters
    ----------
    n : int
        Dimension of the system.
    stable_dims : list of int
        Indices (0-based) of eigen-directions to enforce stability on.
    stability_strength : float
        Maximum magnitude for stable eigenvalues (0 < value < 1).

    Returns
    -------
    A : (n, n) ndarray
        A linear dynamical system matrix with enforced stable modes.
    """
    assert stability_strength > 0 and stability_strength < 1, "stability_strength must be between (0, 1)"
    is_stable = np.zeros(n, dtype=bool)
    if len(stable_dims) > 0:
        is_stable[stable_dims] = True
    unstable_dims = np.where(~is_stable)[0]

    # initialize eigenvalues
    eigvals = np.empty(n)

    # Force stability on requested indices
    for d in stable_dims:
        eigvals[d] = np.random.uniform(-stability_strength, stability_strength)
    # force instability on the others
    for d in unstable_dims:
        eigvals[d] = np.random.choice([-1, 1]) * np.random.uniform(1.5, 3)

    if random_axes:
        # Sample a random orthonormal basis
        Q, _ = np.linalg.qr(np.random.randn(n, n))
    else:
        Q = np.eye(n) # just the unit vectors

    # Construct A = Q Λ Q^{-1} = Q Λ Q^T (since Q orthonormal)
    A = Q @ np.diag(eigvals) @ Q.T

    return A, eigvals