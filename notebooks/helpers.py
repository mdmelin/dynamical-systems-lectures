import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

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
        eigvals[d] = np.random.choice([-1, 1]) * np.random.uniform(1.1, 1.3)

    if random_axes:
        # Sample a random orthonormal basis
        Q, _ = np.linalg.qr(np.random.randn(n, n))
    else:
        Q = np.eye(n) # just the unit vectors

    # Construct A = Q Λ Q^{-1} = Q Λ Q^T (since Q orthonormal)
    A = Q @ np.diag(eigvals) @ Q.T

    return A, eigvals

    
# Helper functions for plotting results from ssm library
def plot_trajectory(z, x, ax=None, ls="-"):
    zcps = np.concatenate(([0], np.where(np.diff(z))[0] + 1, [z.size]))
    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.gca()
    for start, stop in zip(zcps[:-1], zcps[1:]):
        ax.plot(x[start:stop + 1, 0],
                x[start:stop + 1, 1],
                lw=1, ls=ls,
                color=colors[z[start] % len(colors)],
                alpha=1.0)
    return ax

def plot_observations(z, y, ax=None, ls="-", lw=1):

    zcps = np.concatenate(([0], np.where(np.diff(z))[0] + 1, [z.size]))
    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.gca()
    T, N = y.shape
    t = np.arange(T)
    for n in range(N):
        for start, stop in zip(zcps[:-1], zcps[1:]):
            ax.plot(t[start:stop + 1], y[start:stop + 1, n],
                    lw=lw, ls=ls,
                    color=colors[z[start] % len(colors)],
                    alpha=1.0)
    return ax


def plot_most_likely_dynamics(model,
    xlim=(-4, 4), ylim=(-3, 3), nxpts=20, nypts=20,
    alpha=0.8, ax=None, figsize=(3, 3)):
    
    K = model.K
    assert model.D == 2
    x = np.linspace(*xlim, nxpts)
    y = np.linspace(*ylim, nypts)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack((X.ravel(), Y.ravel()))

    # Get the probability of each state at each xy location
    z = np.argmax(xy.dot(model.transitions.Rs.T) + model.transitions.r, axis=1)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    for k, (A, b) in enumerate(zip(model.dynamics.As, model.dynamics.bs)):
        dxydt_m = xy.dot(A.T) + b - xy

        zk = z == k
        if zk.sum(0) > 0:
            ax.quiver(xy[zk, 0], xy[zk, 1],
                      dxydt_m[zk, 0], dxydt_m[zk, 1],
                      color=colors[k % len(colors)], alpha=alpha)

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

    plt.tight_layout()

    return ax