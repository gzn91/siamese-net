import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('seaborn-deep')


def plot_gmm2d(x, y, means, covs, title='GMM'):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    cs = iter(mpl.cm.Set1(np.linspace(0, 1, 10)))
    for i, (mean, cov) in enumerate(zip(means, covs)):
        U, S, V = np.linalg.svd(cov)
        D, W = np.linalg.eigh(cov)

        deg = np.rad2deg(np.arctan2(U[1, 1], U[1, 0]))
        minor, major = 2 * np.sqrt(2 * S)

        c = next(cs)
        plt.scatter(x[y == i, 0], x[y == i, 1], s=40, color=c)
        el = mpl.patches.Ellipse(mean, major, minor, deg, color=c)
        el.set_clip_box(ax.bbox)
        el.set_alpha(0.1)
        ax.add_artist(el)

    ax.set_title(title)


def plot_gmm3d(x, y, means, covs, title='GMM'):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    cs = iter(mpl.cm.Set1(np.linspace(0, 1, 10)))
    for i, (mean, cov) in enumerate(zip(means, covs)):
        U, S, V = np.linalg.svd(cov)
        S = np.sqrt(2 * S)

        u = np.linspace(0.0, 2.0 * np.pi, 100)
        v = np.linspace(0.0, np.pi, 100)
        X = S[0] * np.outer(np.cos(u), np.sin(v))
        Y = S[1] * np.outer(np.sin(u), np.sin(v))
        Z = S[2] * np.outer(np.ones_like(u), np.cos(v))

        tmp = []
        for k in range(3):
            tmp.append(V[0, k] * X + V[1, k] * Y + V[2, k] * Z + mean[k])
        X, Y, Z = tmp

        c = next(cs)
        ax.plot_wireframe(X, Y, Z, rstride=4, cstride=4, color=c, alpha=.1)
        ax.scatter(x[y == i, 0], x[y == i, 1], x[y == i, 2], s=40, color=c)

    ax.set_title(title)

    return fig, ax
