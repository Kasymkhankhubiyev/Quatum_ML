import matplotlib.pyplot as plt
# from sklearn.linear_model import LogisticRegression
import matplotlib as mpl
import numpy as np
# from Circuit import Model


def visualize_dataset(trainX, trainY, testX, testY, name: str) -> None:
    colors = ['blue', 'red', 'yellow', 'green']

    for k in np.unique(trainY):
        plt.plot(trainX[trainY == k, 0], trainX[trainY == k, 1], 'o', label='class {}'.format(k), color=colors[k])

    for k in np.unique(testY):
        plt.plot(testX[testY == k, 0], testX[testY == k, 1], 'o', label='class {}'.format(k+2), color=colors[k+2])

    plt.legend(fontsize=7, ncol=1, facecolor='oldlace', edgecolor='r')
    plt.title(name)
    plt.savefig('Binary_Classification/Normal_distribution/'+name+'_classification_data_set.png')
    plt.close()


def visualize_data(arrayX, arrayY, name: str) -> None:
    colors = ['blue', 'red', 'yellow', 'green']
    for k in np.unique(arrayY):
        plt.plot(arrayX[arrayY == k, 0], arrayX[arrayY == k, 1], 'o', label='class {}'.format(k), color=colors[k])
    plt.legend(fontsize=7, ncol=1, facecolor='oldlace', edgecolor='r')
    plt.title(name)
    plt.savefig('Binary_Classification/Normal_distribution/' + name + '_transformed_data_set.png')
    plt.close()

def plot_double_cake_data(X, Y, ax, num_sectors=None):
    """Plot double cake data and corresponding sectors."""
    x, y = X.T
    cmap = mpl.colors.ListedColormap(["#FF0000", "#0000FF"])
    ax.scatter(x, y, c=Y, cmap=cmap, s=25, marker="s")

    if num_sectors is not None:
        sector_angle = 360 / num_sectors
        for i in range(num_sectors):
            color = ["#FF0000", "#0000FF"][(i % 2)]
            other_color = ["#FF0000", "#0000FF"][((i + 1) % 2)]
            ax.add_artist(
                mpl.patches.Wedge(
                    (0, 0),
                    1,
                    i * sector_angle,
                    (i + 1) * sector_angle,
                    lw=0,
                    color=color,
                    alpha=0.1,
                    width=0.5,
                )
            )
            ax.add_artist(
                mpl.patches.Wedge(
                    (0, 0),
                    0.5,
                    i * sector_angle,
                    (i + 1) * sector_angle,
                    lw=0,
                    color=other_color,
                    alpha=0.1,
                )
            )
            ax.set_xlim(-1, 1)

    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    return ax

def draw_decision_boundary(model, N_gridpoints=14) -> None:

    #разбиваем область на точки
    _xx, _yy = np.linspace(-4, 4, N_gridpoints), np.linspace(-4, 4, N_gridpoints)

    _zz = np.zeros_like(_xx)

    points = [[_xx[i], _yy[i]] for i in range(len(_xx))]

    for idx in np.ndindex(*_xx.shape):
        _zz[idx] = model.predict(points)

    # plot_data = {"_xx": _xx, "_yy": _yy, "_zz": _zz}
    # ax.contourf(
    #     _xx,
    #     _yy,
    #     _zz,
    #     cmap=mpl.colors.ListedColormap(["#FF0000", "#0000FF"]),
    #     alpha=0.2,
    #     levels=[-1, 0, 1],
    # )

    colors = ['pink', 'purple']

    for k in np.unique(_zz):
        plt.scatter(x = _xx[_zz == k], y = _yy[_zz == k], s=30, c=colors[k])

    name = 'decision_boundary_test_plot'

    plt.legend(fontsize=7, ncol=1, facecolor='oldlace', edgecolor='r')
    plt.title(name)
    plt.savefig('Binary_Classification/Normal_distribution/' + name + '_decision_boundary.png')
    plt.close()
