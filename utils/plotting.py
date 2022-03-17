import numpy as np

import torch

torch.manual_seed(0)
torch.set_default_dtype(torch.double)

from scipy.interpolate import interp1d, Rbf

from utils.constants import CATS, ANS, TARGET

import matplotlib.pyplot as plt


def make_square(cat_ind1, cat_ind2, an_ind1, an_ind2, n_steps=97):
    num_dims = len(CATS) + len(ANS)

    cat_xs = np.linspace(0, 1, n_steps).reshape(-1, 1)
    cat_xs = np.hstack((cat_xs, 1 - cat_xs))
    an_xs = np.linspace(0, 1, n_steps).reshape(-1, 1)
    an_xs = np.hstack((an_xs, 1 - an_xs))

    xs = np.hstack(
        (
            np.repeat(cat_xs, an_xs.shape[0], axis=0),
            an_xs[np.array(list(np.arange(an_xs.shape[0])) * cat_xs.shape[0])],
        )
    )

    empty = np.zeros((xs.shape[0], num_dims))
    empty[:, cat_ind1] = xs[:, 0]
    empty[:, cat_ind2] = xs[:, 1]
    empty[:, an_ind1] = xs[:, 2]
    empty[:, an_ind2] = xs[:, 3]

    return torch.tensor(empty).float()


def get_edge_label(cat_ind1, cat_ind2, an_ind1, an_ind2, edge_ind):
    if edge_ind == 0:
        return f"{CATS[cat_ind2]}{ANS[an_ind2 - len(CATS)]} – {CATS[cat_ind1]}{ANS[an_ind2 - len(CATS)]}"
    elif edge_ind == 1:
        return f"{CATS[cat_ind2]}{ANS[an_ind2 - len(CATS)]} – {CATS[cat_ind2]}{ANS[an_ind1 - len(CATS)]}"
    elif edge_ind == 2:
        return f"{CATS[cat_ind2]}{ANS[an_ind1 - len(CATS)]} – {CATS[cat_ind1]}{ANS[an_ind1 - len(CATS)]}"
    elif edge_ind == 3:
        return f"{CATS[cat_ind1]}{ANS[an_ind2 - len(CATS)]} – {CATS[cat_ind1]}{ANS[an_ind1 - len(CATS)]}"


def filter_mask_square_edge(cat_ind1, cat_ind2, an_ind1, an_ind2, edge_ind, xs):
    num_dims = len(CATS) + len(ANS)

    if edge_ind == 0:  # bottom edge
        empty_cols = [
            i
            for i in range(num_dims)
            if i != cat_ind1 and i != cat_ind2 and i != an_ind2
        ]
    elif edge_ind == 1:  # left edge
        empty_cols = [
            i
            for i in range(num_dims)
            if i != cat_ind2 and i != an_ind1 and i != an_ind2
        ]
    elif edge_ind == 2:  # top edge
        empty_cols = [
            i
            for i in range(num_dims)
            if i != cat_ind1 and i != cat_ind2 and i != an_ind1
        ]
    elif edge_ind == 3:  # right edge
        empty_cols = [
            i
            for i in range(num_dims)
            if i != cat_ind1 and i != an_ind1 and i != an_ind2
        ]
    else:
        raise ValueError("invalid edge index")

    return torch.all(xs[:, empty_cols] == 0, axis=1)


def filter_mask_square(cat_ind1, cat_ind2, an_ind1, an_ind2, xs):
    num_dims = len(CATS) + len(ANS)

    empty_cols = [
        i
        for i in range(num_dims)
        if i != cat_ind1 and i != cat_ind2 and i != an_ind1 and i != an_ind2
    ]

    return torch.all(xs[:, empty_cols] == 0, axis=1)


def plot_regressor_square_edges(
    cat_ind1,
    cat_ind2,
    an_ind1,
    an_ind2,
    gp,
    likelihood,
    train_x,
    train_y,
    test_x,
    transform_fn=None,
    unnormalize_fn=None,
    num_samples=3,
):
    edge_ind_to_x_ind = {0: cat_ind1, 1: an_ind1, 2: cat_ind1, 3: an_ind2}

    if transform_fn is None:

        def transform_fn(x_all, xs, ys):
            # f = interp1d(xs, ys)
            f = Rbf(xs, ys)

            return f(x_all)

    if unnormalize_fn is None:
        unnormalize_fn = lambda x: x

    fig, ax = plt.subplots(2, 2, figsize=(16, 12))

    for edge_ind in range(4):
        # training data
        mask = filter_mask_square_edge(
            cat_ind1, cat_ind2, an_ind1, an_ind2, edge_ind, train_x
        )
        filtered_train_x = train_x[mask]
        filtered_train_y = train_y[mask]

        # test data
        mask = filter_mask_square_edge(
            cat_ind1, cat_ind2, an_ind1, an_ind2, edge_ind, test_x
        )
        filtered_test_x = test_x[mask]

        # posterior
        with torch.no_grad():
            output = gp(filtered_test_x)
            mean = output.mean
            lower, upper = output.confidence_region()

            pred = likelihood(output)
            lower, upper = pred.confidence_region()

            # torch.manual_seed(0)
            # samples = output.sample(torch.Size([num_samples]))

        tmp_filtered_xs = filtered_test_x[:, edge_ind_to_x_ind[edge_ind]]
        tmp_filtered_x_all = np.linspace(
            tmp_filtered_xs.min(), tmp_filtered_xs.max(), 501
        )
        tmp_ax = ax[edge_ind // 2][edge_ind % 2]

        tmp_ax.plot(
            tmp_filtered_x_all,
            unnormalize_fn(transform_fn(tmp_filtered_x_all, tmp_filtered_xs, mean)),
            label="mean",
        )

        tmp_ax.fill_between(
            tmp_filtered_x_all,
            unnormalize_fn(transform_fn(tmp_filtered_x_all, tmp_filtered_xs, lower)),
            unnormalize_fn(transform_fn(tmp_filtered_x_all, tmp_filtered_xs, upper)),
            alpha=0.1,
            label="CI",
        )

        tmp_ax.scatter(
            filtered_train_x[:, edge_ind_to_x_ind[edge_ind]],
            unnormalize_fn(filtered_train_y),
            marker="x",
            c="k",
            label="obs",
        )

        tmp_ax.set_xlabel(
            get_edge_label(cat_ind1, cat_ind2, an_ind1, an_ind2, edge_ind)
        )

    ax[0][0].legend()
    ax[0][0].set_ylabel(TARGET)
    ax[1][0].set_ylabel(TARGET)

    return fig, ax


def plot_regressor_square(
    cat_ind1,
    cat_ind2,
    an_ind1,
    an_ind2,
    gp,
    likelihood,
    x_train,
    y_train,
    x_test,
    unnormalize_fn=None,
):
    if unnormalize_fn is None:
        unnormalize_fn = lambda x: x

    xlabel = f"{CATS[cat_ind2]}{ANS[an_ind2 - len(CATS)]} — {CATS[cat_ind1]}{ANS[an_ind2 - len(CATS)]}"
    ylabel = f"{CATS[cat_ind2]}{ANS[an_ind2 - len(CATS)]} — {CATS[cat_ind2]}{ANS[an_ind1 - len(CATS)]}"

    mask = filter_mask_square(cat_ind1, cat_ind2, an_ind1, an_ind2, x_train)
    filtered_x_train = x_train[mask]
    filtered_y_train = y_train[mask]

    filtered_x_test = make_square(cat_ind1, cat_ind2, an_ind1, an_ind2, n_steps=13)
    mask = filter_mask_square(cat_ind1, cat_ind2, an_ind1, an_ind2, x_test)
    valid_x_test = x_test[mask]

    sorted_valid_x_test = sorted(valid_x_test, key=lambda x: (x[..., cat_ind1], x[..., an_ind1]))
    sorted_valid_x_test = torch.vstack(sorted_valid_x_test)

    with torch.no_grad():
        output = gp(sorted_valid_x_test)
        mean = output.mean
        sd = output.stddev

        # pred = likelihood(output)
        # sd = pred.stddev

    mean_interpolator = Rbf(valid_x_test[:, cat_ind1], valid_x_test[:, an_ind1], mean)
    interp_mean = mean_interpolator(
        filtered_x_test[:, cat_ind1], filtered_x_test[:, an_ind1]
    )

    sd_interpolator = Rbf(valid_x_test[:, cat_ind1], valid_x_test[:, an_ind1], sd)
    interp_sd = sd_interpolator(
        filtered_x_test[:, cat_ind1], filtered_x_test[:, an_ind1]
    )

    pad = 20
    fig, ax = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    # predictive mean
    c = ax[0].imshow(
        unnormalize_fn(interp_mean.reshape(13, 13).T),
        origin="lower",
        extent=[0, 1, 0, 1],
        interpolation="gaussian",
    )
    plt.colorbar(c, ax=ax[0])

    for ind in range(filtered_x_train.shape[0]):
        ax[0].text(
            filtered_x_train[ind, cat_ind1],
            filtered_x_train[ind, 5],
            f"{unnormalize_fn(filtered_y_train[ind].item()):.0f}",
            c="r",
            ha="center",
            va="bottom",
            weight="bold",
            fontsize=15,
        )

    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel(ylabel)
    ax[0].set_title("mean", pad=pad)

    # predictive sd
    c = ax[1].imshow(
        unnormalize_fn(interp_sd.reshape(13, 13).T),
        origin="lower",
        extent=[0, 1, 0, 1],
        interpolation="gaussian",
    )
    plt.colorbar(c, ax=ax[1])

    ax[1].scatter(
        filtered_x_train[:, cat_ind1], filtered_x_train[:, 5], c="r", label="obs"
    )

    ax[1].set_xlabel(xlabel)
    ax[1].legend()
    ax[1].set_title("stddev", pad=pad)
