from __future__ import annotations

import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from seaborn import color_palette
from sklearn.neighbors import KernelDensity

from pde_datasets.utils.domains import Interval, Hypercube
from pde_datasets.utils.plotting import add_colorbar
from pde_datasets.modules.fokkerplanck import SineFokkerPlanck
from pde_datasets.modules.burgers import Burgers2D, Burgers1D
from pde_datasets.modules.heat_equation import FundamentalHeatEquation
from pde_datasets.utils import util


def fp():
    sine_fp_settings = dict(
        space_domain=[Interval(-.6, .6)],
        time_domain=Interval(-1, 1),
        noise_sigma=.06,
        sin_scale=10,
        drift_amplitude=1.,
        initial_noise_scale=2e-2
    )

    test = SineFokkerPlanck(**sine_fp_settings)
    spacetime = Hypercube(test.space_domain + [test.time_domain])
    XT_grid = spacetime.get_meshgrid(100)
    XT_flat = XT_grid.reshape(-1, spacetime.dim)
    y = test.analytical_solution(XT_flat)
    y_grid = y.reshape(100, 100)
    # plt.imshow(y_grid, origin='lower')
    # # plt.savefig("plots/fp_1d_solution.png")
    # plt.show()

    X_start = np.random.normal(0, sine_fp_settings["initial_noise_scale"], size=(100_000, 1))
    t_start = -1 * np.ones((X_start.shape[0], 1))
    Xt_start = np.concatenate([X_start, t_start], -1)

    x_solutions, t_solutions = test.simulate_pde(Xt_start, dt=5e-3, save_each=1, T=2)
    xt_solutions = np.stack([x_solutions.flatten(), t_solutions.flatten()], -1)

    kde = KernelDensity(kernel='epanechnikov', bandwidth=0.005).fit(xt_solutions)

    fig, axs = plt.subplots(1, 2, sharex="all", sharey="all")
    axs = axs.flatten()
    norm = mpl.colors.Normalize(vmin=y_grid.min(), vmax=y_grid.max())

    im_args = dict(origin='lower', aspect="auto", extent=[test.space_domain[0].left,
                                                          test.space_domain[0].right,
                                                          test.time_domain.left,
                                                          test.time_domain.right],
                   cmap="mako",
                   norm=norm)
    im = axs[1].imshow(np.exp(kde.score_samples(XT_flat).reshape(100, 100)), **im_args)
    # im = axs[1].hist2d(x_solutions.flatten(), t_solutions.flatten(), bins=100, cmap="mako", density=True, norm=norm)[-1]
    add_colorbar(im, axs[1], fig)

    im = axs[0].imshow(y_grid, **im_args)
    add_colorbar(im, axs[0], fig)

    titles = ["Analytical Solution", "Particle Simulation"]
    for ax, title in zip(axs, titles):
        ax.set_title(title)
    plt.tight_layout()
    # plt.show()
    plt.savefig("plots/fokkerplanck.png")


def burgers_2d(time=0.25):
    reynolds = 100
    X, Y = util.build_mesh(0, 1., 20)

    assert (util.recombine_array2d(util.cut_array2d(X, (10, 5))[-1]) == X).all()
    assert (util.recombine_array2d(util.cut_array2d(Y, (5, 2))[-1]) == Y).all()

    burger = Burgers2D(reynolds_number=reynolds,
                       decimal_precision=100)

    xt = np.stack([X, Y, np.ones_like(X) * time], -1)
    vals_u, vals_v = burger.analytical_solution(xt)
    # vals_u = vals_u.reshape(X.shape)

    from mpl_toolkits import mplot3d
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, vals_u, cmap='viridis')
    ax.set_zlim(-1, 1)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.savefig(f"burgers_2d_t{time:.2f}.png")


def burgers_1d():
    reynolds = 100
    X = np.linspace(-1, 1, 500)
    # t = np.ones_like(X)*.5
    T = np.linspace(1e-5, 1, 200)
    # xts = np.stack([X, t], -1)
    XT = np.stack(np.meshgrid(X, T), -1)
    burger = Burgers1D(reynolds_number=reynolds)

    # sol = burger.analytical_solution(XT, power_num_qmc=12, power_num_splits=7)
    # plt.imshow(sol.T, origin='lower', aspect="auto", cmap="RdYlBu",
    #            extent=[
    #                np.min(T),
    #                np.max(T),
    #                burger.space_domain[0].left,
    #                burger.space_domain[0].right, ],
    #            vmin=-1,
    #            vmax=1)
    # plt.xlabel("Time")
    # plt.ylabel("Space")
    # plt.colorbar(label="u")
    # plt.title("u(t, x)")
    # # plt.plot(X, sol)
    # plt.show()
    fig, axs = plt.subplots(3, 1)
    axs = axs.flatten()

    # for t in [1e-5, .1, .5, 1., 2., 3.]:
    for t in [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]:
        if np.isclose(t, 0.):
            potential = burger.potential_fun_t0(X)
        else:
            potential = burger.potential_fun(X, t=t)
        theta = -2 * burger.mu * np.log(potential)
        diff = (X[1] - X[0]) * (theta[1:] - theta[:-1])
        axs[0].plot(X, theta, label=t)
        # vel = -2*burger.mu *  diff / potential[:-1]
        axs[1].plot(X[:-1], diff, label=t)
        axs[2].plot(X, burger.calc_qmc_solution(X, np.ones(X.shape[0]) * t, power_num_qmc=15, power_num_splits=0),
                    label=t)
    axs[0].set_title("Potential")
    axs[1].set_title("Diff Theta")
    axs[2].set_title("Solution u")
    handles, labels = axs[-1].get_legend_handles_labels()
    for ax in axs:
        ax.legend()
    plt.tight_layout()
    plt.savefig("plots/burgers_1d.png")


def fundamental_heat1d():
    import matplotlib.colors as colors

    diffusivity = 1e-1
    X = np.linspace(-4, 4, 500)
    # t = np.ones_like(X)*.5
    T = np.linspace(1e-5, 1, 200)
    # xts = np.stack([X, t], -1)
    XT = np.stack(np.meshgrid(X, T), -1)

    heat = FundamentalHeatEquation(dim=1, diffusivity=diffusivity)
    sol = heat.analytical_solution(XT) + 1e-10
    plt.imshow(sol.T, origin='lower', aspect="auto", cmap="magma",
               extent=[
                   np.min(T),
                   np.max(T),
                   np.min(X),
                   np.max(X)],
               norm=colors.LogNorm(vmin=sol.min(), vmax=sol.max()))
    plt.colorbar()
    plt.title(f"Heat Equation with diffusivity {diffusivity}")
    plt.savefig("plots/heat_1d.png")


def fundamental_heat3d():
    import matplotlib.colors as colors

    diffusivity = 1e-1
    heat = FundamentalHeatEquation(dim=2, diffusivity=diffusivity)

    grid_dim = 400
    y_min, y_max = -4, 4

    xx = np.linspace(y_min, y_max, grid_dim)
    yy = np.linspace(y_min, y_max, grid_dim)
    XY_grid = np.stack(np.meshgrid(xx, yy), -1)
    XY = XY_grid.reshape(-1, heat.space_dim)
    ts = np.linspace(1e-5, 2)
    t_grid = np.repeat(ts, grid_dim ** heat.space_dim).reshape((ts.shape[0], grid_dim, grid_dim))
    X_grid = np.tile(XY[..., 0], ts.shape[0]).reshape((ts.shape[0], grid_dim, grid_dim))
    Y_grid = np.tile(XY[..., 1], ts.shape[0]).reshape((ts.shape[0], grid_dim, grid_dim))
    XYT_grid = np.stack([X_grid, Y_grid, t_grid], axis=-1)
    sol = heat.analytical_solution(XYT_grid) + 1e-10

    if math.isqrt(heat.space_dim + 1) ** 2 == heat.space_dim + 1:
        fig, axs = plt.subplots(math.isqrt(heat.space_dim + 1), math.isqrt(heat.space_dim + 1))
    else:
        fig, axs = plt.subplots(heat.space_dim + 1, 1)
    axs = axs.flatten()

    for dim, ax in enumerate(axs):
        ax.imshow(sol.mean(dim).T, origin='lower', aspect="auto", cmap="magma",
                  extent=[
                      np.min(ts),
                      np.max(ts),
                      y_min,
                      y_max],
                  norm=colors.LogNorm(vmin=sol.min(), vmax=sol.max()))

    plt.savefig("plots/heat_3d.png")


if __name__ == "__main__":
    print("heat1d")
    fundamental_heat1d()
    print("burgers1d")
    burgers_1d()
    print("burgers2d")
    [burgers_2d(time) for time in [0., 0.25, 0.5, 0.75, ]]
    print("fokkerplanck")
    fp()
    print("heat3d")
    fundamental_heat3d()
