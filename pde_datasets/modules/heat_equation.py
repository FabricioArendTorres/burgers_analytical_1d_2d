from abc import ABC

import numpy as np
from typing import *
from scipy.stats import multivariate_normal, norm
from pde_datasets.utils.domains import *
from pde_datasets.utils import util

from pde_datasets.modules.problem_base import Diffusion


class HeatEquation(Diffusion, ABC):
    def noise_sd(self, x, t) -> np.array:
        return np.sqrt(2 * self._diffusion(x, t))


class FundamentalHeatEquation(HeatEquation):
    def __init__(self, dim=1, diffusivity: float = 1.):
        super().__init__([Interval(-np.inf, np.inf) for d in range(dim)],
                         Interval(0, np.inf), has_analytical_solution=True)
        self.diffusivity = diffusivity

    def get_norm_rv(self, t):
        return multivariate_normal(np.zeros(self.space_dim),
                                   2 * self.diffusivity * t * np.eye(self.space_dim))

    def _diffusion(self, x, t) -> np.array:
        return self.diffusivity

    def analytical_solution(self, xts, **kwargs):
        x, t = util.split_xt_(xts)

        sigma2 = 2 * self.diffusivity * t
        mu = np.zeros_like(x)
        log_pdf = norm(loc=mu, scale=np.sqrt(sigma2)).logpdf(x).sum(-1)
        return np.exp(log_pdf)
        #
        # sum_ = 0.
        # sum_det = np.squeeze(1. - self.space_dim * np.log(np.sqrt(4 * np.pi * self.diffusivity * t)))
        #
        # for dim in range(self.space_dim):
        #     sum_ += -(x[..., [dim]] ** 2)
        #
        # sum_ /= (4 * self.diffusivity * t)
        # sum_ = np.squeeze(sum_)
        # return np.squeeze(np.exp(sum_det + sum_))

        # return np.squeeze(
        #     1. / (np.sqrt(4 * np.pi * self.diffusivity * t)) * np.exp(-(x ** 2) / (4 * self.diffusivity * t)))
