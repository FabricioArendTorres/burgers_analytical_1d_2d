from __future__ import annotations

from abc import abstractmethod
from typing import List

import numpy as np
import sympy as sym
from sympy import lambdify
from scipy.stats import norm

from pde_datasets.modules.problem_base import AdvectionDiffusion
from pde_datasets.utils.domains import Interval


class IndependentNoiseFokkerPlanck(AdvectionDiffusion):
    def drift(self, xt) -> np.array:
        _x = xt[..., :-1]
        _t = xt[..., -1:]
        return self._drift(_x, _t)

    @abstractmethod
    def _drift(self, x, t) -> np.array:
        """
        Basically the advection in FP Equations
        """
        pass

    def _advection(self, _x, _t) -> np.array:
        return self._drift(_x, _t)

    def _diffusion(self, x, t) -> np.array:
        """
        Diffusion Coefficient for Fokker Planck equation, fully defined by the brownian particle noise.
        Returns a 1 dimensional np.array with shape (d,)
        """
        return 0.5 * self.noise_sd(x, t) ** 2


class SineFokkerPlanck(IndependentNoiseFokkerPlanck):

    def __init__(self,
                 space_domain: List[Interval],
                 time_domain: Interval,
                 noise_sigma=.06,
                 sin_scale=10,
                 drift_amplitude=1.,
                 initial_noise_scale=2e-2):
        super().__init__(space_domain, time_domain, has_analytical_solution=True)

        self.noise_sigma = noise_sigma
        self.sin_scale = sin_scale
        self.drift_amplitude = drift_amplitude
        self.initial_noise_scale = initial_noise_scale

        self.t0 = self.get_time_domain().left
        self.tn = self.get_time_domain().right

        self.x_0 = np.array([x.left for x in self.get_space_domain()])
        self.x_n = np.array([x.right for x in self.get_space_domain()])

        t, T = sym.symbols('t T')

        self.mean_sympy = sym.integrate(
            sym.sin(self.sin_scale * t), (t, self.t0, T))
        self.mean_numpy = lambdify(T, self.mean_sympy, 'numpy')

        self.cov_sympy = sym.integrate(
            self.noise_sigma ** 2, (t, self.t0, T)) + self.initial_noise_scale ** 2
        self.cov_numpy = lambdify(T, self.cov_sympy, 'numpy')

    def analytical_solution(self, xts):
        _x = xts[..., :-1]
        _t = xts[..., -1:]
        return self._analytical_solution(_x, _t)

    def _analytical_solution(self, x: np.array, t: np.array):
        """
        Returns analytical solution.
        :param x:
        :return:
        """
        dim = x.shape[-1]

        # mean = np.broadcast_to(, _x.shape[:1])
        mean = self.mean_numpy(t)
        var = self.cov_numpy(t)
        # if dim > 1:
        # #     cov = np.broadcast_to(np.eye(dim)[np.newaxis, ...], (_x.shape[0], dim, dim)) * var[..., np.newaxis]
        prob = 0
        if dim > 1:
            for i in range(dim):
                prob += norm.logpdf(np.squeeze(x[..., i]),
                                    np.squeeze(mean), np.squeeze(var + 1e-6) ** .5)
        else:
            prob = norm.logpdf(np.squeeze(x), np.squeeze(
                mean), np.squeeze(var + 1e-6) ** .5)
        # prob = norm.logpdf(np.squeeze(_x), np.squeeze(mean), np.squeeze(var + 1e-6) ** .5)

        return np.exp(prob.astype(np.float32))

    def _drift(self, x, t) -> np.array:
        # drift = np.zeros(self.get_xt_dim())

        # for i in range(self.get_xt_dim()):
        #     drift[i] = self.drift_amplitude * np.sin(t * self.sin_scale)
        drift_ = self.drift_amplitude * np.sin(t * self.sin_scale)
        if self.get_space_dim() > 1:
            drift = np.repeat(drift_, repeats=self.get_space_dim())
        else:
            drift = drift_
        return drift

    def noise_sd(self, x, t) -> np.array:
        return self.noise_sigma
