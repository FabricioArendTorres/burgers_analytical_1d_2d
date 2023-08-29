from __future__ import annotations

from abc import abstractmethod, ABC
from typing import List, Optional

import numpy as np
from pylatexenc.latex2text import LatexNodes2Text

from pde_datasets.utils.domains import Interval


class Problem:
    def __init__(self, name: str,
                 xt_variables: List[str],
                 space_domain: List[Interval],
                 output_variables: List[str],
                 pde_latex: str,
                 has_analytical_solution: bool,
                 has_particle_simulation: bool,
                 description: Optional[str] = "",
                 ):
        self.has_particle_simulation = has_particle_simulation
        self.has_analytical_solution = has_analytical_solution
        self.description = description
        self.name = name
        self.output_variables = output_variables
        self.xt_variables = xt_variables
        self.space_domain = space_domain
        self.space_dim = len(space_domain)

        self.pde_latex = pde_latex
        self.pde_unicode = LatexNodes2Text().latex_to_text(pde_latex)

    def get_pde_unicode(self) -> str:
        return self.pde_unicode

    def get_space_domain(self) -> List[Interval]:
        return self.space_domain

    def get_space_dim(self):
        return self.space_dim

    def simulate_pde(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'simulate_pde' is not implemented.")

    def analytical_solution(self, xts, **kwargs):
        raise NotImplementedError(
            "The method 'analytical_solution' is not implemented.")


class TimeDependentProblem(Problem, ABC):
    def __init__(self, name: str, xt_variables: List[str],
                 space_domain: List[Interval],
                 time_domain: Interval,
                 output_variables: List[str], pde_latex: str, has_analytical_solution: bool,
                 has_particle_simulation: bool,
                 description: str = ""):
        super().__init__(name, xt_variables, space_domain, output_variables, pde_latex, has_analytical_solution,
                         has_particle_simulation, description)
        self.time_domain = time_domain

    def get_time_domain(self):
        return self.time_domain


class AdvectionDiffusion(TimeDependentProblem):
    def __init__(self, space_domain: List[Interval],
                 time_domain: Interval, has_analytical_solution: bool):
        """
        Equation:

        \frac{\partial c}{\partial t}
            = \mathbf{\nabla} \cdot (D \mathbf{\nabla} c)
                - \mathbf{\nabla} \cdot (\mathbf{v} c)
                + R


        :param space_dim:
        :type space_dim:
        :param has_analytical_solution:
        :type has_analytical_solution:
        :param time_domain:
        :type time_domain:
        :param space_domain:
        :type space_domain:
        :param name:
        :type name:
        """

        super().__init__(name="FokkerPlanck",
                         space_domain=space_domain, time_domain=time_domain,
                         xt_variables=["x", "t"],
                         output_variables=["p"],
                         pde_latex=r"\frac{\partial c}{\partial t}  "
                                   r" - \mathbf{\nabla} \cdot (D \mathbf{\nabla} c) "
                                   r" + \mathbf{\nabla} \cdot (\mathbf{v} c) "
                                   r" = R",
                         has_analytical_solution=has_analytical_solution,
                         has_particle_simulation=True)
        self.delta_t = 1e-05

    def advection(self, xt) -> np.array:
        _x = xt[..., :-1]
        _t = xt[..., -1:]
        return self._advection(_x, _t)

    def diffusion(self, xt) -> np.array:
        _x = xt[..., :-1]
        _t = xt[..., -1:]
        return self._diffusion(_x, _t)

    @abstractmethod
    def noise_sd(self, x, t) -> np.array:
        """
        Noise of underlying Brownian Motion.
        Returns the standard deviations as a 1 dimensional (vector) np.array with shape (d,)
        Note that this implies that there is no correlation in the noise process (hence IndependentNoise)
        """
        pass

    @abstractmethod
    def _advection(self, x, t) -> np.array:
        pass

    @abstractmethod
    def _diffusion(self, x, t) -> np.array:
        pass

    def dW(self, delta_t, num_samples) -> float:
        return np.random.normal(loc=0.0, scale=np.sqrt(delta_t), size=(num_samples, self.get_space_dim()))

    def simulate_pde(self, xt, dt, T, save_each: int = 1):
        _x = xt[..., :-1]
        _t = xt[..., -1:]

        n_particles = xt.shape[0]
        n_steps = int(np.round(T // dt))
        x_list = []
        t_list = []

        cur_x = _x.copy()
        cur_t = _t.copy()
        for i in range(n_steps):
            next_t = cur_t + dt

            velocity = self._advection(cur_x, next_t)
            cur_x = cur_x + dt * velocity + self.noise_sd(cur_x, cur_t) * self.dW(dt, n_particles)
            cur_t = cur_t + dt

            if i % save_each == 0:
                x_list.append(cur_x)
                t_list.append(cur_t)

        return np.array(x_list), np.array(t_list)

class Diffusion(AdvectionDiffusion, ABC):
    def _advection(self, x, t) -> np.array:
        return 0.
