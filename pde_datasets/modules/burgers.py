from abc import ABC

import mpmath as mpm

from itertools import repeat
from scipy.stats import qmc
from tqdm import trange
import numpy as np
from pde_datasets.modules.problem_base import TimeDependentProblem
from pde_datasets.utils.domains import *
from pde_datasets.utils import util


def norm(np_matrix):
    return mpm.norm(mpm.matrix(np_matrix.tolist()))


class BurgersEquation(TimeDependentProblem, ABC):
    def __init__(self,
                 reynolds_number: float,
                 space_domain: List[Interval],
                 time_domain: Interval,
                 has_analytical_solution: bool):
        super().__init__(name="",
                         xt_variables=["x", "t"],
                         output_variables=["u", "v"],
                         pde_latex=r"\frac{\delta u}{\delta t} + u \frac{\delta u}{\delta x} = 0",
                         has_analytical_solution=has_analytical_solution,
                         has_particle_simulation=False,
                         space_domain=space_domain,
                         time_domain=time_domain,
                         description="Burgers equation in 1d and 2d.")
        self.reynolds_number = reynolds_number

    @property
    def mu(self) -> float:
        return 1. / self.reynolds_number

    @property
    def reynolds(self) -> float:
        return self.reynolds_number

    @property
    def lambda_(self) -> mpm.mpf:
        return 1. / (4 * self.mu * mpm.pi)

    def simulate_pde(self, *args, **kwargs):
        super().simulate_pde(*args, **kwargs)


class Burgers1D(BurgersEquation):
    def __init__(self, reynolds_number):
        super().__init__(reynolds_number=reynolds_number,
                         space_domain=[Interval(-1, 1)],
                         time_domain=Interval(0, np.inf),
                         has_analytical_solution=True
                         )

    # @jit(nopython=True)
    def calc_tmp_values(self, x, t, eta):
        # assert eta.shape[1] == x.shape[0] == 1, "Invalid shapes for broadcasting"
        # x_diff = x - eta
        x = np.squeeze(x)
        t = np.squeeze(t)
        eta = np.squeeze(eta)

        x_diff = np.subtract.outer(x, eta).T
        sin_part = np.sin(np.pi * x_diff)
        f_part = np.exp(- np.cos(np.pi * x_diff) / (2 * np.pi * self.mu))
        exp_part = np.exp(-np.divide.outer(eta ** 2,
                                           4 * self.mu * t))
        return sin_part, f_part, exp_part

    def analytical_solution(self, xts, **kwargs):
        prev_shape = xts.shape
        if len(xts.shape) > 2:
            xts = xts.reshape(-1, 2)

        x, t = util.split_xt_(xts)
        assert self.space_domain[0].contains_all(x), "Values are out of space domain.. [-1, 1]"
        assert self.time_domain.contains_all(t), "Values are out of time domain.. [0, np.inf]"

        return self.calc_qmc_solution(x, t,
                                      kwargs.get("power_num_qmc", 16),
                                      kwargs.get("power_num_splits", 6)).reshape(prev_shape[:-1])

    def calc_qmc_solution(self, x, t, power_num_qmc=19, power_num_splits=3, lower_only=False):
        sampler = qmc.Sobol(d=1, scramble=False)

        power_qmc_batch = power_num_qmc - power_num_splits

        total_num_qmc = 2 ** power_num_qmc
        num_splits = 2 ** power_num_splits
        num_qmc_per_batch = 2 ** power_qmc_batch

        # assert total_num_qmc == (num_splits * num_qmc_per_batch),  f"{num_qmc_per_batch} instead of {total_num_qmc}"

        upper_mean = np.zeros_like(np.squeeze(x))
        lower_mean = np.zeros_like(np.squeeze(x))

        for i in trange(num_splits):
            eta = qmc.scale(sampler.random(num_qmc_per_batch),
                            l_bounds=self.space_domain[0].left,
                            u_bounds=self.space_domain[0].right)

            sin_part, f_part, exp_part = self.calc_tmp_values(x, t, eta)

            upper = np.mean(sin_part * f_part * exp_part, axis=0)
            lower = np.mean(f_part * exp_part, axis=0)

            m = i * num_qmc_per_batch
            n = num_qmc_per_batch
            upper_mean = (m / (m + n)) * upper_mean + (n / (m + n)) * upper
            lower_mean = (m / (m + n)) * lower_mean + (n / (m + n)) * lower

        solution_u = -(upper_mean / lower_mean)
        if lower_only:
            return self.space_domain[0].volume * lower_mean
        else:
            return solution_u

    @staticmethod
    def u0(x):
        return -np.sin(np.pi*x)

    @staticmethod
    def u0_integral(x):
        """
        int from 0 to x of u0
        :param x:
        :type x:
        :return:
        :rtype:
        """
        # return 1 - np.cos(x)
        return (np.cos(np.pi*x) -1)/np.pi

    def potential_fun_t0(self, x):
        return np.exp((-1. / (2 * self.mu)) * self.u0_integral(x))

    def gamma(self, x, y, t):
        return ((x - y) ** 2)

    def potential_fun(self, x, t):
        return self.calc_qmc_solution(x, np.ones_like(x)*t, lower_only=True, power_num_qmc=14, power_num_splits=0)

class BurgersAdvection(Burgers1D):
    @staticmethod
    def u0(x):
        return -np.sin(np.pi*x) + 1

    @staticmethod
    def u0_integral(x):
        """
        int from 0 to x of u0
        :param x:
        :type x:
        :return:
        :rtype:
        """
        # return 1 - np.cos(x)
        return x + (np.cos(np.pi*x) -1)/np.pi


    def potential_fun_t0(self, x):
        return np.exp((-1. / (2 * self.mu)) * self.u0_integral(x))


    def calc_tmp_values(self, x, t, eta):
        # assert eta.shape[1] == x.shape[0] == 1, "Invalid shapes for broadcasting"
        # x_diff = x - eta
        x = np.squeeze(x)
        t = np.squeeze(t)
        eta = np.squeeze(eta)

        x_diff = np.subtract.outer(x, eta).T
        sin_part = np.sin(np.pi * x_diff)
        f_part = np.exp(- np.cos(np.pi * x_diff) / (2 * np.pi * self.mu))
        exp_part = np.exp(-np.divide.outer(eta ** 2,
                                           4 * self.mu * t))
        return sin_part, f_part, exp_part

class Burgers2D(BurgersEquation):
    """
    https://github.com/apreziosir/BurgersCFD/blob/master/Taller%204%202018-I.pdf
    https://math.nyu.edu/~tabak/PDEs/The_Burgers-Equation.pdf
    """

    def __init__(self, reynolds_number, decimal_precision: int = 100):
        super().__init__(reynolds_number=reynolds_number,
                         space_domain=[Interval(0, 1), Interval(0, 1)],
                         time_domain=Interval(0, np.inf),
                         has_analytical_solution=True
                         )
        mpm.mp.dps = decimal_precision

    @staticmethod
    def ic_u(xts):
        xy, t = util.split_xt_(xts)
        return np.sin(np.pi * xy[..., 0]) * np.cos(np.pi * xy[..., 1])

    @staticmethod
    def ic_v(xts):
        xy, t = util.split_xt_(xts)
        return np.cos(np.pi * xy[..., 0]) * np.sin(np.pi * xy[..., 1])

    @util.as_ufunc(3, 1)
    def ic_theta(self, x, y):
        return mpm.exp(mpm.cos(mpm.pi * x) * mpm.cos(mpm.pi * y) - 1) / (2 * mpm.pi * self.mu)

    @staticmethod
    def coef_A(m, n):
        if n == m == 0:
            val = 1
        elif (n == 0 and m != 0) or (n != 0 and m == 0):
            val = 2
        else:
            val = 4
        return val

    def coef_E(self, m, n, t):
        return mpm_exp(- (n ** 2 + m ** 2) * mpm.pi ** 2 * self.mu * t)

    def coef_C(self, n, m):
        order1 = (n + m) / 2
        order2 = (n - m) / 2
        return self.bessel(order1) * self.bessel(order2)

    def calc_sum(self, X, Y, t, inner_opt, max_iter=10_000):
        if np.isscalar(X):
            X = np.array([[X]])
        if np.isscalar(Y):
            Y = np.array([[Y]])
        vals1 = (inner_opt(0, 0, X, Y) * 0).copy()

        sum_outer = vals1.copy()

        outer_norms = []
        for n in range(max_iter):
            sum_inner = (vals1 * 0).copy()
            for m in range(max_iter):
                if (n + m) % 2 > 0:
                    continue
                # assert (n+m) % 2 == 0, n+m
                diff = self.coef_A(m, n) * self.coef_E(m, n, t) * self.coef_C(n, m) * inner_opt(m, n, X, Y)
                sum_inner = sum_inner + diff
                if norm(diff) < 1e-30 and m > 0:  # and m > 0 and n > 0:
                    break

            outer_norms.append(norm(sum_inner))
            sum_outer = sum_outer + sum_inner

            if outer_norms[-1] < 1e-30 and n > 0:
                break

        return sum_outer, outer_norms

    def _calc_solution_theta(self, X, Y, t):
        sums, norms = self.calc_sum(X, Y, t=t, inner_opt=_inner_opt_theta)
        return mpm.exp(-2 * self.lambda_) * sums, norms

    def _calc_sum_theta(self, X, Y, t):
        return self.calc_sum(X, Y, t=t, inner_opt=_inner_opt_theta)

    def _calc_sum_u(self, X, Y, t):
        return self.calc_sum(X, Y, t=t, inner_opt=_inner_opt_u)

    def _calc_sum_v(self, X, Y, t):
        return self.calc_sum(X, Y, t=t, inner_opt=_inner_opt_v)

    def analytical_solution(self, xts, **kwargs):
        split_shape = (kwargs.get("num_parallel", 2),)
        prev_shape = xts.shape
        if len(xts.shape) > 2:
            xts = xts.reshape(-1, 3)

        x, t = util.split_xt_(xts)
        assert xts.shape[0] % split_shape[
            0] == 0, f"input rows ({xts.shape[0]}) must be divisible by num_parallel ({split_shape[0]})."
        # if len(np.unique(t)) == 1:
        #     return self.calc_uv_parallel(X=x[..., 0], Y=x[..., 1], t=t.flatten()[0], **kwargs)
        # if len(np.unique(t)) > 1:
        solutions = self.calc_uv_parallel(X=x[..., 0], Y=x[..., 1], t=t,
                                          split_array_shape=split_shape)
        return [np.reshape(sol, prev_shape[:-1]) for sol in solutions]

    def calc_uv_parallel(self, X: np.array, Y: np.array, t: float, split_array_shape=(2, 5)):
        # if len(split_array_shape) == 2:
        #     _, _, X1_blocks = util.cut_array2d(X, split_array_shape)
        #     _, _, X2_blocks = util.cut_array2d(Y, split_array_shape)
        #
        #     # flatten list
        #     X1_b_ll = [item for sublist in X1_blocks for item in sublist]
        #     X2_b_ll = [item for sublist in X2_blocks for item in sublist]
        # elif len(split_array_shape) == 1:
        _, X1_blocks = util.cut_array1d(X, split_array_shape)
        _, X2_blocks = util.cut_array1d(Y, split_array_shape)
        X1_b_ll = X1_blocks
        X2_b_ll = X2_blocks
        # else:
        #     raise ValueError(f"Invalid split_array_shape.")

        if np.isscalar(t):
            t_blocks = repeat(t)
        else:
            if len(split_array_shape) == 1:
                _, t_blocks = util.cut_array1d(np.squeeze(t), split_array_shape)
            else:
                _, t_blocks = util.cut_array2d(np.squeeze(t), split_array_shape)

        sum_theta = util.pool_calculation(self._calc_sum_theta,
                                          zip(X1_b_ll, X2_b_ll, t_blocks))
        sum_u = util.pool_calculation(self._calc_sum_u,
                                      zip(X1_b_ll, X2_b_ll, t_blocks))
        sum_v = util.pool_calculation(self._calc_sum_v,
                                      zip(X1_b_ll, X2_b_ll, t_blocks))
        # theta = mpm.exp(-2 * self.lambda_) * sum_theta
        u = 2 * mpm.pi * self.mu * sum_u / sum_theta
        v = 2 * mpm.pi * self.mu * sum_v / sum_theta

        return np.array(u, dtype="longfloat"), np.array(v, dtype="longfloat")

    def calc_uv(self, x: float, y: float, t: float):
        theta_sum, _ = self._calc_sum_theta(x, y, t)
        u_sum, _ = self._calc_sum_u(x, y, t)
        v_sum, _ = self._calc_sum_v(x, y, t)

        u = 2 * mpm.pi * self.mu * u_sum / theta_sum
        v = 2 * mpm.pi * self.mu * v_sum / theta_sum
        return np.array(u, 'longfloat').squeeze(), np.array(v, 'longfloat').squeeze()

    @util.memoize
    def bessel(self, order):
        return mpm.besseli(order, self.lambda_)


@util.memoize
def cos(n_or_m, x_or_y):
    return mpm.cos(n_or_m * mpm.pi * x_or_y)


@util.memoize
def sin(n_or_m, x_or_y):
    return mpm.sin(n_or_m * mpm.pi * x_or_y)


@util.as_ufunc(4, 1)
def _inner_opt_theta(m, n, x, y):
    return cos(n, x) * cos(m, y)  # mpm.cos(n * mpm.pi * x) * mpm.cos(m * mpm.pi * y)


@util.as_ufunc(1, 1)
def mpm_exp(val):
    return mpm.exp(val)


@util.as_ufunc(4, 1)
def _inner_opt_u(m, n, x, y):
    return n * sin(n, x) * cos(m, y)  # mpm.cos(n * mpm.pi * x) * mpm.cos(m * mpm.pi * y)


@util.as_ufunc(4, 1)
def _inner_opt_v(m, n, x, y):
    return m * cos(n, x) * sin(m, y)  # mpm.cos(n * mpm.pi * x) * mpm.cos(m * mpm.pi * y)
