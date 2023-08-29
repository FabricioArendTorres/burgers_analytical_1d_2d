from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Interval:
    left: float
    right: float

    def get_linspace(self, n: int) -> np.array:
        return np.linspace(self.left, self.right, n)

    def sample_uniformly(self, n: int) -> np.array:
        return np.random.uniform(self.left, self.right, n)

    def contains(self, val: float):
        return self.left <= val <= self.right

    def contains_all(self, val: np.array):
        return np.all((self.left <= val) & (val <= self.right))

    @property
    def volume(self):
        return self.right - self.left

    def __add__(self, o) -> Hypercube:
        return Hypercube([self, o])


@dataclass
class Hypercube:
    intervals: List[Interval]

    def __add___(self, o) -> Hypercube:
        return Hypercube([self, o])

    def get_meshgrid(self, n_per_dim: int):
        return np.stack(np.meshgrid(*[interval.get_linspace(n_per_dim) for interval in self.intervals]), -1)

    @property
    def dim(self):
        return len(self.intervals)

    def get_meshgrid_flat(self, n_per_dim: int):
        return self.get_meshgrid(n_per_dim=n_per_dim).reshape(-1, self.dim)
