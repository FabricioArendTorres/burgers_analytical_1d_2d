from __future__ import annotations

from typing import Callable

import numpy as np

from multiprocessing import Pool, freeze_support

from mpmath import memoize as mpmemoize


def split_xt(func: Callable):
    def wrapper_split_args(ref, xt, *args, **kwargs):
        _x, _t = split_xt_(xt)
        return func(ref, x=_x, t=_t, *args, **kwargs)

    return wrapper_split_args


def memoize(func: Callable):
    return mpmemoize(func)


def split_xt_(xt):
    _x = xt[..., :-1]
    _t = xt[..., -1:]
    return _x, _t


def as_ufunc(nin, nout):
    def _decorator(func):
        return np.frompyfunc(func, nin, nout)

    return _decorator


def build_mesh(start, end, num):
    xx = np.linspace(start, end, num)
    yy = np.linspace(start, end, num)

    X, Y = np.meshgrid(xx, yy)
    return X, Y


def cut_array2d(array, shape):
    arr_shape = np.shape(array)
    xcut = np.linspace(0, arr_shape[0], shape[0] + 1).astype(np.int32)
    ycut = np.linspace(0, arr_shape[1], shape[1] + 1).astype(np.int32)
    blocks = []
    xextent = []
    yextent = []
    for i in range(shape[0]):
        inner_block = []
        for j in range(shape[1]):
            inner_block.append(array[xcut[i]:xcut[i + 1], ycut[j]:ycut[j + 1]])
            xextent.append([xcut[i], xcut[i + 1]])
            yextent.append([ycut[j], ycut[j + 1]])
        blocks.append(inner_block)
    return xextent, yextent, blocks

def cut_array1d(array, shape):
    arr_shape = np.shape(array)
    xcut = np.linspace(0,arr_shape[0],shape[0]+1).astype(np.int32)
    blocks = [];    xextent = [];
    for i in range(shape[0]):
        blocks.append(array[xcut[i]:xcut[i+1], ...])
        xextent.append([xcut[i],xcut[i+1]])
    return xextent, blocks


def recombine_array2d(blocks):
    tmp = np.concatenate(np.array(blocks), axis=-2)
    return np.concatenate(tmp, axis=-1)


def arr2d_from_starmap(L, shape):
    results = [np.array(l[0], dtype="float") for l in L]
    return recombine_array2d(np.array(results).reshape(shape))


def pool_calculation(fun, args):
    with Pool() as pool:
        L = pool.starmap(fun, args)
    # if len(np.array(L[0][0]).shape) > 1:
    #     return arr2d_from_starmap(L, array_shape)
    # else:
    return np.concatenate([np.array(l[0], dtype="float") for l in L])
