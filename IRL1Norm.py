import typing as typ

import numpy as np

import pycsou.abc.operator as pyco
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct




class SquaredL2Norm(pyco.ProxDiffFunc):
    def __init__(self, M: int = None):
        super().__init__(shape=(1, M))
        self._lipschitz = np.inf
        self._diff_lipschitz = 2

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        y = xp.linalg.norm(arr, axis=-1, keepdims=True)
        y2 = xp.power(y, 2, dtype=arr.dtype)
        return y2

    @pycrt.enforce_precision(i="arr")
    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        return 2 * arr

    @pycrt.enforce_precision(i=["arr", "tau"])
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        y = arr / (2 * tau + 1)
        return y

    @pycrt.enforce_precision(i="data", allow_None=True, o=False)
    def asloss(self, data: typ.Optional[pyct.NDArray] = None) -> pyco.ProxFunc:
        if data is None:
            return self
        else:
            return self.argshift(-data)


class IRL1Norm_1(pyco.ProxDiffFunc):
    def __init__(self, W: pyct.NDArray, M: int = None):
        super().__init__(shape=(1, M))
        self.weighted_matrix = W
        self._lipschitz = 1

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        y_ = xp.multiply(self.weighted_matrix, arr).astype(arr.dtype) 
        y = xp.linalg.norm(y_, ord=1, axis=-1, keepdims=True).astype(arr.dtype)
        return y

    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        y = xp.multiply(self.weighted_matrix, xp.multiply(xp.fmax(0, xp.abs(xp.divide(arr, self.weighted_matrix)) - tau), xp.sign(arr)))
        return y



    @pycrt.enforce_precision(i="data", allow_None=True)
    def asloss(self, data: typ.Optional[pyct.NDArray] = None) -> pyco.ProxFunc:
        if data is None:
            return self
        else:
            return self.argshift(-data)




class IRL1Norm_2(pyco.ProxDiffFunc):
    def __init__(self, W: pyct.NDArray, M: int = None):
        super().__init__(shape=(1, M))
        self.weighted_matrix = W
        self._lipschitz = 1

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        y_ = xp.multiply(self.weighted_matrix, arr).astype(arr.dtype) 
        y = xp.linalg.norm(y_, ord=1, axis=-1, keepdims=True).astype(arr.dtype)
        return y

    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        y = xp.multiply(self.weighted_matrix, xp.multiply(xp.fmax(0, xp.divide(arr, self.weighted_matrix) - tau), xp.sign(arr)))
        return y



    @pycrt.enforce_precision(i="data", allow_None=True)
    def asloss(self, data: typ.Optional[pyct.NDArray] = None) -> pyco.ProxFunc:
        if data is None:
            return self
        else:
            return self.argshift(-data)

 
