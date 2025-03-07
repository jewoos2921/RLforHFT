from typing import Sequence, Callable
from scipy.linalg import toeplitz
import numpy as np


def get_decay_toeplitz_matrix(size: int, decay_param: float) -> np.ndarray:
    return toeplitz(
        np.power(decay_param, np.arange(size)),
        np.insert(np.zeros(size - 1), 0, 1.))


def get_generalized_back_prop(dnn_params: Sequence[np.ndarray],
                              fwd_prop: Sequence[np.ndarray],
                              dObj_dOL: np.ndarray,
                              factors: np.ndarray,
                              decay_param: float,
                              hidden_activation_deriv: Callable[[np.ndarray], np.ndarray],
                              output_activation_deriv: Callable[[np.ndarray], np.ndarray]
                              ) -> Sequence[np.ndarray]:
    output = fwd_prop[-1][:, 0]
    layer_inputs = fwd_prop[:-1]

    deriv = (dObj_dOL * output_activation_deriv(output)).reshape(1, -1)
    decay_matrix = get_decay_toeplitz_matrix(len(factors), decay_param)
    back_prop = []
    for l in reversed(range(len(dnn_params))):
        t1 = np.einsum("ij,jk->jik", deriv, layer_inputs[l])
        if decay_param != 0:
            t2 = np.tensordot(decay_matrix, t1, axes=1)
        else:
            t2 = t1
        t3 = np.tensordot(factors, t2, axes=1)
        back_prop.append(t3)

        deriv = (np.dot(dnn_params[l].T, deriv)
                 * hidden_activation_deriv(layer_inputs[l].T))[1:]

    return back_prop[::-1]
