from typing import Sequence, Callable, Tuple, TypeVar
from src.func_approx.func_approx_base import FuncApproxBase
from src.func_approx.dnn_spec import DNNSpec
from scipy.stats import norm
from src.func_approx.eligibility_traces import get_generalized_back_prop
import numpy as np

X = TypeVar('X')


class DNN(FuncApproxBase):
    def __init__(self,
                 feature_funcs: Sequence[Callable[[X], float]],
                 dnn_obj: DNNSpec,
                 reglr_coeff: float = 0.0,
                 learning_rate: float = 0.1,
                 adam: bool = True,
                 adam_decay1: float = 0.9,
                 adam_decay2: float = 0.99,
                 add_unit_feature: bool = True
                 ):
        self.neurons: Sequence[int] = dnn_obj.neurons
        self.hidden_activation: Callable[[np.ndarray], np.ndarray] = dnn_obj.hidden_activation
        self.hidden_activation_deriv: Callable[[np.ndarray], np.ndarray] = dnn_obj.hidden_activation_deriv
        self.output_activation: Callable[[np.ndarray], np.ndarray] = dnn_obj.output_activation
        self.output_activation_deriv: Callable[[np.ndarray], np.ndarray] = dnn_obj.output_activation_deriv
        super().__init__(feature_funcs, reglr_coeff, learning_rate, adam, adam_decay1, adam_decay2, add_unit_feature)

    def init_params(self) -> Sequence[np.ndarray]:
        """These are Xavier input parameters"""
        inp_size = self.num_features
        params = []
        for layer_neurons in self.neurons:
            mat = np.random.rand(layer_neurons, inp_size) / np.sqrt(inp_size)
            params.append(mat)
            inp_size = layer_neurons + 1
        params.append(np.random.randn(1, inp_size) / np.sqrt(inp_size))
        return params

    def init_adam_caches(self) -> Tuple[Sequence[np.ndarray], Sequence[np.ndarray]]:
        return ([np.zeros_like(p) for p in self.params],
                [np.zeros_like(p) for p in self.params])

    def get_forward_prop(self, x_vals_seq: Sequence[X]) -> Sequence[np.ndarray]:
        """

        :param x_vals_seq: a n-length-sequence of input points
        :return: list of length (L+2) where the first (L+1) values each represents the 2-D input arrays (of size n x |I_1] +1),
                for each of the (L+1) layers (L of which are hidden layers), and the last value represents the output of the DNN (as a
                2-D array of length n x 1)
        """
        inp = self.get_feature_vals_pts(x_vals_seq)
        outputs = [inp]
        for this_params in self.params[:-1]:
            out = self.hidden_activation(np.dot(inp, this_params.T))
            inp = np.insert(out, 0, 1., axis=1)
            outputs.append(out)
        outputs.append(self.output_activation(np.dot(inp, self.params[-1].T)))
        return outputs

    def get_func_eval(self, x_vals: X) -> float:
        return self.get_forward_prop([x_vals])[-1][0, 0]

    def get_func_eval_pts(self, x_vals_seq: Sequence[X]) -> np.ndarray:
        return self.get_forward_prop(x_vals_seq)[-1][:, 0]

    def get_back_prop(self, fwd_prop: Sequence[np.ndarray],
                      dObj_dOL: np.ndarray) -> Sequence[np.ndarray]:
        """

        :param fwd_prop:list (of length L+2), the first (L+1) elements of which
                    are n x (|I_l| + 1) 2-D arrays representing the inputs to the (L+1) layers,
                    and the last element is a n x 1 2-D array
        :param dObj_dOL: 1-D array of length n representing the derivative of
                    objective with respect to the output of the DNN.
                    L is the number of hidden layers, n is the number of points
        :return: list (of length L+1) of |O_l| x (|I_l| + 1) 2-D array,
                    i.e., same as the type of self.params
        """
        outputs = fwd_prop[-1]
        layer_inputs = fwd_prop[:-1]
        deriv = (dObj_dOL.reshape(-1, 1) * self.output_activation_deriv(outputs)).T
        back_prop = []
        for l in reversed(range(len(self.params))):
            back_prop.append(np.dot(deriv, layer_inputs[l]))
            deriv = (np.dot(self.params[l].T, deriv) * self.hidden_activation_deriv(layer_inputs[l].T))[1:]

        return back_prop[::-1]

    def get_sum_loss_gradient(self, x_vals_seq: Sequence[X],
                              supervisory_seq: Sequence[float]) -> Sequence[np.ndarray]:
        """
        :param x_vals_seq: list of n data points (x points)
        :param supervisory_seq: list of n supervisory points
        :return:  list (of length L+1) of |O_l| x (|I_l| + 1) 2-D array, i.e., same as the type of self.params
                This function computes the gradient (with respect to w) of
                Loss(w) = \sum_i (f(x_i; w) - y_i)^2 where f is the DNN func
        """
        fwd_prop = self.get_forward_prop(x_vals_seq)
        errors = fwd_prop[-1][:, 0] - supervisory_seq
        return self.get_back_prop(fwd_prop, errors)

    def get_sum_objective_gradient(self, x_vals_seq: Sequence[X],
                                   dObj_dOL: np.ndarray) -> Sequence[np.ndarray]:
        """
        :param x_vals_seq: list of n data points (x points)
        :param dObj_dOL: 1-D array of length n representing the derivative
                of the objective function with respect to the output of the DNN
        :return: list (of length L+1) of |O_l| x (|I_l| + 1) 2-D array,
                 i.e., same as the type of self.params
                This function computes the gradient (with respect to w) of
                g(w) = \sum_i Obj(f(x_i; w))
        """
        fwd_prop = self.get_forward_prop(x_vals_seq)
        return self.get_back_prop(fwd_prop, dObj_dOL)

    def get_el_tr_sum_loss_gradient(self, x_vals_seq: Sequence[X],
                                    supervisory_seq: Sequence[float],
                                    gamma_lambda: float) -> Sequence[np.ndarray]:
        """
        :param x_vals_seq: list of n data points (x points)
        :param supervisory_seq: list of n supervisory points
        :param gamma_lambda: decay discount factor
        :return: list (of length L+1) of |O_l| x (|I_l| + 1) 2-D array,
                         i.e., same as the type of self.params
        """
        fwd_prop = self.get_forward_prop(x_vals_seq)
        errors = fwd_prop[-1][:, 0] - supervisory_seq
        return get_generalized_back_prop(
            dnn_params=self.params,
            fwd_prop=fwd_prop,
            dObj_dOL=np.ones_like(errors),
            factors=errors,
            decay_param=gamma_lambda,
            hidden_activation_deriv=self.hidden_activation_deriv,
            output_activation_deriv=self.output_activation_deriv
        )

    def get_el_tr_sum_objective_gradient(self, x_vals_seq: Sequence[X],
                                         dObj_dOL: np.ndarray,
                                         factors: np.ndarray,
                                         gamma_lambda: float) -> Sequence[np.ndarray]:
        """
        :param x_vals_seq: list of n data points (x points)
        :param dObj_dOL: 1-D array of length n representing the derivative
               of the objective function with respect to the output of the DNN
        :param factors: 1-D array of length n representing the factors
               to be multiplied to dObjective/dparam across the n points
        :param gamma_lambda: decay discount factor
        :return: list (of length L+1) of |O_l| x (|I_l| + 1) 2-D array,
                        i.e., same as the type of self.params
        """
        fwd_prop = self.get_forward_prop(x_vals_seq)
        return get_generalized_back_prop(
            dnn_params=self.params,
            fwd_prop=fwd_prop,
            dObj_dOL=dObj_dOL,
            factors=factors,
            decay_param=gamma_lambda,
            hidden_activation_deriv=self.hidden_activation_deriv,
            output_activation_deriv=self.output_activation_deriv
        )
