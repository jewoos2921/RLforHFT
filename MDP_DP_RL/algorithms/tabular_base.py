from typing import Mapping, Callable, Set
from abc import abstractmethod
from src.algorithms.opt_base import OptBase
from src.processes.policy import Policy
from src.processes.det_policy import DetPolicy
from src.algorithms.helper_funcs import get_pdf_from_samples
from src.utils.standard_typevars import S, A, VFDictType, QFDictType, PolicyType


class TabularBase(OptBase):
    NUM_SAMPLES_PER_ACTION = 10

    @abstractmethod
    def get_init_policy(self) -> Policy:
        pass

    @abstractmethod
    def get_value_func_dict(self, pol: Policy) -> VFDictType:
        pass

    @abstractmethod
    def get_act_value_func_dict(self, pol: Policy) -> QFDictType:
        pass

    @abstractmethod
    def get_optimal_det_policy(self) -> DetPolicy:
        pass

    @abstractmethod
    def get_state_action_dict(self) -> Mapping[S, Set[A]]:
        pass

    def get_value_func(self, polf: PolicyType) -> Callable[[S], float]:
        pol = Policy({s: get_pdf_from_samples(polf(s)(len(v) * TabularBase.NUM_SAMPLES_PER_ACTION)) for s, v in
                      self.get_state_action_dict().items()})

        def vf(state: S, pol=pol) -> float:
            return self.get_value_func_dict(pol)[state]

        return vf

    def get_act_value_func(self, polf: PolicyType) -> Callable[[S], Callable[[A], float]]:
        pol = Policy({s: get_pdf_from_samples(polf(s)(len(v) * TabularBase.NUM_SAMPLES_PER_ACTION)) for s, v in
                      self.get_state_action_dict().items()})

        def qvf(state: S, pol=pol) -> Callable[[A], float]:
            def inner_f(action: A, pol=pol, state=state) -> float:
                return self.get_act_value_func_dict(pol)[state][action]

            return inner_f

        return qvf

    def get_optimal_det_policy_func(self) -> Callable[[S], A]:
        return lambda s: self.get_optimal_det_policy().get_action_for_state(s)
