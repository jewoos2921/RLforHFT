from src.utils.standard_typevars import S, A
from typing import Callable, Sequence, Tuple, Generic


class MDPRepForADPPG(Generic[S, A]):
    def __init__(self,
                 gamma: float,
                 init_states_gen_func: Callable[[int], Sequence[S]],
                 state_reward_gen_func: Callable[[S, A, int], Sequence[Tuple[S, float]]],
                 terminal_state_func: Callable[[S], bool]
                 ):
        self.gamma: float = gamma
        self.init_states_gen_func: Callable[[int], Sequence[S]] = init_states_gen_func
        self.state_reward_gen_func: Callable[[S, A, int], Sequence[Tuple[S, float]]] = state_reward_gen_func
        self.terminal_state_func: Callable[[S], bool] = terminal_state_func
