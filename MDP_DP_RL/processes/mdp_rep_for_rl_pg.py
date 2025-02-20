from src.utils.standard_typevars import S, A
from typing import Callable, Generic, Tuple


class MDPRepForRLPG(Generic[S, A]):
    def __init__(self,
                 gamma: float,
                 init_state_gen_func: Callable[[], S],
                 state_reward_gen_func: Callable[[S, A], Tuple[S, float]],
                 terminal_state_func: Callable[[S], bool]
                 ):
        self.gamma: float = gamma
        self.init_state_gen_func: Callable[[], S] = init_state_gen_func
        self.state_reward_gen_func: Callable[[S, A], Tuple[S, float]] = state_reward_gen_func
        self.terminal_state_func: Callable[[S], bool] = terminal_state_func
