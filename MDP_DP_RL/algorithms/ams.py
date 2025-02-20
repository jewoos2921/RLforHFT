from typing import Mapping, Set, Sequence, Tuple, Generic, Callable, Optional
from src.utils.standard_typevars import S, A
import numpy as np
from random import sample


class AdaptiveMultiStageSampling(Generic[S, A]):
    def __init__(self,
                 start_state: S,
                 actions_sets: Sequence[Set[A]],
                 num_samples: Sequence[int],
                 state_gen_reward_funcs: Sequence[Callable[[S, A], Tuple[Callable[[], S], float]]],
                 terminal_opt_val_func: Callable[[S], float],
                 discount: float) -> None:
        if len(actions_sets) == len(num_samples) == len(state_gen_reward_funcs) and 0. <= discount <= 1. and all(
                len(x) <= y for x, y in zip(actions_sets, num_samples)):
            self.start_state = start_state
            self.actions_sets = actions_sets
            self.num_samples = num_samples
            self.num_time_steps = len(actions_sets)
            self.state_gen_reward_funcs = state_gen_reward_funcs
            self.terminal_opt_val_func = terminal_opt_val_func
            self.discount = discount
        else:
            raise ValueError

    def get_opt_val_and_internals(self, state: S,
                                  time_step: int) -> Tuple[float, Optional[Mapping[A, Tuple[float, int]]]]:
        if time_step == self.num_time_steps:
            ret = (self.terminal_opt_val_func(state), None)
        else:
            actions = self.actions_sets[time_step]
            state_gen_reward = {a: self.state_gen_reward_funcs[time_step](state, a) for a in actions}
            state_gens = {a: x for a, (x, _) in state_gen_reward.items()}
            rewards = {a: y for a, (_, y) in state_gen_reward.items()}
            val_sums = {a: self.get_opt_val_and_internals(state_gens[a](),
                                                          time_step + 1)[0] for a in actions}
            counts = {a: 1 for a in actions}

            for i in range(len(actions), self.num_time_steps[time_step]):
                ucb_vals = {a: rewards[a] + self.discount * val_sums[a] / counts[a]
                               + np.sqrt(2 * np.log(i) / counts[a]) for a in actions}
                max_actions = {a for a, u in ucb_vals.items() if u == max(ucb_vals.values())}
                a_star = sample(max_actions, 1)[0]
                next_state = state_gens[a_star]()
                val_sums[a_star] += self.get_opt_val_and_internals(next_state, time_step + 1)[0]
                counts[a_star] += 1

            ret1 = sum(counts[a] / self.num_samples[time_step] *
                       (rewards[a] + self.discount * val_sums[a] / counts[a]) for a in actions)
            ret2 = {a: (rewards[a] + self.discount * val_sums[a] / counts[a], counts[a]) for a in actions}
            ret = (ret1, ret2)

        return ret
