import FRL.markov_process as mp
import FRL.markov_decision_process as mdp
import FRL.iterate as iterate
import itertools
import math
from typing import Iterable, Iterator, TypeVar, overload

S = TypeVar('S')
A = TypeVar('A')


@overload
def returns(trace: Iterable[mp.TransitionStep[S]],
            y: float,
            tolerance: float) -> Iterator[mp.ReturnStep[S]]:
    pass


@overload
def returns(trace: Iterable[mdp.TransitionStep[S, A]],
            y: float,
            tolerance: float) -> Iterator[mdp.ReturnStep[S, A]]:
    pass


def returns(trace: Iterable[mp.TransitionStep[S]] | Iterable[mdp.TransitionStep[S, A]],
            y: float,
            tolerance: float) -> Iterator[mp.ReturnStep[S] | mdp.ReturnStep[S, A]]:
    """
    Given an iterator of states and rewards, calculate the return of the first N states.
    rewards -- instantaneous rewards
    y -- discount factor (0 < y <= 1)
    tolerance -- a small value-we stop iterating once y^k <= tolerance
    """
    trace = iter(trace)

    max_steps = round(math.log(tolerance) / math.log(y)) if y < 1 else None
    if max_steps is not None:
        trace = itertools.islice(trace, max_steps * 2)

    *transitions, last_transition = list(trace)

    return_steps = iterate.accumulate(
        reversed(transitions),
        func=lambda next, curr: curr.add_return(y, next.return_),
        initial=last_transition.add_return(y, 0)
    )

    return_steps = reversed(list(return_steps))
    if max_steps is not None:
        return_steps = itertools.islice(return_steps, max_steps)

    return return_steps
