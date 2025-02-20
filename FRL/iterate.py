# Finding fixed points of a function using iterators
import itertools
from typing import Iterable, Iterator, Callable, Optional, TypeVar

X = TypeVar('X')
Y = TypeVar('Y')


def iterate(step: Callable[[X], X], start: X) -> Iterator[X]:
    """Find the fixed point of a function f by applying it to its own result, yielding each intermediate value.
    That is, for a function f, iterate(f, state) will give us a generator
    producing:
    state , f(state), f(f(state)),  f(f(f(state))), ...
    """
    state = start
    while True:
        yield state
        state = step(state)


def last(values: Iterator[X]) -> Optional[X]:
    """Return the last value of an iterator, or None if the iterator is empty."""
    try:
        *_, last_element = values
        return last_element
    except ValueError:
        return None


def converge(values: Iterator[X], done: Callable[[X, X], bool]) -> Iterator[X]:
    """Read from an iterator until two consecutive values satisfy the
    given done function or the input iterator ends.
    Raises an error if the input iterator is empty.
    Will loop forever if the input iterator doesn't end 'or' converge.
    """
    a = next(values, None)
    if a is None:
        return

    yield a
    for b in values:
        yield b
        if done(a, b):
            return

        a = b


def converged(values: Iterator[X],
              done: Callable[[X, X], bool]) -> X:
    """Return the last value of an iterator, or None if the iterator is empty."""
    result = last(converge(values, done))
    if result is None:
        raise ValueError("Convergence called on an empty iterator")
    return result


def accumulate(iterable: Iterable[X],
               func: Callable[[Y, X], Y],
               *,
               initial: Optional[Y]) -> Iterator[Y]:
    if initial is not None:
        iterable = itertools.chain([initial], iterable)

    return itertools.accumulate(iterable, func)


if __name__ == '__main__':
    import numpy as np

    x = 0.0
    values = converge(
        iterate(lambda y: np.cos(y), x),
        lambda a, b: np.abs(a - b) < 1e-3
    )
    for i, v in enumerate(values):
        print(f"{i}: {v:.4f}")
