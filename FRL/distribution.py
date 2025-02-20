from __future__ import annotations
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass
import numpy as np
import random
from typing import (
    Callable, Dict, Generic, Iterable, Iterator,
    Tuple, Mapping, Optional, Sequence, TypeVar
)

A = TypeVar('A')
B = TypeVar('B')


# Greek = 가격의 민감도
# alpha, beta, delta, gamma, vega, theta, rho
# https://en.wikipedia.org/wiki/Greek_letter

class Distribution(ABC, Generic[A]):
    """
    Return a random sample from the distribution.
    """

    @abstractmethod
    def sample(self) -> A:
        """Return a random sample from the distribution"""
        pass

    def sample_n(self, n: int) -> Sequence[A]:
        """Return n sample for this distribution.
        This is a convenience method for sampling n samples from this distribution
        """
        return [self.sample() for _ in range(n)]

    @abstractmethod
    def expectation(self, f: Callable[[A], float]) -> float:
        """Return the expectation of f(x) where x is the random variable for
        the distribution and f is an arbitrary function from x to float
        """
        pass

    def map(self, f: Callable[[A], B]) -> Distribution[B]:
        """ Apply a function to the outcomes of this distribution
        """
        return SampledDistribution(lambda: f(self.sample()))

    def apply(self, f: Callable[[A], Distribution[B]]) -> Distribution[B]:
        """ Apply a function that returns a distribution to the outcomes of this distribution.
        this let us express *dependent random variables*
        """

        def sample():
            a = self.sample()
            b_dist = f(a)
            return b_dist.sample()

        return SampledDistribution(sample)


class SampledDistribution(Distribution[A]):
    """
    Return a random sample from the distribution.
    """
    sampler: Callable[[], A]
    expectation_samples: int

    def __init__(self, sampler: Callable[[], A], expectation_samples: int = 10000):
        self.sampler = sampler
        self.expectation_samples = expectation_samples

    def sample(self) -> A:
        return self.sampler()

    def expectation(self, f: Callable[[A], float]) -> float:
        """Return a sampled approximation of the expectation of f(x) for some f."""
        return sum(f(self.sample())
                   for _ in range(self.expectation_samples)) / self.expectation_samples


class Uniform(SampledDistribution[float]):
    """Sample a uniform float between 0 and 1."""

    def __init__(self, expectation_samples: int = 10000):
        super().__init__(sampler=lambda: random.uniform(0, 1), expectation_samples=expectation_samples)


class Poisson(SampledDistribution[int]):
    """
    A Poisson distribution.
    """
    lambda_: float

    def __init__(self, lambda_: float, expectation_samples: int = 1000):
        self.lambda_ = lambda_
        super().__init__(
            sampler=lambda: np.random.poisson(lam=self.lambda_),
            expectation_samples=expectation_samples)


class Gaussian(SampledDistribution[float]):
    """A Gaussian distribution."""
    mu: float
    sigma: float

    def __init__(self, mu: float, sigma: float, expectation_samples: int = 10000):
        self.mu = mu
        self.sigma = sigma
        super().__init__(
            sampler=lambda: np.random.normal(loc=self.mu, scale=self.sigma),
            expectation_samples=expectation_samples)


class Gamma(SampledDistribution[float]):
    """A Gamma distribution with the given shape and scale."""
    alpha: float
    beta: float

    def __init__(self, alpha: float, beta: float, expectation_samples: int = 10000):
        self.alpha = alpha
        self.beta = beta
        super().__init__(
            sampler=lambda: np.random.gamma(shape=self.alpha, scale=1 / self.beta),
            expectation_samples=expectation_samples)


class Beta(SampledDistribution[float]):
    """A Beta distribution with the given shape and scale."""
    alpha: float
    beta: float

    def __init__(self, alpha: float, beta: float, expectation_samples: int = 10000):
        self.alpha = alpha
        self.beta = beta
        super().__init__(
            sampler=lambda: np.random.beta(a=self.alpha, b=self.beta),
            expectation_samples=expectation_samples)


class FiniteDistribution(Distribution[A], ABC):
    """A probability distribution with a finite number of outcomes, which means we can
        render it as a PDF or CDF table. """

    @abstractmethod
    def table(self) -> Mapping[A, float]:
        """Return a tabular representation of the probability density function (PDF) for this distribution."""
        pass

    def probability(self, outcome: A) -> float:
        """Returns the probability of the given outcome according to this distribution."""
        return self.table()[outcome]

    def map(self, f: Callable[[A], B]) -> FiniteDistribution[B]:
        """Return a new distribution that is the result of applying a function
        to each element of distribution."""

        result: Dict[B, float] = defaultdict(float)

        for x, p in self:
            result[f(x)] += p

        return Categorical(result)

    def sample(self) -> A:
        outcomes = list(self.table().keys())
        weights = list(self.table().values())
        return random.choices(outcomes, weights=weights)[0]

    def expectation(self, f: Callable[[A], float]) -> float:
        """Calculate the expected value of the distribution, using the given
            function to turn the outcomes into a value."""
        return sum(p * f(x) for x, p in self)

    def __iter__(self) -> Iterator[Tuple[A, float]]:
        """Return an iterator over the outcomes of this distribution."""
        return iter(self.table().items())

    def __eq__(self, other) -> bool:
        if isinstance(other, FiniteDistribution):
            return self.table() == other.table()
        else:
            return False

    def __repr__(self):
        return repr(self.table())


@dataclass(frozen=True)
class Constant(FiniteDistribution[A]):
    """A distribution that has a single outcome with probability 1."""
    value: A

    def table(self) -> Mapping[A, float]:
        return {self.value: 1}

    def sample(self) -> A:
        return self.value

    def probability(self, outcome: A) -> float:
        return 1. if outcome == self.value else 0.


@dataclass(frozen=True)
class Bernoulli(FiniteDistribution[bool]):
    """A Bernoulli distribution with the given probability of success."""
    rho: float

    def sample(self) -> bool:
        return random.uniform(0, 1) <= self.rho

    def table(self) -> Mapping[bool, float]:
        return {True: self.rho, False: 1 - self.rho}

    def probability(self, outcome: bool) -> float:
        return self.rho if outcome else 1 - self.rho


@dataclass
class Range(FiniteDistribution[int]):
    """Select a random integer in the range [low, high), with low inclusive and high exclusive.
    (this works exactly the same as the normal range function, but differently from random.randit.
    """
    low: int
    high: int

    def __init__(self, a: int, b: Optional[int] = None):
        if b is None:
            b = a
            a = 0

        assert b > a

        self.low = a
        self.high = b

    def sample(self) -> int:
        return random.randint(self.low, self.high)

    def table(self) -> Mapping[int, float]:
        length = self.high - self.low
        return {x: 1 / length for x in range(self.low, self.high)}


class Choose(FiniteDistribution[A]):
    """Select an element of the given list uniformly at random."""
    options: Sequence[A]
    _table: Optional[Mapping[A, float]] = None

    def __init__(self, options: Iterable[A]):
        self.options = list(options)

    def sample(self) -> A:
        return random.choice(self.options)

    def table(self) -> Mapping[A, float]:
        if self._table is None:
            counter = Counter(self.options)
            length = len(self.options)
            self._table = {x: counter[x] / length for x in counter}

        return self._table

    def probability(self, outcome: A) -> float:
        return self.table().get(outcome, 0.0)


class Categorical(FiniteDistribution[A]):
    """Select from a finite set of outcomes with the specified probabilities."""
    probabilities: Mapping[A, float]

    def __init__(self, distribution: Mapping[A, float]):
        total = sum(distribution.values())
        self.probabilities = {outcome: probability / total for outcome, probability in distribution.items()}

    def table(self) -> Mapping[A, float]:
        return self.probabilities

    def probability(self, outcome: A) -> float:
        return self.probabilities.get(outcome, 0.0)
