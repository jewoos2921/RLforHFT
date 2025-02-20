from typing import Iterable, Iterator, TypeVar, Callable
from FRL.distribution import Categorical
from FRL.approximate_dynamic_programming import (ValueFunctionApprox,
                                                QValueFunctionApprox, NTStateDistribution)
from FRL.iterate import last
from FRL.markov_decision_process import MarkovDecisionProcess, Policy, TransitionStep, NonTerminal
from FRL.policy import DeterministicPolicy, RandomPolicy, UniformPolicy
from FRL.returns import returns
import FRL.markov_process as mp

S = TypeVar('S')
A = TypeVar('A')


def mc_prediction(traces: Iterable[Iterable[mp.TransitionStep[S]]],
                  approx_0: ValueFunctionApprox[S],
                  y: float,
                  episode_length_tolerance: float = 1e-6) -> Iterator[ValueFunctionApprox[S]]:
    """
    Evaluate an MRP using the monte carlo method, simulating episodes of the given number of steps.

    :param traces: an iterator of simulation traces from an MRP
    :param approx_0: initial approximation of value function
    :param y: discount rate (0 < y < 1), default: 1
    :param episode_length_tolerance: stop iterating once y^k <= tolerance
    :return:
        returns an iterator with updates to the approximated value function after each episode.
    """
    episodes: Iterator[Iterator[mp.ReturnStep[S]]] = (returns(trace, y, episode_length_tolerance) for trace in traces)
    f = approx_0
    yield f

    for episode in episodes:
        f = last(f.iterate_updates([(step.state, step.return_)] for step in episode))
        yield f


def greedy_policy_from_qvf(
        q: QValueFunctionApprox[S, A],
        actions: Callable[[NonTerminal[S]], Iterable[A]]
) -> DeterministicPolicy[S, A]:
    """ Return the policy that takes the optimal action at each state based
    on the givne approximation of the process's Q function."""

    def optimal_action(s: S) -> A:
        _, a = q.argmax((NonTerminal(s), a) for a in actions(NonTerminal(s)))
        return a

    return DeterministicPolicy(optimal_action)


def epsilon_greedy_policy(
        q: QValueFunctionApprox[S, A],
        mdp: MarkovDecisionProcess[S, A],
        epsilon: float = 0.0) -> Policy[S, A]:
    def explore(s: S, mdp=mdp) -> Iterable[A]:
        return mdp.actions(NonTerminal(s))

    return RandomPolicy(Categorical({
        UniformPolicy(explore): epsilon,
        greedy_policy_from_qvf(q, mdp.actions): 1 - epsilon
    }))


def glie_mc_control(
        mdp: MarkovDecisionProcess[S, A],
        states: NTStateDistribution[S],
        approx_0: QValueFunctionApprox[S, A],
        y: float,
        e_as_func_of_episodes: Callable[[int], float],
        episode_length_tolerance: float = 1e-6) -> Iterator[QValueFunctionApprox[S, A]]:
    """Evaluate an MDP using the monte carlo method, simulating episodes of the given number of steps.

    :param mdp: an MDP
    :param states: distribution over non-terminal states
    :param approx_0: initial approximation of value function
    :param y: discount rate (0 < y < 1), default: 1
    :param e_as_func_of_episodes: function that returns the number of episodes
        to simulate for each iteration
    :param episode_length_tolerance: stop iterating once y^k <= tolerance
    :return:
        returns an iterator with updates to the approximated value function after each episode.
    """
    q: QValueFunctionApprox[S, A] = approx_0
    p: Policy[S, A] = epsilon_greedy_policy(q, mdp, epsilon=1.0)
    yield q

    num_episodes: int = 0
    while True:
        trace: Iterable[TransitionStep[S]] = mp.simulate_actions(states, p)
        num_episodes += 1
        for step in returns(trace, y, episode_length_tolerance):
            q = q.update([((step.state, step.action), step.return_)])
