from operator import itemgetter
import itertools
from typing import Callable, Iterable, Iterator, TypeVar, Sequence, Set, Tuple
import numpy as np
from FRL.approximate_dynamic_programming import (ValueFunctionApprox, QValueFunctionApprox, NTStateDistribution,
                                                 extended_vf)
from FRL.distribution import Categorical
import FRL.iterate as iterate
import FRL.markov_process as mp
from FRL.function_approx import LinearFunctionApprox, Weights
from FRL.monte_carlo import greedy_policy_from_qvf
from FRL.markov_decision_process import (MarkovDecisionProcess,
                                         TransitionStep,
                                         NonTerminal)
from FRL.policy import Policy, DeterministicPolicy
from FRL.experience_replay import ExperienceReplayMemory

S = TypeVar("S")


def td_prediction(transitions: Iterable[mp.TransitionStep[S]],
                  approx_0: ValueFunctionApprox[S],
                  gamma: float) -> Iterator[ValueFunctionApprox[S]]:
    '''
    Evaluate an MRP using TD(0) using the given sequence of
       transitions.

       Each value this function yields represents the approximated value
       function for the MRP after an additional transition.

    Arguments:
         transitions -- a sequence of transitions from an MRP which don't
                        have to be in order or from the same simulation
         approx_0 -- initial approximation of value function
         γ -- discount rate (0 < γ ≤ 1)
    '''

    def step(v: ValueFunctionApprox[S], transition: mp.TransitionStep[S]) -> ValueFunctionApprox[S]:
        return v.update([(transition.state, transition.reward + gamma * extended_vf(v, transition.next_state))])

    return iterate.accumulate(transitions, step, initial=approx_0)


def batch_td_prediction(transitions: Iterable[mp.TransitionStep[S]],
                        approx_0: ValueFunctionApprox[S],
                        gamma: float,
                        convergence_tolerance: float = 1e-5) -> ValueFunctionApprox[S]:
    """Transitions is a finite iterable"""

    def step(v: ValueFunctionApprox[S], tr_seq: Iterable[mp.TransitionStep[S]]) -> ValueFunctionApprox[S]:
        return v.update([(tr.state, tr.reward + gamma * extended_vf(v, tr.next_state)) for tr in tr_seq])

    def done(a: ValueFunctionApprox[S],
             b: ValueFunctionApprox[S],
             convergence_tolerance: float = convergence_tolerance) -> bool:
        return b.within(a, convergence_tolerance)

    return iterate.converged(
        iterate.accumulate(itertools.repeat(list(transitions)), step, initial=approx_0), done)


def least_squares_td(
        transitions: Iterable[mp.TransitionStep[S]],
        feature_functions: Sequence[Callable[[NonTerminal[S]], float]],
        gamma: float,
        epsilon: float) -> LinearFunctionApprox[NonTerminal[S]]:
    """Transitions is a finite iterable"""
    num_features: int = len(feature_functions)
    a_inv: np.ndarray = np.eye(num_features) / epsilon
    b_vec: np.ndarray = np.zeros(num_features)

    for tr in transitions:
        phi1: np.ndarray = np.array([f(tr.state) for f in feature_functions])
        if isinstance(tr.next_state, NonTerminal):
            phi2 = phi1 - gamma * np.array([f(tr.next_state) for f in feature_functions])
        else:
            phi2 = phi1
        temp: np.ndarray = a_inv.T.dot(phi2)
        a_inv = a_inv - np.outer(a_inv.dot(phi1), temp) / (1 + phi1.dot(temp))
        b_vec += phi1 * tr.reward

    opt_wts: np.ndarray = a_inv.dot(b_vec)
    return LinearFunctionApprox.create(
        feature_functions, weights=Weights.create(opt_wts))


A = TypeVar("A")


def epsilon_greedy_action(q: QValueFunctionApprox[S, A],
                          nt_state: NonTerminal[S],
                          actions: Set[A],
                          epsilon: float) -> A:
    greedy_action: A = max(((a, q((nt_state, a))) for a in actions), key=itemgetter(1))[0]

    return Categorical(
        {a: epsilon / len(actions) + (1 - epsilon if a == greedy_action else 0.) for a in actions}).sample()


def glie_sarsa(mdp: MarkovDecisionProcess[S, A],
               states: NTStateDistribution[S], approx_0: QValueFunctionApprox[S, A],
               gamma: float,
               epsilon_as_func_of_episodes: Callable[[int], float],
               max_episode_length: int) -> Iterator[QValueFunctionApprox[S, A]]:
    q: QValueFunctionApprox[S, A] = approx_0
    yield q
    num_episodes: int = 0
    while True:
        num_episodes += 1
        e: float = epsilon_as_func_of_episodes(num_episodes)
        state: NonTerminal[S] = states.sample()
        action: A = epsilon_greedy_action(q, state, set(mdp.actions(state)), e)
        steps: int = 0
        while isinstance(state, NonTerminal) and steps < max_episode_length:
            next_state, reward = mdp.step(state, action).sample()
            if isinstance(next_state, NonTerminal):
                next_action: A = epsilon_greedy_action(q, next_state, set(mdp.actions(next_state)), e)
                q = q.update([((state, action), reward + gamma * q((next_state, next_action)))])
                action = next_action

            else:
                q = q.update([((state, action), reward)])
            yield q
            steps += 1
            state = next_state


PolicyFromQType = Callable[[QValueFunctionApprox[S, A], MarkovDecisionProcess[S, A]], Policy[S, A]]


def q_learning(mdp: MarkovDecisionProcess[S, A],
               policy_from_q: PolicyFromQType,
               states: NTStateDistribution[S],
               approx_0: QValueFunctionApprox[S, A],
               gamma: float,
               max_episode_length: int) -> Iterator[QValueFunctionApprox[S, A]]:
    q: QValueFunctionApprox[S, A] = approx_0
    yield q
    num_episodes: int = 0
    while True:
        state: NonTerminal[S] = states.sample()
        num_episodes += 1
        steps: int = 0
        while isinstance(state, NonTerminal) and steps < max_episode_length:
            policy: Policy[S, A] = policy_from_q(q, mdp)
            action: A = policy.act(state).sample()
            next_state, reward = mdp.step(state, action).sample()
            next_returns: float = max(q((next_state, a)) for a in mdp.actions(next_state)) if isinstance(
                next_state, NonTerminal) else 0.
            q = q.update([((state, action), reward + gamma * next_returns)])
            yield q
            steps += 1
            state = next_state


def q_learning_external_transitions(
        transitions: Iterable[TransitionStep[S, A]],
        actions: Callable[[NonTerminal[S]], Iterable[A]],
        approx_0: QValueFunctionApprox[S, A],
        gamma: float) -> Iterator[QValueFunctionApprox[S, A]]:
    def step(q: QValueFunctionApprox[S, A],
             transition_s: TransitionStep[S, A]) -> QValueFunctionApprox[S, A]:
        next_return: float = max(
            q((transition_s.next_state, a))
            for a in actions(transition_s.next_state)
        ) if isinstance(transition_s.next_state, NonTerminal) else 0.
        return q.update([((transition_s.state, transition_s.action),
                          transition_s.reward + gamma * next_return)])

    return iterate.accumulate(transitions, step, initial=approx_0)


def q_learning_experience_replay(mdp: MarkovDecisionProcess[S, A],
                                 policy_from_q: PolicyFromQType,
                                 states: NTStateDistribution[S],
                                 approx_0: QValueFunctionApprox[S, A],
                                 gamma: float,
                                 max_episode_length: int,
                                 mini_batch_size: int,
                                 weight_decay_half_life: float) -> Iterator[QValueFunctionApprox[S, A]]:
    exp_replay: ExperienceReplayMemory[TransitionStep[S, A]] = \
        ExperienceReplayMemory(time_weights_func=lambda t: 0.5 ** (t / weight_decay_half_life))

    q: QValueFunctionApprox[S, A] = approx_0
    yield q
    while True:
        state: NonTerminal[S] = states.sample()
        steps: int = 0
        while isinstance(state, NonTerminal) and steps < max_episode_length:
            policy: Policy[S, A] = policy_from_q(q, mdp)
            action: A = policy.act(state).sample()
            next_state, reward = mdp.step(state, action).sample()
            exp_replay.add_data(TransitionStep(state, action, next_state, reward))
            trs: Sequence[TransitionStep[S, A]] = exp_replay.sample_mini_batch(mini_batch_size)
            q = q.update(
                [((tr.state, tr.action),
                  tr.reward + gamma * (max(q((tr.next_state, a)) for a in mdp.actions(tr.next_state))
                                       if isinstance(tr.next_state, NonTerminal) else 0.))
                 for tr in trs])
            yield q
            steps += 1
            state = next_state


def least_squares_tdq(
        transitions: Iterable[TransitionStep[S, A]],
        feature_functions: Sequence[Callable[[Tuple[NonTerminal[S], A]], float]],
        target_policy: DeterministicPolicy[S, A],
        gamma: float,
        epsilon: float) -> LinearFunctionApprox[Tuple[NonTerminal[S], A]]:
    num_features: int = len(feature_functions)
    a_inv: np.ndarray = np.eye(num_features) / epsilon
    b_vec: np.ndarray = np.zeros(num_features)
    for tr in transitions:
        phi1: np.ndarray = np.array([f((tr.state, tr.action)) for f in feature_functions])
        if isinstance(tr.next_state, NonTerminal):
            phi2 = phi1 - gamma * np.array([f((tr.next_state, target_policy.action_for(tr.next_state).sample()))
                                            for f in feature_functions])
        else:
            phi2 = phi1

        temp: np.ndarray = a_inv.T.dot(phi2)
        a_inv = a_inv - np.outer(a_inv.dot(phi1), temp) / (1 + phi1.dot(temp))
        b_vec += phi1 * tr.reward

    opt_wts: np.ndarray = a_inv.dot(b_vec)
    return LinearFunctionApprox.create(feature_functions, weights=Weights.create(opt_wts))


def least_squares_policy_iteration(
        transitions: Iterable[TransitionStep[S, A]],
        actions: Callable[[NonTerminal[S]], Iterable[A]],
        feature_functions: Sequence[Callable[[Tuple[NonTerminal[S], A]], float]],
        initial_target_policy: DeterministicPolicy[S, A],
        gamma: float,
        epsilon: float) -> Iterator[LinearFunctionApprox[Tuple[NonTerminal[S], A]]]:
    target_policy: DeterministicPolicy[S, A] = initial_target_policy
    transitions_seq: Sequence[TransitionStep[S, A]] = list(transitions)
    while True:
        q: LinearFunctionApprox[Tuple[NonTerminal[S], A]] = least_squares_tdq(transitions_seq,
                                                                              feature_functions,
                                                                              target_policy,
                                                                              gamma,
                                                                              epsilon)
        target_policy = greedy_policy_from_qvf(q, actions)
        yield q
