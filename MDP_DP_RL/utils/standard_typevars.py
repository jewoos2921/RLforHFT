from typing import Callable, Sequence, Mapping, Tuple, TypeVar
from src.utils.generic_typevars import S, A

VFType = Callable[[S], float]
QFType = Callable[[S], Callable[[A], float]]
PolicyType = Callable[[S], Callable[[int], Sequence[A]]]

VFDictType = Mapping[S, float]
QFDictType = Mapping[S, Mapping[A, float]]
PolicyActDictType = Callable[[S], Mapping[A, float]]

SSf = Mapping[S, Mapping[S, float]]
SSTff = Mapping[S, Mapping[S, Tuple[float, float]]]
STSff = Mapping[S, Tuple[Mapping[S, float], float]]
SAf = Mapping[S, Mapping[A, float]]
SASf = Mapping[S, Mapping[A, Mapping[S, float]]]
SASTff = Mapping[S, Mapping[A, Mapping[S, Tuple[float, float]]]]
SATSff = Mapping[S, Mapping[A, Tuple[Mapping[S, float], float]]]
