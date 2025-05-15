from dataclasses import dataclass
from typing import Callable, Generic, NewType, TypeVar, TypeVarTuple

import numpy as np

DType = TypeVar("DType")
Shape = TypeVarTuple("Shape")

VocabSize = NewType("VocabSize", int)
SequenceLength = NewType("SequenceLength", int)


class Array(Generic[DType, *Shape], list[DType]):
    pass


@dataclass
class Datum:
    tokens: Array[int, SequenceLength]
    weight: float
    active: bool


def smc(
    q: Callable[[Array[int, SequenceLength]], Array[float, VocabSize]],
    il: Callable[[Array[int, SequenceLength]], Array[float, VocabSize]],
    M: int,
    tau: float,
    eos_token_id: int,
    vocab_size: VocabSize,
) -> list[tuple[Array[int, SequenceLength], float]]:
    data = [
        Datum(
            tokens=Array([]),  # Example token array
            weight=1.0,
            active=True,
        )
        for _ in range(M)
    ]
    while any(datum.active for datum in data):
        for datum in filter(lambda datum: datum.active, data):
            sampled_token = int(
                np.random.choice(
                    a=range(vocab_size),
                    p=q(datum.tokens),
                )
            )
            # TODO: use functional form of phi and utilize caching
            # to make sure we only need to compute something about
            # the new token
            # TODO: use log weights for numerical stability
            # TODO: need to add token and reweight when we get eos?
            # if so, then we need to modify phi for the eos case
            datum.weight *= sum(
                np.array(q(datum.tokens + [sampled_token]))
                * np.array(il(datum.tokens + [sampled_token]))
            )
            if sampled_token == eos_token_id:
                datum.active = False
            else:
                datum.tokens.append(sampled_token)

        data = resample(data, tau)

    W = sum([datum.weight for datum in data])
    import itertools

    gb = itertools.groupby(
        sorted(data, key=lambda datum: datum.tokens), key=lambda datum: datum.tokens
    )
    P = []
    for tokens, datum_group in gb:
        weight = sum([datum.weight for datum in datum_group])
        P.append((tokens, weight / W))
    return P


def resample(
    data: list[Datum],
    tau: float,
) -> list[Datum]:
    M = len(data)
    W = sum([datum.weight for datum in data])
    M_eff = W**2 / sum([datum.weight**2 for datum in data])
    # Resample if effective sample size is less than tau * M
    if M_eff < tau * M:
        new_data: list[Datum] = []
        for _ in range(M):
            R = int(
                np.random.choice(
                    range(M),
                    p=[datum.weight / W for datum in data],
                )
            )
            new_data.append(
                Datum(
                    # !IMPORTANT! make sure to copy the tokens
                    tokens=Array(data[R].tokens),
                    weight=W / M,
                    active=data[R].active,
                )
            )
        return new_data
    return data
