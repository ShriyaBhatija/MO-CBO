from typing import Set, List, Tuple, FrozenSet, AbstractSet

from .model import CausalDiagram
from .utils import pop, only, combinations


def CC(G: CausalDiagram, X: str):
    """ an X containing c-component of G  """
    return G.c_component(X)


def MISs(G: CausalDiagram, Y: List[str]) -> FrozenSet[FrozenSet[str]]:
    """ All minimal intervention sets """
    II = G.V - set(Y)
    assert II <= G.V
    assert not any(y in II for y in Y)

    G = G[G.An(Y)]
    Ws = G.causal_order(backward=True)
    Ws = only(Ws, II)
    return subMISs(G, Y, frozenset(), Ws)


def subMISs(G: CausalDiagram, Y: List[str], Xs: FrozenSet[str], Ws: List[str]) -> FrozenSet[FrozenSet[str]]:
    """ subroutine for MISs -- this creates a recursive call tree with n, n-1, n-2, ... widths """
    out = frozenset({Xs})
    for i, W_i in enumerate(Ws):
        H = G.do({W_i})
        H = H[H.An(Y)]
        out |= subMISs(H, Y, Xs | {W_i}, only(Ws[i + 1:], H.V))
    return out


def bruteforce_POMISs(G: CausalDiagram, Y: List[str]) -> FrozenSet[FrozenSet[str]]:
    """ This computes a complete set of POMISs in a brute-force way """
    return frozenset({frozenset(IB(G.do(Ws), Y))
                      for Ws in combinations(list(G.V - set(Y)))})


def MUCT(G: CausalDiagram, Y: List[str]) -> FrozenSet[str]:
    """ Minimal Unobserved Confounder's Territory """
    H = G[G.An(Y)]

    Qs = set(Y)
    Ts = frozenset(set(Y))
    Ys = set(Y)

    while Qs:
        Q1 = pop(Qs)
        if Q1 in Ys:
            Ys = Ys - {Q1}
        Ws = CC(H, Q1)
        Ts |= Ws
        Qs = (Qs | H.de(Ws)) - Ts
        Qs |= Ys

    return Ts


def IB(G: CausalDiagram, Y: List[str]) -> FrozenSet[str]:
    """ Interventional Border """
    Zs = MUCT(G, Y)
    return G.pa(Zs) - Zs


def MUCT_IB(G: CausalDiagram, Y: List[str]) -> Tuple[FrozenSet[str], FrozenSet[str]]:
    Zs = MUCT(G, Y)
    return Zs, G.pa(Zs) - Zs