"""Graph ranking: global PageRank (stored at index time) and local personalized PageRank (per query)."""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Callable, Iterable


def pagerank(nodes: Iterable[int], edges: Iterable[tuple[int, int]],
             damping: float = 0.85, iters: int = 40, tol: float = 1e-9) -> dict[int, float]:
    """Classic directed PageRank. Importance flows src -> dst, so widely depended-upon code ranks high."""
    nodes = list(nodes)
    n = len(nodes)
    if n == 0:
        return {}
    out: dict[int, list[int]] = defaultdict(list)
    for s, d in edges:
        out[s].append(d)
    rank = {i: 1.0 / n for i in nodes}
    dangling_nodes = [i for i in nodes if i not in out]
    for _ in range(iters):
        new = {i: (1 - damping) / n for i in nodes}
        share = damping * sum(rank[i] for i in dangling_nodes) / n
        for s, ds in out.items():
            c = damping * rank[s] / len(ds)
            for d in ds:
                new[d] += c
        for i in nodes:
            new[i] += share
        delta = sum(abs(new[i] - rank[i]) for i in nodes)
        rank = new
        if delta < tol:
            break
    return rank


def personalized_pagerank(nbrs: Callable[[int], list[tuple[int, float]]], seeds: dict[int, float],
                          alpha: float = 0.15, eps: float = 1e-5) -> dict[int, float]:
    """Local push approximation of random-walk-with-restart from `seeds`.

    Touches only nodes near the seeds, so cost is independent of repo size.
    `nbrs(u)` returns [(neighbour, weight)]; weights let callers down-weight hub nodes.
    """
    p: dict[int, float] = defaultdict(float)
    r: dict[int, float] = defaultdict(float)
    for s, m in seeds.items():
        r[s] += m
    queue = deque(seeds)
    queued = set(seeds)
    while queue:
        u = queue.popleft()
        queued.discard(u)
        ru = r[u]
        nb = nbrs(u)
        if not nb:
            p[u] += ru
            r[u] = 0.0
            continue
        if ru < eps * len(nb):
            continue
        total_w = sum(w for _, w in nb)
        p[u] += alpha * ru
        spread = (1 - alpha) * ru
        r[u] = 0.0
        for v, w in nb:
            r[v] += spread * w / total_w
            if v not in queued and r[v] >= eps * max(1, len(nbrs(v))):
                queue.append(v)
                queued.add(v)
    return p
