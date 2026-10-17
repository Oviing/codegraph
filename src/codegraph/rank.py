"""Graph ranking: global PageRank (stored at index time) and local personalized PageRank (per query)."""
from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable, Iterable


def pagerank(nodes: Iterable[int], edges: Iterable[tuple[int, int]],
             damping: float = 0.85, iters: int = 60, tol: float = 1e-7,
             stats: dict | None = None, init: dict[int, float] | None = None) -> dict[int, float]:
    """Classic directed PageRank. Importance flows src -> dst, so widely depended-upon code ranks high.

    `tol` is the L1 change per iteration. Power iteration with dangling nodes sheds mass at only
    ~`damping` per step, so a tolerance far below that needs hundreds of passes for little gain;
    1e-7 is the point where the ranking stops visibly changing. Pass a dict as `stats` to find out
    whether the run converged inside the budget rather than assuming it.
    """
    nodes = list(nodes)
    n = len(nodes)
    if n == 0:
        if stats is not None:
            stats.update(iterations=0, converged=True, delta=0.0, nodes=0)
        return {}
    pos = {v: i for i, v in enumerate(nodes)}
    out: list[list[int]] = [[] for _ in range(n)]
    for s, d in edges:  # ignore edges pointing at symbols that are no longer in the graph
        si, di = pos.get(s), pos.get(d)
        if si is not None and di is not None:
            out[si].append(di)
    # dense lists beat dicts here: this loop runs iters x edges times and is the one full pass
    sources = [(i, o, damping / len(o)) for i, o in enumerate(out) if o]
    dangling = [i for i, o in enumerate(out) if not o]
    rank = [1.0 / n] * n
    if init:
        # warm start from the previous run: an incremental reindex then converges in a few passes.
        # Nodes without a prior score (new symbols) start at the uniform value.
        rank = [init.get(v) or 1.0 / n for v in nodes]
        total = sum(rank)
        rank = [r / total for r in rank]
    delta, it = 0.0, 0
    for _step in range(1, iters + 1):
        it = _step
        base = (1 - damping) / n + damping * sum(rank[i] for i in dangling) / n
        new = [base] * n
        for i, ds, w in sources:
            c = rank[i] * w
            for d in ds:
                new[d] += c
        delta = sum(map(abs, map(float.__sub__, new, rank)))
        rank = new
        if delta < tol:
            break
    if stats is not None:
        stats.update(iterations=it, converged=delta < tol, delta=delta, nodes=n)
    return dict(zip(nodes, rank, strict=True))


def personalized_pagerank(nbrs: Callable[[int], list[tuple[int, float]]], seeds: dict[int, float],
                          alpha: float = 0.15, eps: float = 1e-5,
                          max_nodes: int = 20_000) -> dict[int, float]:
    """Local push approximation of random-walk-with-restart from `seeds`.

    Only nodes near the seeds are visited, so cost does not grow with repo size -- except for hub
    seeds, which is why the walk is hard-capped. `nbrs(u)` returns [(neighbour, weight)]; weights
    let callers down-weight hub nodes.

    Mass handling: a node whose residual falls below the threshold keeps it and stops. When the
    `max_nodes` cap is reached the walk stops expanding and the leftover is banked onto the nodes
    it belonged to, so scores stay comparable; any residue from nodes past the cap is added to the
    highest-scoring node rather than dropped, so total mass is preserved.
    """
    p: dict[int, float] = defaultdict(float)
    r: dict[int, float] = defaultdict(float)
    for s, m in seeds.items():
        r[s] += m
    queue = deque(seeds)
    queued = set(seeds)
    seen = set(seeds)
    overflow = 0.0

    def reachable(u: int) -> bool:
        """True while the result may still grow by one node."""
        return len(seen) < max_nodes

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
            p[u] += ru  # below threshold: stop here, but keep the mass
            r[u] = 0.0
            continue
        total_w = sum(w for _, w in nb)
        p[u] += alpha * ru
        spread = (1 - alpha) * ru
        r[u] = 0.0
        for v, w in nb:
            r[v] += spread * w / total_w
            if v in queued or r[v] < eps * max(1, len(nbrs(v))):
                continue
            if not reachable(v):
                overflow += r[v]
                r[v] = 0.0
                continue
            seen.add(v)
            queue.append(v)
            queued.add(v)
    for v, rv in r.items():  # anything still pending when the queue drains
        if rv:
            if len(seen) >= max_nodes:
                overflow += rv
            else:
                p[v] += rv
    if overflow and p:
        p[max(p, key=p.get)] += overflow
    return p
