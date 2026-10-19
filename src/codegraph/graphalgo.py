"""Graph-theory algorithms behind the UI lenses. Pure stdlib, deterministic, no I/O.

Nodes are any sortable hashables (symbol ids). Graphs are plain dicts so callers can build them from SQL
rows without an adapter: `edges` is a list of (a, b, weight) for the undirected algorithms and `out`
is {node: iterable of successors} for the directed ones.
"""
from __future__ import annotations

import heapq
from collections import defaultdict, deque

EPS = 1e-12


def _renumber(label: dict, by_size: bool = True) -> dict:
    """Relabel communities 0..k-1, biggest first, so ids are stable and the largest is always 0."""
    size: dict = defaultdict(int)
    for lb in label.values():
        size[lb] += 1
    order = sorted(size, key=lambda lb: (-size[lb], lb) if by_size else (lb,))
    ix = {lb: i for i, lb in enumerate(order)}
    return {n: ix[lb] for n, lb in label.items()}


def louvain(nodes, edges, resolution: float = 1.0, max_levels: int = 10) -> list[dict]:
    """Louvain modularity optimisation (Blondel et al. 2008) with a resolution parameter.

    Returns one partition per level, finest first: [{node: community}, ...]. Every level maps the
    *original* nodes, so levels[-1] is the coarsest. Nodes without edges are left out. Resolution > 1
    favours smaller communities, < 1 larger ones. Visiting order is sorted, so the result is reproducible.
    """
    adj: dict = defaultdict(lambda: defaultdict(float))
    loops: dict = defaultdict(float)
    for a, b, w in edges:
        if a == b:
            loops[a] += w
        else:
            adj[a][b] += w
            adj[b][a] += w
    cur = sorted(n for n in set(nodes) if adj.get(n) or loops.get(n))
    member = {n: n for n in cur}  # original node -> node of the current (aggregated) graph
    levels: list[dict] = []
    for _ in range(max_levels):
        deg = {i: sum(adj[i].values()) + 2 * loops.get(i, 0.0) for i in cur}
        m2 = sum(deg.values())
        if m2 <= 0:
            break
        comm = {i: i for i in cur}
        tot = dict(deg)
        improved = False
        for _sweep in range(50):
            moved = 0
            for i in cur:
                ci, ki = comm[i], deg[i]
                wc: dict = defaultdict(float)
                for j, w in adj[i].items():
                    wc[comm[j]] += w
                tot[ci] -= ki
                best = ci
                best_gain = wc.get(ci, 0.0) - resolution * tot[ci] * ki / m2
                for c in sorted(wc):
                    g = wc[c] - resolution * tot[c] * ki / m2
                    if g > best_gain + EPS:
                        best, best_gain = c, g
                tot[best] += ki
                if best != ci:
                    comm[i] = best
                    moved += 1
                    improved = True
            if not moved:
                break
        if not improved:
            break
        comm = _renumber(comm)
        member = {o: comm[member[o]] for o in member}
        levels.append(dict(member))
        k = len(set(comm.values()))
        if k == len(cur):
            break
        nadj: dict = defaultdict(lambda: defaultdict(float))
        nloops: dict = defaultdict(float)
        for i in cur:
            ci = comm[i]
            nloops[ci] += loops.get(i, 0.0)
            for j, w in adj[i].items():
                cj = comm[j]
                if ci == cj:
                    nloops[ci] += w / 2  # an internal edge is seen from both ends
                else:
                    nadj[ci][cj] += w
        adj, loops, cur = nadj, nloops, list(range(k))
    return levels


def modularity(part: dict, edges, resolution: float = 1.0) -> float:
    """Newman modularity of a partition; used to compare runs and in tests."""
    deg: dict = defaultdict(float)
    inside = 0.0
    total = 0.0
    for a, b, w in edges:
        total += w
        deg[a] += w
        deg[b] += w
        if part.get(a) is not None and part.get(a) == part.get(b):
            inside += w
    if total <= 0:
        return 0.0
    tot: dict = defaultdict(float)
    for n, d in deg.items():
        if n in part:
            tot[part[n]] += d
    return inside / total - resolution * sum(t * t for t in tot.values()) / (4 * total * total)


def fold_small(part: dict, edges, min_size: int = 3) -> dict:
    """Merge communities smaller than `min_size` into the neighbour they are most connected to."""
    adj: dict = defaultdict(lambda: defaultdict(float))
    for a, b, w in edges:
        if a != b:
            adj[a][b] += w
            adj[b][a] += w
    part = dict(part)
    size: dict = defaultdict(int)
    for c in part.values():
        size[c] += 1
    for n in sorted(part):
        if size[part[n]] >= min_size:
            continue
        votes: dict = defaultdict(float)
        for m, w in adj.get(n, {}).items():
            if m in part and size[part[m]] >= min_size:
                votes[part[m]] += w
        if votes:
            size[part[n]] -= 1
            part[n] = min(votes, key=lambda c: (-votes[c], c))
            size[part[n]] += 1
    return _renumber(part)


def sccs(nodes, out) -> list[list]:
    """Strongly connected components (iterative Tarjan). `out` maps node -> iterable of successors."""
    index: dict = {}
    low: dict = {}
    on: set = set()
    stack: list = []
    comps: list = []
    n = 0
    for root in sorted(nodes):
        if root in index:
            continue
        index[root] = low[root] = n
        n += 1
        stack.append(root)
        on.add(root)
        work = [(root, iter(sorted(out.get(root, ()))))]
        while work:
            v, it = work[-1]
            for w in it:
                if w not in index:
                    index[w] = low[w] = n
                    n += 1
                    stack.append(w)
                    on.add(w)
                    work.append((w, iter(sorted(out.get(w, ())))))
                    break
                if w in on:
                    low[v] = min(low[v], index[w])
            else:
                work.pop()
                if work:
                    u = work[-1][0]
                    low[u] = min(low[u], low[v])
                if low[v] == index[v]:
                    comp = []
                    while True:
                        w = stack.pop()
                        on.discard(w)
                        comp.append(w)
                        if w == v:
                            break
                    comps.append(comp)
    return comps


def betweenness(adj, pivots: int | None = None) -> dict:
    """Betweenness centrality (Brandes) on an unweighted undirected graph {node: iterable of neighbours}.

    With `pivots` set, only that many evenly spaced sources are used and the sum is scaled up: the
    standard sampling estimate, which keeps big graphs affordable while ranking hubs the same way.
    """
    nodes = sorted(adj)
    nbr = {v: sorted(set(adj[v]) - {v}) for v in nodes}
    sources = nodes
    if pivots and pivots < len(nodes):
        step = len(nodes) / pivots
        sources = [nodes[int(i * step)] for i in range(pivots)]
    bc = dict.fromkeys(nodes, 0.0)
    for s in sources:
        order = []
        preds: dict = defaultdict(list)
        sigma = dict.fromkeys(nodes, 0)
        dist = dict.fromkeys(nodes, -1)
        sigma[s], dist[s] = 1, 0
        q = deque([s])
        while q:
            v = q.popleft()
            order.append(v)
            for w in nbr[v]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    q.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    preds[w].append(v)
        delta = dict.fromkeys(nodes, 0.0)
        for w in reversed(order):
            for v in preds[w]:
                delta[v] += sigma[v] / sigma[w] * (1 + delta[w])
            if w != s:
                bc[w] += delta[w]
    scale = len(nodes) / len(sources) if sources else 1.0
    return {v: b * scale / 2 for v, b in bc.items()}  # undirected: each pair was counted twice


def kcore(adj) -> dict:
    """Core number of every node (Batagelj-Zaversnik, heap variant): the largest k such that the node
    survives repeatedly peeling off nodes with fewer than k neighbours."""
    nbr = {v: set(ns) - {v} for v, ns in adj.items()}
    deg = {v: len(ns) for v, ns in nbr.items()}
    heap = [(d, v) for v, d in deg.items()]
    heapq.heapify(heap)
    core: dict = {}
    k = 0
    while heap:
        d, v = heapq.heappop(heap)
        if v in core or d != deg[v]:
            continue
        k = max(k, d)
        core[v] = k
        for u in nbr[v]:
            if u not in core:
                deg[u] -= 1
                heapq.heappush(heap, (deg[u], u))
    return core


def shortest_path(out, inn, a, b, directed_first: bool = True) -> list:
    """Fewest-hop path a -> b. Follows edge direction first (a calls ... calls b) and only if that fails
    treats the graph as undirected. Returns [] when b is unreachable."""
    def bfs(step):
        prev = {a: None}
        q = deque([a])
        while q:
            v = q.popleft()
            if v == b:
                path = []
                while v is not None:
                    path.append(v)
                    v = prev[v]
                return path[::-1]
            for w in step(v):
                if w not in prev:
                    prev[w] = v
                    q.append(w)
        return []
    if a == b:
        return [a]
    if directed_first:
        p = bfs(lambda v: sorted(out.get(v, ())))
        if p:
            return p
    return bfs(lambda v: sorted(set(out.get(v, ())) | set(inn.get(v, ()))))
