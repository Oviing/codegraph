"""Community detection over the structural graph: groups of symbols that belong together.

Weighted label propagation, stdlib only. Leiden/Louvain give slightly better modularity but need a
dependency (igraph/networkx); for "which cluster is this symbol in" label propagation is enough, runs
in near-linear time and is deterministic (fixed visiting order, ties broken by smallest label).
"""
from __future__ import annotations

from collections import defaultdict

CONF_WEIGHT = {"extracted": 1.0, "inferred": 0.6, "ambiguous": 0.25}
_FRESH = 1 << 40  # label space for nodes with no previous community
MIN_SIZE = 3  # smaller groups are folded into the neighbouring community they connect to most
MAX_SWEEPS = 12


def label_propagation(nodes: list[int], edges: list[tuple[int, int, float]],
                      hub_degree: int = 200, init: dict[int, int] | None = None) -> dict[int, int]:
    """Return {node: community id}. Nodes with more than `hub_degree` neighbours (utility hubs such as
    a logger) vote for nobody and are not voted for, so they cannot glue unrelated areas together."""
    adj: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for a, b, w in edges:
        if a != b:
            adj[a][b] += w
            adj[b][a] += w
    hubs = {n for n, nb in adj.items() if len(nb) > hub_degree}
    order = sorted(nodes, key=lambda n: (-len(adj.get(n, ())), n))  # dense areas settle first
    if init:  # warm start: keep last run's grouping, new nodes start alone (offset avoids clashes)
        label = {n: init[n] if init.get(n, -1) >= 0 else _FRESH + n for n in nodes}
    else:
        label = {n: n for n in nodes}
    for _ in range(MAX_SWEEPS):
        moved = 0
        for n in order:
            if n in hubs or n not in adj:
                continue
            votes: dict[int, float] = defaultdict(float)
            for m, w in adj[n].items():
                if m not in hubs:
                    votes[label[m]] += w
            if not votes:
                continue
            best = min(votes, key=lambda lb: (-votes[lb], lb))
            if best != label[n] and votes[best] > votes.get(label[n], 0.0):
                label[n] = best
                moved += 1
        if not moved:
            break
    # fold tiny groups into their strongest neighbour, then renumber 0..k-1 by size
    size: dict[int, int] = defaultdict(int)
    for lb in label.values():
        size[lb] += 1
    for n in order:
        if size[label[n]] < MIN_SIZE and n in adj:
            votes = defaultdict(float)
            for m, w in adj[n].items():
                if size[label[m]] >= MIN_SIZE:
                    votes[label[m]] += w
            if votes:
                size[label[n]] -= 1
                label[n] = min(votes, key=lambda lb: (-votes[lb], lb))
                size[label[n]] += 1
    ranked = sorted({lb for lb in label.values()}, key=lambda lb: (-size[lb], lb))
    renum = {lb: i for i, lb in enumerate(ranked)}
    return {n: renum[lb] for n, lb in label.items()}


def detect(db) -> dict:
    """Assign symbols.community from the edge table. Singletons (no edges) get -1."""
    ids = [r[0] for r in db.execute("SELECT id FROM symbols")]
    edges = [(s, d, CONF_WEIGHT.get(c, 0.6)) for s, d, c in db.execute("SELECT src, dst, conf FROM edges")]
    prev = {r[0]: r[1] for r in db.execute("SELECT id, community FROM symbols")}
    label = label_propagation(ids, edges, init=prev)
    linked = {x for a, b, _ in edges for x in (a, b)}
    db.executemany("UPDATE symbols SET community=? WHERE id=?",
                   [(label[i] if i in linked else -1, i) for i in ids])
    return {"communities": len({label[i] for i in linked}), "linked_symbols": len(linked)}
