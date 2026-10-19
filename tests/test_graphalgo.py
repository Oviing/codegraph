import itertools

from codegraph import graphalgo as ga


def clique(ids, w=1.0):
    return [(a, b, w) for a, b in itertools.combinations(ids, 2)]


def two_cliques(bridge=1.0):
    return list(range(10)), clique(range(5)) + clique(range(5, 10)) + [(4, 5, bridge)]


def test_louvain_finds_planted_communities():
    nodes, edges = two_cliques()
    part = ga.louvain(nodes, edges)[-1]
    assert len({part[i] for i in range(5)}) == 1
    assert len({part[i] for i in range(5, 10)}) == 1
    assert part[0] != part[5]
    assert ga.modularity(part, edges) > 0.4


def test_louvain_is_deterministic_and_skips_isolated_nodes():
    nodes, edges = two_cliques()
    assert ga.louvain(nodes + [99], edges) == ga.louvain(nodes + [99], edges)
    assert 99 not in ga.louvain(nodes + [99], edges)[-1]
    assert ga.louvain([], []) == []


def test_louvain_resolution_controls_granularity():
    # a ring of 6 triangles joined by single edges: low resolution merges, high resolution splits
    edges, nodes = [], list(range(18))
    for t in range(6):
        edges += clique([3 * t, 3 * t + 1, 3 * t + 2])
        edges.append((3 * t + 2, (3 * t + 3) % 18, 1.0))
    coarse = len(set(ga.louvain(nodes, edges, resolution=0.3)[-1].values()))
    fine = len(set(ga.louvain(nodes, edges, resolution=3.0)[-1].values()))
    assert coarse < fine


def test_fold_small_merges_tiny_groups_into_neighbour():
    nodes, edges = two_cliques()
    edges.append((0, 100, 1.0))  # a pendant node forms (or joins) a tiny group
    part = ga.fold_small({**ga.louvain(nodes + [100], edges)[-1], 100: 7}, edges, min_size=2)
    assert part[100] == part[0]


def test_sccs_find_cycles_only():
    out = {1: [2], 2: [3], 3: [1, 4], 4: [5], 5: []}
    comps = [sorted(c) for c in ga.sccs(out.keys(), out)]
    assert [1, 2, 3] in comps and [4] in comps and [5] in comps
    assert len(comps) == 3


def test_sccs_deep_chain_does_not_recurse():
    out = {i: [i + 1] for i in range(5000)}
    out[5000] = []
    assert len(ga.sccs(out.keys(), out)) == 5001


def test_betweenness_path_graph_peaks_in_the_middle():
    adj = {i: [j for j in (i - 1, i + 1) if 0 <= j <= 4] for i in range(5)}
    bc = ga.betweenness(adj)
    assert bc[2] == max(bc.values()) and bc[0] == bc[4] == 0
    assert bc[2] == 4  # pairs routed through the centre: (0,3)(0,4)(1,3)(1,4)
    sampled = ga.betweenness(adj, pivots=5)
    assert sampled == bc


def test_kcore():
    adj = {0: [1, 2, 3], 1: [0, 2, 3], 2: [0, 1, 3], 3: [0, 1, 2, 4], 4: [3]}
    core = ga.kcore(adj)
    assert core[4] == 1
    assert core[0] == core[1] == core[2] == core[3] == 3


def test_shortest_path_prefers_direction_then_falls_back():
    out = {1: [2], 2: [3], 3: [], 4: [3]}
    inn = {2: [1], 3: [2, 4]}
    assert ga.shortest_path(out, inn, 1, 3) == [1, 2, 3]
    assert ga.shortest_path(out, inn, 1, 4) == [1, 2, 3, 4]  # only reachable ignoring direction
    assert ga.shortest_path(out, inn, 1, 99) == []
    assert ga.shortest_path(out, inn, 1, 1) == [1]
