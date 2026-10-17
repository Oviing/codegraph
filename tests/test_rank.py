"""Ranking: PageRank sanity and the promise that a local query stays local."""
from codegraph.rank import pagerank, personalized_pagerank


def chain(n):
    """0 -> 1 -> ... -> n-1, a path graph."""
    return list(range(n)), [(i, i + 1) for i in range(n - 1)]


def in_star(n):
    """Everyone points at node 0: the shape of a function many callers depend on."""
    return list(range(n)), [(i, 0) for i in range(1, n)]


def test_empty_graph():
    assert pagerank([], []) == {}


def test_hub_outranks_a_leaf():
    nodes, edges = in_star(6)
    r = pagerank(nodes, edges)
    assert max(r, key=r.get) == 0
    assert r[0] > r[5]


def test_scores_sum_to_one():
    nodes, edges = in_star(8)
    assert abs(sum(pagerank(nodes, edges).values()) - 1.0) < 1e-9


def test_dangling_nodes_do_not_lose_mass():
    nodes, edges = chain(5)
    assert abs(sum(pagerank(nodes, edges).values()) - 1.0) < 1e-9


def test_convergence_is_reported():
    nodes, edges = in_star(8)
    stats = {}
    pagerank(nodes, edges, stats=stats)
    assert stats["converged"] is True
    assert stats["iterations"] <= 60
    assert stats["nodes"] == 8


def test_non_convergence_is_reported_rather_than_hidden():
    # a dense graph with a tight tolerance will not settle inside the iteration budget
    nodes, edges = chain(60)  # a long chain converges slowly
    stats = {}
    pagerank(nodes, edges, iters=2, tol=1e-14, stats=stats)
    assert stats["converged"] is False
    assert stats["iterations"] == 2


def test_edges_to_unknown_nodes_are_ignored():
    # a stale edge must not corrupt the result
    r = pagerank([1, 2], [(1, 2), (1, 999)])
    assert set(r) == {1, 2}
    assert abs(sum(r.values()) - 1.0) < 1e-9


def test_ppr_mass_falls_off_with_distance():
    nbrs = {i: [(i + 1, 1.0)] for i in range(10)}
    p = personalized_pagerank(lambda u: nbrs.get(u, []), {0: 1.0})
    reached = [p[i] for i in range(6) if i in p]
    assert reached == sorted(reached, reverse=True)


def test_ppr_ranks_a_hub_seed_first():
    nbrs = {i: [(0, 1.0)] for i in range(1, 12)}
    nbrs[0] = []
    p = personalized_pagerank(lambda u: nbrs.get(u, []), {0: 1.0})
    assert max(p, key=p.get) == 0


def test_ppr_spreads_along_edges():
    nbrs = {0: [(1, 1.0), (2, 1.0)], 1: [], 2: []}
    p = personalized_pagerank(lambda u: nbrs.get(u, []), {0: 1.0})
    assert p[1] > 0 and p[2] > 0


def test_ppr_stops_at_the_node_cap():
    """A hub seed must not drag the whole repository into the result."""
    n = 5000
    nbrs = {i: ([(i + 1, 1.0)] if i + 1 < n else []) for i in range(n)}
    nbrs[0] = [(i, 1.0) for i in range(1, 200)]
    p = personalized_pagerank(lambda u: nbrs.get(u, []), {0: 1.0}, max_nodes=50)
    assert len(p) <= 50


def test_ppr_keeps_mass_when_pruned():
    """Whatever is left over when the cap is hit must be kept, not silently discarded."""
    n = 400
    nbrs = {i: [(j, 1.0) for j in range(n) if j != i] for i in range(n)}
    p = personalized_pagerank(lambda u: nbrs.get(u, []), {0: 1.0}, max_nodes=20)
    assert sum(p.values()) > 0.5  # most of the restart mass is accounted for


def test_ppr_without_seeds():
    assert personalized_pagerank(lambda u: [], {}) == {}


def test_ppr_with_a_single_seed_keeps_all_mass():
    nbrs = {i: [(j, 1.0) for j in range(i + 1, 6)] for i in range(6)}
    p = personalized_pagerank(lambda u: nbrs.get(u, []), {0: 1.0})
    assert abs(sum(p.values()) - 1.0) < 1e-9
