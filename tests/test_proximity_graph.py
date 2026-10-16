"""Proximity network: graph, breadth-first search and the indicators built on it."""

import pytest

from proximity_graph import Graph, proximity_graph
from tactical_analysis import AnalysisContext, FrameSnapshot, PlayerPosition, TacticalAnalyzer, network_snapshot
from tactical_insights import TacticalInsightGenerator
from team_assignment import TEAM_A, TEAM_B


def chain(*nodes):
    graph = Graph()
    for a, b in zip(nodes, nodes[1:]):
        graph.add_edge(a, b)
    return graph


def test_bfs_counts_hops_and_finds_shortest_paths():
    graph = chain(1, 2, 3, 4)
    graph.add_edge(1, 4)  # a shortcut closes the ring
    graph.add_node(9)     # isolated
    assert graph.bfs(1) == {1: 0, 2: 1, 4: 1, 3: 2}
    assert graph.shortest_path(2, 4) in ([2, 1, 4], [2, 3, 4])
    assert graph.shortest_path(1, 9) == []
    assert graph.isolated() == [9]
    assert graph.edge_count == 4 and graph.degree(1) == 2


def test_components_and_diameter():
    graph = chain(1, 2, 3, 4, 5)
    graph.add_edge(10, 11)
    graph.add_node(20)
    components = graph.connected_components()
    assert [sorted(group) for group in components] == [[1, 2, 3, 4, 5], [10, 11], [20]]
    assert graph.diameter() == 4
    assert Graph().diameter() == 0 and Graph().connected_components() == []


def test_proximity_graph_links_only_close_team_mates():
    points = [(1, 0.10, 0.50), (2, 0.25, 0.50), (3, 0.40, 0.50), (4, 0.90, 0.50)]
    graph = proximity_graph(points, aspect_ratio=1.0, link_distance=0.2)
    assert sorted(map(sorted, graph.edges())) == [[1, 2], [2, 3]]
    assert graph.isolated() == [4]
    # Vertical distances use the aspect ratio: 0.3 of the height is 0.15 of the width when height/width = 0.5.
    vertical = proximity_graph([(1, 0.5, 0.2), (2, 0.5, 0.5)], aspect_ratio=0.5, link_distance=0.2)
    assert vertical.edge_count == 1


def _frame(team_a, team_b, frame=0):
    players = [PlayerPosition(i + 1, TEAM_A, x, y, 0.9) for i, (x, y) in enumerate(team_a)]
    players += [PlayerPosition(101 + i, TEAM_B, x, y, 0.9) for i, (x, y) in enumerate(team_b)]
    return FrameSnapshot(frame, frame / 25, players, ball=None)


SPLIT = [(0.10, 0.2), (0.10, 0.4), (0.10, 0.6), (0.55, 0.3), (0.55, 0.5), (0.95, 0.9)]  # 3 groups, 1 isolated
COMPACT = [(0.60, 0.4), (0.65, 0.5), (0.70, 0.4), (0.70, 0.6), (0.62, 0.6)]


def test_indicators_and_issue_for_a_split_team():
    context = AnalysisContext([_frame(SPLIT, COMPACT, frame) for frame in range(10)], aspect_ratio=1.0)
    indicators = {indicator.full_name: indicator for indicator in TacticalAnalyzer(by_phase=False).analyze(context)}
    assert indicators["team_a.network_components"].value == 3
    assert indicators["team_a.isolated_players"].value == 1
    assert indicators["team_b.network_components"].value == 1
    assert indicators["team_a.network_diameter"].value == pytest.approx(2)
    issues, _ = TacticalInsightGenerator().generate(list(indicators.values()), team_separation=1.0)
    split = [issue for issue in issues if issue["code"] == "DISCONNECTED_TEAM"]
    assert [issue["team"] for issue in split] == [TEAM_A]
    assert "BFS" in split[0]["evidence"]


def test_network_snapshot_is_ready_to_draw():
    context = AnalysisContext([_frame(SPLIT[:2], COMPACT, 0), _frame(SPLIT, COMPACT, 1)], aspect_ratio=1.0)
    network = network_snapshot(context)
    assert network["frame_number"] == 1  # the frame with most players
    team_a = network["teams"][TEAM_A]
    assert team_a["groups"] == 3 and team_a["isolated"] == [6]
    assert {node["track_id"] for node in team_a["nodes"]} == {1, 2, 3, 4, 5, 6}
    assert team_a["hops"]["1"]["3"] == 2 and "4" not in team_a["hops"]["1"]
    assert network_snapshot(AnalysisContext([], aspect_ratio=1.0)) is None
