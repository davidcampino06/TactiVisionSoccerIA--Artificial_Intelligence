"""Proximity network between team-mates: a graph explored with breadth-first search (BFS).

Every visible player of a team is a NODE. Two team-mates are joined by an EDGE when they stand
closer than ``LINK_DISTANCE`` (a fraction of the image width): close enough to support each other
with a short pass or a cover. On that graph, BFS answers the tactical questions:

    connected components  how many separate groups the team is split into (1 = all linked)
    isolated players      nodes without edges: nobody near to support them
    hops (BFS distance)   how many short passes are needed to go from one player to another
    diameter              the longest of those shortest routes inside a group

Complexity with n players and m links: building the graph O(n^2) (every pair is measured),
one BFS O(n + m), all components O(n + m), every pair of hops (n BFS runs) O(n * (n + m)).
With 11 players per team this is a few hundred operations per frame.

Positions are image coordinates (see tactical_analysis.py): the distances are image relative.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from math import hypot
from typing import Generic, TypeVar

N = TypeVar("N", bound=Hashable)

# Two team-mates closer than this (fraction of the image width) are linked.
LINK_DISTANCE = 0.20


class _FifoQueue(Generic[N]):
    """Queue for the BFS: an array plus the index of the front. Enqueue and dequeue are O(1)."""

    def __init__(self) -> None:
        self._items: list[N] = []
        self._front = 0

    def enqueue(self, item: N) -> None:
        self._items.append(item)

    def dequeue(self) -> N:
        item = self._items[self._front]
        self._front += 1
        return item

    def __bool__(self) -> bool:
        return self._front < len(self._items)


class Graph(Generic[N]):
    """Undirected graph stored as adjacency lists: node -> set of neighbours."""

    def __init__(self) -> None:
        self._adjacency: dict[N, set[N]] = {}

    def add_node(self, node: N) -> None:
        self._adjacency.setdefault(node, set())

    def add_edge(self, a: N, b: N) -> None:
        if a == b:
            return
        self.add_node(a)
        self.add_node(b)
        self._adjacency[a].add(b)
        self._adjacency[b].add(a)

    @property
    def nodes(self) -> list[N]:
        return list(self._adjacency)

    def neighbors(self, node: N) -> set[N]:
        return self._adjacency[node]

    def degree(self, node: N) -> int:
        return len(self._adjacency[node])

    @property
    def edge_count(self) -> int:
        return sum(len(neighbours) for neighbours in self._adjacency.values()) // 2

    def edges(self) -> list[tuple[N, N]]:
        seen: set[frozenset] = set()
        result = []
        for node, neighbours in self._adjacency.items():
            for other in neighbours:
                pair = frozenset((node, other))
                if pair not in seen:
                    seen.add(pair)
                    result.append((node, other))
        return result

    def bfs(self, start: N) -> dict[N, int]:
        """Hops from ``start`` to every node it can reach (start = 0). O(n + m)."""
        distance = {start: 0}
        queue: _FifoQueue[N] = _FifoQueue()
        queue.enqueue(start)
        while queue:
            node = queue.dequeue()
            for neighbour in self._adjacency[node]:
                if neighbour not in distance:  # first visit = shortest route
                    distance[neighbour] = distance[node] + 1
                    queue.enqueue(neighbour)
        return distance

    def shortest_path(self, start: N, goal: N) -> list[N]:
        """Fewest-hops route from start to goal (empty when they are not connected)."""
        parent: dict[N, N | None] = {start: None}
        queue: _FifoQueue[N] = _FifoQueue()
        queue.enqueue(start)
        while queue:
            node = queue.dequeue()
            if node == goal:
                path = [node]
                while parent[path[-1]] is not None:
                    path.append(parent[path[-1]])
                return path[::-1]
            for neighbour in self._adjacency[node]:
                if neighbour not in parent:
                    parent[neighbour] = node
                    queue.enqueue(neighbour)
        return []

    def connected_components(self) -> list[list[N]]:
        """Groups of nodes linked among themselves, biggest first. O(n + m)."""
        seen: set[N] = set()
        components = []
        for node in self._adjacency:
            if node not in seen:
                reached = self.bfs(node)
                seen.update(reached)
                components.append(list(reached))
        return sorted(components, key=len, reverse=True)

    def isolated(self) -> list[N]:
        return [node for node, neighbours in self._adjacency.items() if not neighbours]

    def all_hops(self) -> dict[N, dict[N, int]]:
        """BFS from every node: hops between every connected pair. O(n * (n + m))."""
        return {node: self.bfs(node) for node in self._adjacency}

    def diameter(self) -> int:
        """Longest shortest route inside any group (0 when there are no links)."""
        return max((max(hops.values()) for hops in self.all_hops().values()), default=0)


def proximity_graph(points: Iterable[tuple[N, float, float]], aspect_ratio: float,
                    link_distance: float = LINK_DISTANCE) -> Graph[N]:
    """Graph of one team: ``points`` are (track_id, x, y) in image ratios. Vertical distances
    are converted to width units with the aspect ratio (height / width), as in the indicators."""
    graph: Graph[N] = Graph()
    players = list(points)
    for node, _, _ in players:
        graph.add_node(node)
    for index, (a, ax, ay) in enumerate(players):
        for b, bx, by in players[index + 1:]:
            if hypot(ax - bx, (ay - by) * aspect_ratio) <= link_distance:
                graph.add_edge(a, b)
    return graph
