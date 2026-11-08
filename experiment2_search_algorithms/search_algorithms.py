from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from math import inf
from typing import Iterable


Grid = list[list[str]]
Point = tuple[int, int]


class TreeNode:
    def __init__(self, value=None, children=None):
        self.value = value
        self.children = children if children is not None else []


def manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _valid_point(point: Point, grid: Grid) -> bool:
    row, col = point
    return 0 <= row < len(grid) and 0 <= col < len(grid[0])


def _open_cell(point: Point, grid: Grid) -> bool:
    return _valid_point(point, grid) and str(grid[point[0]][point[1]]).upper() != "X"


def _neighbors(point: Point, grid: Grid) -> Iterable[Point]:
    row, col = point
    for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):
        nxt = (row + dr, col + dc)
        if _open_cell(nxt, grid):
            yield nxt


def _rebuild_path(parent: dict[Point, Point], start: Point, end: Point) -> list[Point]:
    path = [end]
    while path[-1] != start:
        path.append(parent[path[-1]])
    return path[::-1]


def astar_path(
    start: Point,
    end: Point,
    grid: Grid,
    heuristic=None,
    stats: dict | None = None,
) -> list[Point]:
    if not grid or not grid[0] or not _open_cell(start, grid) or not _open_cell(end, grid):
        return []
    if start == end:
        return [start]

    def h(point: Point) -> float:
        return heuristic(point, end) if heuristic else manhattan(point, end)

    sequence = count()
    frontier = [(h(start), 0, next(sequence), start)]
    g_score = {start: 0}
    parent: dict[Point, Point] = {}
    expanded = 0
    generated = 1

    while frontier:
        _, current_g, _, current = heappop(frontier)
        if current_g != g_score.get(current):
            continue
        expanded += 1
        if current == end:
            if stats is not None:
                stats.update(expanded=expanded, generated=generated)
            return _rebuild_path(parent, start, end)

        for neighbor in _neighbors(current, grid):
            new_g = current_g + 1
            if new_g < g_score.get(neighbor, inf):
                g_score[neighbor] = new_g
                parent[neighbor] = current
                generated += 1
                heappush(frontier, (new_g + h(neighbor), new_g, next(sequence), neighbor))

    if stats is not None:
        stats.update(expanded=expanded, generated=generated)
    return []


def _terrain_weight(cell: str) -> float:
    text = str(cell)
    if text.upper() == "X":
        return inf
    if text.upper() in {"S", "E"}:
        return 0
    try:
        return float(text)
    except ValueError:
        return 1


def weighted_heuristic(point: Point, end: Point, grid: Grid) -> float:
    weights = []
    for row in grid:
        for cell in row:
            weight = _terrain_weight(cell)
            if 0 < weight < inf:
                weights.append(weight)
    step_cost = min(weights, default=1)
    distance = manhattan(point, end)
    return max(0, distance - 1) * step_cost


def _weighted_astar(start: Point, end: Point, grid: Grid, use_heuristic: bool):
    if not grid or not grid[0] or not _open_cell(start, grid) or not _open_cell(end, grid):
        return -1, []
    if start == end:
        return 0, [start]

    sequence = count()
    frontier = [(0, 0, next(sequence), start)]
    g_score = {start: 0}
    parent: dict[Point, Point] = {}

    while frontier:
        _, current_g, _, current = heappop(frontier)
        if current_g != g_score.get(current):
            continue
        if current == end:
            return int(current_g), _rebuild_path(parent, start, end)

        for neighbor in _neighbors(current, grid):
            step = _terrain_weight(grid[neighbor[0]][neighbor[1]])
            new_g = current_g + step
            if new_g < g_score.get(neighbor, inf):
                g_score[neighbor] = new_g
                parent[neighbor] = current
                estimate = weighted_heuristic(neighbor, end, grid) if use_heuristic else 0
                heappush(frontier, (new_g + estimate, new_g, next(sequence), neighbor))
    return -1, []


def astar_weighted_cost(start: Point, end: Point, grid: Grid) -> int:
    return _weighted_astar(start, end, grid, True)[0]


def weighted_astar_path(start: Point, end: Point, grid: Grid) -> tuple[int, list[Point]]:
    return _weighted_astar(start, end, grid, True)


def _descendant_count(node: TreeNode) -> int:
    return sum(1 + _descendant_count(child) for child in node.children)


def _alpha_beta(
    node: TreeNode,
    maximizing: bool,
    alpha: float,
    beta: float,
    stats: dict,
) -> float:
    stats["visited"] += 1
    if not node.children:
        return node.value

    value = -inf if maximizing else inf
    for index, child in enumerate(node.children):
        candidate = _alpha_beta(child, not maximizing, alpha, beta, stats)
        if maximizing:
            value = max(value, candidate)
            alpha = max(alpha, value)
        else:
            value = min(value, candidate)
            beta = min(beta, value)
        if alpha >= beta:
            stats["pruned"] += sum(
                _descendant_count(rest) + 1 for rest in node.children[index + 1 :]
            )
            break
    return value


def alpha_beta_search(
    root: TreeNode,
    alpha: float = -inf,
    beta: float = inf,
    stats: dict | None = None,
) -> float:
    local_stats = {"visited": 0, "pruned": 0}
    result = _alpha_beta(root, True, alpha, beta, local_stats)
    if stats is not None:
        stats.update(local_stats)
    return result


def _minimax(node: TreeNode, maximizing: bool, stats: dict) -> float:
    stats["visited"] += 1
    if not node.children:
        return node.value
    values = [_minimax(child, not maximizing, stats) for child in node.children]
    return max(values) if maximizing else min(values)


def minimax_search(root: TreeNode, stats: dict | None = None) -> float:
    local_stats = {"visited": 0, "pruned": 0}
    result = _minimax(root, True, local_stats)
    if stats is not None:
        stats.update(local_stats)
    return result


TREASURE_LABELS = ("A", "B", "T1", "T2", "T3", "T4")
POSITIONS = {
    "A": (0, 0),
    "B": (6, 6),
    "T1": (0, 6),
    "T2": (2, 2),
    "T3": (4, 4),
    "T4": (6, 0),
}
TREASURE_VALUES = {"T1": 9, "T2": 6, "T3": 7, "T4": 7}
TREASURE_MAP: Grid = [
    ["A", "1", "1", "X", "2", "1", "T1"],
    ["1", "X", "6", "X", "1", "X", "1"],
    ["1", "1", "T2", "1", "1", "1", "1"],
    ["X", "1", "X", "8", "X", "1", "X"],
    ["1", "1", "1", "1", "T3", "1", "1"],
    ["1", "X", "9", "X", "1", "X", "1"],
    ["T4", "1", "1", "X", "1", "1", "B"],
]


@dataclass(frozen=True)
class TreasureState:
    positions: tuple[Point, Point]
    remaining: tuple[str, ...]
    score_a: int = 0
    score_b: int = 0


def treasure_distance_matrix(
    grid: Grid = TREASURE_MAP,
    positions: dict[str, Point] = POSITIONS,
) -> tuple[tuple[str, ...], list[list[int]]]:
    matrix = []
    for source in TREASURE_LABELS:
        row = []
        for target in TREASURE_LABELS:
            cost = astar_weighted_cost(positions[source], positions[target], grid)
            row.append(cost)
        matrix.append(row)
    return TREASURE_LABELS, matrix


def _next_treasure_state(
    state: TreasureState,
    target: str,
    player: int,
    distances: dict[tuple[str, str], int],
) -> TreasureState:
    positions = list(state.positions)
    scores = [state.score_a, state.score_b]
    labels = ("A", "B") if player == 0 else ("B", "A")
    p_label = labels[0]
    t_label = labels[1]
    gain = TREASURE_VALUES[target] - distances[(p_label, target)]
    scores[player] += gain
    positions[player] = POSITIONS[target]
    return TreasureState(
        positions=(positions[0], positions[1]),
        remaining=tuple(item for item in state.remaining if item != target),
        score_a=scores[0],
        score_b=scores[1],
    )


def _ranked_targets(
    state: TreasureState,
    order: tuple[str, ...] | None,
) -> tuple[str, ...]:
    if order is None:
        return state.remaining
    return tuple(item for item in order if item in state.remaining)


def _treasure_search(
    state: TreasureState,
    depth_left: int,
    maximizing: bool,
    alpha: float,
    beta: float,
    distances: dict[tuple[str, str], int],
    order: tuple[str, ...],
    use_pruning: bool,
    stats: dict,
) -> float:
    stats["visited"] += 1
    if depth_left == 0 or not state.remaining:
        return state.score_a - state.score_b

    player = 0 if maximizing else 1
    targets = _ranked_targets(state, order)
    value = -inf if maximizing else inf

    for index, target in enumerate(targets):
        child = _next_treasure_state(state, target, player, distances)
        candidate = _treasure_search(
            child,
            depth_left - 1,
            not maximizing,
            alpha,
            beta,
            distances,
            order,
            use_pruning,
            stats,
        )
        if maximizing:
            value = max(value, candidate)
            alpha = max(alpha, value)
        else:
            value = min(value, candidate)
            beta = min(beta, value)
        if use_pruning and alpha >= beta:
            stats["pruned"] += len(targets) - index - 1
            break
    return value


def solve_treasure_game(
    depth: int,
    use_pruning: bool = True,
    order: tuple[str, ...] = ("T1", "T2", "T3", "T4"),
    grid: Grid = TREASURE_MAP,
    positions: dict[str, Point] = POSITIONS,
) -> dict:
    if depth < 1:
        raise ValueError("depth must be at least 1")
    _, matrix = treasure_distance_matrix(grid, positions)
    distances = {
        (source, target): matrix[i][j]
        for i, source in enumerate(TREASURE_LABELS)
        for j, target in enumerate(TREASURE_LABELS)
    }
    initial = TreasureState(
        positions=(positions["A"], positions["B"]),
        remaining=tuple(order),
    )
    stats = {"visited": 0, "pruned": 0}
    best_target = None
    best_value = -inf

    for target in order:
        child = _next_treasure_state(initial, target, 0, distances)
        candidate = _treasure_search(
            child,
            depth - 1,
            False,
            -inf,
            inf,
            distances,
            order,
            use_pruning,
            stats,
        )
        if candidate > best_value:
            best_value = candidate
            best_target = target

    return {
        "depth": depth,
        "best_target": best_target,
        "best_value": int(best_value),
        "visited": stats["visited"],
        "pruned": stats["pruned"],
    }
