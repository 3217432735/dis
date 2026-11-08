from __future__ import annotations

import json
from pathlib import Path

from search_algorithms import (
    TreeNode,
    alpha_beta_search,
    astar_path,
    astar_weighted_cost,
    minimax_search,
    solve_treasure_game,
    treasure_distance_matrix,
    weighted_astar_path,
)


MAZE = [
    ["S", ".", ".", "X", "."],
    [".", "X", ".", ".", "."],
    [".", "X", ".", "X", "."],
    [".", ".", ".", "X", "E"],
    ["X", "X", ".", ".", "."],
]

WEIGHTED_TERRAIN = [
    ["S", "1", "1", "8", "1"],
    ["1", "8", "3", "1", "1"],
    ["1", "8", "1", "8", "1"],
    ["1", "1", "1", "8", "1"],
    ["8", "8", "1", "1", "E"],
]

CUSTOM_MAZE = [
    ["S", ".", "X", ".", ".", ".", ".", ".", ".", "."],
    [".", ".", "X", ".", "X", "X", "X", "X", "X", "."],
    [".", ".", ".", ".", "X", ".", ".", ".", "X", "."],
    ["X", "X", "X", ".", "X", ".", "X", ".", "X", "."],
    [".", ".", ".", ".", ".", ".", "X", ".", "X", "."],
    [".", "X", "X", "X", "X", "X", "X", ".", "X", "."],
    [".", ".", ".", ".", ".", ".", ".", ".", "X", "."],
    [".", "X", "X", "X", "X", "X", "X", "X", "X", "."],
    [".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
    [".", ".", ".", ".", ".", ".", ".", ".", ".", "E"],
]


def build_sample_tree() -> TreeNode:
    leaf = lambda value: TreeNode(value=value)
    max1 = TreeNode(children=[leaf(3), leaf(5)])
    max2 = TreeNode(children=[leaf(6), leaf(9)])
    max3 = TreeNode(children=[leaf(1), leaf(2)])
    min1 = TreeNode(children=[max1, max2])
    min2 = TreeNode(children=[max2, max3])
    return TreeNode(children=[min1, min2])


def ordered_tree(order: str = "natural") -> TreeNode:
    root = build_sample_tree()
    first_min = root.children[0]
    second_min = root.children[1]
    if order == "poor":
        first_min.children.reverse()
    elif order == "good":
        second_min.children.reverse()
    return root


def main() -> None:
    maze_stats = {}
    maze_path = astar_path((0, 0), (3, 4), MAZE, stats=maze_stats)
    no_heuristic_stats = {}
    astar_path((0, 0), (3, 4), MAZE, heuristic=lambda a, b: 0, stats=no_heuristic_stats)

    custom_zero = {}
    astar_path((0, 0), (9, 9), CUSTOM_MAZE, heuristic=lambda a, b: 0, stats=custom_zero)
    custom_manhattan = {}
    astar_path((0, 0), (9, 9), CUSTOM_MAZE, stats=custom_manhattan)

    weighted_cost, weighted_path = weighted_astar_path((0, 0), (4, 4), WEIGHTED_TERRAIN)

    tree_stats = {}
    tree_value = alpha_beta_search(build_sample_tree(), stats=tree_stats)
    minimax_stats = {}
    minimax_value = minimax_search(build_sample_tree(), stats=minimax_stats)
    order_tests = {}
    for name in ("natural", "poor", "good"):
        order_stats = {}
        order_value = alpha_beta_search(ordered_tree(name), stats=order_stats)
        order_tests[name] = {"value": order_value, **order_stats}

    labels, distances = treasure_distance_matrix()
    treasure_results = {}
    for depth in (2, 4):
        treasure_results[str(depth)] = {
            "minimax": solve_treasure_game(depth, use_pruning=False),
            "alpha_beta": solve_treasure_game(depth, use_pruning=True),
        }

    results = {
        "maze": {
            "path": maze_path,
            "path_length": len(maze_path) - 1,
            "zero_heuristic": no_heuristic_stats,
            "manhattan": maze_stats,
        },
        "weighted": {
            "cost": weighted_cost,
            "path": weighted_path,
            "unreachable_cost": astar_weighted_cost(
                (0, 0), (1, 1), [["S", "X"], ["X", "E"]]
            ),
        },
        "alpha_beta": {
            "value": tree_value,
            "stats": tree_stats,
            "minimax_value": minimax_value,
            "minimax_stats": minimax_stats,
            "order_tests": order_tests,
        },
        "treasure": {
            "labels": labels,
            "distance_matrix": distances,
            "results": treasure_results,
        },
        "custom": {
            "zero_heuristic": custom_zero,
            "manhattan": custom_manhattan,
        },
    }

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("题目一：", maze_path)
    print("题目二：", weighted_cost)
    print("题目三：", tree_value)
    print("题目四：")
    for depth, values in treasure_results.items():
        print(
            f"  H={depth} Minimax={values['minimax']['best_target']},"
            f" {values['minimax']['best_value']}; "
            f"Alpha-beta={values['alpha_beta']['best_target']},"
            f" {values['alpha_beta']['best_value']}"
        )


if __name__ == "__main__":
    main()
