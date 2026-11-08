import unittest

from experiment import MAZE, WEIGHTED_TERRAIN, build_sample_tree
from search_algorithms import (
    TreeNode,
    alpha_beta_search,
    astar_path,
    astar_weighted_cost,
    minimax_search,
    solve_treasure_game,
    treasure_distance_matrix,
)


class SearchAlgorithmTests(unittest.TestCase):
    def test_maze_shortest_path(self):
        expected = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4)]
        self.assertEqual(astar_path((0, 0), (3, 4), MAZE), expected)

    def test_maze_without_solution(self):
        grid = [["S", "X"], ["X", "E"]]
        self.assertEqual(astar_path((0, 0), (1, 1), grid), [])

    def test_weighted_terrain(self):
        self.assertEqual(astar_weighted_cost((0, 0), (4, 4), WEIGHTED_TERRAIN), 7)

    def test_weighted_without_solution(self):
        grid = [["S", "X"], ["X", "E"]]
        self.assertEqual(astar_weighted_cost((0, 0), (1, 1), grid), -1)

    def test_alpha_beta_sample(self):
        self.assertEqual(alpha_beta_search(build_sample_tree()), 5)
        self.assertEqual(minimax_search(build_sample_tree()), 5)

    def test_alpha_beta_custom_tree(self):
        root = TreeNode(
            children=[
                TreeNode(children=[TreeNode(value=2), TreeNode(value=8)]),
                TreeNode(children=[TreeNode(value=1), TreeNode(value=4)]),
            ]
        )
        self.assertEqual(alpha_beta_search(root), 2)

    def test_treasure_distance_matrix(self):
        labels, matrix = treasure_distance_matrix()
        self.assertEqual(labels, ("A", "B", "T1", "T2", "T3", "T4"))
        self.assertEqual(len(matrix), 6)
        self.assertTrue(all(len(row) == 6 for row in matrix))
        self.assertTrue(all(matrix[i][i] == 0 for i in range(6)))

    def test_treasure_expected_decisions(self):
        result2 = solve_treasure_game(2)
        result4 = solve_treasure_game(4)
        self.assertEqual((result2["best_target"], result2["best_value"]), ("T2", -1))
        self.assertEqual((result4["best_target"], result4["best_value"]), ("T3", 3))

    def test_minimax_matches_alpha_beta(self):
        for depth in (2, 4):
            self.assertEqual(
                solve_treasure_game(depth, False)["best_value"],
                solve_treasure_game(depth, True)["best_value"],
            )


if __name__ == "__main__":
    unittest.main()
