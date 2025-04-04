import unittest
import numpy as np
from src.main.assignment_3 import *

class TestAssignment3(unittest.TestCase):

    def test_gaussian_elimination(self):
        A = np.identity(3)
        b = np.array([2, -1, 1])
        expected = np.array([2.0, -1.0, 1.0])
        np.testing.assert_allclose(solve_by_gaussian_elimination(A, b), expected, rtol=1e-6)

    def test_lu_decomposition(self):
        A = np.array([[1, 1, 0, 3], [2, 1, -1, 1], [3, -1, -1, 2], [-1, 2, 3, -1]])
        L, U = lu_decomposition(A)
        det = matrix_determinant(U)
        self.assertAlmostEqual(det, 39.0, places=6)
        np.testing.assert_allclose(L @ U, A, rtol=1e-5)

    def test_diagonal_dominance(self):
        A = np.array([[9, 0, 5, 2, 1], [3, 9, 1, 2, 1], [0, 1, 7, 2, 3], [4, 2, 3, 12, 2], [3, 2, 4, 0, 8]])
        self.assertFalse(is_diagonally_dominant(A))

    def test_positive_definite(self):
        A = np.array([[2, 2, 1], [2, 3, 0], [1, 0, 2]])
        self.assertTrue(is_positive_definite(A))

if __name__ == '__main__':
    unittest.main()