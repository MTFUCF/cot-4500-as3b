import numpy as np

def solve_by_gaussian_elimination(A, b):
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    n = len(b)
    x = np.zeros(n)

    for k in range(n):
        max_index = np.argmax(abs(A[k:, k])) + k
        A[[k, max_index]] = A[[max_index, k]]
        b[[k, max_index]] = b[[max_index, k]]

        for i in range(k + 1, n):
            factor = A[i][k] / A[k][k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i][i]

    return x

def lu_decomposition(A):
    n = A.shape[0]
    L = np.eye(n)
    U = A.astype(float)
    for i in range(n):
        for j in range(i+1, n):
            factor = U[j, i] / U[i, i]
            L[j, i] = factor
            U[j] -= factor * U[i]
    return L, U

def matrix_determinant(U):
    return np.prod(np.diag(U))

def is_diagonally_dominant(A):
    A = np.abs(A)
    for i in range(A.shape[0]):
        if A[i, i] < np.sum(A[i]) - A[i, i]:
            return False
    return True

def is_positive_definite(A):
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False

def run_all():
    A1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    b1 = np.array([2, -1, 1])
    x1 = solve_by_gaussian_elimination(A1, b1)
    print(x1)
    print()

    A2 = np.array([[1, 1, 0, 3], [2, 1, -1, 1], [3, -1, -1, 2], [-1, 2, 3, -1]])
    L, U = lu_decomposition(A2)
    det = matrix_determinant(U)
    print(det)
    print()
    print(L)
    print()
    print(U)
    print()

    A3 = np.array([[9, 0, 5, 2, 1], [3, 9, 1, 2, 1], [0, 1, 7, 2, 3], [4, 2, 3, 12, 2], [3, 2, 4, 0, 8]])
    print(is_diagonally_dominant(A3))
    print()

    A4 = np.array([[2, 2, 1], [2, 3, 0], [1, 0, 2]])
    print(is_positive_definite(A4))

if __name__ == "__main__":
    run_all()