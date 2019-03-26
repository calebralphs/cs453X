import numpy as np

def problem1 (A, B):
    return A + B

def problem2 (A, B, C):
    return np.dot(A, B) - C

def problem3 (A, B, C):
    return A * B + C.T

def problem4 (x, y):
    return np.dot(x.T, y)

def problem5 (A):
    return np.zeros(A.shape)

def problem6 (A):
    return np.ones(len(A))

def problem7 (A, alpha):
    return A + alpha * np.eye(len(A))

def problem8 (A, i, j):
    return A[i, j]

def problem9 (A, i):
    return np.sum(A[i])

def problem10 (A, c, d):
    return np.mean(A[(a >= c) & (a <= d)])

def problem11 (A, k):
    eig_vals, eig_vecs = np.linalg.eig(A)
    eig_vals_idx = np.argsort(eig_vals)[::-1][:k]
    return eig_vecs[eig_vals_idx]

def problem12 (A, x):
    return np.linalg.solve(A, x)

def problem13 (A, x):
    return np.linalg.solve(A.T, x.T).T