import pykokkos as pk
import numpy as np


@pk.workunit
def kernel_A(i: int, X: pk.View1D[float], B: pk.View1D[float]):
    B[i] = X[i] * 2.0

@pk.workunit
def kernel_C(i: int, X: pk.View1D[float], D: pk.View1D[float]):
    D[i] = X[i] + 10.0


def main():
    N = 10
    a = np.random.rand(N)
    b = np.random.rand(N)
    d = np.random.rand(N)
    print(a)
    print(b)

    pk.parallel_for(N, kernel_A, X=a, B=b)
    pk.parallel_for(N, kernel_C, X=a, D=d)

    pk.flush()

    print(a)
    print(b)


main()
