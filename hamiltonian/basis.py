import numpy as np


class Basis():
    """return basis with all number states
    L: number of lattice sites
    N: number of particles
    |4000>, |3100>, ..., |0004> for L=N=4"""

    def __init__(self, L, N):
        self.N = N
        self.L = L


        def sums(length, total_sum):
            # source https://stackoverflow.com/questions/7748442/generate-all-possible-lists-of-length-n-that-sum-to-s-in-python
            if length == 1:
                yield [total_sum,]
            else:
                for value in range(total_sum + 1):
                    for permutation in sums(length - 1, total_sum - value):
                        yield [value,] + permutation

        self.basis_list = list(sums(L, N))
        self.basis_list = np.array(self.basis_list[::-1])
        self.length = self.basis_list.shape[0]


    def nstate_index(self, nstate_in):

        for i, nstate in enumerate(self.basis_list):
            if (nstate_in == nstate).all():
                return i

        raise ValueError(f'nstate: {nstate_in} not found in basis')
