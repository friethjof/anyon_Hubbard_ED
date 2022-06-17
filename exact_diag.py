import os
import itertools
import math
import time
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg
import scipy.linalg


# author: Friethjof Theel
# date: 15.06.2022
# last modeified: 16.06.2022


#===============================================================================
class Basis():
    """return basis with all number states
    L: number of lattice sites
    N: number of particles
    |4000>, |3100>, ..., |0004> for L=N=4"""

    def __init__(self, L, N):
        self.N = N
        self.L = L


        def sums(length, total_sum):
            # source https://stackoverflow.com/posts/7748851/edit
            if length == 1:
                yield [total_sum,]
            else:
                for value in range(total_sum + 1):
                    for permutation in sums(length - 1, total_sum - value):
                        yield [value,] + permutation

        self.basis_list = list(sums(L, N))
        self.basis_list = self.basis_list[::-1]
        self.length = len(self.basis_list)


#===============================================================================
class AnyonHubbardHamiltonian():
    """
    H = -J sum_j^L-1  b_j^t exp(i*theta*n_j) b_j+1
                    + b_j+1^t exp(-i*theta*n_j) b_j
        + U/2 sum_j^L  n_j(n_j-1)

    bosonic operators
    b_j   | n > = sqrt(n)   | n-1 >
    b_j^t | n > = sqrt(n+1) | n+1 >
    """

    def __init__(self, J, U, theta, basis):
        # basis : instance of class Bais
        self.J = J
        self.U = U
        self.theta = theta
        self.basis = basis

        #-----------------------------------------------------------------------
        def hop(basis1, basis2):
            "sum_j^L-1 b_j^t exp(i*theta*n_j) b_j+1"

            sum_hop = 0
            for j in range(basis.L-1):
                b1 = basis1[::]
                b2 = basis2[::]
                # b_j+1
                if b2[j+1] == 0:
                    hop_jp1 = 0
                else:
                    hop_jp1 = math.sqrt(b2[j+1])
                    b2[j+1] -= 1
                # exp(i * theta * n_j)
                exp_j = np.exp(1j*theta*b2[j])

                # b_j^t
                if b2[j] == basis.N:
                    hop_j = 0
                else:
                    hop_j = math.sqrt(b2[j]+1)
                    b2[j] += 1
                if b1 == b2:
                    sum_hop = hop_j*exp_j*hop_jp1

            return sum_hop


        #-----------------------------------------------------------------------
        def hop_t(basis1, basis2):
            "sum_j^L-1 b_j+1^t exp(-i*theta*n_j) b_j"

            sum_hop = 0
            for j in range(basis.L-1):
                b1 = basis1[::]
                b2 = basis2[::]

                # b_j
                if b2[j] == 0:
                    hop_j = 0
                else:
                    hop_j = math.sqrt(b2[j])
                    b2[j] -= 1

                # exp(i * theta * n_j)
                exp_j = np.exp(-1j*theta*b2[j])

                # b_j+1^t
                if b2[j+1] == basis.N:
                    hop_jp1 = 0
                else:
                    hop_jp1 = math.sqrt(b2[j+1]+1)
                    b2[j+1] += 1

                if b1 == b2:
                    sum_hop = hop_j*exp_j*hop_jp1

            return sum_hop


        #-----------------------------------------------------------------------
        def onsite_int(basis1, basis2):
            "U/2 sum_j^L  n_j(n_j-1)"

            if basis1 != basis2:
                return 0

            sum_int = 0
            for j in range(basis.L):

                if basis1[j] == 0:
                    continue
                else:
                    sum_int += basis1[j]*(basis1[j] - 1)

            return sum_int

        # calculate all matrix elements: H_mn = <m| H | n>
        self.hamilt = np.zeros((basis.length, basis.length), dtype=complex)
        for i, b_m in enumerate(basis.basis_list):
            for j, b_n in enumerate(basis.basis_list):

                H_mn = (-J*hop(b_m, b_n)
                        - J*hop_t(b_m, b_n)
                        + U/2.0*onsite_int(b_m, b_n))
                self.hamilt[i, j] = H_mn

        # check whether hamiltonian is hermitian
        assert np.allclose(self.hamilt, np.conjugate(self.hamilt.T))

        eval, evec = linalg.eig(self.hamilt)
        self.eval = eval
        self.evec = evec.T
        idx = eval.argsort()
        self.eval_sort = eval[idx]
        self.evec_sort = (evec[:, idx]).T

        assert all([np.vdot(el, el) for el in evec])




#===============================================================================
class Propagation():
    """
    |psi(t)> = sum_n exp(-i*E_n*t) |n> <n|psi_0>
    n> : eigenvectors
    """

    def __init__(self, hamilt, psi0, tmax, tstep):
        # hamilt : object of hamilt class
        # psi0 : normalized array
        # tmax, tstep: floats, time specifications


        # check whether psi_0 is normalized
        assert np.vdot(psi_0, psi_0) == 1
        self.psi0 = psi0
        self.hamilt = hamilt
        self.time = np.arange(0, tmax, tstep)

        psi_t = []
        for t in self.time:
            psi_tstep = np.zeros(hamilt.evec[0].shape, dtype=complex)
            for eval_n, evec_n in zip(hamilt.eval, hamilt.evec):
                psi_tstep += np.exp(-1j*eval_n*t)*evec_n*np.vdot(evec_n, psi_0)
            psi_t.append(psi_tstep)
            assert np.abs(np.vdot(psi_tstep, psi_tstep) - 1+0j) < 1e-8
        self.psi_t = np.array(psi_t)


    #---------------------------------------------------------------------------
    # def _oneBodyDensity(self, subsys_ind):
    #     """
    #     subsys_ind : list of indices corresponding to lattice indices
    #     calculate one-body density of system rho_A
    #         rho_A|i,j = <psi(t)| b_i^t b_j |psi(t)>,
    #     i,j indices of subsystem.
    #     """
    #     N = self.hamilt.basis.N
    #     basis_list = self.hamilt.basis.basis_list
    #
    #     rhoA_t = []
    #     for psi_tstep in self.psi_t:  # loop over all times
    #         # get density operator of subsystem A
    #         rhoA = np.zeros((len(subsys_ind), len(subsys_ind)), dtype=complex)
    #         for i in subsys_ind:
    #             for j in subsys_ind:
    #                 # calculate rho_A|i,j = <psi(t)| b_i^t b_j |psi(t)>
    #                 #   = (<4,0,0,0|c_0 + <3,1,0,0|c_1 +...)
    #                 #     b_i^t b_j (d_0|4,0,0,0> + d_1|3,1,0,0> +...)
    #                 sum_rhoA = 0
    #                 for b_n, psi_coeff_n in zip(basis_list, psi_tstep):
    #                     # b_i^t b_j | b_n>
    #                     b1 = b_n[::]
    #                     # b_j
    #                     if b1[j] == 0:  # hop_j = 0
    #                         continue
    #                     else:
    #                         hop_j = math.sqrt(b1[j])
    #                         b1[j] -= 1
    #                     # b_i^t
    #                     if b1[i] == N:  # hop_jp1 = 0
    #                         continue
    #                     else:
    #                         hop_i = math.sqrt(b1[i]+1)
    #                         b1[i] += 1
    #                     for b_m, psi_coeff_m in zip(basis_list, psi_tstep):
    #                         if b_m == b1:
    #                             sum_rhoA += 1/N*(hop_i*hop_j
    #                                 *np.conjugate(psi_coeff_m)*psi_coeff_n)
    #                             break
    #                 rhoA[i, j] = sum_rhoA
    #             rhoA_t.append(rhoA)
    #     print('needs to by verfied')


    #---------------------------------------------------------------------------
    def numOp(self, site_i):
        "psi_t = <psi(t)| b_i^t b^i |psi(t)>"
        path_nop_dict = Path('dict_numOp.npz')
        basis_list = self.hamilt.basis.basis_list
        L = self.hamilt.basis.L
        if path_nop_dict.is_file():
            nop_dict = dict(np.load(path_nop_dict))
            if f'site_{site_i}' in nop_dict.keys():
                return nop_dict[f'site_{site_i}']
        else:
            nop_dict = {}

        nop = []
        for psi_tstep in self.psi_t:
            nop_t = 0
            for psi_coeff, b_m in zip(psi_tstep, basis_list):
                nop_t += np.abs(psi_coeff)**2*b_m[site_i]
            nop.append(nop_t)

        nop_dict[f'site_{site_i}'] = nop
        nop_dict[f'time'] = self.time
        np.savez(path_nop_dict, **nop_dict)
        return nop


    def numOp_cplot(self, fig_name='num_op_cplot.png'):
        plt.rc('text', usetex=True)
        fig, ax = plt.subplots()
        L = self.hamilt.basis.L
        x, y = np.meshgrid(range(1, L+1), self.time)
        numOp_mat = np.array([self.numOp(i) for i in range(L)])
        im = ax.pcolormesh(x, y, np.transpose(numOp_mat), cmap='turbo')
        [ax.axvline(i+0.5, c='black', lw=2) for i in range(1, L)]
        ax.tick_params(labelsize=12)
        ax.set_xticks(list(range(1, L+1)))
        ax.set_xlabel('site i', fontsize=14)
        ax.set_ylabel('time', fontsize=14)
        cbar = fig.colorbar(im)
        cbar.ax.tick_params(labelsize=12)
        cbar.ax.set_ylabel(r"$\langle \Psi|b_i^{\dagger}b_i| \Psi\rangle$",
            fontsize=14)
        # plt.show()
        plt.savefig(fig_name)
        plt.close()
        subprocess.call(['convert', fig_name, '-trim', fig_name])


    def numOp_lplot(self, fig_name='num_op.png'):
        plt.rc('text', usetex=True)
        fig, ax = plt.subplots()
        for site_i in range(L):
            plt.plot(self.time, class_prop.numOp(site_i),
                label=f'site {site_i+1}')
        plt.legend()
        ax.tick_params(labelsize=12)
        ax.set_xlabel('time', fontsize=14)
        ax.set_ylabel(r"$\langle \Psi|b_i^{\dagger}b_i| \Psi\rangle$",
            fontsize=14)
        # plt.show()
        plt.savefig(fig_name)
        plt.close()
        subprocess.call(['convert', fig_name, '-trim', fig_name])


    #---------------------------------------------------------------------------
    def rho_A(self, subsysA_ind):
        """Calculate the reduced density operator of subsystem A by tracing out
        the dofs of subsystem B.

        rho_A = tr_B(|psi(t)><psi(t)|)
              = sum_basisB <b_n| (|3,1,0,0><1,2,1,0| + ...) |b_n>
        """
        N = self.hamilt.basis.N
        L = self.hamilt.basis.L
        basis_list = self.hamilt.basis.basis_list
        subsysB_ind = [el for el in range(L) if el not in subsysA_ind]

        # basis_B = [[0, 0],...,[0, N], [1, 0],...,[1, N-1], ..., [N, 0]]
        basis_A = [np.array(list(el)) for el in itertools.product(
            range(N+1), repeat=len(subsysA_ind)) if sum(el) <= N]
        basis_B = [np.array(list(el)) for el in itertools.product(
            range(N+1), repeat=len(subsysB_ind)) if sum(el) <= N]

        rhoA_t = []
        for t, psi_tstep in zip(self.time, self.psi_t):  # loop over all times
            print(f'time: {t}')
            # rho_A|k,l = sum_ijkl <nB_i| ( |nA_k>|nB_k> <nA_l|<nB_l| ) |nB_j>
            #   = sum_kl |nA_k> <nA_l| * sum_ij <nB_i|nB_k> <nB_l|nB_j>

            rho_A = np.zeros((len(basis_A), len(basis_A)), dtype=complex)


            for bB_i in basis_B:
                # bB_i = <0, 0|,  |0, 0>
                for k, (basis_k, coeff_k) in enumerate(zip(basis_list,
                    psi_tstep)):
                    # basis_k = |2, 0, 1, 1>
                    # basisA_k = |2, 0> ; basisB_k = |1, 1>
                    basisB_k = np.array([basis_k[x] for x in subsysB_ind])
                    basisA_k = np.array([basis_k[x] for x in subsysA_ind])
                    for l, (basis_l, coeff_l) in enumerate(zip(basis_list,
                        psi_tstep)):
                        # basis_l = <3, 0, 0, 1|
                        # basisA_l = <3, 0| ; basisB_l = <0, 1|
                        basisB_l = np.array([basis_l[x] for x in subsysB_ind])
                        basisA_l = np.array([basis_l[x] for x in subsysA_ind])
                        # basisB_l, basisB_k match with bB_i,i exactly
                        # len(basis_A)-times
                        count = 0
                        if count < len(basis_A) and (bB_i==basisB_k).all()\
                            and (bB_i==basisB_l).all():
                            count += 1
                            # assign coefficient to the right index of rhoA
                            bool_break = False
                            for i, bA_i in enumerate(basis_A):
                                for j, bA_j in enumerate(basis_A):
                                    # there exists only one combination of
                                    # bA_i,j that matches basisA_l,k
                                    if (bA_i==basisA_k).all() and\
                                        (bA_j==basisA_l).all():
                                        rho_A[i, j] +=\
                                            coeff_k*np.conjugate(coeff_l)
                                        bool_break = True
                                        break
                                if bool_break:
                                    break

            rhoA_t.append(rho_A)
            # rhoA_2 = np.matmul(rho_A, rho_A)
            # print(np.abs(np.sum(np.abs(rho_A-rhoA_2))))
            # assert np.abs(np.sum(np.abs(rho_A-rhoA_2))) < 1e-8

        return np.array(rhoA_t)


    def _vN_entropy(self, subsys_ind):
        """
        subsys_ind : list of indices corresponding to lattice indices
        Calculate the von Neumann entropy of a given subsystem A.
            S = sum_i -n_i*log(n_i).
        with n_i corresponding to the eigenvalues of A
        """
        rhoA_t = self.rho_A(subsys_ind)
        s_ent = []
        for rhoA in rhoA_t:  # loop over all times
            # calculate von Neumann entropy
            eval, _ = linalg.eig(rhoA)
            # verify that all eigenvalues have no imaginary contribution
            assert all([el.imag < 1e-8 for el in eval])
            # entropy = -np.trace(np.matmul(rhoA, scipy.linalg.logm(rhoA)))
            entropy = sum([-el*np.log(el) for el in eval.real if 1e-8 < abs(el)])
            s_ent.append(entropy)
        return np.array(s_ent)


    def bipartite_ent(self):
        path_sentA_dict = Path('dict_SentA.npz')
        if path_sentA_dict.is_file():
            SentA_dict = dict(np.load(path_sentA_dict))
            return SentA_dict['SentA']

        N = self.hamilt.basis.N
        L = self.hamilt.basis.L
        lattice_ind = list(range(L//2))
        entropy = self._vN_entropy(lattice_ind)
        np.savez(path_sentA_dict, SentA=entropy, time=self.time)
        return entropy


    def plot_bipartite_ent(self, fig_name='S_ent_A.png'):
        plt.rc('text', usetex=True)
        fig, ax = plt.subplots()
        plt.plot(self.time, class_prop.bipartite_ent())
        ax.tick_params(labelsize=12)
        ax.set_xlabel('time', fontsize=14)
        ax.set_ylabel(r"$S_A(t)$", fontsize=14)
        # plt.show()
        plt.savefig(fig_name)
        plt.close()
        subprocess.call(['convert', fig_name, '-trim', fig_name])



#===============================================================================
# parameters
#===============================================================================
N = 4       # number of particles
L = 8       # number of sites
J = 1.0     # hopping term
U = 1.0     # on-site interaction
# theta = math.pi/2.0   # complex phase
theta = 0   # complex phase
clock_start = time.time()


#===============================================================================
# create and diagonalize full many-body Hamiltonian
#===============================================================================
class_basis = Basis(L, N)

class_hamilt = AnyonHubbardHamiltonian(J, U, theta, class_basis)



#===============================================================================
# propagation
#===============================================================================
# define initial state: |N/2, 0, ..., 0, N/2>
assert L%2==0  # even number of lattice sites
nstate_ini = [0]*L
nstate_ini[0] = N/2
nstate_ini[-1] = N/2
psi0_ind = [i for i, el in enumerate(class_basis.basis_list) if el == nstate_ini]
assert len(psi0_ind) == 1

psi_0 = np.zeros((class_basis.length), dtype=complex)
psi_0[psi0_ind[0]] = 1

# propagate
class_prop = Propagation(class_hamilt, psi_0, 10.1, 0.1)

# measure ellapsed time
clock_end = time.time()
np.savez('calculation_time.npz', time_ellapsed=clock_end-clock_start)


#===============================================================================
# plot observables
#===============================================================================
class_prop.numOp_cplot()
class_prop.numOp_lplot()
class_prop.plot_bipartite_ent()

