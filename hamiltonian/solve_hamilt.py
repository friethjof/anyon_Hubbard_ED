import os
import itertools
import shutil
import math
import time
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import scipy

from hamiltonian.basis import Basis
from helper import operators


# author: Friethjof Theel
# date: 15.06.2022
# last modified: Feb 2023


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

    def __init__(self, path_basis, L, N, J, U, theta):
        # basis : instance of class Bais

        self.path_basis = path_basis
        self.L = L
        self.N = N
        self.J = J
        self.U = U
        self.theta = theta
        self.basis = Basis(L, N)


        path_hamil_npz = path_basis/'hamilt_spectrum.npz'
        if path_hamil_npz.is_file():
            self.hamilt = np.load(path_hamil_npz)['hamilt_mat']
            self.evals = np.load(path_hamil_npz)['evals']
            self.evecs = np.load(path_hamil_npz)['evecs']

        else:

            # measure ellapsed time
            clock_start = time.time()
            #-------------------------------------------------------------------
            # calculate all matrix elements: H_mn = <m| H | n>
            self.hamilt = np.zeros(
                (self.basis.length, self.basis.length),
                dtype=complex
                )
            for i, b_m in enumerate(self.basis.basis_list):
                for j, b_n in enumerate(self.basis.basis_list):
                    H_mn = (- J*operators.sum_hop(L, N, theta, b_m, b_n)
                            - J*operators.sum_hop_t(L, N, theta, b_m, b_n)
                            + U/2.0*operators.sum_onsite_int(L, b_m, b_n))

                    self.hamilt[i, j] = H_mn

            # check whether hamiltonian is hermitian
            assert np.allclose(self.hamilt, np.conjugate(self.hamilt.T))

            # Do not use np.linalg.eig here!
            # --> it might be that eigenvectors of degenerate eigenvalues
            # are not orthogonal!
            eval, evec = np.linalg.eigh(self.hamilt)
            idx = eval.argsort()
            self.evals = eval[idx]
            self.evecs = (evec[:, idx]).T

            assert all([(np.vdot(el, el) - 1) < 1e-8 for el in self.evecs])
            assert np.max(np.abs(self.evals.imag)) < 1e-8
            self.evals = self.evals.real

            # check orthonormality
            ortho_bool = True
            for i in range(self.basis.length):
                for j in range(self.basis.length):
                    sp = np.vdot(self.evecs[i], self.evecs[j])
                    if i == j and np.abs(sp - 1) > 1e-10:
                        ortho_bool = False

                    if i != j and np.abs(sp) > 1e-10:
                        ortho_bool = False
            assert ortho_bool

            clock_end = time.time()
            time_ellapsed = clock_end-clock_start
            print('time needed for diagonalization:', time_ellapsed)

            np.savez(path_hamil_npz,
                hamilt_mat=self.hamilt,
                evals=self.evals,
                evecs=self.evecs,
                basis_list=self.basis.basis_list,
                basis_length=self.basis.length,
                L=self.L,
                N=self.N,
                J=self.J,
                U=self.U,
                theta=self.theta
                )


    #===========================================================================
    def ground_state(self):
        return self.evecs[0]


    #===========================================================================
    def energy_spectrum(self):
        return self.evals


    #===========================================================================
    def numOp(self, psi, site_i):
        "psi_t = <psi(t)| b_i^t b^i |psi(t)>"
        basis_list = self.basis.basis_list
        nop = 0
        for psi_coeff, b_m in zip(psi, basis_list):
            nop += np.abs(psi_coeff)**2*b_m[site_i]
        return nop


    # def plot_gs_numOp(self):
    #     """Plot numOperator of the ground state """
    #     fig, ax = plt.subplots()
    #     L = self.L
    #     latt = range(1, L+1)
    #     numOp = np.array([self.numOp(self.ground_state(), i) for i in range(L)])
    #     ax.plot(latt, numOp)
    #     [ax.axvline(i+0.5, c='black', lw=2) for i in range(1, L)]
    #     ax.tick_params(labelsize=12)
    #     ax.set_xticks(list(range(1, L+1)))
    #     ax.set_xlabel('site i', fontsize=14)
    #     ax.set_ylabel(r"$\langle \Psi|b_i^{\dagger}b_i| \Psi\rangle$",
    #         fontsize=14)
    #     plt.show()
    #     # path_fig = (self.path_run/fig_name).resolve()
    #     # plt.savefig(path_fig)
    #     # plt.close()
    #     # if self.bool_convert_trim:
    #     #     subprocess.call(['convert', path_fig, '-trim', path_fig])
    #

    #===========================================================================
    def get_bibj_correlator(self, psi, i, j):
        r"""\langle b_i^\dagger b_j \rangle
            = <psi | b_i^\dagger b_j | psi >
            = (<4,0,0,0|c_1 + <3,1,0,0|c_2 + ... ) | b_i^\dagger b_j | psi >
            """
        bibj_op = 0
        for ind_1, basis1 in enumerate(self.basis.basis_list):
            for ind_2, basis2 in enumerate(self.basis.basis_list):

                # apply bi^dagger b_j
                coeff_bibj = operators.bi_dagger_bj(basis1, basis2, i, j,
                                                    self.N)

                if coeff_bibj == 0:
                    continue
                else:
                    bibj_op += coeff_bibj*np.conjugate(psi[ind_1])*psi[ind_2]

        return bibj_op


    def get_momentum(self, psi, k_mom, L):
        momentum = 0
        for m in range(1, L+1):
            for n in range(1, L+1):
                corr_mn = self.get_bibj_correlator(psi, m-1, n-1)
                momentum += np.exp(1j*k_mom*(m - n))*corr_mn
        momentum = np.array(momentum)/L

        if 1e-8 < np.max(momentum.imag):
            print(self.path_run, k_mom)
            print('momentum has an imaginary part > 1e-8! --raise')
            raise
        else:
            momentum = momentum.real

        return momentum


    def get_momentum_distribution(self, psi):
        L = self.L
        k_range = np.linspace(-np.pi*1, np.pi*1, 100)
        mom_mat = np.array([self.get_momentum(psi, k, L).real for k in k_range])
        return k_range, mom_mat


    # def plot_gs_momOp(self):
    #     r"""Fourier transform of the correlation function
    #     	\langle \hat{n}_k^{(b)} \rangle
    #         = 1/L \sum_m,n=1^L e^{ik(m - n)} \langle b_i^\dagger b_j \rangle
    #     """
    #
    #     k_range, mom_mat = self.get_momentum_distribution(self.ground_state())
    #     print(np.vdot(k_range, mom_mat))
    #     plt.plot(k_range, mom_mat)
    #     plt.show()
    #     exit()


    # #===========================================================================
    # def get_ninj_correlator(self, psi, i, j):
    #     r"""\langle b_i^\dagger b_j \rangle
    #         = <psi | b_i^\dagger b_j | psi >
    #         = (<4,0,0,0|c_1 + <3,1,0,0|c_2 + ... ) | b_i^\dagger b_j | psi >
    #         """
    #     ninj_op = 0
    #     for ind_1, basis1 in enumerate(self.basis.basis_list):
    #         for ind_2, basis2 in enumerate(self.basis.basis_list):
    #
    #             # apply bi^dagger b_j
    #             coeff_ninj = ni_nj(basis1, basis2, i, j, self.N)
    #
    #             if coeff_ninj == 0:
    #                 continue
    #             else:
    #                 ninj_op += coeff_ninj*np.conjugate(psi[ind_1])*psi[ind_2]
    #
    #     return ninj_op


    #===========================================================================
    # K - Operator
    #===========================================================================
    def get_K_mat(self):
        """ symmetry operator
            PRL 118, 120401 (2017)
            https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.118.120401
            Eq. (12)
        """

        path_npz = self.path_basis/'K_operator.npz'

        # if path_npz.is_file():
        #     K_dict = dict(np.load(path_npz))
        #     return K_dict['K_mat'], K_dict['evals'], K_dict['evecs']


        K_mat = np.zeros((self.basis.length, self.basis.length), dtype=complex)

        for i, basis1 in enumerate(self.basis.basis_list):
            for j, basis2 in enumerate(self.basis.basis_list):
                K_mat[i, j] = operators.K_operator(self.L, self.theta, basis1,
                                                   basis2)

        eval, evec = scipy.linalg.eig(K_mat)
        idx = eval.argsort()
        evals = eval[idx]
        evecs = (evec[:, idx]).T

        np.savez(path_npz, K_mat=K_mat, evals=evals, evecs=evecs)

        return K_mat, evals, evecs


    def K_mat_polar_coord(self, nstate_str=None):
        """Get entries of K-operator. Count the non-zero values and group then
        according to their degeneracy"""


        K_mat, evals, evecs = self.get_K_mat()
        K_mat[np.abs(K_mat) == 0] = np.nan
        cmplx_set, counts = np.unique(K_mat.round(8), return_counts=True,
                                      equal_nan=True)
        cmplx_angle_set = np.angle(cmplx_set)
        idx = cmplx_angle_set.argsort()
        cmplx_angle_set = cmplx_angle_set[idx]
        counts = counts[idx]
        nan_ind = np.where(cmplx_angle_set == np.nan)


        if nstate_str is not None:
            nstate = [eval(el) for el in nstate_str]
            nstate_ind = self.basis.basis_list.index(nstate)
            print(self.basis.basis_list[nstate_ind])
            exit()
        raise

        if np.isnan(cmplx_angle_set[-1]):
            return cmplx_angle_set[:-1], counts[:-1]
        else:
            return cmplx_angle_set, counts


    def K_eigvals_polar_coord(self):
        """The eigenvalues of the K-operator should have absolute value of 1,
        return the argument of the complex number"""

        K_mat, evals, evecs = self.get_K_mat()

        assert np.allclose(np.ones(evals.shape[0]), np.abs(evals))

        cmplx_set, counts = np.unique(evals.round(8), return_counts=True)
        cmplx_angle_set = np.angle(cmplx_set)
        idx = cmplx_angle_set.argsort()
        cmplx_angle_set = cmplx_angle_set[idx]
        counts = counts[idx]

        return cmplx_angle_set, counts

        # print(sum([-1/165*np.log(1/165) for el in range(165)]))
        #
        # plt.plot(cmplx_angle_set, occurence)
        # plt.show()
        # exit()





    # def K_op_black_diag(self):
    #     """Make block diagonalization of K-Operator"""
    #
    #     K_mat, evals, evecs = self.get_K_mat()
