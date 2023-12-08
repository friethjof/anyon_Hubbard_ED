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
import scipy.linalg

import path_dirs
from hamiltonian.basis import Basis
from helper import operators
from helper import other_tools


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

    def __init__(self, bc, L, N, J, U, theta):
        # basis : instance of class Bais

        self.path_basis = path_dirs.get_path_basis(bc, L, N, U, theta, J)

        self.bc = bc
        self.L = L
        self.N = N
        self.J = J
        self.U = U
        self.theta = theta
        self.basis = Basis(L, N)

        self._hamilt = None
        self._evals = None
        self._evecs = None


    #===========================================================================
    # build and solve hamiltonian
    #===========================================================================
    def make_diagonalization(self, load_key):
        path_hamil_npz = self.path_basis/'hamilt_spectrum.npz'

        if path_hamil_npz.is_file():
            if load_key == 'hamilt_mat':
                self._hamilt = np.load(path_hamil_npz)['hamilt_mat']
            elif load_key == 'evals':
                self._evals = np.load(path_hamil_npz)['evals']
            elif load_key == 'evecs':
                self._evecs = np.load(path_hamil_npz)['evecs']
            else:
                NotImplementedError
        else:

            # measure ellapsed time
            print('basis size:', self.basis.length)
            clock_1 = time.time()
            #-------------------------------------------------------------------
            # calculate all matrix elements: H_mn = <m| H | n>
            self._hamilt = np.zeros(
                (self.basis.length, self.basis.length),
                dtype=complex
                )
            for i, b_m in enumerate(self.basis.basis_list):
                for j, b_n in enumerate(self.basis.basis_list):

                    self._hamilt[i, j] = operators.get_hamilt_mn(
                        bc = self.bc,
                        L = self.L,
                        J = self.J,
                        U = self.U,
                        N = self.N,
                        theta = self.theta,
                        b_m = b_m,
                        b_n = b_n
                    )


            # check whether hamiltonian is hermitian
            assert np.allclose(self._hamilt, np.conjugate(self._hamilt.T))
            clock_2 = time.time()
            print('matrix hamiltonian created, '
                f'time: {other_tools.time_str(clock_2-clock_1)}')
            # Do not use np.linalg.eig here!
            # --> it might be that eigenvectors of degenerate eigenvalues
            # are not orthogonal!
            eval, evec = np.linalg.eigh(self._hamilt)
            idx = eval.argsort()
            self._evals = eval[idx]
            self._evecs = (evec[:, idx]).T

            # assert all([(np.vdot(el, el) - 1) < 1e-8 for el in self._evecs])
            assert np.max(np.abs(self._evals.imag)) < 1e-8
            self._evals = self._evals.real

            clock_3 = time.time()
            print(f'matrix hamiltonian has been diagonalized, '
                f'time: {other_tools.time_str(clock_3-clock_2)}')
            # check orthonormality
            ortho_bool = True
            for i in range(self.basis.length):
                for j in range(self.basis.length):
                    sp = np.vdot(self._evecs[i], self._evecs[j])
                    if i == j and np.abs(sp - 1) > 1e-10:
                        ortho_bool = False

                    if i != j and np.abs(sp) > 1e-10:
                        ortho_bool = False
            assert ortho_bool

            clock_4 = time.time()
            print(f'orthonormality has been checked, '
                f'time: {other_tools.time_str(clock_4-clock_3)}')

            np.savez(path_hamil_npz,
                hamilt_mat=self._hamilt,
                evals=self._evals,
                evecs=self._evecs,
                basis_list=self.basis.basis_list,
                basis_length=self.basis.length,
                L=self.L,
                N=self.N,
                J=self.J,
                U=self.U,
                theta=self.theta
            )
            print(path_hamil_npz.stat().st_size)
            print(f'File size: {path_hamil_npz.stat().st_size/(1024**2):.3f} MB')

            clock_5 = time.time()
            print(f'Total time consumption: '
                f'{other_tools.time_str(clock_5-clock_1)}')


    def hamilt(self):
        if self._hamilt is None:
            self.make_diagonalization('hamilt_mat')
        return self._hamilt


    def evals(self):
        if self._evals is None:
            self.make_diagonalization('evals')
        return self._evals


    def evecs(self):
        if self._evecs is None:
            self.make_diagonalization('evecs')
        return self._evecs

    #===========================================================================
    # analysis functions
    #===========================================================================
    def ground_state(self):
        return self.evecs()[0]


    def get_eigenstate_nOp_E0(self):
        """Get eigenstates which are degenerated with an eigenenergy E=0
        \rangle \phi | \hat{n}_i | \phi \rangle

        Returns
        arr : eigenvectors
        """
        evevs_E0 = self.get_eigenstates_E0()
        evecs_nOp_E0 = [self.eigenstate_nOp(evec) for evec in evevs_E0]
        return np.array(evecs_nOp_E0)


    def eigenstate_nOp_1b(self):
        """Calculate the one-body density for the eigenstates"""
        return [self.eigenstate_nOp(evec) for evec in self.evecs()]


    def eigenstate_nOp_2b(self):
        """Calculate the two-body density for the eigenstates"""

        evecs = self.evecs()

        evec_2b_list = []
        for evec in evecs:
            num_2b_mat = np.zeros((self.L, self.L))
            for m in range(0, self.L):
                for n in range(0, self.L):
                    for psi_coeff, b_m in zip(evec, self.basis.basis_list):
                        num_2b_mat[m, n] += (
                            np.abs(psi_coeff)**2
                            *b_m[m]
                            *b_m[n]
                        )

            evec_2b_list.append(num_2b_mat)

        return evec_2b_list


    def eigenstate_corr_2b(self):
        """Calculate the two-body density for the eigenstates"""

        evecs = self.evecs()
        bjbibibj_mats = operators.get_bjbibibj_mats(self.basis.basis_list, self.L)

        evec_2b_list = []
        for evec in evecs:
            corr_mat = np.zeros((self.L, self.L), dtype=complex)
            for m in range(0, self.L):
                for n in range(0, self.L):
                    corr_mat[m, n] = np.vdot(
                        evecs,
                        bjbibibj_mats[m, n, :, :].dot(evecs)
                    )

            assert np.max(np.abs(corr_mat.imag)) < 1e-8
            evec_2b_list.append(corr_mat.real)

        return evec_2b_list


    #===========================================================================
    # checkerboard pattern in 2-body density
    #===========================================================================
    def get_2b_checkerboard(self, ni, ninj):
        """"Calculate $\sum_{ij}(-1)^{i-j}(1-2\hat{n_i})(1-2\hat{n_j})$ """

        O_sum = 0
        for i in range(self.L):
            for j in range(self.L):
                O_sum += (-1)**(i-j)*(1 - 2*ni[i] - 2*ni[j] + 4*ninj[i,j])
        return O_sum


    def evecs_2b_checkerboard(self):
        """"Calculate $\sum_{ij}(-1)^{i-j}(1-2\hat{n_i})(1-2\hat{n_j})$ """
        evec_1b_list = self.eigenstate_nOp_1b()
        evec_2b_list = self.eigenstate_nOp_2b()

        checkerboard_list = []
        for ni, ninj in zip(evec_1b_list, evec_2b_list):
            checkerboard_list.append(self.get_2b_checkerboard(ni, ninj))

        return self.evals(), checkerboard_list


    #===========================================================================
    # E0 degeneracy
    #===========================================================================
    def energy_degeneracy(self):
        """Determine the degeneracies of the energies.

        Returns
        dict : degeneracies
        """

        return other_tools.find_degeneracies(self.evals())


    def eigenstate_nOp(self, state):
        """Get a state in nstate-representation and apply number operator
        \rangle \phi | \hat{n}_i | \phi \rangle

        Returns
        arr : eigenvector on lattive site
        """

        nop = []
        for site_i in range(self.L):
            nop_i = 0
            for psi_coeff, b_m in zip(state, self.basis.basis_list):
                nop_i += np.abs(psi_coeff)**2*b_m[site_i]
            nop.append(nop_i)
        nop = np.array(nop)
        assert np.max(np.abs(nop.imag)) < 1e-8
        nop = nop.real

        assert np.abs(np.sum(nop) - self.N) < 1e-9

        return nop


    def get_eigenstates_E0(self):
        """Get eigenstates which are degenerated with an eigenenergy E=0
        """

        dict_degen = other_tools.find_degeneracies(self.evals())
        E0_found = False
        for k, v in dict_degen.items():
            if abs(eval(k)) < 1e-10:
                ind_list = v
                E0_found = True
                break

        if not E0_found:
            raise ValueError
        evevs_E0 = [self.evecs()[i] for i in ind_list]

        return np.array(evevs_E0)


    def get_eigenstate_nOp_E0(self):
        """Get eigenstates which are degenerated with an eigenenergy E=0
        \rangle \phi | \hat{n}_i | \phi \rangle

        Returns
        arr : eigenvectors
        """
        evevs_E0 = self.get_eigenstates_E0()
        evecs_nOp_E0 = [self.eigenstate_nOp(evec) for evec in evevs_E0]
        return np.array(evecs_nOp_E0)


    def get_E0_degeneracy(self):
        path_npz = self.path_basis/'hamilt_E0_degeneracy.npz'
        if path_npz.is_file():
            return np.load(path_npz)['E0_degeneracy']

        dict_degen = self.energy_degeneracy()

        E0_degeneracy = np.nan
        for key, val in dict_degen.items():
            E_val = eval(key)
            if abs(E_val) < 1e-10:
                E0_degeneracy = len(val)
                break

        np.savez(path_npz, E0_degeneracy=E0_degeneracy)
        return E0_degeneracy


    #===========================================================================
    def nstate_eigenstate_SVN(self, nstate_str=None):

        svn_list = []
        for evecs_nstate in self.evecs().T:
            svn = sum([-el*np.log(el) for el in np.abs(evecs_nstate)**2
                       if el > 0])
            svn_list.append(svn)

        if nstate_str is None:
            return np.array(svn_list)

        nstate_eval = np.array([eval(el) for el in nstate_str.split('-')])

        bool_succ = False
        for svn, nstate_i in zip(svn_list, self.basis.basis_list):
            if (nstate_eval == nstate_i).all():
                bool_succ = True
                break
        assert bool_succ
        return svn


    #===========================================================================
    def bipartite_SVN(self, psi, subsysA_ind):
        """Write |Psi> = sum_ij C_ij |Psi_i^A> |Psi_j^B>
        Where |Psi_i^A> |Psi_j^B> correspond to the left and right part:
        |0,0,1,0,1,0> = |0,0> |1,0,1,0>
        Associate both composite basis states with the respective basis of A and
        B and create the rectangular matrix C_ij. From there do a Schmidt
        -decomposition.
        """

        subsysA = list(range(0, subsysA_ind))
        subsysB = list(range(subsysA_ind, self.L))

        basis_A = np.array([list(el) for el in itertools.product(
            range(self.N+1), repeat=len(subsysA)) if sum(el) <= self.N])
        basis_B = np.array([list(el) for el in itertools.product(
            range(self.N+1), repeat=len(subsysB)) if sum(el) <= self.N])


        C_ij = np.zeros((len(basis_A), len(basis_B)), dtype=complex)
        for psi_coeff, basis_full in zip(psi, self.basis.basis_list):


            basis_fullA = basis_full[0:subsysA_ind]
            basis_fullB = basis_full[subsysA_ind:]
            ind_i = (basis_A==basis_fullA).all(axis=1).nonzero()
            ind_j = (basis_B==basis_fullB).all(axis=1).nonzero()
            assert len(ind_i) == 1
            assert len(ind_j) == 1
            ind_i = ind_i[0][0]
            ind_j = ind_j[0][0]

            C_ij[ind_i, ind_j] = psi_coeff

        sc_list = other_tools.make_schmidt_decomp(C_ij)
        lambda_list = [el[0]**2 for el in sc_list]
        Mlam = len(lambda_list)
        cA_list = [el[1] for el in sc_list]
        cB_list = [el[2] for el in sc_list]

        svn = sum([-el*np.log(el) for el in lambda_list if el != 0])

        return svn


    def bipartite_SVN_list_evec0(self):
        l_list = list(range(1,self.L))
        svn_list = []
        for l in l_list:
            svn = self.bipartite_SVN(
                psi=self.evecs()[0],
                subsysA_ind=l
            )
            svn_list.append(svn)
        return l_list, svn_list


    def bipartite_SVN_all_eigstates(self):
        l_list = list(range(1,self.L))
        svn_lol = []
        for l in l_list:
            svn_list = []
            for evec in self.evecs():
                svn = self.bipartite_SVN(
                    psi=evec,
                    subsysA_ind=l
                )
                svn_list.append(svn)
            svn_lol.append(svn_list)

        return l_list, np.array(svn_lol)


    def bipartite_SVN_E0_subspace(self):
        l_list = list(range(1,self.L))
        svn_lol = []
        for l in l_list:
            svn_list = []
            for evec in self.get_eigenstates_E0():
                svn = self.bipartite_SVN(
                    psi=evec,
                    subsysA_ind=l
                )
                svn_list.append(svn)
            svn_lol.append(svn_list)

        return l_list, np.array(svn_lol)


    #===========================================================================
    def get_eigenstate_nOp_E0_K_basis(self):
        evecs_E0 = self.get_eigenstates_E0()

        K_mat_nstate, _, _ = self.get_K_mat()

        K_mat_E0_basis = np.zeros((evecs_E0.shape[0], evecs_E0.shape[0]),
                                  dtype=complex)
        for i, evec_i in enumerate(evecs_E0):
            for j, evec_j in enumerate(evecs_E0):
                K_mat_E0_basis[i, j] = np.vdot(evec_i, K_mat_nstate.dot(evec_j))

        eval, evec = np.linalg.eigh(K_mat_E0_basis)
        idx = eval.argsort()
        K_evals_E0_basis = eval[idx]
        K_evecs_E0_basis = (evec[:, idx]).T

        # transorm into number state basis
        K_evecs_nstate = []
        for coeff in K_evecs_E0_basis:
            evec_eff = 0
            for c, evec_E0 in zip(coeff, evecs_E0):
                evec_eff += c*evec_E0
            K_evecs_nstate.append(evec_eff)


        K_evecs_nOp = [self.eigenstate_nOp(evec) for evec in K_evecs_nstate]
        return K_evals_E0_basis, np.array(K_evecs_nOp)


    #===========================================================================
    def get_energy_occup_basis(self, occupation_list):
        """Obtain energy occupation number basis
        Single particle eigenstates:
            c_{\nu,j} = sqrt(2/(L+1)) sin(pi*\nu*j/(L+1))
            b'_\nu = \sum_{j=1}^L c_{\nu,j} b_j
            |{\eta_1, ..., \eta_L}> =
                    \Prod_{\nu=1}^L (b'^dagger)^{\eta_\nu}/sqrt(\eta_\nu!) |0>

            Then return |{\eta_1, ..., \eta_L}> in number basis
        """

        assert len(occupation_list) == self.L
        assert sum(occupation_list) == self.N

        C_tensor = np.zeros((self.basis.length, self.basis.length))
        for i in range(self.basis.length):
            for j in range(self.basis.length):
                c_ij = np.sqrt(2/(self.L + 1)) * np.sin(np.pi*(i+1)*(j+1)/(self.L + 1))
                C_tensor[i, j] = c_ij


        #-----------------------------------------------------------------------
        def apply_cij_bj(i, j, coeff, number_list):
            n_list_out = number_list.copy()
            coeff *= C_tensor[i, j]*math.sqrt(number_list[j] + 1)
            n_list_out[j] += 1
            return coeff, n_list_out


        #-----------------------------------------------------------------------
        def apply_b_tilde(i, n_vec_in, basis_in):
            """Apply \tilde{b}
                --> creates one particle
                --> update number state basis
            """

            N_new = basis_in.N + 1
            basis_new = Basis(self.L, N_new)
            n_vec_new = np.zeros(basis_new.length)
            for coeff, n_list in zip(n_vec_in, basis_in.basis_list):
                for j in range(self.L):
                    coeff_new, n_list_new = apply_cij_bj(i, j, coeff, n_list)

                    # add new coeff to new basis state
                    ind = basis_new.nstate_index(n_list_new)
                    n_vec_new[ind] += coeff_new
            return n_vec_new, basis_new


        #-----------------------------------------------------------------------
        basis = Basis(self.L, 0)
        n_vec = np.array([1.0])
        for i, n in enumerate(occupation_list):
            for _ in range(n):
                n_vec, basis = apply_b_tilde(i, n_vec, basis)
            n_vec *= 1/np.sqrt(math.factorial(n))

        assert basis.length == self.basis.length
        assert abs(1 - np.vdot(n_vec, n_vec)) < 1e-8
        # print(np.vdot(n_vec, self.hamilt().dot(n_vec)))

        return n_vec


    def ovlp_energy_occup_basis_E0_subspace(self):
        """get all |{\eta_1, ..., \eta_L}> in nstate representation and
        calculate overlap with E0 subspace"""

        evecs_E0 = self.get_eigenstates_E0()

        E0_overlap = []
        for nstate in self.basis.basis_list:

            ebasis = self.get_energy_occup_basis(nstate)

            E0_overlap.append(
                np.sum([np.abs(np.vdot(ebasis, E0_evec))**2 for E0_evec in evecs_E0])
            )


        return np.array(E0_overlap)


    def get_eigenstate_nOp_E0_energy_basis(self):
        """Return occupation density the E_0 eigenstates build from the
        energy occupation number basis.
        For L = 6 and N = 2:
            Return nOp for:
                |{\eta_1, ..., \eta_L}> = |1,0,0,0,0,1>
                |{\eta_1, ..., \eta_L}> = |0,1,0,0,1,0>
                |{\eta_1, ..., \eta_L}> = |0,0,1,1,0,0>
        For L = 6 and N = 4:
            Return nOp for:
                |{\eta_1, ..., \eta_L}> = |2,0,0,0,0,2>
                |{\eta_1, ..., \eta_L}> = |0,2,0,0,2,0>
                |{\eta_1, ..., \eta_L}> = |0,0,2,2,0,0>
                |{\eta_1, ..., \eta_L}> = |1,1,0,0,1,1>
                |{\eta_1, ..., \eta_L}> = |1,0,1,1,0,1>
                |{\eta_1, ..., \eta_L}> = |0,1,1,1,1,0>
        """

        if self.L == 6 and self.N == 2:
            nOp_lol = [
                self.eigenstate_nOp(self.get_energy_occup_basis([1,0,0,0,0,1])),
                self.eigenstate_nOp(self.get_energy_occup_basis([0,1,0,0,1,0])),
                self.eigenstate_nOp(self.get_energy_occup_basis([0,0,1,1,0,0]))
            ]
            legend_list = [
                r'$\tilde{\eta}_1=\tilde{\eta}_6=1$',
                r'$\tilde{\eta}_2=\tilde{\eta}_5=1$',
                r'$\tilde{\eta}_3=\tilde{\eta}_4=1$',
            ]

        elif self.L == 6 and self.N == 4:
            nOp_lol = [
                self.eigenstate_nOp(self.get_energy_occup_basis([2,0,0,0,0,2])),
                self.eigenstate_nOp(self.get_energy_occup_basis([0,2,0,0,2,0])),
                self.eigenstate_nOp(self.get_energy_occup_basis([0,0,2,2,0,0])),
                self.eigenstate_nOp(self.get_energy_occup_basis([1,1,0,0,1,1])),
                self.eigenstate_nOp(self.get_energy_occup_basis([1,0,1,1,0,1])),
                self.eigenstate_nOp(self.get_energy_occup_basis([0,1,1,1,1,0]))
            ]
            legend_list = [
                r'$\tilde{\eta}_1=\tilde{\eta}_6=2$',
                r'$\tilde{\eta}_2=\tilde{\eta}_5=2$',
                r'$\tilde{\eta}_3=\tilde{\eta}_4=2$',
                r'$\tilde{\eta}_1=\tilde{\eta}_2=\tilde{\eta}_5=\tilde{\eta}_6=1$',
                r'$\tilde{\eta}_1=\tilde{\eta}_3=\tilde{\eta}_4=\tilde{\eta}_6=1$',
                r'$\tilde{\eta}_2=\tilde{\eta}_3=\tilde{\eta}_4=\tilde{\eta}_5=1$'
            ]

        else:
            raise NotImplementedError('L=', self.L, ' N=', self.N)

        return nOp_lol, legend_list


    #===========================================================================
    def eigenstate_E0_perturbation(self, U_max, steps, order):
        """Expand degenerate eigenstates of E=0 with Brillouin-Wigner
        perturbation theory"""

        assert self.U == 0

        #-----------------------------------------------------------------------
        # get eigenvectors of unperturbed Hamiltonian, U=0
        dict_degen = other_tools.find_degeneracies(self.evals())
        E0_found = False
        for k, v in dict_degen.items():
            if abs(eval(k)) < 1e-10:
                ind_list = v
                E0_found = True
                break
        if not E0_found:
            raise ValueError

        evals_E0 = [self.evals()[i] for i in ind_list]
        evecs_E0 = [self.evecs()[i] for i in ind_list]
        evecs_E_compl = [self.evecs()[i] for i in range(self.basis.length)
                         if i not in ind_list]
        evals_E_compl = [self.evals()[i] for i in range(self.basis.length)
                         if i not in ind_list]

        #-----------------------------------------------------------------------
        # get H0 and V
        H0_mat = np.zeros((self.basis.length, self.basis.length), dtype=complex)
        V_mat = np.zeros((self.basis.length, self.basis.length), dtype=complex)
        for i, b_m in enumerate(self.basis.basis_list):
            for j, b_n in enumerate(self.basis.basis_list):

                H0_mat[i, j] = operators.get_hamilt_mn(self.L, self.J, 0,
                                                       self.N, self.theta,
                                                       b_m, b_n)

                V_mat[i, j] = operators.get_hamilt_mn(self.L, 0, 1, self.N,
                                                      self.theta, b_m, b_n)

        #-----------------------------------------------------------------------
        # create H_eff in basis of evevs_E0 as function of lambda

        n0_eff_lol, evals_eff_lol = [], []
        U_range = np.linspace(0, U_max, steps)
        for U in U_range:
            print(f'U={round(U, 8)}')
            H_eff_U = operators.get_perturbed_H_eff(
                H0_mat=H0_mat,
                V_mat=V_mat,
                evals_E=evals_E0,
                evecs_E=evecs_E0,
                evals_E_compl=evals_E_compl,
                evecs_E_compl=evecs_E_compl,
                lam=U,
                order=order
            )

            eval_eff, coeff_eff = np.linalg.eigh(H_eff_U)
            idx = eval_eff.argsort()
            evals_eff = eval_eff[idx]
            coeffs_eff = (coeff_eff[:, idx]).T

            # transorm into number state basis
            evecs_eff = []
            for coeff in coeffs_eff:
                evec_eff = 0
                for c, evec_E0 in zip(coeff, evecs_E0):
                    evec_eff += c*evec_E0
                evecs_eff.append(evec_eff)


            n0_eff_lol.append([self.eigenstate_nOp(evec) for evec in evecs_eff])
            evals_eff_lol.append(evals_eff)

        return U_range, np.array(evals_eff_lol), np.array(n0_eff_lol)


    #===========================================================================
    def onsite_int_nstate_exp(self):
        U_energy = []
        for nstate in self.basis.basis_list:
            sum_onsite = operators.sum_onsite_int(self.L, nstate, nstate)
            U_energy.append(sum_onsite)
        return np.array(U_energy)


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


    # def energy_spectrum_mom(self):
    #     """ Try: E_k = 1/sqrt(L) sum_k b_k exp(ikj) E_j
    #     Where E_j is eigenvalue of
    #     """
    #     assert self.bc == 'periodic'





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


    # def get_K_dagger_mat(self):
    #     """ symmetry operator
    #         PRL 118, 120401 (2017)
    #         https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.118.120401
    #         Eq. (12)
    #     """
    #
    #     # path_npz = self.path_basis/'K_dagger_operator.npz'
    #
    #     # if path_npz.is_file():
    #     #     K_dict = dict(np.load(path_npz))
    #     #     return K_dict['K_mat'], K_dict['evals'], K_dict['evecs']
    #
    #
    #     K_mat = np.zeros((self.basis.length, self.basis.length), dtype=complex)
    #
    #     for i, basis1 in enumerate(self.basis.basis_list):
    #         for j, basis2 in enumerate(self.basis.basis_list):
    #             K_mat[i, j] = operators.K_dagger_operator(self.L, self.theta,
    #                                                       basis1, basis2)
    #
    #     eval, evec = scipy.linalg.eig(K_mat)
    #     idx = eval.argsort()
    #     evals = eval[idx]
    #     evecs = (evec[:, idx]).T
    #
    #     # np.savez(path_npz, K_mat=K_mat, evals=evals, evecs=evecs)
    #
    #     return K_mat, evals, evecs



    def K_mat_polar_coord(self):
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

        if np.isnan(cmplx_angle_set[-1]):
            return cmplx_angle_set[:-1], counts[:-1]
        else:
            return cmplx_angle_set, counts


    def K_eigvals_polar_coord(self, nstate_str=None):
        """The eigenvalues of the K-operator should have absolute value of 1,
        return the argument of the complex number"""

        K_mat, evals, evecs = self.get_K_mat()

        assert np.allclose(np.ones(evals.shape[0]), np.abs(evals))

        cmplx_set, counts = np.unique(evals.round(8), return_counts=True)
        cmplx_angle_set = np.angle(cmplx_set)
        idx = cmplx_angle_set.argsort()
        cmplx_angle_set = cmplx_angle_set[idx]
        counts = counts[idx]

        if nstate_str is not None:
            nstate = [eval(el) for el in nstate_str]
            nstate_ind = self.basis.basis_list.index(nstate)
            print(self.basis.basis_list[nstate_ind])
            exit()
            raise

        return cmplx_angle_set, counts

        # print(sum([-1/165*np.log(1/165) for el in range(165)]))
        #
        # plt.plot(cmplx_angle_set, occurence)
        # plt.show()
        # exit()


    def H_in_K_basis(self):
        """Make block diagonalization of K-Operator"""

        K_mat, evals, evecs = self.get_K_mat()

        hamilt_Kbasis = np.zeros((self.hamilt().shape), dtype=complex)
        for i, evec_i in enumerate(evecs):
            for j, evec_j in enumerate(evecs):
                # eigenvectors of K have still T-operator
                hamilt_Kbasis[i, j] = np.vdot(evec_i, np.conjugate(self.hamilt().dot(evec_j)))


        print(hamilt_Kbasis)
        x, y = np.meshgrid(range(self.basis.length), range(self.basis.length))
        im = plt.pcolormesh(x, y, np.abs(hamilt_Kbasis))
        plt.colorbar(im)
        plt.show()
        exit()


    def get_pair_mat(self):
        """Calculate pair-operator in number state basis
        \nu_p = \sum_j \hat{n}_j (\hat{n}_j - 1)/2
        """

        pair_mat = np.zeros((self.basis.length, self.basis.length))

        for i, basis1 in enumerate(self.basis.basis_list):
            for j, basis2 in enumerate(self.basis.basis_list):
                pair_mat[i, j] = operators.pair_operator(self.L, basis1, basis2)

        eval, evec = scipy.linalg.eig(pair_mat)
        idx = eval.argsort()
        evals = eval[idx]
        evecs = (evec[:, idx]).T

        # print(pair_mat)
        # x, y = np.meshgrid(range(self.basis.length), range(self.basis.length))
        # im = plt.pcolormesh(x, y, pair_mat)
        # plt.colorbar(im)
        # plt.show()
        # exit()


        return pair_mat, evals, evecs



    # #===========================================================================
    # # rho A --- working alternative
    # #===========================================================================
    # def rho_A(self, subsysA_ind):
    #     """Calculate the reduced density operator of subsystem A by tracing out
    #     the dofs of subsystem B.
    #
    #     rho_A = tr_B(|psi(t)><psi(t)|)
    #           = sum_basisB <b_n| (|3,1,0,0><1,2,1,0| + ...) |b_n>
    #     """
    #     N = self.N
    #     L = self.L
    #     basis_list = self.basis.basis_list
    #     subsysB_ind = [el for el in range(L) if el not in subsysA_ind]
    #
    #     # basis_B = [[0, 0],...,[0, N], [1, 0],...,[1, N-1], ..., [N, 0]]
    #     basis_A = [np.array(list(el)) for el in itertools.product(
    #         range(N+1), repeat=len(subsysA_ind)) if sum(el) <= N]
    #     basis_B = [np.array(list(el)) for el in itertools.product(
    #         range(N+1), repeat=len(subsysB_ind)) if sum(el) <= N]
    #
    #     time = [0]
    #     psi_test = np.zeros(self.basis.length)
    #     psi_test[1] = 1
    #     psi_t = [self.evecs()[0]]
    #     # psi_t = [psi_test]
    #
    #     rhoA_t = []
    #     for t, psi_tstep in zip(time, psi_t):  # loop over all times
    #         print(f'time: {t}')
    #         # rho_A|k,l = sum_ijkl <nB_i| ( |nA_k>|nB_k> <nA_l|<nB_l| ) |nB_j>
    #         #   = sum_kl |nA_k> <nA_l| * sum_ij <nB_i|nB_k> <nB_l|nB_j>
    #
    #         rho_A = np.zeros((len(basis_A), len(basis_A)), dtype=complex)
    #
    #
    #         for bB_i in basis_B:
    #             # bB_i = <0, 0|,  |0, 0>
    #             for k, (basis_k, coeff_k) in enumerate(zip(basis_list,
    #                 psi_tstep)):
    #                 # basis_k = |2, 0, 1, 1>
    #                 # basisA_k = |2, 0> ; basisB_k = |1, 1>
    #                 basisB_k = np.array([basis_k[x] for x in subsysB_ind])
    #                 basisA_k = np.array([basis_k[x] for x in subsysA_ind])
    #                 for l, (basis_l, coeff_l) in enumerate(zip(basis_list,
    #                     psi_tstep)):
    #                     # basis_l = <3, 0, 0, 1|
    #                     # basisA_l = <3, 0| ; basisB_l = <0, 1|
    #                     basisB_l = np.array([basis_l[x] for x in subsysB_ind])
    #                     basisA_l = np.array([basis_l[x] for x in subsysA_ind])
    #                     if (bB_i==basisB_k).all() and (bB_i==basisB_l).all():
    #                         # assign coefficient to the right index of rhoA
    #                         bool_break = False
    #                         for i, bA_i in enumerate(basis_A):
    #                             for j, bA_j in enumerate(basis_A):
    #                                 # there exists only one combination of
    #                                 # bA_i,j that matches basisA_l,k
    #                                 if (bA_i==basisA_k).all() and\
    #                                     (bA_j==basisA_l).all():
    #                                     rho_A[i, j] +=\
    #                                         coeff_k*np.conjugate(coeff_l)
    #                                     bool_break = True
    #                                     break
    #                             if bool_break:
    #                                 break
    #
    #         rhoA_t.append(rho_A)
    #         # rhoA_2 = np.matmul(rho_A, rho_A)
    #         # print(np.abs(np.sum(np.abs(rho_A-rhoA_2))))
    #         # assert np.abs(np.sum(np.abs(rho_A-rhoA_2))) < 1e-8
    #
    #     return np.array(rhoA_t)
    #
    #
    # def _vN_entropy(self, subsys_ind):
    #     """
    #     subsys_ind : list of indices corresponding to lattice indices
    #     Calculate the von Neumann entropy of a given subsystem A.
    #         S = sum_i -n_i*log(n_i).
    #     with n_i corresponding to the eigenvalues of A
    #     """
    #     rhoA_t = self.rho_A(subsys_ind)
    #     s_ent = []
    #     for rhoA in rhoA_t:  # loop over all times
    #         # calculate von Neumann entropy
    #         eval, _ = np.linalg.eig(rhoA)
    #         # verify that all eigenvalues have no imaginary contribution
    #         assert all([el.imag < 1e-8 for el in eval])
    #         # entropy = -np.trace(np.matmul(rhoA, linalg.logm(rhoA)))
    #         entropy = sum([-el*np.log(el) for el in eval.real if 1e-8 < abs(el)])
    #         s_ent.append(entropy)
    #     return np.array(s_ent)
    #
    #
    # def bipartite_ent(self):
    #     # path_sentA_dict = self.path_prop/f'dict_SentA_{self.dum_Tf_dt}.npz'
    #     # if path_sentA_dict.is_file():
    #     #     SentA_dict = dict(np.load(path_sentA_dict))
    #     #     return SentA_dict['SentA']
    #
    #     N = self.N
    #     L = self.L
    #     lattice_ind = list(range(L//2))
    #     entropy = self._vN_entropy(lattice_ind)
    #     # np.savez(path_sentA_dict, SentA=entropy, time=self.time)
    #     return entropy
