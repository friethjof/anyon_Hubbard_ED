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
import scipy.optimize

import path_dirs
from hamiltonian.solve_hamilt import AnyonHubbardHamiltonian
from helper import operators
from propagation import initialize_prop



#===============================================================================
class Propagation(AnyonHubbardHamiltonian):
    """Propagation of a given initial state.

    |psi(t)> = sum_n exp(-i*E_n*t) |n> <n|psi_0>
    n> : eigenvectors
    """

    def __init__(self, bc, L, N, J, U, theta, psi0_str, Tprop, dtprop):
        """Initialize parameters.

        Args:
            bc (str): boundary conditions of lattice, open (hard-wall) | periodic
            L (int): number of lattice sites
            N (int): number of atoms
            J (float): hopping constant
            U (float): on-site interaction
            theta (float): statistical angle

            psi0_str (str): specifies initial state for propagation
            Tprop (float): final time of propagation
            dtprop (float): time steps of propagation
            _psi_t (arr): place holder for psi-file

        """

        super().__init__(bc, L, N, J, U, theta)



        if Tprop is not None:
            self.Tprop = Tprop
            self.dtprop = dtprop
            self.time = np.arange(0, Tprop+dtprop, dtprop)
            self.dum_Tf_dt = path_dirs.get_dum_Tf_dt(Tprop, dtprop)

        self.psi0_str = psi0_str

        # get initial state
        if psi0_str is not None:
            self.path_prop = self.path_basis/psi0_str
            self.path_prop.mkdir(parents=True, exist_ok=True)
            self.psi0, self.nstate0_str = initialize_prop.get_psi0(
                psi0_str, self.basis.basis_list, self.evecs())


        self._psi_t = None





    #===========================================================================
    # propagate intial state
    #===========================================================================
    def make_propagation(self):
        path_psi_npz = self.path_prop/f'psi_{self.dum_Tf_dt}.npz'

        # if False:
        if (path_psi_npz.is_file() and
                np.load(path_psi_npz)['Tprop'] == self.Tprop and
                np.load(path_psi_npz)['dtprop'] == self.dtprop):
            self._psi_t = np.load(path_psi_npz)['psi_t']
        else:
            # make propagation
            psi_t = []
            for t in self.time:
                # print(f't = {t:.1f}')
                psi_tstep = np.zeros(self.basis.length, dtype=complex)
                for eval_n, evec_n in zip(self.evals(), self.evecs()):

                    psi_tstep += (
                        np.exp(-1j*eval_n*t)
                        *evec_n*np.vdot(evec_n, self.psi0)
                    )

                psi_t.append(psi_tstep)
                assert np.abs(np.vdot(psi_tstep, psi_tstep) - 1+0j) < 1e-8

            self._psi_t = np.array(psi_t)

            np.savez(
                path_psi_npz,
                time=self.time,
                psi_t=self._psi_t,
                psi0=self.psi0,
                psi0_str=self.psi0_str,
                nstate0_str=self.nstate0_str,
                Tprop=self.Tprop,
                dtprop=self.dtprop,
                L=self.L,
                N=self.N,
                J=self.J,
                U=self.U,
                theta=self.theta,
                basis_list=self.basis.basis_list
            )

            path_logf = self.path_prop/f'psi_{self.dum_Tf_dt}_log_file.txt'
            initialize_prop.write_log_file(
                path_logf, self.L, self.N, self.J, self.U, self.theta,
                self.psi0_str, self.nstate0_str, self.Tprop, self.dtprop)


    def psi_t(self):
        if self._psi_t is None:
            self.make_propagation()
        return self._psi_t


    #===========================================================================
    # number operator
    #===========================================================================
    def num_op_site(self, site_i):
        "psi_t = <psi(t)| b_i^t b^i |psi(t)>"
        path_nop_dict = self.path_prop/f'dict_numOp_{self.dum_Tf_dt}.npz'
        if path_nop_dict.is_file():
            nop_dict = dict(np.load(path_nop_dict))
            time = nop_dict['time']
            if abs(time[-1] - self.time[-1]) < 1e-6:
                if f'site_{site_i}' in nop_dict.keys():
                    return time, nop_dict[f'site_{site_i}']
            else:
                nop_dict = {}
        else:
            nop_dict = {}

        nop = []
        for psi_tstep in self.psi_t():
            nop_t = 0
            for psi_coeff, b_m in zip(psi_tstep, self.basis.basis_list):
                nop_t += np.abs(psi_coeff)**2*b_m[site_i]
            nop.append(nop_t)
        nop = np.array(nop)
        assert np.max(np.abs(nop.imag)) < 1e-8
        nop = nop.real

        nop_dict[f'site_{site_i}'] = nop
        nop_dict[f'time'] = self.time
        np.savez(path_nop_dict, **nop_dict)
        return self.time, nop


    def num_op_mat(self):
        """Return lists which are easy to plot"""
        L = self.L
        expV_list, label_list = [], []
        for i in range(L):
            time, expV = self.num_op_site(i)
            expV_list.append(expV)
        return time, np.array(expV_list)


    def get_nop_time(self, t_instance):
        time, expV_list = self.num_op_mat()
        time_ind = np.argmin(np.abs(time - t_instance))
        return np.array(expV_list)[:, time_ind]


    def num_op_bound_state(self):
        """Works only for 2 Atoms!
        Separate psi wave vector in a bound and scatter part.
        Define bound states where number states have an occupations on the same
        or neighbouring sites, i.e., |..., 2, ....> or |..., 1, 1, ....>

        n_i^bound = <psi_bound(t)| n_i |psi_bound(t)>
        n_i^scatter = <psi_scatter(t)| n_i |psi_scatter(t)>
        """
        path_npz = self.path_prop/f'dict_numOp_bound_state_{self.dum_Tf_dt}.npz'

        if path_npz.is_file():
            nop_dict = dict(np.load(path_npz))
            time = nop_dict['time']
            if abs(time[-1] - self.time[-1]) < 1e-6:
                return time, nop_dict['num_op_bound'], nop_dict['num_op_scatter']

        assert self.N == 2

        #-----------------------------------------------------------------------
        # get indices of number states corresponding to bound states
        nstate_ind_bound, nstate_ind_scatter = [], []
        for i, nstate in enumerate(self.basis.basis_list):
            check = False
            for j, n in enumerate(nstate):
                if n == 2:
                    nstate_ind_bound.append(i)
                    check = True
                    break
                elif n == 1:
                    if nstate[j+1] == 1:
                        nstate_ind_bound.append(i)
                        check = True
                        break
                    else:
                        nstate_ind_scatter.append(i)
                        check = True
                        break
            if not check:
                raise


        #-----------------------------------------------------------------------
        # calculate site population
        nop_bound = np.zeros((self.L, self.time.shape[0]))
        nop_scatter = np.zeros((self.L, self.time.shape[0]))
        for t, psi_tstep in enumerate(self.psi_t()):
            for site_i in range(self.L):

                for j in nstate_ind_bound:
                    psi_coeff = psi_tstep[j]
                    b_m = self.basis.basis_list[j]
                    nop_bound[site_i, t] += np.abs(psi_coeff)**2*b_m[site_i]

                for j in nstate_ind_scatter:
                    psi_coeff = psi_tstep[j]
                    b_m = self.basis.basis_list[j]
                    nop_scatter[site_i, t] += np.abs(psi_coeff)**2*b_m[site_i]

        return nop_bound, nop_scatter



    def root_mean_square(self):
        'calculate sqrt(1/L sum_i (i * n_i)**2)'

        nop_bound, nop_scatter = self.num_op_bound_state()

        rms_bound, rms_scatter = [], []
        for n_bound, n_scatter in zip(nop_bound.T, nop_scatter.T):

            x_grid = np.arange(-self.L/2+0.5, self.L/2+0.5, 1)
            rms_b = np.sqrt( np.mean( (n_bound * x_grid**2) ))
            rms_s = np.sqrt( np.mean( (n_scatter * x_grid**2) ))

            rms_bound.append(rms_b)
            rms_scatter.append(rms_s)

        return np.array(rms_bound), np.array(rms_scatter)


    def root_mean_square_linear_fit(self, t_interval):
        """ Make linear fit to root_mean_squares of bound and scatter states
        Return: slope and error
        """
        assert len(t_interval) == 2

        rms_bound, rms_scatter = self.root_mean_square()

        ind_min = np.argmin(np.abs(self.time - t_interval[0]))
        ind_max = np.argmin(np.abs(self.time - t_interval[1]))

        time_red = self.time[ind_min:ind_max+1]
        rms_bound = rms_bound[ind_min:ind_max+1]
        rms_scatter = rms_scatter[ind_min:ind_max+1]

        res_bound = scipy.stats.linregress(time_red, rms_bound)
        res_scatter = scipy.stats.linregress(time_red, rms_scatter)

        return res_bound, res_scatter



    #===========================================================================
    # natural population
    #===========================================================================
    def get_natpop(self):
        """https://journals.aps.org/pra/pdf/10.1103/PhysRevA.101.063617
        Eq.(3),(4)
        rho_1 = \sum_ij <psi(t)| b_i^dagger b_j |psi(t)>
              = sum_i n_i p_i

        return
            arr : natpop shape -> N_orb x time
            arr : natorb shape -> N_orb x L x time
        """

        path_dict = self.path_prop/f'dict_natpop_{self.dum_Tf_dt}.npz'
        if path_dict.is_file():
            obs_dict = dict(np.load(path_dict))
            return obs_dict['natpop'], obs_dict['natorb']
        else:
            obs_dict = {}

        bibj_op_mat = np.zeros(
            (self.L, self.L, self.basis.length, self.basis.length),
            dtype=complex
        )
        for i in range(self.L):
            for j in range(self.L):
                bibj_op_mat[i, j, :, :] = operators.get_bibj_mat(
                      self.basis.basis_list, i, j)

        natpop = []
        natorbs = []
        for t, psi_tstep in zip(self.time, self.psi_t()):
            psi_tstep = psi_tstep/np.sqrt(self.N)
            bibj_mat = np.zeros((self.L, self.L), dtype=complex)
            for i in range(self.L):
                for j in range(self.L):
                    bibj_mat[i, j] = np.vdot(psi_tstep, (bibj_op_mat[i, j].dot(psi_tstep)))


            assert np.allclose(bibj_mat, np.conjugate(bibj_mat.T))

            eval, evec = np.linalg.eigh(bibj_mat)
            idx = eval.argsort()[::-1]
            evals = eval[idx]
            evecs = (evec[:, idx]).T
            assert np.max(np.abs(evals.imag)) < 1e-10

            natpop.append(evals.real)
            natorbs.append(evecs)

        # reshape
        natpop = np.array(natpop).T
        natorbs = np.swapaxes(np.swapaxes(np.array(natorbs), 0, 1), 1, 2)

        assert np.allclose(np.ones(self.time.shape[0]), np.sum(natpop, axis=0))

        np.savez(path_dict, natpop=natpop, natorb=natorbs)

        return natpop, natorbs


    def natpop_SVN(self):
        natpop, _ = self.get_natpop()
        svn = []
        for npop in natpop.T:
            svn.append(sum([-el*np.log(el) for el in npop if el > 0]))
        svn = np.array(svn)
        svn_max = np.log(self.L)
        return svn, svn_max


        svn = [np.sum(npop[:, t]) for t, _ in enumerate(self.time)]
        return np.array(svn)


    #===========================================================================
    # two-site density <n_i n_j>
    #===========================================================================
    def num_op_ninj(self, site_i, site_j, bool_save=True):
        "psi_t = <psi(t)| n_j n^i |psi(t)>"
        path_dict = self.path_prop/f'dict_ninj_{self.dum_Tf_dt}.npz'
        obs_name = f'site_{site_i}_{site_j}'
        if path_dict.is_file():
            obs_dict = dict(np.load(path_dict))
            time = obs_dict['time']
            if obs_name in obs_dict.keys():
                return time, obs_dict[obs_name]
        else:
            obs_dict = {}

        # print(obs_name)
        basis_list = self.basis.basis_list

        obs_list = []
        for psi_tstep in self.psi_t():
            obs_t = 0
            for psi_coeff, b_m in zip(psi_tstep, basis_list):
                obs_t += (
                    np.abs(psi_coeff)**2
                    *b_m[site_i]
                    *b_m[site_j]
                )
            obs_list.append(obs_t)
        obs = np.array(obs_list)

        assert np.max(np.abs(obs.imag)) < 1e-8
        obs = obs.real

        obs_dict[obs_name] = obs
        obs_dict[f'time'] = self.time
        if bool_save:
            np.savez(path_dict, **obs_dict)
        return self.time, obs


    def lists_ninj(self, bool_save=True):
        """Note that <n_i n_j> is almost the same"""

        expV_list, label_list = [], []
        for i in range(self.L):
            for j in range(self.L):
                time, expV = self.num_op_ninj(i, j, bool_save)
                expV_list.append(expV)
                label_list.append(f'{i+1} {j+1}')
        return time, expV_list, label_list


    def get_ninj_mat_time(self, t_instance):
        L = self.L
        time, expV_list, label_list = self.lists_ninj()
        time_ind = np.argmin(np.abs(time - t_instance))
        ninj_mat = np.zeros((L, L))
        for i, lab in enumerate(label_list):
            k, l = int(lab.split()[0]) - 1, int(lab.split()[1]) - 1
            ninj_mat[k, l] = expV_list[i][time_ind]

        x, y = np.meshgrid(range(L), range(L))
        return ninj_mat


    #===========================================================================
    # checkerboard pattern in 2-body density
    #===========================================================================
    def checkerboard_2bd(self, bool_save=True):
        """"Calculate $\sum_{ij}(-1)^{i-j}(1-2\hat{n_i})(1-2\hat{n_j})$ """
        time, lists_ninj, label_list = self.lists_ninj(bool_save)
        _, num_op_mat = self.num_op_mat()


        checkerboard_list = []
        for t, _ in enumerate(time):
            ni = num_op_mat[:, t]
            ninj = np.zeros((self.L, self.L))
            for i, lab in enumerate(label_list):
                k, l = int(lab.split()[0]) - 1, int(lab.split()[1]) - 1
                ninj[k, l] = lists_ninj[i][t]

            check_val = self.get_2b_checkerboard(ni, ninj)

            checkerboard_list.append(check_val)

        return checkerboard_list


    #===========================================================================
    def E0_subspace_overlap(self):
        """Cost function from the optimization routine"""

        assert self.U == 0.0 and self.theta == 0.0

        evecs_E0 = self.get_eigenstates_E0()


        cost_list = []
        for psi in self.psi_t():
            x = np.linalg.lstsq(evecs_E0.T, psi, rcond=None)[0]
            vec_min = np.abs((evecs_E0.T).dot(x) - psi)
            length_vec = np.abs(np.vdot(vec_min, vec_min))**2
            cost = length_vec
            cost_list.append(cost)

        return cost_list



    #===========================================================================
    # momentum operator
    #===========================================================================
    # def op_bibj(self, site_i, site_j):
    #     "psi_t = <psi(t)| b_i^\dagger b^j |psi(t)>"
    #     path_dict = self.path_prop/f'dict_bitbj_{self.dum_Tf_dt}.npz'
    #     obs_name = f'site_{site_i}_{site_j}'
    #     if path_dict.is_file():
    #         obs_dict = dict(np.load(path_dict))
    #         if obs_name in obs_dict.keys():
    #             return obs_dict[obs_name]
    #     else:
    #         obs_dict = {}
    #
    #     bibj_mat = operators.get_bibj_mat(
    #         self.basis.basis_list, site_i, site_j)
    #     obs_list = []
    #     for psi_tstep in self.psi_t():
    #         obs_list.append(np.vdot(psi_tstep, bibj_mat.dot(psi_tstep)))
    #
    #     obs = np.array(obs_list)
    #
    #     obs_dict[obs_name] = obs
    #     obs_dict[f'time'] = self.time
    #     np.savez(path_dict, **obs_dict)
    #     return obs


    # def get_k_momentum(self, k_mom):
    #     momentum = 0
    #     # for m in range(1, self.L+1):
    #     #     for n in range(1, self.L+1):
    #     print(k_mom)
    #     for m in range(0, self.L):
    #         for n in range(0, self.L):
    #             corr_mn = self.op_bibj(m-1, n-1)
    #             momentum += np.exp(1j*k_mom*(m - n))*corr_mn
    #     momentum = np.array(momentum)/self.L
    #
    #     if 1e-8 < np.max(momentum.imag):
    #         raise ValueError('momentum has an imaginary part > 1e-8! --raise')
    #
    #     return momentum.real


    # def momentum_distribution(self):
    #     k_range = np.linspace(-np.pi*1, np.pi*1, 100)
    #     mom_mat = np.array([self.get_k_momentum(k) for k in k_range])
    #     return k_range, mom_mat

    def momentum_distribution_krange(self, nq_list):

        assert self.bc == 'open'

        # get correlator
        bibj_mats = operators.get_bibj_mats(self.basis.basis_list, self.L)
        corr_mat = np.zeros(
            (self.L, self.L, self.time.shape[0]),
            dtype=complex
        )
        for m in range(0, self.L):
            for n in range(0, self.L):
                for t, psi_tstep in enumerate(self.psi_t()):
                    corr_mat[m, n, t] = np.vdot(
                        psi_tstep,
                        bibj_mats[m, n, :, :].dot(psi_tstep)
                    )


        q_range = nq_list*np.pi/(self.L + 1)

        mom_mat = []
        for q in q_range:
            momentum = 0
            for m in range(0, self.L):
                for n in range(0, self.L):
                    momentum += np.sin(m*q)*np.sin(n*q)*corr_mat[m, n, :]

            momentum = 2/(self.L + 1)*np.array(momentum)

            if 1e-8 < np.max(momentum.imag):
                raise ValueError('momentum has an imaginary part > 1e-8! --raise')
            mom_mat.append(momentum.real)


        return q_range, np.array(mom_mat)


    def momentum_distribution_cont(self):
        nq_list = np.linspace(-10, 10, 100)
        return self.momentum_distribution_krange(nq_list)


    def momentum_distribution_discrete(self):
        nq_list = np.array(range(-self.L, self.L+1))
        return self.momentum_distribution_krange(nq_list)



    def get_bjbibibj_exp(self):
        """Calculate C_ij = < b_j^dagger b_i^dagger b_i b_j > and return
        the time-dependent C_ij-matrix."""

        bjbibibj_mats = operators.get_bjbibibj_mats(self.basis.basis_list, self.L)

        corr_mat = np.zeros(
            (self.L, self.L, self.time.shape[0]),
            dtype=complex
        )
        for m in range(0, self.L):
            for n in range(0, self.L):
                for t, psi_tstep in enumerate(self.psi_t()):

                    corr_mat[m, n, t] = np.vdot(
                        psi_tstep,
                        bjbibibj_mats[m, n, :, :].dot(psi_tstep)
                    )


        assert np.max(np.abs(corr_mat.imag)) < 1e-8

        return corr_mat.real


    def get_bjbibibj_time(self, t_instance):
        corr_mat = self.get_bjbibibj_exp()
        time_ind = np.argmin(np.abs(self.time - t_instance))
        return corr_mat[:, :, time_ind]

    #===========================================================================
    # rho A
    #===========================================================================
    def rho_A(self, subsysA_ind):
        """Calculate the reduced density operator of subsystem A by tracing out
        the dofs of subsystem B.

        rho_A = tr_B(|psi(t)><psi(t)|)
              = sum_basisB <b_n| (|3,1,0,0><1,2,1,0| + ...) |b_n>
        """
        N = self.N
        L = self.L
        basis_list = self.basis.basis_list
        subsysB_ind = [el for el in range(L) if el not in subsysA_ind]

        # basis_B = [[0, 0],...,[0, N], [1, 0],...,[1, N-1], ..., [N, 0]]
        basis_A = [np.array(list(el)) for el in itertools.product(
            range(N+1), repeat=len(subsysA_ind)) if sum(el) <= N]
        basis_B = [np.array(list(el)) for el in itertools.product(
            range(N+1), repeat=len(subsysB_ind)) if sum(el) <= N]

        rhoA_t = []
        for t, psi_tstep in zip(self.time, self.psi_t()):  # loop over all times
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
                        if (bB_i==basisB_k).all() and (bB_i==basisB_l).all():
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
            eval, _ = np.linalg.eig(rhoA)
            # verify that all eigenvalues have no imaginary contribution
            assert all([el.imag < 1e-8 for el in eval])
            # entropy = -np.trace(np.matmul(rhoA, linalg.logm(rhoA)))
            entropy = sum([-el*np.log(el) for el in eval.real if 1e-8 < abs(el)])
            s_ent.append(entropy)
        return np.array(s_ent)


    def bipartite_ent(self):
        path_sentA_dict = self.path_prop/f'dict_SentA_{self.dum_Tf_dt}.npz'
        if path_sentA_dict.is_file():
            SentA_dict = dict(np.load(path_sentA_dict))
            return SentA_dict['SentA']

        N = self.N
        L = self.L
        lattice_ind = list(range(L//2))
        entropy = self._vN_entropy(lattice_ind)
        np.savez(path_sentA_dict, SentA=entropy, time=self.time)
        return entropy


    def plot_bipartite_ent(self, fig_name='S_ent_A.png'):
        fig, ax = plt.subplots()
        plt.plot(self.time, self.bipartite_ent())
        ax.tick_params(labelsize=12)
        ax.set_xlabel('time', fontsize=14)
        ax.set_ylabel(r"$S_A(t)$", fontsize=14)
        # plt.show()
        path_fig = (self.path_prop/fig_name).resolve()
        plt.savefig(path_fig)
        plt.close()
        if self.bool_convert_trim:
            subprocess.call(['convert', path_fig, '-trim', path_fig])


    #===========================================================================
    # number state projection
    #===========================================================================
    def nstate_projection(self):
        """Get number state list: [2, 0, 0, 2]
        Get the overlap with psi_prop, i.e. return the time-dependent
        coefficient corresponding to this state"""

        nstate_mat = np.abs(self.psi_t())**2

        count_nstate = len([el for el in nstate_mat.T if np.max(np.abs(el))>1e-10])
        print('number of number states which are nonzero', count_nstate)

        return nstate_mat


    def nstate_SVN(self):
        path_dict = self.path_prop/f'dict_nstate_SVN_{self.dum_Tf_dt}.npz'
        if path_dict.is_file():
            obs_dict = dict(np.load(path_dict))
            return obs_dict['svn'], obs_dict['svn_max_val']
        else:
            obs_dict = {}
        nstate_mat = self.nstate_projection()

        svn = []
        for i in range(nstate_mat.shape[0]):
            svn.append(sum([-el*np.log(el) for el in nstate_mat[i, :]
                if el > 0]))
        svn = np.array(svn)

        svn_max_val = np.log(nstate_mat.shape[1])

        obs_dict['time'] = self.time
        obs_dict['svn'] = svn
        obs_dict['svn_max_val'] = svn_max_val
        np.savez(path_dict, **obs_dict)

        return svn, svn_max_val


    def nstate_SVN_horizontal_fit(self, tmin=30, tmax=None):
        svn, svn_max_val = self.nstate_SVN()

        def hline(t, a):
            return a

        tind_min = np.argmin(np.abs(self.time - tmin))
        if tmax is not None:  # take last time step
            tind_max = np.argmin(np.abs(self.time - tmax))
        else:
            tind_max = None

        svn_cut = svn[tind_min:tind_max]
        popt, pcov = scipy.optimize.curve_fit(
            f=hline,
            xdata=self.time[tind_min:tind_max],
            ydata=svn_cut,
            p0=svn_max_val)
        svn_fit = popt[0]
        # svn_err = np.sqrt(np.sum(np.abs(svn_cut - svn_fit)**2)/svn_cut.shape[0])
        svn_err = np.sum(np.abs(svn_cut - svn_fit)**2)/svn_cut.shape[0]

        return svn_fit, svn_err, svn_max_val


    #===========================================================================
    # projection on eigenstates
    #===========================================================================
    def eigenstate_projection(self):
        "Project on eigenstates of the Hamiltonian"

        #-----------------------------------------------------------------------
        eig_proj = []
        for t, psi_tstep in zip(self.time, self.psi_t()):

            eig_proj_t = []
            energy = 0
            for eval, evec in zip(self.evals(), self.evecs()):
                expV = np.abs(np.vdot(evec, psi_tstep))**2
                eig_proj_t.append(expV)
                energy += expV*eval
            eig_proj.append(eig_proj_t)
            # H_exp = np.vdot(psi_tstep, self.hamilt().dot(psi_tstep))
            # ana_list.append(np.cos(2*t)*np.sin(2*t))
            # assert np.abs(H_exp.imag) < 1e-10
            # H_exp = H_exp.real
            # print(energy, np.abs(energy - H_exp))
            # energy_list.append(H_exp)
        eig_proj = np.array(eig_proj)

        return eig_proj



    #===========================================================================
    # symmetry operators
    #===========================================================================
    def K_operator(self):
        'Expectation value of K-operator (see operators)'

        path_dict = self.path_prop/f'dict_K_operator_{self.dum_Tf_dt}.npz'
        # if path_dict.is_file():
        #     obs_dict = dict(np.load(path_dict))
        #     time = obs_dict['time']
        #     if 'K_pop' in obs_dict.keys():
        #         return time, obs_dict['K_op']
        # else:
        #     obs_dict = {}

        K_mat, K_evals, K_evecs = self.get_K_mat()

        # Kdagger_HK = K_mat@(self.hamilt()@(np.conjugate(K_mat.T)))
        # print('K^dagger H K = H :', np.allclose(Kdagger_HK, self.hamilt()))
        # False expected

        K_exp_list = []
        eig_proj = []
        for t, psi_tstep in zip(self.time, self.psi_t()):

            # expectation value
            expV = np.vdot(psi_tstep, K_mat.dot(np.conjugate(psi_tstep)))
            K_exp_list.append(expV)

            if self.L == 2 and self.N == 2:
                # assume psi0 = |1, 1>
                ana_res = np.cos(4*t)
                print(np.allclose(ana_res, expV))

            # overlap of psi_t with eigenstates of K
            eig_proj_t = []
            for eval, evec in zip(K_evals, K_evecs):
                expV = np.abs(np.vdot(evec, psi_tstep))**2
                eig_proj_t.append(expV)
            eig_proj.append(eig_proj_t)
        eig_proj = np.array(eig_proj)
        K_exp = np.array(K_exp_list)

        n_pop_eig = len([el for el in eig_proj.T if np.max(np.abs(el)) > 1e-10])
        print(f'# eigenstates of K which overlap at least once with psi(t): {n_pop_eig}')
        print(f'upper bound for entanglement: {np.log(n_pop_eig)}')


        # x, y = np.meshgrid(range(eig_proj.shape[1]), self.time)
        # import matplotlib.colors as colors
        # im = plt.pcolormesh(x, y, eig_proj, shading='nearest', norm = colors.LogNorm(vmin=1e-3, vmax=1), cmap='Greys')
        # plt.colorbar(im)
        # plt.show()


        K_exp_angle = np.array([np.pi if np.abs(np.abs(el) - np.pi) < 1e-8
            else el for el in np.angle(K_exp)])

        # cmplx_angle_set, counts = self.K_eigvals_polar_coord()


        # plt.plot(self.time, np.abs(K_exp))
        # plt.hlines(cmplx_angle_set, xmin=0, xmax=50, color='gray')
        # plt.scatter(self.time, K_exp_angle)
        # plt.show()
        # exit()
        #
        # assert np.max(np.abs(obs.imag)) < 1e-8
        # obs = obs.real

        # obs_dict['K_op'] = obs
        # obs_dict[f'time'] = self.time
        # np.savez(path_dict, **obs_dict)
        return K_exp, K_exp_angle


    #---------------------------------------------------------------------------
    def pair_operator(self):
        path_dict = self.path_prop/f'dict_pair_op_{self.dum_Tf_dt}.npz'
        if path_dict.is_file():
            obs_dict = dict(np.load(path_dict))
            return obs_dict['pair']
        else:
            obs_dict = {}


        pair_mat, pair_evals, pair_evecs = self.get_pair_mat()

        pair_exp_list = []
        # eig_proj = []
        for t, psi_tstep in zip(self.time, self.psi_t()):

            expV = np.vdot(psi_tstep, pair_mat.dot(psi_tstep))
            pair_exp_list.append(expV)

            # eig_proj_t = []
            # for eval, evec in zip(pair_evals, pair_evecs):
            #     proj = np.abs(np.vdot(evec, psi_tstep))**2
            #     eig_proj_t.append(proj)
            # eig_proj.append(eig_proj_t)

        # eig_proj = np.array(eig_proj)
        # print(len([el for el in eig_proj[0] if el > 1e-10]))
        # print(len([el for el in eig_proj[-1] if el > 1e-10]))
        pair_expV = np.array(pair_exp_list)

        if np.max(np.abs(pair_expV.imag)) < 1e-8:
            pair_expV = pair_expV.real
        else:
            raise ValueError

        obs_dict['time'] = self.time
        obs_dict['pair'] = pair_expV
        np.savez(path_dict, **obs_dict)
        return pair_expV


    def pair_op_horizontal_fit(self, tmin=30, tmax=None):

        pair_expV = self.pair_operator()

        def hline(t, a):
            return a


        tind_min = np.argmin(np.abs(self.time - tmin))
        if tmax is not None:  # take last time step
            tind_max = np.argmin(np.abs(self.time - tmax))
        else:
            tind_max = None

        pair_cut = pair_expV[tind_min:tind_max]
        popt, pcov = scipy.optimize.curve_fit(
            f=hline,
            xdata=self.time[tind_min:tind_max],
            ydata=pair_cut,
            p0=2)
        pair_fit = popt[0]
        pair_err = np.sum(np.abs(pair_cut - pair_fit)**2)/pair_cut.shape[0]

        return pair_expV, pair_fit, pair_err
