import os
import itertools
import shutil
import math
import time
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np

from hamiltonian.solve_hamilt import AnyonHubbardHamiltonian
from helper import operators
from propagation import initialize_prop



#===============================================================================
class Propagation(AnyonHubbardHamiltonian):
    """Propagation of a given initial state.

    |psi(t)> = sum_n exp(-i*E_n*t) |n> <n|psi_0>
    n> : eigenvectors
    """

    def __init__(self, path_basis, L, N, J, U, theta, psi0_str, Tprop, dtprop):
        """Initialize parameters.

        Args:
            path_basis (Path): path where hamilt eigenspectrum is stored
            L (int): number of lattice sites
            N (int): number of atoms
            J (float): hopping constant
            U (float): on-site interaction
            theta (float): statistical angle

            psi_ini (str): specifies initial state for propagation
            Tprop (float): final time of propagation
            dtprop (float): time steps of propagation

        """

        super().__init__(path_basis, L, N, J, U, theta)

        self.path_prop = path_basis/psi0_str
        self.path_prop.mkdir(parents=True, exist_ok=True)

        self.time = np.arange(0, Tprop+dtprop, dtprop)
        self.Tprop = Tprop
        # get initial state
        self.psi0, nstate0_str = initialize_prop.get_psi0(
            psi0_str, self.basis.basis_list, self.evecs)

        # K_mat, K_evals, K_evecs = self.get_K_mat()
        # self.psi0 = K_evecs[-1]
        #-----------------------------------------------------------------------
        path_psi_npz = self.path_prop/f'psi_Tf_{Tprop}.npz'
        # if False:
        if (path_psi_npz.is_file() and
            np.load(path_psi_npz)['Tprop'] == Tprop and
            np.load(path_psi_npz)['dtprop'] == dtprop):
            self.time = np.load(path_psi_npz)['time']
            self.psi_t = np.load(path_psi_npz)['psi_t']
        else:
            psi_t = []
            for t in self.time:
                # print(f't = {t:.1f}')
                psi_tstep = np.zeros(self.basis.length, dtype=complex)
                for eval_n, evec_n in zip(self.evals, self.evecs):

                    psi_tstep += (
                        np.exp(-1j*eval_n*t)
                        *evec_n*np.vdot(evec_n, self.psi0)
                    )

                psi_t.append(psi_tstep)
                assert np.abs(np.vdot(psi_tstep, psi_tstep) - 1+0j) < 1e-8

            self.psi_t = np.array(psi_t)

            np.savez(path_psi_npz, time=self.time, psi_t=self.psi_t,
                     psi0=self.psi0, psi0_str=psi0_str, Tprop=Tprop,
                     dtprop=dtprop, L=L, N=N, J=J, U=U, theta=theta,
                     basis_list=self.basis.basis_list)

            initialize_prop.write_log_file(self.path_prop, L, N, J, U, theta,
                                           psi0_str, nstate0_str, Tprop, dtprop)



    #===========================================================================
    # number operator
    #===========================================================================
    def num_op_site(self, site_i):
        "psi_t = <psi(t)| b_i^t b^i |psi(t)>"
        path_nop_dict = self.path_prop/f'dict_numOp_Tf_{self.Tprop}.npz'
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
        for psi_tstep in self.psi_t:
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
        return time, expV_list


    def get_nop_time(self, t_instance):
        time, expV_list = self.num_op_mat()
        time_ind = np.argmin(np.abs(time - t_instance))
        return np.array(expV_list)[:, time_ind]


    #===========================================================================
    # natural population
    #===========================================================================
    def get_natpop(self):
        "<b_i^\dagger b_j>"

        natpop = []
        natorbs = []
        for psi_tstep in self.psi_t:
            bibj_mat = np.zeros((self.L, self.L), dtype=complex)
            for i in range(self.L):
                for j in range(self.L):
                    bibj_mat[i, j] = operators.get_bibj_correlator(
                        psi_tstep, self.basis.basis_list, i, j)

            assert np.allclose(bibj_mat, np.conjugate(bibj_mat.T))
            eval, evec = np.linalg.eigh(self.hamilt)
            idx = eval.argsort()
            evals = eval[idx]
            evecs = (evec[:, idx]).T

            natpop.append(evals)
            natorbs.append(evecs)

        return np.array(natpop), np.array(natorbs)


    #===========================================================================
    # two-site density <n_i n_j>
    #===========================================================================
    def num_op_ninj(self, site_i, site_j):
        "psi_t = <psi(t)| n_j n^i |psi(t)>"
        path_dict = self.path_prop/f'dict_ninj_Tf_{self.Tprop}.npz'
        obs_name = f'site_{site_i}_{site_j}'
        if path_dict.is_file():
            obs_dict = dict(np.load(path_dict))
            time = obs_dict['time']
            if obs_name in obs_dict.keys():
                return time, obs_dict[obs_name]
        else:
            obs_dict = {}

        basis_list = self.basis.basis_list

        obs_list = []
        for psi_tstep in self.psi_t:
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
        np.savez(path_dict, **obs_dict)
        return self.time, obs


    def lists_ninj(self):
        """Note that <n_i n_j> is almost the same"""

        expV_list, label_list = [], []
        for i in range(self.L):
            for j in range(self.L):
                time, expV = self.num_op_ninj(i, j)
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
    # momentum operator
    #===========================================================================
    def op_bibj(self, site_i, site_j):
        "psi_t = <psi(t)| b_i^\dagger b^j |psi(t)>"
        path_dict = self.path_prop/f'dict_bitbj_Tf_{self.Tprop}.npz'
        obs_name = f'site_{site_i}_{site_j}'
        if False:# path_dict.is_file():
            obs_dict = dict(np.load(path_dict))
            time = obs_dict['time']
            if obs_name in obs_dict.keys():
                return time, obs_dict[obs_name]
        else:
            obs_dict = {}

        obs_list = []
        for psi_tstep in self.psi_t:
            obs_list.append(operators.get_bibj_correlator(
                psi_tstep, self.basis.basis_list, site_i, site_j))

        obs = np.array(obs_list)

        obs_dict[obs_name] = obs
        obs_dict[f'time'] = self.time
        np.savez(path_dict, **obs_dict)
        return self.time, obs


    def get_k_momentum(self, k_mom):
        momentum = 0
        # for m in range(1, self.L+1):
        #     for n in range(1, self.L+1):
        for m in range(0, self.L):
            for n in range(0, self.L):
                corr_mn = self.op_bibj(m-1, n-1)[1]
                momentum += np.exp(1j*k_mom*(m - n))*corr_mn
        momentum = np.array(momentum)/self.L

        if 1e-8 < np.max(momentum.imag):
            raise ValueError('momentum has an imaginary part > 1e-8! --raise')

        return momentum.real


    def momentum_distribution(self):
        k_range = np.linspace(-np.pi*1, np.pi*1, 100)
        mom_mat = np.array([self.get_k_momentum(k) for k in k_range])
        return k_range, mom_mat


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
        path_sentA_dict = Path(self.path_prop/'dict_SentA.npz')
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

        nstate_mat = np.abs(self.psi_t)**2

        count_nstate = len([el for el in nstate_mat.T if np.max(np.abs(el))>1e-10])
        print('number of number states which are nonzero', count_nstate)

        return nstate_mat


    def nstate_SVN(self):
        nstate_mat = self.nstate_projection()
        svn = []
        for i in range(nstate_mat.shape[0]):
            svn.append(sum([-el*np.log(el) for el in nstate_mat[i, :]
                if el > 0]))
        svn = np.array(svn)

        svn_max = np.log(nstate_mat.shape[1])

        return svn, svn_max


    #===========================================================================
    # projection on eigenstates
    #===========================================================================
    def eigenstate_projection(self):
        "Project on eigenstates of the Hamiltonian"

        #-----------------------------------------------------------------------
        eig_proj = []
        for t, psi_tstep in zip(self.time, self.psi_t):

            eig_proj_t = []
            energy = 0
            for eval, evec in zip(self.evals, self.evecs):
                expV = np.abs(np.vdot(evec, psi_tstep))**2
                eig_proj_t.append(expV)
                energy += expV*eval
            eig_proj.append(eig_proj_t)
            # H_exp = np.vdot(psi_tstep, self.hamilt.dot(psi_tstep))
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

        path_dict = self.path_prop/f'dict_K_operator_Tf_{self.Tprop}.npz'
        if path_dict.is_file():
            obs_dict = dict(np.load(path_dict))
            time = obs_dict['time']
            if 'K_pop' in obs_dict.keys():
                return time, obs_dict['K_op']
        else:
            obs_dict = {}

        K_mat, K_evals, K_evecs = self.get_K_mat()
        KHK = (K_mat.T)@(np.conjugate(self.hamilt@K_mat))
        print('K^T H K = H :', np.allclose(KHK, self.hamilt))

        K_exp_list = []
        eig_proj = []
        for t, psi_tstep in zip(self.time, self.psi_t):

            eig_proj_t = []
            energy = 0
            for eval, evec in zip(K_evals, K_evecs):
                expV = np.abs(np.vdot(evec, psi_tstep))**2
                eig_proj_t.append(expV)
                energy += expV*eval
            eig_proj.append(eig_proj_t)
        eig_proj = np.array(eig_proj)

            # K_expV = np.vdot(psi_tstep, K_mat.dot(np.conjugate(psi_tstep)))
            # assert np.abs(K_expV.imag) < 1e-10
            # H_exp = H_exp.real
            # print(energy, np.abs(energy - K_expV))
            # K_exp_list.append(K_expV)

        # K_exp = np.array(K_exp_list)
        # K_exp_angle = np.array([np.pi if np.abs(np.abs(el) - np.pi) < 1e-8
        #     else el for el in np.angle(K_expV)])


        # eigstate_mat = self.eigenstate_projection()
        # n_pop_eig = len([el for el in eigstate_mat[0] if el > 1e-10])
        # svn_predict = np.log(n_pop_eig)
        # print(n_pop_eig, svn_predict)

        n_pop_eig0 = len([el for el in eig_proj[0] if el > 1e-10])
        n_pop_eig = len([el for el in eig_proj.T if np.max(np.abs(el)) > 1e-10])
        print(n_pop_eig0)
        print(n_pop_eig)
        print(np.log(n_pop_eig))
        print(np.log(n_pop_eig))



        x, y = np.meshgrid(range(eig_proj.shape[1]), self.time)
        import matplotlib.colors as colors
        im = plt.pcolormesh(x, y, eig_proj, shading='nearest', norm = colors.LogNorm(vmin=1e-3, vmax=1), cmap='Greys')
        plt.colorbar(im)
        plt.show()

        # count = 0
        # for K_evec in K_evecs:
        #     ovlp_
        #     if np.vdot(self.psi0, K_evec) > 1e-6:
        #         count += 1

        print('number of eigenstates of K which overlap with psi0', count)



        K_exp_list = []
        # ana_list = []
        for t, psi_tstep in zip(self.time, self.psi_t):
            expV = np.vdot(psi_tstep, K_mat.dot(np.conjugate(psi_tstep)))
            K_exp_list.append(expV)
            # ana_list.append(np.cos(2*t)**2 - np.sin(2*t)**2)
        K_exp = np.array(K_exp_list)
        K_exp_angle = np.array([np.pi if np.abs(np.abs(el) - np.pi) < 1e-8
            else el for el in np.angle(K_exp)])


        cmplx_angle_set, counts = self.K_eigvals_polar_coord()


        plt.plot(self.time, np.abs(K_exp))
        plt.hlines(cmplx_angle_set, xmin=0, xmax=50, color='gray')
        plt.scatter(self.time, K_exp_angle)
        plt.show()
        exit()

        assert np.max(np.abs(obs.imag)) < 1e-8
        obs = obs.real

        obs_dict['K_op'] = obs
        obs_dict[f'time'] = self.time
        np.savez(path_dict, **obs_dict)
        return self.time, obs


    #---------------------------------------------------------------------------
    def K_operator_K_dagger(self):
        """\mathcal{K} + \mathcal{K}^\dagger
        """


        K_mat, K_evals, K_evecs = self.get_K_mat()
        Kt_mat, Kt_evals, Kt_evecs = self.get_K_dagger_mat()


        K_exp_list = []
        for t, psi_tstep in zip(self.time, self.psi_t):

            expV_a = np.vdot(psi_tstep, K_mat.dot(np.conjugate(psi_tstep)))
            expV_b = np.dot(psi_tstep, Kt_mat.dot(psi_tstep))

            K_exp_list.append(expV_a + expV_b)


        K_exp = np.array(K_exp_list)
        if np.max(np.abs(K_exp.imag)) < 1e-8:
            K_exp = K_exp.real
        else:
            raise ValueError

        return K_exp

    #---------------------------------------------------------------------------
    def pair_operator(self):

        pair_mat, pair_evals, pair_evecs = self.get_pair_mat()

        pair_exp_list = []
        # eig_proj = []
        for t, psi_tstep in zip(self.time, self.psi_t):

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
        pair_exp = np.array(pair_exp_list)

        if np.max(np.abs(pair_exp.imag)) < 1e-8:
            pair_exp = pair_exp.real
        else:
            raise ValueError

        return pair_exp
