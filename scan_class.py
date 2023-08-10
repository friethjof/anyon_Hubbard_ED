import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from hamiltonian.solve_hamilt import AnyonHubbardHamiltonian
from propagation.propagate import Propagation

import path_dirs
from helper import other_tools
from helper import plot_helper
from helper import operators


class ScanClass():
    """Organize exact diagonalization method, plotting, scaning of paramters.
    """

    def __init__(self, N=None, L=None, J=1, U=None, theta=None,
                 psi0_str=None, Tprop=None, dtprop=None):
        """Initialize parameters.

        Args:
            L (int): number of lattice sites
            N (int): number of atoms
            J (float): hopping constant
            U (float): on-site interaction
            theta (float): statistical angle

            psi0 (str): specifies initial state for propagation
            Tprop (float): final time of propagation
            dtprop (float): time steps of propagation
        """

        self.L = L
        self.N = N
        self.U = U
        self.J = J  # is assumed to be always 1
        self.theta = theta

        self.psi0_str = psi0_str
        self.Tprop = Tprop
        self.dtprop = dtprop




    #===========================================================================
    # ground state analysis
    #===========================================================================
    def gs_vary_U_theta(self, U_list, theta_list, obs_name):
        """Vary the parameters U and theta and plot a given observable

        Args:
            U_list (list): List of U values.
            theta_list (list): List of theta values.
            obs_name (str): specifies the observable

        Returns:
            None
        """

        assert self.L is not None
        assert self.N is not None

        for U in U_list:
            for theta in theta_list:

                #---------------------------------------------------------------
                # solve hamilt
                path_basis = other_tools.get_path_basis(
                    path_dirs.path_data_top, self.L, self.N, U, theta)

                hamilt_class = AnyonHubbardHamiltonian(
                    path_basis=path_basis,
                    L=self.L,
                    N=self.N,
                    J=self.J,
                    U=U,
                    theta=theta
                )

                #---------------------------------------------------------------
                # get figure parameters
                path_fig = path_dirs.path_fig_top/f'L{self.L}_N{self.N}/gs_{obs_name}'
                theta_ = f'{theta/math.pi:.3f}'.replace('.', '_')
                U_ = f'{U:.3f}'.replace('.', '_')
                fig_name = path_fig/f'U_{U_}_thpi_{theta_}_{obs_name}'
                title = plot_helper.make_title(self.L, self.N, U, theta)


                #---------------------------------------------------------------
                if obs_name == 'eigen_spectrum':
                    evals = hamilt_class.energy_spectrum()
                    # np.savez('evals_test.npz', evals=evals)
                    plot_helper.plot_lines(
                        fig_name=fig_name,
                        x_lists=[range(evals.shape[0])],
                        y_lists=[evals],
                        fig_args={'xlabel':'state', 'ylabel':'energy',
                                  'title':title}
                    )

                #---------------------------------------------------------------
                if obs_name == 'eigenstate_E0':
                    # plot eigenstates corresponding to degeneracy
                    evevs_E0 = hamilt_class.get_eigenstate_nOp_E0()

                    plot_helper.plot_lines(
                        fig_name=fig_name,
                        x_lists=[range(1, self.L+1)],
                        y_lists=evevs_E0,
                        fig_args={'xlabel':'site i', 'ylabel':
                                  r'$\langle \phi | \hat{n}_i|\phi\rangle$',
                                  'title':title}
                    )

                #---------------------------------------------------------------
                if obs_name == 'K_operator':
                    #-----------------------------------------------------------
                    # fig_name1 = fig_name.parent/(fig_name.name + '_eigVal_hist')
                    # cmplx_ang_set, counts = hamilt_class.K_eigvals_polar_coord()
                    # cmplx_ang_set = cmplx_ang_set/np.pi
                    # plot_helper.plot_histogram(
                    #     fig_name1, cmplx_ang_set, counts,
                    #     fig_args={'xlabel':r'$\phi/\pi$', 'ylabel':'counts',
                    #               'title':title})

                    #-----------------------------------------------------------
                    # fig_name2 = fig_name.parent/(fig_name.name + '_mat')
                    # K_mat, _, _ = hamilt_class.get_K_mat()
                    # K_mat[np.abs(K_mat) == 0] = np.nan
                    # basis_range = range(hamilt_class.basis.length +1)
                    # plot_helper.make_cplot(
                    #     fig_name2, basis_range, basis_range, np.angle(K_mat).T,
                    #     fig_args={'ylabel':r'$\langle\vec{n}_i|$',
                    #               'xlabel':r'$|\vec{n}_j\rangle$',
                    #               'title':title, 'shading':'flat',
                    #               'clabel':r'$arg(K)$', 'cmap':''})

                    #-----------------------------------------------------------
                    # fig_name3 = fig_name.parent/(fig_name.name + '_mat_hist')
                    # cmplx_ang_set, counts = hamilt_class.K_mat_polar_coord(
                    #     nstate_str='20000002')
                    # plot_helper.plot_histogram(
                    #     fig_name3, cmplx_ang_set, counts,
                    #     fig_args={'xlabel':r'$\phi/\pi$', 'ylabel':'counts',
                    #               'title':title})


                    K_bloack_diag = hamilt_class.H_in_K_basis()






    #===========================================================================
    # propagation analysis
    #===========================================================================
    def prop_vary_U_theta(self, U_list, theta_list, obs_name, arg_dict=None):
        """Vary the parameters U and theta and plot a given observable

        Args:
            U_list (list): List of U values.
            theta_list (list): List of theta values.
            obs_name (str): specifies the observable
            arg_dict (None|dict): optional input arguments

        Returns:
            None
        """

        assert self.L is not None
        assert self.N is not None
        assert self.psi0_str is not None
        assert self.Tprop is not None
        assert self.dtprop is not None


        for U in U_list:
            for theta in theta_list:

                #---------------------------------------------------------------
                # get propagation
                path_basis = other_tools.get_path_basis(
                    path_dirs.path_data_top, self.L, self.N, U, theta)

                prop_class = Propagation(
                    path_basis=path_basis,
                    L=self.L,
                    N=self.N,
                    J=self.J,
                    U=U,
                    theta=theta,
                    psi0_str=self.psi0_str,
                    Tprop=self.Tprop,
                    dtprop=self.dtprop
                )


                #---------------------------------------------------------------
                # get paths
                path_fig = (path_dirs.path_fig_top/f'L{self.L}_N{self.N}'/
                            f'{self.psi0_str}_Tf_{self.Tprop}/{obs_name}')
                theta_ = f'{theta/math.pi:.3f}'.replace('.', '_')
                U_ = f'{U:.3f}'.replace('.', '_')
                fig_name = path_fig/f'U_{U_}_thpi_{theta_}_{obs_name}.png'
                title = plot_helper.make_title(self.L, self.N, U, theta,
                                               psi0=self.psi0_str)


                #---------------------------------------------------------------
                if obs_name == 'num_op':
                    time, num_op = prop_class.num_op_mat()
                    plot_helper.num_op_cplot(fig_name, prop_class.time, self.L,
                                             num_op, title)

                #---------------------------------------------------------------
                if obs_name == 'natpop':
                    natpop, _ = prop_class.get_natpop()
                    plot_helper.plot_lines(
                        fig_name=fig_name,
                        x_lists=[prop_class.time],
                        y_lists=natpop.T,
                        fig_args={'xlabel':'time', 'ylabel':'natpop',
                                  'title':title}
                    )


                #---------------------------------------------------------------
                if obs_name == 'num_op_2b':
                    t_list = arg_dict['t_list']
                    mat_list = [prop_class.get_ninj_mat_time(t) for t in t_list]
                    num_op = prop_class.get_nop_time(t_list[-1])

                    print(mat_list[-1])
                    print(np.sum(mat_list[-1], axis=0)/2)
                    print(num_op)

                    for i in range(self.L):
                        for j in range(self.L):
                            print(i+1, j+1, mat_list[-1][i, j] - num_op[i]*num_op[j])
                    exit()
                    # import matplotlib.pyplot as plt
                    # plt.plot(np.sum(mat_list[-1], axis=0)/4)
                    # plt.show()
                    # exit()

                    plot_helper.make_5cplots(
                        fig_name, t_list, self.L, mat_list,
                        fig_args={
                            'xlabel':'site i',
                            'ylabel':'site j',
                            'clabel':r"$\langle \Psi|\hat{n}_i \hat{n}_j| \Psi\rangle$",
                            'title_list':[f't={t}' for t in t_list],
                            'main_title':title
                        }
                    )


                #---------------------------------------------------------------
                if obs_name == 'mom_op':
                    k_range, mom_mat = prop_class.momentum_distribution()
                    print(mom_mat)
                    plot_helper.make_cplot(
                        fig_name, k_range, prop_class.time, mom_mat.T, fig_args={
                        'xlabel':'$k$', 'ylabel':'$t$', 'main_title':title,
                        'clabel':r"$\langle \Psi|\hat{n}_k \hat{n}_j| \Psi\rangle$"}
                    )


                #---------------------------------------------------------------
                if obs_name == 'K_operator':
                    time, K_op = prop_class.K_operator()
                    print(K_op)


                #---------------------------------------------------------------
                if obs_name == 'K_operator_K_dagger':
                    K_sum_expV = prop_class.K_operator_K_dagger()  # prop
                    _, K_sum_evals, _ = prop_class.get_K_K_dagger_mat()  # GS
                    dict_degen = other_tools.find_degeneracies(K_sum_evals)
                    for key, val in dict_degen.items():
                        print(key, len(val))


                    plot_helper.plot_lines(
                        fig_name=fig_name,
                        x_lists=[prop_class.time],
                        y_lists=[K_sum_expV],
                        fig_args={'xlabel':'time', 'ylabel':
                                  r'$\mathcal{K} + \mathcal{K}^\dagger$',
                                  'title':title, 'ylim':[-2.0, 2.0]}
                    )


                #---------------------------------------------------------------
                if obs_name == 'pair_operator':
                    pair_exp = prop_class.pair_operator()  # prop
                    _, pair_evals, _ = prop_class.get_pair_mat()  # GS
                    dict_degen = other_tools.find_degeneracies(pair_evals)
                    for key, val in dict_degen.items():
                        print(key, len(val))
                    exit()

                    plot_helper.plot_lines(
                        fig_name=fig_name,
                        x_lists=[prop_class.time],
                        y_lists=[pair_exp],
                        fig_args={'xlabel':'time', 'ylabel':r'$\nu_p$',
                                  'title':title, 'ylim':[0.5, 3.5]}
                    )



                #---------------------------------------------------------------
                if obs_name == 'nstate':
                    nstate_mat = prop_class.nstate_projection()

                    count = 0
                    for nstate_t in nstate_mat.T:
                        if np.max(nstate_t) > 0.05:
                            count +=1
                    print(count)


                    basis_range = range(prop_class.basis.length)
                    plot_helper.make_cplot(
                        fig_name, basis_range, prop_class.time, nstate_mat,
                        fig_args={
                        'xlabel':r'$|\vec{n}_i\rangle$', 'ylabel':r'$t$',
                        'title':title, 'cmap':'Greys', 'lognorm':'',
                        'clabel':r"$|\langle \vec{n}_i | \Psi(t)\rangle|^2$"}
                    )


                if obs_name == 'nstate_SVN':
                    svn, svn_max = prop_class.nstate_SVN()
                    eigstate_mat = prop_class.eigenstate_projection()
                    n_pop_eig = len([el for el in eigstate_mat[0] if el > 1e-10])
                    svn_predict = np.log(n_pop_eig)
                    # print(n_pop_eig)
                    # print(eigstate_mat.shape)
                    # svn_proj = [sum([(-el*np.log(el)) for el in p_vec if el > 0]) for p_vec in eigstate_mat]
                    # plt.plot(svn_proj)
                    # plt.plot(svn)
                    # plt.show()
                    # plt.close()
                    # print(n_pop_eig, svn_predict)
                    # print(svn_max)
                    plot_helper.plot_lines(
                        fig_name=fig_name,
                        x_lists=[prop_class.time],
                        y_lists=[svn],
                        fig_args={'xlabel':'time', 'ylabel':'SVN',
                                  'title':title, 'hline':svn_max,
                                  'hline2':svn_predict}
                    )




                #---------------------------------------------------------------
                if obs_name == 'eigenstate_projection':
                    eigstate_mat = prop_class.eigenstate_projection()
                    n_pop_eig = len([el for el in eigstate_mat[0] if el > 1e-10])
                    tot = eigstate_mat.shape[1]
                    title += f'\n # populated eigenstates {n_pop_eig} of {tot}'

                    basis_range = range(prop_class.basis.length)
                    plot_helper.make_cplot(
                        fig_name, basis_range, prop_class.time, eigstate_mat,
                        fig_args={
                        'xlabel':r'$\phi_i^{eig}$', 'ylabel':r'$t$',
                        'title':title, 'cmap':'Greys', 'lognorm':'',
                        'clabel':r"$|\langle \phi_i^{eig} |\Psi(t)\rangle|^2$"}
                    )
