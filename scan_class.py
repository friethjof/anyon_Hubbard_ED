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

    def __init__(self, bc=None, N=None, L=None, J=1, U=None, theta=None,
                 psi0_str=None, Tprop=None, dtprop=None):
        """Initialize parameters.

        Args:
            bc (str): boundary conditions of lattice, open (hard-wall) | periodic
            L (int): number of lattice sites
            N (int): number of atoms
            J (float): hopping constant
            U (float): on-site interaction
            theta (float): statistical angle

            psi0 (str): specifies initial state for propagation
            Tprop (float): final time of propagation
            dtprop (float): time steps of propagation
        """

        self.bc = bc
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
    def gs_vary_U_theta(self, U_list, theta_list, obs_name, args_dict=None):
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
                hamilt_class = AnyonHubbardHamiltonian(
                    bc=self.bc,
                    L=self.L,
                    N=self.N,
                    J=self.J,
                    U=U,
                    theta=theta
                )

                #---------------------------------------------------------------
                # get figure parameters
                path_fig = path_dirs.get_path_fig_top(self.bc, self.L, self.N)/(
                                                      f'gs_{obs_name}')
                U_theta_ = path_dirs.get_dum_name_U_theta(U, theta, self.J)
                fig_name = path_fig/f'{U_theta_}_{obs_name}'
                title = plot_helper.make_title(self.L, self.N, U, theta)


                #---------------------------------------------------------------
                if obs_name == 'eigen_spectrum':
                    evals = hamilt_class.evals()
                    print(evals)
                    exit()
                    with open(f'L{self.L}_N{self.N}_{U_theta_}', 'w') as f:
                        for ev in evals:
                            f.write(f'{ev}\n')

                    # np.savez('evals_test.npz', evals=evals)
                    plot_helper.plot_lines(
                        fig_name=fig_name,
                        x_lists=[range(evals.shape[0])],
                        y_lists=[evals],
                        fig_args={'xlabel':'state', 'ylabel':'energy',
                                  'title':title}
                    )


                #---------------------------------------------------------------
                if obs_name == 'eigen_spectrum_mom':
                    krange, e_k_mat = hamilt_class.energy_spectrum_mom()

                    # np.savez('evals_test.npz', evals=evals)
                    plot_helper.plot_lines(
                        fig_name=fig_name,
                        x_lists=[krange],
                        y_lists=e_k_mat,
                        fig_args={'xlabel':'$k$', 'ylabel':'energy',
                                  'title':title}
                    )


                #---------------------------------------------------------------
                # <n_i n_j>
                if obs_name == 'eigenstate_num_op_2b':

                    evals = hamilt_class.evals()
                    evec_2b_list = hamilt_class.eigenstate_nOp_2b()

                    for i, evec_2b in enumerate(evec_2b_list):
                        e_ =f'E_{evals[i]:.3f}'.replace('.', '_')
                        plot_helper.make_cplot(
                            fig_name=path_fig/U_theta_/f'eigenstate{i+1}_{e_}',
                            xarr=np.arange(1, self.L+1),
                            yarr=np.arange(1, self.L+1),
                            mat=evec_2b,
                            fig_args={
                                'title':f'E={evals[i]:.3f}, {title}',
                                'xlabel':'site i',
                                'ylabel':'site j',
                                'clabel':r"$\langle n_i n_j \rangle$",
                            }
                        )


                #---------------------------------------------------------------
                # <b_j^t b_i^t b_i b_j>
                if obs_name == 'eigenstate_corr_2b':

                    evals = hamilt_class.evals()
                    evec_2b_list = hamilt_class.eigenstate_corr_2b()

                    for i, evec_2b in enumerate(evec_2b_list):
                        e_ =f'E_{evals[i]:.3f}'.replace('.', '_')
                        plot_helper.make_cplot(
                            fig_name=path_fig/U_theta_/f'eigenstate{i+1}_{e_}',
                            xarr=np.arange(1, self.L+1),
                            yarr=np.arange(1, self.L+1),
                            mat=evec_2b,
                            fig_args={
                                'title':f'E={evals[i]:.3f}, {title}',
                                'xlabel':'site i',
                                'ylabel':'site j',
                                'clabel':r"$\langle b_j^\dagger b_i^\dagger b_ib_j \rangle$",
                            }
                        )


                #---------------------------------------------------------------
                # E0 eigenstates number op <n_i>
                if obs_name == 'eigenstate_E0':
                    
                    evecs_E0 = hamilt_class.get_eigenstate_nOp_E0()

                    plot_helper.plot_lines(
                        fig_name=fig_name,
                        x_lists=[range(1, self.L+1)],
                        y_lists=evecs_E0,
                        fig_args={'xlabel':'site i', 'ylabel':
                                  r'$\langle \phi | \hat{n}_i|\phi\rangle$',
                                  'title':title}
                    )


                #---------------------------------------------------------------
                # E0 eigenstates in number state basis
                if obs_name == 'eigenstate_E0_nstate':
                    
                    evecs_E0 = hamilt_class.get_eigenstates_E0()
                    nstate_population = np.sum(np.abs(evecs_E0)**2, axis=0)
                    for i in np.argsort(nstate_population):
                        print(hamilt_class.basis.basis_list[i], nstate_population[i])

                    print()
                    U_energy = hamilt_class.onsite_int_nstate_exp()
                    idx = np.argsort(U_energy)
                    plot_helper.plot_scatter(
                        fig_name=fig_name,
                        x_list=list(range(hamilt_class.basis.length)),
                        y_lists=[nstate_population[idx]],
                        fig_args={
                            'xlabel':'nstate ',
                            'ylabel':'overlap',
                            'title':title,
                            'xticklabels':hamilt_class.basis.basis_list[idx]
                        }
                    )


                #---------------------------------------------------------------
                # E0 eigenstates in K-basis
                if obs_name == 'eigenstate_E0_K_basis':
                   
                    K_evals_E0, K_evecs_nOp = hamilt_class.get_eigenstate_nOp_E0_K_basis()

                    plot_helper.plot_lines(
                        fig_name=fig_name,
                        x_lists=[range(1, self.L+1)],
                        y_lists=K_evecs_nOp,
                        label_list=[f'{i+1}, $K eval={el:.2f}$' for i, el in enumerate(K_evals_E0)],
                        fig_args={
                            'xlabel':'site i',
                            'ylabel': r'$\langle \phi | \hat{n}_i|\phi\rangle$',
                            'title':title
                        }
                    )


                #---------------------------------------------------------------
                # eigenstates in 'energy'-basis
                if obs_name == 'eigenstate_E0_energy_basis':
                    nOp_lol, legend_list = hamilt_class.get_eigenstate_nOp_E0_energy_basis()

                    plot_helper.plot_lines(
                        fig_name=fig_name,
                        x_lists=[range(1, self.L+1)],
                        y_lists=nOp_lol,
                        label_list=legend_list,
                        fig_args={
                            'xlabel':'site i',
                            'ylabel': (
                                r'$\langle \{ \tilde{\eta}_1\cdots \tilde{\eta}_6\}'
                                + r'| \hat{n}_i|'
                                + r'\{ \tilde{\eta}_1\cdots \tilde{\eta}_6\} \rangle$'
                            ),
                            'title':title
                        }
                    )


                #---------------------------------------------------------------
                if obs_name == 'eigenstate_E0_perturbation':
                    U_max, steps = args_dict['U_max'], args_dict['steps']
                    order = args_dict['order']
                    U_max_ = f'_U_max_{U_max}_steps_{steps}'.replace('.', '_')
                    fig_name = fig_name.parent/(fig_name.name
                        + f'_U_max_{U_max}_steps_{steps}_order_{order}'
                        .replace('.', '_'))

                    U_range, evals_eff_mat, n0_eff_mat = \
                        hamilt_class.eigenstate_E0_perturbation(U_max, steps,
                                                                order)


                    label_list = [f'$U={round(el, 8)}$' for el in U_range]

                    plot_helper.plot_scatter(
                        fig_name=fig_name.parent/(fig_name.name + '_energies'),
                        x_list=[range(1, evals_eff_mat.shape[1]+1)],
                        y_lists=evals_eff_mat,
                        fig_args={'xlabel':'$i$', 'ylabel': '$E_i$',
                                  'title':title, 'legend_list':label_list}
                    )

                    # plot site occupations
                    for i in range(n0_eff_mat.shape[1]):
                        fig_nOp = fig_name.parent/(fig_name.name + f'_n_op_{i+1}')
                        plot_helper.plot_lines(
                            fig_name=fig_nOp,
                            x_lists=[range(1, self.L+1)],
                            y_lists=n0_eff_mat[:, i, :],
                            label_list=label_list,
                            fig_args={'xlabel':'site i', 'ylabel':
                                      r'$\langle \phi | \hat{n}_i|\phi\rangle$',
                                      'title':title}
                        )


                #---------------------------------------------------------------
                if obs_name == 'nstate_eigenstate_SVN':
                    # plot eigenstates corresponding to degeneracy
                    svn_list = hamilt_class.nstate_eigenstate_SVN()
                    basis_l = hamilt_class.basis.length
                    svn_max = -np.log(1/basis_l)
                    U_energy = hamilt_class.onsite_int_nstate_exp()
                    idx = np.argsort(U_energy)

                    plot_helper.plot_scatter(
                        fig_name=fig_name,
                        x_list=list(range(basis_l)),
                        y_lists=[svn_list[idx]],
                        fig_args={
                            'ylabel':'counts',
                            'ylim':[0, svn_max+0.2],
                            'title':title,
                            'hline':svn_max
                        }
                    )


                #---------------------------------------------------------------
                if obs_name == 'bipartite_SVN':

                    l_list, svn_list = hamilt_class.bipartite_SVN_list_evec0()
                    plot_helper.plot_lines(
                        fig_name=fig_name.parent/(fig_name.name+'_GS'),
                        x_lists=[l_list],
                        y_lists=[svn_list],
                        fig_args={
                            'xlabel':'$l$',
                            'ylabel':'$S^{\mathrm{vN}}$',
                            'title':title
                        }
                    )

                    l_list, svn_mat = hamilt_class.bipartite_SVN_E0_subspace()
                    plot_helper.plot_lines(
                        fig_name=fig_name.parent/(fig_name.name+'_E0_subspace'),
                        x_lists=[l_list],
                        y_lists=svn_mat.T,
                        fig_args={
                            'xlabel':'$l$',
                            'ylabel':'$S^{\mathrm{vN}}$',
                            'title':title
                        }
                    )

                    eval = hamilt_class.evals()
                    l_list, svn_mat = hamilt_class.bipartite_SVN_all_eigstates()
                    plot_helper.make_cplot(
                        fig_name=fig_name.parent/(fig_name.name+'_all_eigstates'),
                        xarr=l_list,
                        yarr=eval,
                        mat=svn_mat.T,
                        fig_args={
                            'xlabel':'$l$',
                            'ylabel':'energy',
                            'title':title,
                            'clabel':'$S^{\mathrm{vN}}$'
                        }
                    )


                #---------------------------------------------------------------
                if obs_name == 'K_operator':
                    #-----------------------------------------------------------
                    fig_name1 = fig_name.parent/(fig_name.name + '_eigVal_hist')
                    cmplx_ang_set, counts = hamilt_class.K_eigvals_polar_coord()
                    cmplx_ang_set = cmplx_ang_set/np.pi
                    plot_helper.plot_scatter(
                        fig_name=fig_name1,
                        x_list=cmplx_ang_set,
                        y_lists=[counts],
                        fig_args={'xlabel':r'$\phi/\pi$',
                                  'ylabel':'counts',
                                  'title':title}
                    )

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


                    # K_bloack_diag = hamilt_class.H_in_K_basis()






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

        assert self.bc is not None
        assert self.L is not None
        assert self.N is not None
        assert self.psi0_str is not None
        assert self.Tprop is not None
        assert self.dtprop is not None


        for U in U_list:
            for theta in theta_list:
                print(f'U={U}, theta/pi={theta/np.pi:.3f}')

                #---------------------------------------------------------------
                # get propagation
                prop_class = Propagation(
                    bc=self.bc,
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
                path_fig = path_dirs.get_path_fig_top_dyn(
                    self.bc, self.L, self.N, self.psi0_str, self.Tprop,
                    self.dtprop)
                U_theta_ = path_dirs.get_dum_name_U_theta(U, theta, self.J)

                fig_name = path_fig/f'{obs_name}'/f'{U_theta_}_{obs_name}'
                title = plot_helper.make_title(self.L, self.N, U, theta,
                                               psi0=self.psi0_str)


                #---------------------------------------------------------------
                # <n_i>(t)
                if obs_name == 'num_op':
                    time, num_op = prop_class.num_op_mat()
                    plot_helper.num_op_cplot(fig_name, prop_class.time, self.L,
                                             num_op, title)


                #---------------------------------------------------------------
                if obs_name == 'num_op_bound_state':
                    num_op_bound, num_op_scatter = prop_class.num_op_bound_state()

                    plot_helper.num_op_2cplot(
                        fig_name=fig_name,
                        time=prop_class.time,
                        L=self.L,
                        mat1=num_op_bound,
                        mat2=num_op_scatter,
                        fig_args={
                            'title1':r'$\langle \Psi_{bound}^{d\leq2} | \hat{n}_i |\Psi_{bound}^{d\leq2} \rangle$',
                            'title2':r'$\langle \Psi_{scatter}^{d>2} | \hat{n}_i |\Psi_{scatter}^{d>2} \rangle$',
                            'main_title':title
                        }
                    )


                #---------------------------------------------------------------
                if obs_name == 'num_op_bound_state_RMS':
                    rms_bound, rms_scatter = prop_class.root_mean_square()
                    res_bound, res_scatter = \
                        prop_class.root_mean_square_linear_fit(t_interval=[1, 4])
                    anno = (
                        f'slope bound $ = {res_bound.slope:.2f}' + r'\pm ' +
                        f'{res_bound.stderr:.3f}$\n' +
                        f'slope scatter $ = {res_scatter.slope:.2f}' + r'\pm ' +
                        f'{res_scatter.stderr:.3f}$'
                    )
                    fit_bound = res_bound.intercept + res_bound.slope*prop_class.time
                    fit_scatter = res_scatter.intercept + res_scatter.slope*prop_class.time

                    plot_helper.plot_lines(
                        fig_name=fig_name,
                        x_lists=[prop_class.time],
                        y_lists=[rms_bound, rms_scatter, fit_bound, fit_scatter],
                        label_list = ['bound', 'scatter', 'fit bound', 'fit scatter'],
                        fig_args={
                            'xlabel':'time',
                            'ylabel':r'RMS size (sites)',
                            'title':title,
                            'xlim':[1, 4],
                            'anno':anno,
                            'style_list':[
                                {'marker':'o', 'color':'tomato'},
                                {'marker':'s', 'color':'cornflowerblue'},
                                {'color':'gray', 'zorder':0},
                                {'color':'gray', 'zorder':1}
                            ]
                        }
                    )


                #---------------------------------------------------------------
                if obs_name == 'natpop':
                    natpop, natorb = prop_class.get_natpop()
                    natorb = [np.abs(orb)**2 for orb in natorb]
                    plot_helper.plot_lines(
                        fig_name=fig_name,
                        x_lists=[prop_class.time],
                        y_lists=natpop,
                        fig_args={'xlabel':'time', 'ylabel':'natpop',
                                  'title':title}
                    )

                    plot_helper.make_4cplots_time(
                        fig_name.parent/(fig_name.name[:-3] + 'orb'),
                        prop_class.time, self.L, natorb[:5],
                        fig_args={
                            'xlabel':'site i',
                            'ylabel':'time',
                            'clabel':r"$\phi^{nat}_j$",
                            'title_list':[f'j={j}' for j in range(1,5)],
                            # 'main_title': title
                        }
                    )


                #---------------------------------------------------------------
                # <n_i n_j>(t)
                if obs_name == 'num_op_2b':
                    t_list = arg_dict['t_list']
                    mat_list = [prop_class.get_ninj_mat_time(t) for t in t_list]
                    num_op = prop_class.get_nop_time(t_list[-1])

                    fig_args={
                        'xlabel':'site i',
                        'ylabel':'site j',
                        'clabel':r"$\langle \Psi|\hat{n}_i \hat{n}_j| \Psi\rangle$",
                    }

                    if len(t_list) == 1:
                        fig_args['title'] = f't={t_list[0]}; ' + title
                        plot_helper.make_cplot(
                            fig_name, np.arange(1, self.L+1),
                            np.arange(1, self.L+1), mat_list[0],
                            fig_args=fig_args
                        )
                    elif len(t_list) == 5:
                        fig_args['title_list'] = [f't={t}' for t in t_list]
                        fig_args['main_title'] = title
                        plot_helper.make_5cplots(
                            fig_name.parent/(fig_name.name + '_5plots'),
                            t_list, self.L, mat_list, fig_args
                        )
                    else:
                        raise NotImplementedError


                #---------------------------------------------------------------
                # <b_j^t b_i^t b_i b_j>(t)
                if obs_name == 'op_2b_corr':
                    t_list = arg_dict['t_list']
                    mat_list = [prop_class.get_bjbibibj_time(t) for t in t_list]
                    num_op = prop_class.get_nop_time(t_list[-1])

                    fig_args={
                        'xlabel':'site i',
                        'ylabel':'site j',
                        'clabel':r"$\langle b_j^\dagger b_i^\dagger b_ib_j \rangle$",
                    }

                    if len(t_list) == 1:
                        fig_args['title'] = f't={t_list[0]}; ' + title
                        plot_helper.make_cplot(
                            fig_name, np.arange(1, self.L+1),
                            np.arange(1, self.L+1), mat_list[0],
                            fig_args=fig_args
                        )
                    elif len(t_list) == 5:
                        fig_args['title_list'] = [f't={t}' for t in t_list]
                        fig_args['main_title'] = title
                        plot_helper.make_5cplots(
                            fig_name.parent/(fig_name.name + '_5plots'),
                            t_list, self.L, mat_list, fig_args
                        )
                    else:
                        raise NotImplementedError


                #---------------------------------------------------------------
                if obs_name == 'momentum_distribution_cont':
                    q_range, mom_mat = prop_class.momentum_distribution_cont()

                    plot_helper.make_cplot(
                        fig_name, q_range/np.pi, prop_class.time, mom_mat.T,
                        fig_args={
                            'xlabel':'$q/\pi$', 'ylabel':'$t$',
                            'title':title,
                            'clabel':r"$\langle \hat{n}_k \rangle$"
                        }
                    )


                #---------------------------------------------------------------
                if obs_name == 'momentum_distribution_discrete':
                    q_range, mom_mat = prop_class.momentum_distribution_discrete()

                    plot_helper.make_cplot(
                        fig_name, q_range/np.pi, prop_class.time, mom_mat.T,
                        fig_args={
                            'xlabel':'$q/\pi$', 'ylabel':'$t$',
                            'title':title,
                            'clabel':r"$\langle \hat{n}_k \rangle$"
                        }
                    )


                #---------------------------------------------------------------
                if obs_name == 'K_operator':
                    # static properties
                    _, K_evals, _ = prop_class.get_K_mat()
                    K_evals_cmplx_angle_set, _ = prop_class.K_eigvals_polar_coord()
                    # dynamic properties
                    K_exp, K_exp_angle  = prop_class.K_operator()
                    K_exp_abs = np.abs(K_exp)


                    #-----------------------------------------------------------
                    plot_helper.polar_corrd(
                        fig_name=fig_name,
                        time=prop_class.time,
                        abs_vals=K_exp_abs,
                        angle_vals=K_exp_angle,
                        fig_args={'xlabel':'time',
                                  'label_dum':r'\langle\mathcal{K}\rangle',
                                  'title':title, 'hline':K_evals_cmplx_angle_set}
                    )

                    #-----------------------------------------------------------
                    fig_name2 = fig_name.parent/(fig_name.name + '_cmplx_pane')
                    plot_helper.complx_plane(
                        fig_name=fig_name2,
                        cmplx_exp=K_exp,
                        fig_args={'obs':r'\mathcal{K}',
                                  'scatter':K_evals,
                                  'unit_circle':True}
                    )


                #---------------------------------------------------------------
                if obs_name == 'pair_operator':
                    pair_exp = prop_class.pair_operator()  # prop
                    _, pair_evals, _ = prop_class.get_pair_mat()  # GS
                    dict_degen = other_tools.find_degeneracies(pair_evals)
                    for key, val in dict_degen.items():
                        print(key, len(val))

                    plot_helper.plot_lines(
                        fig_name=fig_name,
                        x_lists=[prop_class.time],
                        y_lists=[pair_exp],
                        fig_args={'xlabel':'time', 'ylabel':r'$\nu_p$',
                                  'title':title, 'ylim':[0.5, 3.5]}
                    )


                #---------------------------------------------------------------
                # |<n_i|Psi(t)>|^2
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


                if obs_name == 'checkerboard_2bd':
                    check_list = prop_class.checkerboard_2bd()
                    plot_helper.plot_lines(
                        fig_name=fig_name,
                        x_lists=[prop_class.time],
                        y_lists=[check_list],
                        label_list=[''],
                        fig_args={
                            'xlabel':r'$t$',
                            'ylabel':r'$\sum_{ij}(-1)^{i-j}(1-2\hat{n_i})(1-2\hat{n_j})$',
                            'title':title,
                        }
                    )


                #---------------------------------------------------------------
                # min||Av - \Psi(t)||
                if obs_name == 'E0_subspace_overlap':
                    cost_list = prop_class.E0_subspace_overlap()
                    print(fig_name)
                    plot_helper.plot_lines(
                        fig_name=fig_name,
                        x_lists=[prop_class.time],
                        y_lists=[cost_list],
                        label_list=[''],
                        fig_args={
                            'xlabel':r'$t$',
                            'ylabel':r'$\mathrm{min}||Av - \Psi(t_{\mathrm{fin}})||$',
                            'title':title,
                        }
                    )


    #===========================================================================
    # mutli plot
    #===========================================================================
    def plot_gs_multi_lines(self, U_list, obs_name, theta_list, args_dict={}):
        """Make plot for different values of U and in dependence of theta """

        assert self.bc is not None
        assert self.L is not None
        assert self.N is not None



        for U in U_list:
            l_list = []
            obs_list = []

            for theta in theta_list:
                print(f'U={U}, theta/pi={theta/np.pi:.3f}')

                #---------------------------------------------------------------
                # solve hamilt
                hamilt_class = AnyonHubbardHamiltonian(
                    bc=self.bc,
                    L=self.L,
                    N=self.N,
                    J=self.J,
                    U=U,
                    theta=theta
                )

                #---------------------------------------------------------------
                if obs_name == 'spectrum':
                    evals = hamilt_class.evals()
                    obs = evals[:args_dict['n_evals']]

                #---------------------------------------------------------------
                if obs_name == 'nstate_eigenstate_SVN':
                    svn_list = hamilt_class.nstate_eigenstate_SVN()
                    U_energy = hamilt_class.onsite_int_nstate_exp()
                    idx = np.argsort(U_energy)
                    obs = svn_list[idx]


                #---------------------------------------------------------------
                if obs_name == 'bipartite_SVN':
                    l_list, svn_list = hamilt_class.bipartite_SVN_list_evec0()
                    obs = svn_list


                #---------------------------------------------------------------
                if obs_name == 'evec_2b_checkerboard':
                    e_list, cb_list = hamilt_class.evecs_2b_checkerboard()
                    l_list.append(e_list)
                    obs = cb_list


                #---------------------------------------------------------------
                obs_list.append(obs)


            #-------------------------------------------------------------------
            # get figure parameters
            path_fig = path_dirs.get_path_fig_top(self.bc, self.L, self.N)/(
                                                  f'gs_multi_{obs_name}')
            U_ = f'U_{U:.3f}'.replace('.', '_')
            fig_name = path_fig/f'{U_}_{obs_name}'
            title = plot_helper.make_title(self.L, self.N, U, theta=None,
                                           psi0=self.psi0_str)

            legend_list = [r'$\theta/pi='+f'{theta_list[0]/np.pi:.2f}$'] + [
                f'${el/np.pi:.2f}$' for el in theta_list[1:]]


            #-------------------------------------------------------------------
            if obs_name == 'spectrum':

                path_fig = fig_name.parent/(
                    fig_name.name
                    + f'_thpi_min_{theta_list[0]/np.pi:.3f}'.replace('.', '_')
                    + f'_max_{theta_list[-1]/np.pi:.3f}'.replace('.', '_')
                    + f'_step_{args_dict["dth"]/np.pi:.3f}'.replace('.', '_')
                    + f'_n_evals_{args_dict["n_evals"]}'
                )

                plot_helper.plot_scatter(
                    fig_name=path_fig,
                    x_list=np.array(theta_list)/np.pi,
                    y_lists=np.array(obs_list).T,
                    fig_args={
                        'xlabel':r'$\theta/\pi$',
                        'ylabel':r'$E$',
                        'title':title,
                        **args_dict
                    }
                )

            #-------------------------------------------------------------------
            if obs_name == 'nstate_eigenstate_SVN':

                basis_l = hamilt_class.basis.length
                svn_max = -np.log(1/basis_l)

                plot_helper.plot_scatter(
                    fig_name=fig_name,
                    x_list=list(range(basis_l)),
                    y_lists=obs_list,
                    fig_args={
                        'ylabel':'SVN',
                        'ylim':[0, svn_max+0.2],
                        'title':title,
                        'hline':svn_max,
                        'legend_list':legend_list
                    }
                )

            #-------------------------------------------------------------------
            if obs_name == 'bipartite_SVN':

                plot_helper.plot_lines(
                    fig_name=fig_name,
                    x_lists=[l_list],
                    y_lists=obs_list,
                    label_list=legend_list,
                    fig_args={
                        'xlabel':'$l$',
                        'ylabel':'$S^{\mathrm{vN}}$',
                        'title':title,
                    }
                )

            #-------------------------------------------------------------------
            if obs_name == 'evec_2b_checkerboard':

                plot_helper.plot_scatter(
                    fig_name=fig_name,
                    x_list=l_list,
                    y_lists=obs_list,
                    fig_args={
                        'xlabel':r'$\epsilon_i$',
                        'ylabel':r'$\sum_{ij}(-1)^{i-j}(1-2\hat{n_i})(1-2\hat{n_j})$',
                        'title':title,
                        'legend_list':legend_list
                    }
                )



    def plot_multi_lines(self, U_list, obs_name, theta_list=None):
        """Make plot for different values of U and in dependence of theta """

        assert self.bc is not None
        assert self.L is not None
        assert self.N is not None
        assert self.psi0_str is not None
        assert self.Tprop is not None
        assert self.dtprop is not None


        if theta_list is None:
            theta_list = np.arange(0, 1.1, 0.1)*np.pi


        for U in U_list:

            obs_list = []
            err_list = []

            for theta in theta_list:
                print(f'U={U}, theta/pi={theta/np.pi:.3f}')

                #---------------------------------------------------------------
                # get propagation
                prop_class = Propagation(
                    bc=self.bc,
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
                path_fig = path_dirs.get_path_fig_top_dyn(
                    self.bc, self.L, self.N, self.psi0_str, self.Tprop,
                    self.dtprop)
                U_theta_ = path_dirs.get_dum_name_U_theta(U, theta, self.J)



                #---------------------------------------------------------------
                if obs_name == 'bound_scatter_root_mean_square_slope':
                    res_bound, res_scatter = \
                        prop_class.root_mean_square_linear_fit(t_interval=[1, 4])

                    obs = [res_bound.slope, res_scatter.slope]
                    err = [res_bound.stderr, res_scatter.stderr]


                #---------------------------------------------------------------
                if obs_name == 'bound_scatter_root_mean_square_slope_ratio':
                    res_bound, res_scatter = \
                        prop_class.root_mean_square_linear_fit(t_interval=[1, 4])

                    obs = res_bound.slope / res_scatter.slope
                    err = np.sqrt(
                        (res_bound.stderr/res_scatter.slope)**2
                        + (-1*(res_bound.slope/res_scatter.slope)*res_scatter.stderr)**2
                    )


                #---------------------------------------------------------------
                obs_list.append(obs)
                err_list.append(err)



            #-------------------------------------------------------------------
            U_ = f'U_{U:.3f}'.replace('.', '_')
            fig_name = path_fig/f'{obs_name}'/f'{U_}_{obs_name}'
            title = plot_helper.make_title(self.L, self.N, U, theta=None,
                                           psi0=self.psi0_str)



            if obs_name == 'bound_scatter_root_mean_square_slope':
                plot_helper.plot_errorbar(
                    fig_name=fig_name,
                    x_list=theta_list/np.pi,
                    y_lists=np.array(obs_list).T,
                    yerr_lists=np.array(err_list).T,
                    fig_args={
                        'xlabel':r'$\theta/\pi$',
                        'legend_list':[r'$v_{bound}$', r'$v_{scatter}$'],
                        'title':title,
                        'ms':5
                        }
                )


            if obs_name == 'bound_scatter_root_mean_square_slope_ratio':
                plot_helper.plot_errorbar(
                    fig_name=fig_name,
                    x_list=theta_list/np.pi,
                    y_lists=[obs_list],
                    yerr_lists=[err_list],
                    fig_args={
                        'xlabel':r'$\theta/\pi$',
                        'ylabel':r'$v_{bound} / v_{scatter}$',
                        'title':title,
                        'ms':5
                        }
                )
