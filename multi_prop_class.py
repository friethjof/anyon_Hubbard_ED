import os
import gc
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


class MultiPropClass():
    """Organize exact diagonalization method, plotting, scaning of paramters.
    """

    def __init__(self, bc=None, N=None, L=None, J=1, U=None, theta=None,
        Tprop=None, dtprop=None, psi0s_str=None):
        """Initialize parameters.

        Args:
            bc (str): boundary conditions of lattice, open (hard-wall) | periodic
            L (int): number of lattice sites
            N (int): number of atoms
            J (float): hopping constant
            U (float): on-site interaction
            theta (float): statistical angle

            Tprop (float): final time of propagation
            dtprop (float): time steps of propagation

            psi0s_str (str): specifies initial states for propagation

        """

        self.bc = bc
        self.L = L
        self.N = N
        self.U = U
        self.J = J
        self.theta = theta
        self.time = np.arange(0, Tprop+dtprop, dtprop)

        self.Tprop = Tprop
        self.dtprop = dtprop

        self.psi0s_str = psi0s_str



    #===========================================================================
    # propagation analysis
    #===========================================================================
    def prop_vary_U_theta(self, U_list, theta_list, obs_name, args_dict=None):
        """Vary the parameters U and theta and plot a given observable,
        thereby, scan over different initial states

        Args:
            U_list (list): List of U values.
            theta_list (list): List of theta values.
            obs_name (str): specifies the observable
            args_dict (None|dict): optional input arguments

        Returns:
            None
        """

        assert self.bc is not None
        assert self.L is not None
        assert self.N is not None
        assert self.Tprop is not None
        assert self.dtprop is not None

        if self.J == 0:
            if len(U_list) != 1 or U_list[0] != 1:
                print('J=0: set U_list=[1]')
                U_list = [1]

        for U in U_list:
            for theta in theta_list:
                self.U = U  # has to updated for each set of propagations
                self.theta = theta  #

                #---------------------------------------------------------------
                # get paths
                path_fig = path_dirs.get_path_fig_top_dyn_multi(
                    self.bc, self.L, self.N, self.psi0s_str, self.Tprop,
                    self.dtprop)
                U_theta_ = path_dirs.get_dum_name_U_theta(U, theta, self.J)

                fig_name = path_fig/f'{U_theta_}_{obs_name}.png'
                title = plot_helper.make_title(self.L, self.N, U, theta)


                #---------------------------------------------------------------
                if obs_name == 'SVN_max':
                    psi0_list, svn_list, svn_max_list, svn_max_val = \
                        self.get_obs_multi(obs_name)

                    plot_helper.plot_scatter(
                        fig_name=fig_name,
                        x_list=range(svn_max_list.shape[0]),
                        y_lists=[svn_max_list],
                        fig_args={'xlabel':'nstates', 'ylabel':'$S_{vN}^{max}$',
                                  'hline':svn_max_val, 'title':title,
                                  'xticklabels':psi0_list}
                    )

                    plot_helper.plot_lines(
                        fig_name=fig_name.parent/(fig_name.name[:-4]+'_time.png'),
                        x_lists=[self.time],
                        y_lists=svn_list,
                        fig_args={'xlabel':'time', 'ylabel':'SVN',
                                  'title':title, 'hline':svn_max_val}
                    )

                #---------------------------------------------------------------
                if obs_name == 'SVN_fit':

                    psi0_list, svn_fit_list, svn_fit_err, svn_max_val = \
                        self.get_obs_multi(obs_name, args_dict)
                    svn_fit_err = np.sqrt(svn_fit_err)

                    plot_helper.plot_errorbar(
                        fig_name=fig_name,
                        x_list=range(svn_fit_list.shape[0]),
                        y_lists=[svn_fit_list],
                        yerr_lists=[svn_fit_err],
                        fig_args={'xlabel':'nstates', 'ylabel':r'$\bar{S}_{vN}$',
                                  'hline':svn_max_val, 'title':title,
                                  'xticklabels':psi0_list}
                    )


                #---------------------------------------------------------------
                if obs_name == 'natpop_SVN_max':

                    psi0_list, svn_list, svn_max_list, svn_max_val = \
                        self.get_obs_multi(obs_name)

                    plot_helper.plot_scatter(
                        fig_name=fig_name,
                        x_list=range(svn_max_list.shape[0]),
                        y_lists=[svn_max_list],
                        fig_args={'xlabel':'nstates', 'ylabel':'$S_{vN}^{max}$',
                                  'hline':svn_max_val, 'title':title,
                                  'xticklabels':psi0_list}
                    )

                    plot_helper.plot_lines(
                        fig_name=fig_name.parent/(fig_name.name[:-4]+'_time.png'),
                        x_lists=[self.time],
                        y_lists=svn_list,
                        fig_args={'xlabel':'time', 'ylabel':'SVN',
                                  'title':title, 'hline':svn_max_val}
                    )


                #---------------------------------------------------------------
                if obs_name == 'pair_fit':

                    psi0_list, pair_list, pair_fit_list, pair_fit_err = \
                        self.get_obs_multi(obs_name, args_dict)
                    pair_fit_err = np.sqrt(pair_fit_err)

                    plot_helper.plot_errorbar(
                        fig_name=fig_name,
                        x_list=range(pair_fit_list.shape[0]),
                        y_lists=[pair_fit_list],
                        yerr_lists=[pair_fit_err],
                        fig_args={'xlabel':'nstates', 'xticklabels':psi0_list,
                                  'ylabel':r'$\bar{\nu}_{p}$', 'title':title}
                    )

                    plot_helper.plot_lines(
                        fig_name=fig_name.parent/(fig_name.name[:-4]+'_time.png'),
                        x_lists=[self.time],
                        y_lists=pair_list,
                        fig_args={'xlabel':'time', 'ylabel':r'$\nu_p$',
                                  'title':title}
                    )



                #---------------------------------------------------------------
                # clean up
                self.U = None
                self.theta = None
                gc.collect()


    def prop_vary_U_or_theta(self, par2var, U_list, theta_list, obs_name,
                             args_dict=None):
        """Vary the parameters U and theta and plot a given observable,
        thereby, scan over different initial states

        Args:
            par2var (str): either 'U' or 'theta'
            U_list (list): List of U values.
            theta_list (list): List of theta values.
            obs_name (str): specifies the observable
            args_dict (None|dict): optional input arguments

        Returns:
            None
        """

        assert self.bc is not None
        assert self.L is not None
        assert self.N is not None
        assert self.Tprop is not None
        assert self.dtprop is not None


        if par2var == 'theta':
            par1_list, par2_list = U_list, theta_list
        elif par2var == 'U':
            par1_list, par2_list = theta_list, U_list
        else:
            raise NotImplementedError


        #-----------------------------------------------------------------------
        for par1 in par1_list:

            obs_lol, err_lol = [], []

            for par2 in par2_list:  # <-- par2var

                if par2var == 'theta':
                    self.U = par1
                    self.theta = par2

                elif par2var == 'U':
                    self.U = par2
                    self.theta = par1



                #---------------------------------------------------------------
                if obs_name == 'SVN_max':
                    psi0_list, svn_list, svn_max_list, svn_max_val = \
                        self.get_obs_multi(obs_name)
                    obs_lol.append(svn_max_list)

                #---------------------------------------------------------------
                elif obs_name == 'SVN_fit':
                    psi0_list, svn_fit_list, svn_fit_err, svn_max_val = \
                        self.get_obs_multi(obs_name, args_dict)

                    obs_lol.append(svn_fit_list)
                    err_lol.append(np.sqrt(svn_fit_err))

                #---------------------------------------------------------------
                elif obs_name == 'natpop_SVN_max':
                    psi0_list, svn_list, svn_max_list, svn_max_val = \
                        self.get_obs_multi(obs_name)
                    obs_lol.append(svn_max_list)

                #---------------------------------------------------------------
                elif obs_name == 'pair_fit':
                    psi0_list, pair_list, pair_fit_list, pair_fit_err = \
                        self.get_obs_multi(obs_name, args_dict)
                    obs_lol.append(pair_fit_list)
                    err_lol.append(np.sqrt(pair_fit_err))

                #---------------------------------------------------------------
                elif obs_name == 'pair_exp':
                    pair_exp = self.get_obs_multi(obs_name)
                    assert len(pair_exp) == 1
                    obs_lol.append(pair_exp[0])


                #---------------------------------------------------------------
                else:
                    raise NotImplementedError


                #---------------------------------------------------------------
                # clean up
                if par2var == 'theta': self.theta = None
                if par2var == 'U': self.U = None


            #===================================================================
            if 'return' in args_dict.keys() and args_dict['return']:
                assert len(par1_list) == 1
                return psi0_list, obs_lol, err_lol

            #===================================================================
            # get paths
            path_fig_ = path_dirs.get_path_fig_top_dyn_multi(
                self.bc, self.L, self.N, self.psi0s_str, self.Tprop, self.dtprop)


            if par2var == 'theta':
                legend_list = [r'$\theta= ' + f'{th/math.pi:.3f}\pi$'
                               for th in theta_list]
                path_fig = path_fig_/f'{obs_name}_vary_theta'
                U_ = f'{self.U:.3f}'.replace('.', '_')
                fig_name = path_fig/f'U_{U_}_{obs_name}.png'
                title = plot_helper.make_title(self.L, self.N, U=self.U, theta=None)

            elif par2var == 'U':
                legend_list = [f'$U={U}$' for U in U_list]
                path_fig = path_fig_/f'{obs_name}_vary_U'
                theta_ = f'{self.theta/math.pi:.3f}'.replace('.', '_')
                fig_name = path_fig/f'thpi_{theta_}_{obs_name}.png'
                title = plot_helper.make_title(self.L, self.N, U=None, theta=self.theta)

            if par2var == 'theta':
                hamilt_class = AnyonHubbardHamiltonian(
                    bc=self.bc,
                    L=self.L,
                    N=self.N,
                    J=self.J,
                    U=self.U,
                    theta=0
                )

                assert len(psi0_list) == hamilt_class.basis.basis_list.shape[0]
                U_energy = hamilt_class.onsite_int_nstate_exp()
                idx = np.argsort(U_energy)
                print(idx)
                psi0_list = np.array(psi0_list)[idx]
                obs_lol = [el[idx] for el in obs_lol]
                if len(err_lol) > 0:
                    err_lol = [el[idx] for el in err_lol]

                if (self.U >= 5 or self.J == 0):
                    U_set, counts = np.unique(U_energy.round(8), return_counts=True)
                    snv_predict = [np.log(el) for el in counts]
                    hline_list = [snv_predict, counts]
                else:
                    hline_list = None
            else:
                hline_list = None


            self.U = None
            self.theta = None


            #===================================================================
            # plot for each theta-value
            #===================================================================

            if obs_name in ['SVN_max', 'natpop_SVN_max']:

                plot_helper.plot_scatter(
                    fig_name=fig_name,
                    x_list=range(svn_max_list.shape[0]),
                    y_lists=obs_lol,
                    fig_args={'xlabel':'nstates', 'ylabel':'$S_{vN}^{max}$',
                              'hline':svn_max_val, 'title':title,
                              'xticklabels':psi0_list,
                              'legend_list':legend_list}
                )

            elif obs_name in ['SVN_fit', 'pair_fit']:
                print(fig_name)
                fig_args={'xlabel':'nstates', 'title':title,
                          'xticklabels':psi0_list, 'legend_list':legend_list,
                          'hline_list':hline_list,
                          'ylim':[0, 3.7]}

                if obs_name == 'SVN_fit':
                    fig_args['hline'] = svn_max_val
                    fig_args['ylabel'] = r'$\bar{S}_{vN}$'
                elif obs_name == 'pair_fit':
                    fig_args['ylabel'] = r'$\bar{\nu}_p$'

                plot_helper.plot_errorbar(
                    fig_name=fig_name,
                    x_list=range(obs_lol[0].shape[0]),
                    y_lists=obs_lol,
                    yerr_lists=err_lol,
                    fig_args=fig_args
                )

            elif obs_name == 'pair_exp':
                # print(time.shape)
                # print(np.array(obs_lol).shape)
                # exit()
                plot_helper.plot_lines(
                    fig_name=fig_name,
                    x_lists=[self.time],
                    y_lists=obs_lol,
                    label_list=legend_list,
                    fig_args={'xlabel':r'$t \ [\hbar/J]$',
                              'ylabel':r'$\bar{\nu}_p$',
                              'title':title}
                )

            #-------------------------------------------------------------------
            gc.collect()


    #===========================================================================
    # get propagations
    #===========================================================================
    def get_propagations(self):
        #-----------------------------------------------------------------------
        # get Hamiltonian
        #-----------------------------------------------------------------------
        hamilt_class = AnyonHubbardHamiltonian(
            bc=self.bc,
            L=self.L,
            N=self.N,
            J=self.J,
            U=self.U,
            theta=self.theta
        )

        #-----------------------------------------------------------------------
        # get all psi0 strings
        #-----------------------------------------------------------------------
        psi0_str_list = []
        if self.psi0s_str == 'all_number_states':
            psi0_list_out = []
            for nstate0 in hamilt_class.basis.basis_list:
                psi0_str = f"psi0_nstate_{'-'.join(map(str, nstate0))}"
                psi0_str_list.append(psi0_str)
                psi0_list_out.append('-'.join(map(str, nstate0)))

        elif 'psi0_nstate' in self.psi0s_str:
            # psi0_str is in a form consistent with initialize.py-get_psi0()
            psi0_list_out = [self.psi0s_str]
            psi0_str_list = [self.psi0s_str]
        else:
            NotImplementedError

        #-----------------------------------------------------------------------
        # collect propagation classes
        #-----------------------------------------------------------------------
        prop_class_list = []
        for psi0_str in psi0_str_list:
            print(psi0_str)
            prop_class = Propagation(
                bc=self.bc,
                L=self.L,
                N=self.N,
                J=self.J,
                U=self.U,
                theta=self.theta,
                psi0_str=psi0_str,
                Tprop=self.Tprop,
                dtprop=self.dtprop
            )


            prop_class_list.append(prop_class)

        return psi0_list_out, prop_class_list


    #===========================================================================
    # get multi obs
    #===========================================================================
    def get_obs_multi(self, obs_name, args_dict=None):

        path_npz_dir = path_dirs.get_path_basis_multi_dyn(
            self.bc, self.L, self.N, self.U, self.theta, self.J, self.psi0s_str,
            self.Tprop, self.dtprop
        )




        # load data if possible
        if obs_name in ['SVN_max', 'natpop_SVN_max']:
            path_npz = path_npz_dir/f'{obs_name}.npz'
            if path_npz.is_file():
                psi0_list = np.load(path_npz)['psi0_list']
                svn_list = np.load(path_npz)['svn_list']
                svn_max_val = np.load(path_npz)['svn_max_val']
                svn_max_list = np.load(path_npz)['svn_max_list']
                return psi0_list, svn_list, svn_max_list, svn_max_val
            else:
                svn_list, svn_max_list = [], []

        elif obs_name == 'SVN_fit':
            if args_dict is None:
                path_npz = path_npz_dir/'SVN_fit.npz'
                tmin, tmax = 30, None
            else:
                tmin, tmax = args_dict['tmin'], args_dict['tmax']
                assert isinstance(tmin, int) and isinstance(tmax, int)
                path_npz = path_npz_dir/f'SVN_fit_tmin_{tmin}_tmax_{tmax}.npz'
            if path_npz.is_file():
                psi0_list = np.load(path_npz)['psi0_list']
                svn_fit_list = np.load(path_npz)['svn_fit_list']
                svn_fit_err_list = np.load(path_npz)['svn_fit_err_list']
                svn_max_val = np.load(path_npz)['svn_max_val']
                return psi0_list, svn_fit_list, svn_fit_err_list, svn_max_val
            else:
                svn_fit_list, svn_fit_err_list = [], []

        elif obs_name == 'pair_fit':
            path_npz = path_npz_dir/'pair_fit.npz'
            if path_npz.is_file():
                psi0_list = np.load(path_npz)['psi0_list']
                pair_list = np.load(path_npz)['pair_list']
                pair_fit_list = np.load(path_npz)['pair_fit_list']
                pair_fit_err_list = np.load(path_npz)['pair_fit_err_list']
                return psi0_list, pair_list, pair_fit_list, pair_fit_err_list
            else:
                pair_list, pair_fit_list, pair_fit_err_list = [], [], []

        elif obs_name == 'pair_exp':
            path_npz = path_npz_dir/'pair_exp.npz'
            pair_list = []

        else:
            raise NotImplementedError(obs_name)


        psi0_list, prop_class_list = self.get_propagations()


        # collect data
        for i, prop_class in enumerate(prop_class_list):

            print(i, prop_class.psi0_str)

            if obs_name == 'SVN_max':
                svn, svn_max_val = prop_class.nstate_SVN()
                svn_max_list.append(np.max(svn))
                svn_list.append(svn)

            elif obs_name == 'natpop_SVN_max':
                svn, svn_max_val = prop_class.natpop_SVN()
                svn_max_list.append(np.max(svn))
                svn_list.append(svn)

            elif obs_name == 'SVN_fit':
                svn_fit, svn_err, svn_max_val =\
                    prop_class.nstate_SVN_horizontal_fit(tmin, tmax)
                svn_fit_list.append(svn_fit)
                svn_fit_err_list.append(svn_err)

            elif obs_name == 'pair_fit':
                pair_expV, pair_fit, pair_err =\
                    prop_class.pair_op_horizontal_fit()
                pair_list.append(pair_expV)
                pair_fit_list.append(pair_fit)
                pair_fit_err_list.append(pair_err)

            elif obs_name == 'pair_exp':
                pair_list.append(prop_class.pair_operator())

            gc.collect()

        # save and return data
        dict_save = {'time':self.time, 'psi0_list':psi0_list,}

        if obs_name in ['SVN_max', 'natpop_SVN_max']:
            dict_save['svn_list'] = svn_list
            dict_save['svn_max_list'] = svn_max_list
            dict_save['svn_max_val'] = svn_max_val
            np.savez(path_npz, **dict_save)
            return (psi0_list, np.array(svn_list), np.array(svn_max_list),
                    svn_max_val)

        elif obs_name == 'SVN_fit':
            dict_save['svn_fit_list'] = svn_fit_list
            dict_save['svn_fit_err_list'] = svn_fit_err_list
            dict_save['svn_max_val'] = svn_max_val
            np.savez(path_npz, **dict_save)
            return (psi0_list, np.array(svn_fit_list),
                    np.array(svn_fit_err_list), svn_max_val)

        elif obs_name == 'pair_fit':
            dict_save['pair_list'] = pair_list
            dict_save['pair_fit_list'] = pair_fit_list
            dict_save['pair_fit_err_list'] = pair_fit_err_list
            np.savez(path_npz, **dict_save)
            return (psi0_list, np.array(pair_list),
                    np.array(pair_fit_list), np.array(pair_fit_err_list))

        elif obs_name == 'pair_exp':
            return pair_list
