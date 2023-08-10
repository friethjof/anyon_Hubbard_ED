import os
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

    def __init__(self, N=None, L=None, J=1, U=None, theta=None, Tprop=None,
        dtprop=None, psi0s_str=None):
        """Initialize parameters.

        Args:
            L (int): number of lattice sites
            N (int): number of atoms
            J (float): hopping constant
            U (float): on-site interaction
            theta (float): statistical angle

            Tprop (float): final time of propagation
            dtprop (float): time steps of propagation

            psi0s_str (str): specifies initial states for propagation

        """

        self.L = L
        self.N = N
        self.U = U
        self.J = J  # is assumed to be always 1
        self.theta = theta
        self.time = np.arange(0, Tprop+dtprop, dtprop)

        self.Tprop = Tprop
        self.dtprop = dtprop

        self.psi0s_str = psi0s_str

        # parameters which are assigned during the for-loop
        self.path_basis = None
        self.path_fig = None
        self.fig_name = None
        self.title = None


    #===========================================================================
    # get propagations
    #===========================================================================
    def get_propagations(self):
        #-----------------------------------------------------------------------
        # get Hamiltonian
        #-----------------------------------------------------------------------
        hamilt_class = AnyonHubbardHamiltonian(
            path_basis=self.path_basis,
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

        else:
            NotImplementedError

        #-----------------------------------------------------------------------
        # collect propagation classes
        #-----------------------------------------------------------------------
        prop_class_list = []
        for psi0_str in psi0_str_list:
            print(psi0_str)
            prop_class = Propagation(
                path_basis=self.path_basis,
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
    # propagation analysis
    #===========================================================================
    def prop_vary_U_theta(self, U_list, theta_list, obs_name, arg_dict=None):
        """Vary the parameters U and theta and plot a given observable,
        thereby, scan over different initial states

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
        assert self.Tprop is not None
        assert self.dtprop is not None


        for U in U_list:
            for theta in theta_list:
                self.U = U
                self.theta = theta

                self.path_basis = other_tools.get_path_basis(
                    path_dirs.path_data_top, self.L, self.N, U, theta)
                self.path_npz_dir = self.path_basis/f'multi_psi0_{self.psi0s_str}'
                os.makedirs(self.path_npz_dir, exist_ok=True)

                #---------------------------------------------------------------
                # get paths
                path_fig = (path_dirs.path_fig_top/f'L{self.L}_N{self.N}'/
                            f'scan_psi0_Tf_{self.Tprop}/{obs_name}')
                theta_ = f'{theta/math.pi:.3f}'.replace('.', '_')
                U_ = f'{U:.3f}'.replace('.', '_')
                fig_name = path_fig/f'U_{U_}_thpi_{theta_}_{obs_name}.png'
                title = plot_helper.make_title(self.L, self.N, U, theta)


                #---------------------------------------------------------------
                if obs_name == 'SVN_max':

                    psi0_list, svn_list, svn_max_list, svn_max_val = \
                        self.get_SVN_max()

                    plot_helper.plot_scatter(
                        fig_name=fig_name,
                        x_list=range(svn_max_list.shape[0]),
                        y_lists=[svn_max_list],
                        fig_args={'xlabel':'nstates', 'ylabel':'$S_{vN}^{max}$',
                                  'hline':svn_max_val, 'title':title,
                                  'xticklabels':psi0_list}
                    )

                    plot_helper.plot_lines(
                        fig_name=fig_name.parent/(fig_name.name[:-3]+'_time.png'),
                        x_lists=[self.time],
                        y_lists=svn_list,
                        fig_args={'xlabel':'time', 'ylabel':'SVN',
                                  'title':title, 'hline':svn_max_val}
                    )


                #---------------------------------------------------------------
                # clean up
                self.U = None
                self.theta = None
                self.path_basis = None
                self.path_npz_dir = None



    def prop_vary_U_or_theta(self, par2var, U_list, theta_list, obs_name,
                             arg_dict=None):
        """Vary the parameters U and theta and plot a given observable,
        thereby, scan over different initial states

        Args:
            par2var (str): either 'U' or 'theta'
            U_list (list): List of U values.
            theta_list (list): List of theta values.
            obs_name (str): specifies the observable
            arg_dict (None|dict): optional input arguments

        Returns:
            None
        """

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

            obs_lol = []

            for par2 in par2_list:  # <-- par2var

                if par2var == 'theta':
                    self.U = par1
                    self.theta = par2

                elif par2var == 'U':
                    self.U = par2
                    self.theta = par1


                self.path_basis = other_tools.get_path_basis(
                    path_dirs.path_data_top, self.L, self.N, self.U, self.theta)
                self.path_npz_dir = self.path_basis/f'multi_psi0_{self.psi0s_str}'
                os.makedirs(self.path_npz_dir, exist_ok=True)


                if obs_name == 'SVN_max':
                    psi0_list, svn_list, svn_max_list, svn_max_val = \
                        self.get_SVN_max()

                    obs_lol.append(svn_max_list)

                #---------------------------------------------------------------
                # clean up
                if par2var == 'theta': self.theta = None
                if par2var == 'U': self.U = None

                self.path_basis = None
                self.path_npz_dir = None


            #===================================================================
            # get paths

            path_fig_ = (path_dirs.path_fig_top/f'L{self.L}_N{self.N}'/
                        f'scan_psi0_Tf_{self.Tprop}')


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


            self.U = None
            self.theta = None


            #===================================================================
            # plot for each theta-value
            #===================================================================
            if obs_name == 'SVN_max':

                plot_helper.plot_scatter(
                    fig_name=fig_name,
                    x_list=range(svn_max_list.shape[0]),
                    y_lists=obs_lol,
                    fig_args={'xlabel':'nstates', 'ylabel':'$S_{vN}^{max}$',
                              'hline':svn_max_val, 'title':title,
                              'xticklabels':psi0_list,
                              'legend_list':legend_list}
                )


    #===========================================================================
    # plotting routines
    #===========================================================================
    def get_SVN_max(self):

        path_npz = self.path_npz_dir/'SVN_max.npz'

        if path_npz.is_file():
            psi0_list = np.load(path_npz)['psi0_list']
            svn_list = np.load(path_npz)['svn_list']
            svn_max_val = np.load(path_npz)['svn_max_val']
            svn_max_list = np.load(path_npz)['svn_max_list']
            return psi0_list, svn_list, svn_max_list, svn_max_val

        psi0_list, prop_class_list = self.get_propagations()

        svn_list, svn_max_list = [], []
        for prop_class in prop_class_list:
            svn, svn_max_val = prop_class.nstate_SVN()
            svn_max_list.append(np.max(svn))
            svn_list.append(svn)

        np.savez(
            path_npz,
            time=self.time,
            psi0_list=psi0_list,
            svn_list=svn_list,
            svn_max_list=svn_max_list,
            svn_max_val=svn_max_val
        )

        return psi0_list, np.array(svn_list), np.array(svn_max_list), svn_max_val
