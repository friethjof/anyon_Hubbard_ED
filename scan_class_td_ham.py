import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from propagation.propagate_td_ham import PropagationTD

import path_dirs
from helper import other_tools
from helper import plot_helper
from helper import operators


class ScanClassTD():
    """Organize exact diagonalization method, plotting, scaning of paramters.
    """

    def __init__(self, bc=None, N=None, L=None, J=1, U=None, theta=None,
                 psi0_str=None, Tprop=None, dtprop=None):
        """Initialize parameters.

        Args:
            bc (str): boundary conditions of lattice, open (hard-wall) | periodic
            L (int): number of lattice sites
            N (int): number of atoms
            J (int): hopping constant
            U (float|fct): on-site interaction
            theta (float|fct): statistical angle

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


                #---------------------------------------------------------------
                # get propagation
                prop_class = PropagationTD(
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


                # #---------------------------------------------------------------
                # # get paths
                # path_fig = path_dirs.get_path_fig_top_dyn(
                #     self.bc, self.L, self.N, self.psi0_str, self.Tprop,
                #     self.dtprop)
                # U_theta_ = path_dirs.get_dum_name_U_theta(U, theta, self.J)
                #
                # fig_name = path_fig/f'{obs_name}'/f'{U_theta_}_{obs_name}'
                # title = plot_helper.make_title(self.L, self.N, U, theta,
                #                                psi0=self.psi0_str)


                #---------------------------------------------------------------
                if obs_name == 'num_op':
                    fig_name = Path('./test_gpop.png')

                    time, num_op = prop_class.num_op_mat()
                    plot_helper.num_op_cplot(fig_name, prop_class.time, self.L,
                                             num_op, '')

                    plt.plot(prop_class.time, sum(num_op[:3]))
                    plt.show()
                    exit()
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
