import time
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import scipy

import path_dirs
from hamiltonian.solve_hamilt import AnyonHubbardHamiltonian
from propagation.propagate import Propagation
from helper import operators
from propagation import initialize_prop



#===============================================================================
class PropagationTD(Propagation):
    """Propagation of a given initial state.

    |psi(t)> = sum_n exp(-i*E_n*t) |n> <n|psi_0>
    n> : eigenvectors
    """

    def __init__(self, bc, L, N, J, U, theta, psi0_str, Tprop, dtprop,
                 ):
        """Initialize parameters.

        Args:
            bc (str): boundary conditions of lattice, open (hard-wall) | periodic
            L (int): number of lattice sites
            N (int): number of atoms
            J (float): hopping constant
            U (float|fct): on-site interaction
            theta (float|fct): statistical angle

            psi0_str (str): specifies initial state for propagation
            Tprop (float): final time of propagation
            dtprop (float): time steps of propagation
            _psi_t (arr): place holder for psi-file

        """
        if isinstance(U, Callable):
            self.U_fct = U
        else:
            self.U_fct = lambda t: U  # constact function  U_fct(t) = U

        if isinstance(theta, Callable):
            self.theta_fct = theta
        else:
            self.theta_fct = lambda t: theta


        super().__init__(bc, L, N, J, self.U_fct(0), self.theta_fct(0),
                         psi0_str, Tprop, dtprop)



    #===========================================================================
    # propagate intial state
    #===========================================================================
    def make_td_rhs(self):
        "RHS: -i*H(t) | psi>"

        def func(t, psi):
            hamilt = np.zeros(
                (self.basis.length, self.basis.length),
                dtype=complex
            )
            for i, b_m in enumerate(self.basis.basis_list):
                for j, b_n in enumerate(self.basis.basis_list):

                    hamilt[i, j] = operators.get_hamilt_mn(
                        bc = self.bc,
                        L = self.L,
                        J = self.J,
                        U = self.U_fct(t),
                        N = self.N,
                        theta = self.theta_fct(t),
                        b_m = b_m,
                        b_n = b_n
                    )
            return (-1j)*hamilt.T.dot(psi)

        return func


    def make_propagation(self):
        # path_psi_npz = self.path_prop/f'psi_{self.dum_Tf_dt}.npz'

        if False:
        # if (path_psi_npz.is_file() and
        #         np.load(path_psi_npz)['Tprop'] == self.Tprop and
        #         np.load(path_psi_npz)['dtprop'] == self.dtprop):
            self._psi_t = np.load(path_psi_npz)['psi_t']
        else:
            # make propagation
            integrator = scipy.integrate.complex_ode(self.make_td_rhs())
            integrator.set_integrator('dop853', nsteps=1e8,
                atol=1e-10, rtol=1e-10)
            integrator.set_initial_value(self.psi0, self.time[0])


            psi_t = [self.psi0]
            for t in self.time[1:]:
                # print(f't = {t:.1f}')
                psi_tstep = integrator.integrate(t)
                psi_t.append(psi_tstep)

            self._psi_t = np.array(psi_t)

            # np.savez(
            #     path_psi_npz,
            #     time=self.time,
            #     psi_t=self._psi_t,
            #     psi0=self.psi0,
            #     psi0_str=self.psi0_str,
            #     nstate0_str=self.nstate0_str,
            #     Tprop=self.Tprop,
            #     dtprop=self.dtprop,
            #     L=self.L,
            #     N=self.N,
            #     J=self.J,
            #     U=self.U,
            #     theta=self.theta,
            #     basis_list=self.basis.basis_list
            # )
            #
            # path_logf = self.path_prop/f'psi_{self.dum_Tf_dt}_log_file.txt'
            # initialize_prop.write_log_file(
            #     path_logf, self.L, self.N, self.J, self.U, self.theta,
            #     self.psi0_str, self.nstate0_str, self.Tprop, self.dtprop)


    def psi_t(self):
        if self._psi_t is None:
            self.make_propagation()
        return self._psi_t


    #===========================================================================
    # number operator
    #===========================================================================
    def num_op_site(self, site_i):
        "psi_t = <psi(t)| b_i^t b^i |psi(t)>"
        nop = []
        for psi_tstep in self.psi_t():
            nop_t = 0
            for psi_coeff, b_m in zip(psi_tstep, self.basis.basis_list):
                nop_t += np.abs(psi_coeff)**2*b_m[site_i]
            nop.append(nop_t)
        nop = np.array(nop)
        assert np.max(np.abs(nop.imag)) < 1e-8
        nop = nop.real

        return self.time, nop


    def num_op_mat(self):
        """Return lists which are easy to plot"""
        L = self.L
        expV_list = []
        for i in range(L):
            time, expV = self.num_op_site(i)
            expV_list.append(expV)
        return time, np.array(expV_list)



    def get_E0_subpace_U0_theta0_cost_list(self, path_basis=None):
        """Estimate the cost for a given input psi"""

        if path_basis is not None:
            path_npz = path_basis/'E0_subpace_U0_theta0_cost_list.npz'
            if path_npz.is_file():
                return np.load(path_npz)['cost_list']

        assert self.U_fct(self.time[-1]) == 0.0 and self.theta_fct(self.time[-1]) == 0.0

        hamil_class = AnyonHubbardHamiltonian(
            bc=self.bc,
            L=self.L,
            N=self.N,
            J=self.J,
            U=0,
            theta=0
        )
        evecs_E0 = hamil_class.get_eigenstates_E0()

        cost_list = []
        for psi in self.psi_t():

            x = np.linalg.lstsq(evecs_E0.T, psi, rcond=None)[0]
            vec_min = np.abs((evecs_E0.T).dot(x) - psi)
            length_vec = np.abs(np.vdot(vec_min, vec_min))**2
            cost = length_vec

            cost_list.append(cost)


        if path_basis is not None:
            np.savez(path_npz, cost_list=cost_list)

        return cost_list


