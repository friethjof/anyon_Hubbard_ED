import time
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import minimize

from hamiltonian.solve_hamilt import AnyonHubbardHamiltonian
from propagation.propagate import Propagation
from propagation.propagate_td_ham import PropagationTD

import path_dirs
from helper import operators



#===============================================================================
class ClassOptimize():
    """Propagation of a given initial state.

    |psi(t)> = sum_n exp(-i*E_n*t) |n> <n|psi_0>
    n> : eigenvectors
    """

    def __init__(self, bc, L, N, J, U_ini, theta_ini, U_fin, theta_fin,
                 psi0_str, psifinal_str, Tprop, dtprop, version, N_interp=None,
                 ini_guess=None):
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

        """
        self.U_ini = U_ini
        self.theta_ini = theta_ini
        self.U_fin = U_fin
        self.theta_fin = theta_fin

        self.path_psi = path_dirs.get_path_basis_optimize(
            bc, L, N, U_ini, theta_ini, J,
            psi0_str, psifinal_str,
            N_interp, Tprop, ini_guess, version
        )

        self.prop_class_ini = Propagation(bc, L, N, J, U_ini, theta_ini,
                                         psi0_str, Tprop, dtprop)
        self.psi0_str = psi0_str
        self.psi0 = self.prop_class_ini.psi0
        self.time = self.prop_class_ini.time
        self.psifinal_str = psifinal_str

        self.N_interp = N_interp
        self.ini_guess = ini_guess
        self.version = version  # defines version of initial guess

        #-----------------------------------------------------------------------
        # optimization parameters
        # self.par_n_steps = 2
        # self.par_step_width = 0.05

        #-----------------------------------------------------------------------
        self.temp_dict = {}
        self.time_interp = None



    #===========================================================================
    def get_ramp_fct(self, t1, t2, a, b):
        return lambda t: a + (t-t1)*(b-a)/(t2-t1)


    def get_cost_val(self, psi_in):
        """Estimate the cost for a given input psi"""


        if self.psifinal_str == 'E0_subpace_U0_theta0':
            assert self.U_fin == 0.0 and self.theta_fin == 0.0
            if 'E0_subpace_U0_theta0' not in self.temp_dict.keys():
                hamil_class = AnyonHubbardHamiltonian(
                    bc=self.prop_class_ini.bc,
                    L=self.prop_class_ini.L,
                    N=self.prop_class_ini.N,
                    J=self.prop_class_ini.J,
                    U=0,
                    theta=0
                )
                evecs_E0 = hamil_class.get_eigenstates_E0()
                self.temp_dict['E0_subpace_U0_theta0'] = evecs_E0
            else:
                evecs_E0 = self.temp_dict['E0_subpace_U0_theta0']

            # cost = 1 - np.sum(
            #     [np.abs(np.vdot(psi_in, E0_evec))**2 for E0_evec in evecs_E0]
            # )

            x = np.linalg.lstsq(evecs_E0.T, psi_in, rcond=None)[0]
            vec_min = np.abs((evecs_E0.T).dot(x) - psi_in)
            length_vec = np.abs(np.vdot(vec_min, vec_min))**2
            cost = length_vec

        return cost


    #===========================================================================
    # Optimization bfgs
    #===========================================================================
    def get_psi_spline(self, x, time):
        """"Get input array where even elements correspond to U values and the
        odd ones to theta. Intepolate these with spline and feed them into
        the Time-dependent propagation class,
            return
                psi, U_array, theta_array
        """

        U_vals = np.array([self.U_ini] + list(x[0::2]) + [self.U_fin])
        theta_vals = np.array([self.theta_ini] + list(x[1::2]) + [self.theta_fin])

        print('U: ', U_vals)
        print('th:', theta_vals)

        U_fct = InterpolatedUnivariateSpline(self.time_interp, U_vals)
        theta_fct = InterpolatedUnivariateSpline(self.time_interp, theta_vals)

        prop_class_TD = PropagationTD(
            bc=self.prop_class_ini.bc,
            L=self.prop_class_ini.L,
            N=self.prop_class_ini.N,
            J=self.prop_class_ini.J,
            U=U_fct,
            theta=theta_fct,
            psi0_str=self.psi0_str,
            Tprop=None,
            dtprop=None,
        )
        prop_class_TD.time = time

        return prop_class_TD.psi_t(), U_fct(time), theta_fct(time)


    #---------------------------------------------------------------------------
    def perform_optimzation_bfgs(self):

        path_res = self.path_psi/'optimization_result.npz'

        if path_res.is_file():
            dict_load = np.load(path_res)
            return [
                dict_load['success'],
                [
                    dict_load['time_interp'],
                    dict_load['U_interp'],
                    dict_load['th_interp']
                ],
                [
                    dict_load['time'],
                    dict_load['psi_list'],
                    dict_load['U_list'],
                    dict_load['theta_list'],
                    dict_load['cost_list']
                ]
            ]


        #-----------------------------------------------------------------------
        def func(x):
            time_red = np.array([self.time[0], self.time[-1]])
            psi, _, _ = self.get_psi_spline(x, time_red)
            print('cost', self.get_cost_val(psi[-1]))
            print()
            return self.get_cost_val(psi[-1])


        #-----------------------------------------------------------------------
        self.time_interp = np.linspace(self.time[0], self.time[-1], self.N_interp+2)

        # x0 = np.random.uniform(low=-0.1, high=0.1, size=(2*self.N_interp,))

        if self.version == 1:
            x0 = np.zeros((2*self.N_interp))
            x0[0::2] = self.ini_guess
            x0[1::2] = self.ini_guess
        else:
            raise ValueError('version of initial guess has to be specified')


        bounds = ((-10, 10),(None, None),)*self.N_interp
        print(bounds)
        res = minimize(func, x0, method='L-BFGS-B', bounds=bounds)
        print(res)
        U_interp = np.array([self.U_ini] + list(res.x[0::2]) + [self.U_fin])
        th_interp = np.array([self.theta_ini] + list(res.x[1::2]) + [self.theta_fin])


        psi_list, U_list, theta_list = self.get_psi_spline(res.x, self.time)
        cost_list = np.array([self.get_cost_val(psi) for psi in psi_list])


        np.savez(
            path_res,
            success=res.success,
            time_interp=self.time_interp,
            U_interp=U_interp,
            th_interp=th_interp,
            time=self.time,
            psi_list=psi_list,
            cost_list=cost_list,
            U_list=U_list,
            theta_list=theta_list
        )

        return [
            res.success,
            [self.time_interp, U_interp, th_interp],
            [self.time, psi_list, U_list, theta_list, cost_list]
        ]


    def get_res_name(self, res_name, bool_load_only=False):
        if res_name not in self.temp_dict.keys() and not bool_load_only:
            succ, coarse_list, fine_list = self.perform_optimzation_bfgs()
            self.temp_dict['success'] = succ
            self.temp_dict['time_interp'] = coarse_list[0]
            self.temp_dict['U_interp'] = coarse_list[1]
            self.temp_dict['th_interp'] = coarse_list[2]
            self.temp_dict['time'] = fine_list[0]
            self.temp_dict['psi_list'] = fine_list[1]
            self.temp_dict['U_list'] = fine_list[2]
            self.temp_dict['theta_list'] = fine_list[3]
            self.temp_dict['cost_list'] = fine_list[4]
        try:
            return self.temp_dict[res_name]
        except:
            if bool_load_only:
                return None
            else:
                raise ValueError(res_name, 'is not an output name of ',
                                 'perform_optimzation_bfgs')



    def get_res(self, res_name_list):
        return [self.get_res_name(el) for el in res_name_list]


    #
    # #===========================================================================
    # # Optimization brute force
    # #===========================================================================
    # def optimization_step_bf(self, psi_now, U_now, theta_now, t_now, t_next):
    #     """Continue the propaation always from psi_now and vary U and theta
    #     Evaluate cost fun  turn best (U_next, theta_next)
    #     """
    #
    #     # define parameter range for mass and omeg
    #     U_1 = U_now - self.par_n_steps*self.par_step_width
    #     U_2 = U_now + self.par_n_steps*self.par_step_width + 1e-6
    #     th_1 = theta_now - self.par_n_steps*self.par_step_width
    #     th_2 = theta_now + self.par_n_steps*self.par_step_width + 1e-6
    #
    #     U_list = np.arange(U_1, U_2, self.par_step_width)
    #     theta_list = np.arange(th_1, th_2, self.par_step_width)
    #
    #     psi_list = []
    #     cost_corrd = []
    #     for U_i in U_list:
    #         for th_i in theta_list:
    #             U_i = round(U_i, 6)
    #             th_i = round(th_i, 6)
    #
    #             "ramp time-dependent parameters"
    #             U_fct_i = self.get_ramp_fct(t_now, t_next, U_now, U_i)
    #             theta_fct_i = self.get_ramp_fct(t_now, t_next, theta_now, th_i)
    #
    #
    #             prop_class_TD = PropagationTD(
    #                 bc=self.prop_class_ini.bc,
    #                 L=self.prop_class_ini.L,
    #                 N=self.prop_class_ini.N,
    #                 J=self.prop_class_ini.J,
    #                 U=U_fct_i,
    #                 theta=theta_fct_i,
    #                 psi0_str=None,
    #                 Tprop=None,
    #                 dtprop=0.1,
    #             )
    #             prop_class_TD.time = np.array([t_now, t_next])
    #             prop_class_TD.psi0 = psi_now
    #
    #             psi_i = prop_class_TD.psi_t()[-1]
    #
    #             cost = self.get_cost_val(psi_i)
    #
    #             psi_list = psi_i
    #             cost_corrd.append([U_i, th_i, cost])
    #
    #
    #     cost_mat = np.array(cost_corrd)
    #     ind_min = cost_mat[:, 2].argmin()
    #
    #     psi_next = psi_list[ind_min]
    #     U_next = cost_mat[ind_min, 0]
    #     theta_next = cost_mat[ind_min, 1]
    #     cost = cost_mat[ind_min, 2]
    #
    #     return psi_next, U_next, theta_next, cost
    #
    #
    #
    #
    # def perform_optimzation_brute_force(self):
    #     """  """
    #
    #     psi_list = [self.psi0]
    #     t_now = self.time[0]
    #     U_list = [self.U_ini]
    #     theta_list = [self.theta_ini]
    #     cost_list = [self.get_cost_val(self.psi0)]
    #
    #
    #     while t_now < self.time[-1]:
    #         print(f't_now = {t_now}')
    #
    #         # estimate end of propagation time
    #         t_next = t_now + 0.1
    #
    #         #-------------------------------------------------------------------
    #         psi_next, U_next, theta_next, cost = self.optimization_step_bf(
    #             psi_now=psi_list[-1],
    #             U_now=U_list[-1],
    #             theta_now=theta_list[-1],
    #             t_now=t_now,
    #             t_next=t_next
    #         )
    #
    #         #-------------------------------------------------------------------
    #         psi_list.append(psi_next)
    #         theta_list.append(theta_next)
    #         U_list.append(U_next)
    #         cost_list.append(cost)
    #
    #         t_now = t_next
    #
    #
    #
    #
    #     plt.plot(self.time, cost_list)
