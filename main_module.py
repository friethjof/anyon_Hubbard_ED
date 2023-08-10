import math

# author: Friethjof Theel
# date: 15.06.2022
# last modified: Jul. 2023


from scan_class import ScanClass
from multi_prop_class import MultiPropClass





#-------------------------------------------------------------------------------
# initialize scan class
#-------------------------------------------------------------------------------
par_dict = {
    'L':4,
    'N':4,
    'Tprop':50,
    'dtprop':0.1
}



#===============================================================================
# scan_class
#===============================================================================
scan_class = ScanClass(**par_dict)
U_list = [0.0, 1.0, 2.0, 5.0, 7.0, 10.0]
theta_list = [el*math.pi for el in [0.0, 0.2, 0.5, 0.8, 1.0]]


#-------------------------------------------------------------------------------
'ground state analysis'
# scan_class.gs_vary_U_theta(U_list, theta_list, 'eigen_spectrum')
# scan_class.gs_vary_U_theta(U_list, theta_list, 'K_operator')
# scan_class.gs_vary_U_theta(U_list, theta_list, 'eigenstate_E0')


#-------------------------------------------------------------------------------
# initialize propagation
#-------------------------------------------------------------------------------
# scan_class.psi0_str = 'psi0_n00n'
# scan_class.psi0_str = 'pi0_eigstate_160'
scan_class.psi0_str = 'psi0_nstate_0-4-0-0'

scan_class.prop_vary_U_theta(U_list, theta_list, 'num_op')
# scan_class.prop_vary_U_theta(U_list, theta_list, 'num_op_2b',
#                              arg_dict={'t_list':[0, 1, 2, 3, 5]})
# scan_class.prop_vary_U_theta(U_list, theta_list, 'K_operator')
# scan_class.prop_vary_U_theta(U_list, theta_list, 'K_operator_K_dagger')
# scan_class.prop_vary_U_theta(U_list, theta_list, 'mom_op')
# scan_class.prop_vary_U_theta(U_list, theta_list, 'nstate')
# scan_class.prop_vary_U_theta(U_list, theta_list, 'nstate_SVN')
# scan_class.prop_vary_U_theta(U_list, theta_list, 'natpop')
# scan_class.prop_vary_U_theta(U_list, theta_list, 'eigenstate_projection')
# scan_class.prop_vary_U_theta(U_list, theta_list, 'pair_operator')

# class_prop.plot_nstate_proj([2, 0, 0, 2])
# class_prop.plot_nstate_proj([0, 2, 2, 0])
# class_prop.plot_nstate_proj([1, 2, 1, 0])
# class_prop.plot_nstate_proj([1, 2, 0, 1])
# class_prop.numOp_cplot()
# class_prop.numOp_lplot()
# class_prop.plot_ninj_mat_time(t_instance=2)
# # class_prop.plot_bipartite_ent()



#===============================================================================
# compare multiple initializations
#===============================================================================
par_dict['psi0s_str'] = 'all_number_states'

multi_class = MultiPropClass(**par_dict)
# multi_class.prop_vary_U_theta(U_list, theta_list, 'SVN_max')
multi_class.prop_vary_U_or_theta('U', U_list, theta_list, 'SVN_max')
multi_class.prop_vary_U_or_theta('theta', U_list, theta_list, 'SVN_max')
