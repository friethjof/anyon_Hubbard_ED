import math

# author: Friethjof Theel
# date: 15.06.2022
# last modified: Jul. 2023


from scan_class import ScanClass





#-------------------------------------------------------------------------------
# initialize scan class
#-------------------------------------------------------------------------------
par_dict = {
    'N':4,
    'L':8,
}
scan_class = ScanClass(**par_dict)

U_list = [0.0]
theta_list = [el*math.pi for el in [0.0]]

#-------------------------------------------------------------------------------
'ground state analysis'
# scan_class.gs_vary_U_theta(U_list, theta_list, 'eigen_spectrum')
# scan_class.gs_vary_U_theta(U_list, theta_list, 'K_operator')


#-------------------------------------------------------------------------------
# initialize propagation
#-------------------------------------------------------------------------------
scan_class.psi0_str = 'psi0_n00n'
scan_class.Tprop = 5
scan_class.dtprop = 0.1

# scan_class.prop_vary_U_theta(U_list, theta_list, 'num_op')
scan_class.prop_vary_U_theta(U_list, theta_list, 'num_op_2b',
                             arg_dict={'t_list':[0, 1, 2, 3, 5]})
# scan_class.prop_vary_U_theta(U_list, theta_list, 'K_operator')



# class_prop.plot_nstate_proj([2, 0, 0, 2])
# class_prop.plot_nstate_proj([0, 2, 2, 0])
# class_prop.plot_nstate_proj([1, 2, 1, 0])
# class_prop.plot_nstate_proj([1, 2, 0, 1])
# class_prop.numOp_cplot()
# class_prop.numOp_lplot()
# class_prop.plot_ninj_mat_time(t_instance=2)
# # class_prop.plot_bipartite_ent()
