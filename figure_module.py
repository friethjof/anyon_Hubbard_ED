import math

from advanced_figures import plot_spectrum
from advanced_figures import plot_spectrum_degeneracy
from advanced_figures import plot_E0_degeneracies
from advanced_figures import compare_E0_subspace
from advanced_figures import plot_nstate_ovlp_E0_subspace_L6_N2
from advanced_figures import plot_nstate_ovlp_E0_subspace_L6_N4
from advanced_figures import plot_energyOccupNstate_ovlp_E0_subspace_L6_N2
# from advanced_figures import plot_energyOccupNstate_ovlp_E0_subspace_L6_N4
from advanced_figures import plot_edge_eigenstate
from advanced_figures import plot_two_site_correlation
from advanced_figures import test_two_site_correlation
from advanced_figures import plot_td_pair_op
from advanced_figures import plot_td_pair_op_diff
from advanced_figures import plot_num_op_coll_diff
from advanced_figures import plot_L8_N4_svn_psi0_set1

from advanced_figures import plot_svn_overview_L8_N2




# plot_nstate_ovlp_E0_subspace_L6_N2.make_plot()
# plot_energyOccupNstate_ovlp_E0_subspace_L6_N2.make_plot()
# plot_nstate_ovlp_E0_subspace_L6_N4.make_plot()
# plot_E0_degeneracies.make_plot()

compare_E0_subspace.make_plot(L=6, N=2)

# plot_svn_overview_L8_N2.make_plot()

# plot_spectrum.make_plot(L=8, N=2, U=100)


# plot_spectrum.make_plot(L=8, N=4, U=0)
# plot_spectrum.make_plot(L=8, N=4, U=1)
# plot_spectrum.make_plot(L=8, N=4, U=10)
# plot_spectrum.make_plot(L=8, N=4, U=100)

# plot_spectrum_degeneracy.make_plot(U=0, theta=0.0*math.pi, bc_name='obc')
# plot_spectrum_degeneracy.make_plot(U=0, theta=0.2*math.pi, bc_name='obc')
# plot_spectrum_degeneracy.make_plot(U=0, theta=1.0*math.pi, bc_name='obc')
# plot_spectrum_degeneracy.make_plot(U=1, theta=0.0*math.pi, bc_name='obc')
# plot_spectrum_degeneracy.make_plot(U=1, theta=0.2*math.pi, bc_name='obc')
# plot_spectrum_degeneracy.make_plot(U=1, theta=1.0*math.pi, bc_name='obc')

# plot_edge_eigenstate.make_plot(theta=0.5*math.pi)

# plot_two_site_correlation.make_plot(L=8, N=4, U=0.2, psi0_str='psi0_n00n')

# test_two_site_correlation.make_plot()

# plot_td_pair_op.make_plot()

# plot_num_op_coll_diff.make_plot()

# plot_td_pair_op_diff.make_plot()

# plot_L8_N4_svn_psi0_set1.make_plot()
