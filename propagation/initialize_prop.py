import math

import numpy as np


def get_psi0(basis_list, psi0_str):
    """Create from input string an array corresponding to the intitial state.

    Args:
        basis_list (list): list of basis states
        psi0_str (str): specifier for initial state

    Returns:
        psi_ini (arr): array corresponding to the initial state
        nstate0_str (str): initial number state, e.g. [1, 2, 3, 4]
    """

    L = len(basis_list[0])
    N = sum(basis_list[0])
    basis_length = len(basis_list)

    if psi0_str == 'psi0_n00n':
        # define initial state: |N/2, 0, ..., 0, N/2>
        assert L%2==0  # even number of lattice sites
        nstate_ini = [0]*L
        nstate_ini[0] = N/2
        nstate_ini[-1] = N/2
        psi0_ind = [i for i, el in enumerate(basis_list) if el == nstate_ini]
        assert len(psi0_ind) == 1

        psi0 = np.zeros((basis_length), dtype=complex)
        psi0[psi0_ind[0]] = 1
        nstate0_str = str(basis_list[psi0_ind[0]])

    elif psi0_str == 'psi0_3100':
        assert N == 4 and L == 4
        nstate_ini = [0]*L
        nstate_ini[0] = 3
        nstate_ini[1] = 1

        psi0_ind = [i for i, el in enumerate(basis_list) if el == nstate_ini]
        assert len(psi0_ind) == 1

        psi0 = np.zeros((basis_length), dtype=complex)
        psi0[psi0_ind[0]] = 1
        nstate0_str = str(basis_list[psi0_ind[0]])

    else:
        raise NotImplementedError

    return psi0, nstate0_str


def write_log_file(path_prop, L, N, J, U, theta, psi0_str, nstate0_str, Tprop,
                   dtprop):
    """Writes log file.

    Args:
        ...

    Returns:
        None
    """

    with open(path_prop/'log_file.txt', 'w') as lf:
        lf.write('')
        lf.write(f'L = {L}\t\t\t\t# number of sites\n')
        lf.write(f'N = {N}\t\t\t\t# number of particles\n')
        lf.write(f'J = {J}\t\t\t\t# hopping term\n')
        lf.write(f'U = {U}\t\t\t\t# on-site interaction\n')
        lf.write(f'theta = {theta}\t\t\t# theta/pi\n')
        lf.write(f'theta_pi = {theta/math.pi}\t\t# complex phase\n\n')
        lf.write(f"psi0 = '{psi0_str}'\n")
        lf.write(f'psi0 corresponds to {nstate0_str}\n\n')
        lf.write(f'Tprop = {Tprop}\t\t# final propagation time\n')
        lf.write(f'dtprop = {dtprop}\t\t# propagation time steps\n')
