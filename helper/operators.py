import math
import numpy as np
from numba import njit



@njit
def nstate_allclose(arr1, arr2):
    'Provide an alternative to compare two number states'

    all_close = True
    for el1, el2 in zip(arr1, arr2):
        if np.abs(el1 - el2) > 1e-10:
            all_close = False
            break

    return all_close


@njit
def hop_term_j(j, N, theta, basis1, basis2):
    "b_j^t exp(i*theta*n_j) b_j+1"

    b1 = basis1.copy()
    b2 = basis2.copy()

    # b_j+1
    if b2[j+1] == 0:
        return 0

    hop_jp1 = math.sqrt(b2[j+1])
    b2[j+1] -= 1

    # exp(i * theta * n_j)
    exp_j = np.exp(1j*theta*b2[j])

    # b_j^t
    if b2[j] == N:
        return 0

    hop_j = math.sqrt(b2[j]+1)
    b2[j] += 1

    if not nstate_allclose(b1, b2):
        return 0

    return hop_j*exp_j*hop_jp1


@njit
def hop_term_L(N, theta, basis1, basis2):
    "b_L^t exp(i*theta*n_L) b_1"

    b1 = basis1.copy()
    b2 = basis2.copy()

    # b_j+1
    if b2[0] == 0:
        return 0

    hop_1 = math.sqrt(b2[0])
    b2[0] -= 1

    # exp(i * theta * n_j)
    exp_L = np.exp(1j*theta*b2[-1])

    # b_j^t
    if b2[-1] == N:
        return 0

    hop_L = math.sqrt(b2[-1]+1)
    b2[-1] += 1

    if not nstate_allclose(b1, b2):
        return 0

    return hop_L*exp_L*hop_1


@njit
def hop_term_j_dagger(j, N, theta, basis1, basis2):
    "b_j+1^t exp(-i*theta*n_j) b_j"


    b1 = basis1.copy()
    b2 = basis2.copy()

    # b_j
    if b2[j] == 0:
        return 0

    hop_j = math.sqrt(b2[j])
    b2[j] -= 1

    # exp(-i * theta * n_j)
    exp_j = np.exp(-1j*theta*b2[j])

    # b_j+1^t
    if b2[j+1] == N:
        return 0

    hop_jp1 = math.sqrt(b2[j+1]+1)
    b2[j+1] += 1

    if not nstate_allclose(b1, b2):
        return 0

    return hop_jp1*hop_j*exp_j


@njit
def hop_term_L_dagger(N, theta, basis1, basis2):
    "b_1^t exp(-i*theta*n_L) b_L"


    b1 = basis1.copy()
    b2 = basis2.copy()

    # b_j
    if b2[-1] == 0:
        return 0

    hop_L = math.sqrt(b2[-1])
    b2[-1] -= 1

    # exp(-i * theta * n_j)
    exp_L = np.exp(-1j*theta*b2[-1])

    # b_j+1^t
    if b2[0] == N:
        return 0

    hop_1 = math.sqrt(b2[0]+1)
    b2[0] += 1

    if not nstate_allclose(b1, b2):
        return 0

    return hop_1*exp_L*hop_L



#-----------------------------------------------------------------------
@njit
def sum_hop(L, N, theta, basis1, basis2):
    "sum_j^L-1 b_j^t exp(i*theta*n_j) b_j+1"

    sum_hop = 0
    for j in range(L-1):
        sum_hop += hop_term_j(j, N, theta, basis1, basis2)

    return sum_hop


def sum_hop_t(L, N, theta, basis1, basis2):
    "sum_j^L-1 b_j+1^t exp(-i*theta*n_j) b_j"

    sum_hop = 0
    for j in range(L-1):
        sum_hop += hop_term_j_dagger(j, N, theta, basis1, basis2)

    return sum_hop



#-----------------------------------------------------------------------
@njit
def sum_onsite_int(L, basis1, basis2):
    "U/2 sum_j^L  n_j(n_j-1)"

    if not nstate_allclose(basis1, basis2):
        return 0

    sum_int = 0
    for j in range(L):

        if basis1[j] == 0:
            continue
        else:
            sum_int += basis1[j]*(basis1[j] - 1)

    return sum_int


#-----------------------------------------------------------------------
def get_hamilt_mn(bc, L, J, U, N, theta, b_m, b_n):
    if bc == 'open':
        return (
            - J*sum_hop(L, N, theta, b_m, b_n)
            - J*sum_hop_t(L, N, theta, b_m, b_n)
            + U/2.0*sum_onsite_int(L, b_m, b_n)
        )

    elif bc == 'periodic':
        return (
            - J*sum_hop(L, N, theta, b_m, b_n)
            - J*hop_term_L(N, theta, b_m, b_n)
            - J*sum_hop_t(L, N, theta, b_m, b_n)
            - J*hop_term_L_dagger(N, theta, b_m, b_n)
            + U/2.0*sum_onsite_int(L, b_m, b_n)
        )

    else:
        raise NotImplementedError


#-----------------------------------------------------------------------
def get_perturbed_H_eff(H0_mat, V_mat, evals_E, evecs_E, evals_E_compl,
                        evecs_E_compl, lam, order):

    if order >= 2:
        VphiphiV_list = [
            V_mat.dot((np.outer(evec, evec.conjugate())).dot(V_mat))
            for evec in evecs_E_compl
        ]

    def get_second_pert_term(E_n, psi_n, psi_m):
        mat_sum = 0
        for l, E_l in enumerate(evals_E_compl):
            mat_sum += VphiphiV_list[l]/(E_n - E_l)

        return lam**2 * np.vdot(psi_n, mat_sum.dot(psi_m))


    H_eff = np.zeros((len(evecs_E), len(evecs_E)), dtype=complex)
    for i, psi_n in enumerate(evecs_E):
        for j, psi_m in enumerate(evecs_E):

            H0_mn = np.vdot(psi_n, H0_mat.dot(psi_m))
            V_mn = lam*np.vdot(psi_n, V_mat.dot(psi_m))

            H_eff[i, j] = H0_mn + V_mn

            if order >= 2:
                Heff2_mn = get_second_pert_term(evals_E[i], psi_n, psi_m)
                H_eff[i, j] += Heff2_mn

    return H_eff


#-----------------------------------------------------------------------
@njit
def bi_dagger_bj(basis1, basis2, i, j, N):
    "<1,2,0,1| b_i^t b_j | 2,2,0,0>"

    b1 = basis1.copy()
    b2 = basis2.copy()

    # b_j |2>
    if b2[j] == 0:
        return 0
    else:
        bj = math.sqrt(b2[j])
        b2[j] -= 1

    # b_i^t b_j |2>
    if b2[i] == N:
        return 0
    else:
        bi_dagger = math.sqrt(b2[i] + 1)
        b2[i] += 1

    if nstate_allclose(b1, b2):
        return bi_dagger*bj
    else:
        return 0



#-----------------------------------------------------------------------
@njit
def bj_dagger_bi_dagger_bi_bj(basis1, basis2, i, j, N):
    "<1,2,0,1| b_j^t b_i^t b_i b_j | 2,2,0,0>"

    b1 = basis1.copy()
    b2 = basis2.copy()

    # b_j |2>
    if b2[j] == 0:
        return 0
    else:
        bj = math.sqrt(b2[j])
        b2[j] -= 1

    # b_i |2>'
    if b2[i] == 0:
        return 0
    else:
        bi = math.sqrt(b2[i])
        b2[i] -= 1

    # b_i^t |2>''
    if b2[i] == N:
        return 0
    else:
        bi_dagger = math.sqrt(b2[i] + 1)
        b2[i] += 1

    # b_j^t |2>'''
    if b2[j] == N:
        return 0
    else:
        bj_dagger = math.sqrt(b2[j] + 1)
        b2[j] += 1

    if nstate_allclose(b1, b2):
        return bj_dagger*bi_dagger*bj*bj
    else:
        return 0



#-----------------------------------------------------------------------
def K_operator(L, theta, basis1, basis2):
    """ symmetry operator
        PRL 118, 120401 (2017)
        https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.118.120401
        Eq. (12)

        K = exp(i\theta sum_{j=1}^L n_j(n_j-1)/2) * I * T

        I : inversion operator: |1, 2, 3, 4>  -->  |4, 3, 2, 1>
        T : time-reversal op: np.conjugate(|1, 2, 3, 4>)

        Apply time-reversal operator separately!
    """

    b1 = basis1.copy()
    b2 = basis2.copy()

    # spatial inversion
    b2 = np.flip(b2)

    if np.allclose(b1, b2):
        sum_njnj = sum([b2[j]*(b2[j] - 1) for j in range(L)])
        exp_term = np.exp(1j*theta*sum_njnj/2)
        return exp_term

    else:
        return 0


#-----------------------------------------------------------------------
def pair_operator(L, basis1, basis2):
    """ \nu_p = \sum_j \hat{n}_j (\hat{n}_j - 1)/2  """

    if np.allclose(basis1, basis2):
        sum_njnj = sum([basis1[j]*(basis1[j] - 1)/2 for j in range(L)])
        return sum_njnj

    else:
        return 0

#===============================================================================
# momentum operator
#===============================================================================
def get_bibj_mat(basis_list, i, j, N):
    r"""\langle b_i^\dagger b_j \rangle
        = <psi | b_i^\dagger b_j | psi >
        = (<4,0,0,0|c_1 + <3,1,0,0|c_2 + ... ) | b_i^\dagger b_j | psi >
        """

    bibj_mat = np.zeros((len(basis_list), len(basis_list)), dtype=complex)
    for ind_1, basis1 in enumerate(basis_list):
        for ind_2, basis2 in enumerate(basis_list):
            # apply bi^dagger b_j
            coeff_bibj = bi_dagger_bj(basis1, basis2, i, j, N)

            bibj_mat[ind_1, ind_2] = coeff_bibj

    return bibj_mat


def get_bibj_mats(basis_list, L):

    N = sum(basis_list[0])

    bibj_mats = np.zeros(
        (L, L, len(basis_list), len(basis_list)),
        dtype=complex
    )

    for m in range(0, L):
        for n in range(0, L):
            bibj_mats[m, n, :, :] = get_bibj_mat(basis_list, m, n, N)

    return bibj_mats


#===============================================================================
def get_bjbibibj_mat(basis_list, i, j, N):

    bjbibibj_mat = np.zeros((len(basis_list), len(basis_list)), dtype=complex)
    for ind_1, basis1 in enumerate(basis_list):
        for ind_2, basis2 in enumerate(basis_list):
            # apply bi^dagger b_j
            coeff_bibj = bj_dagger_bi_dagger_bi_bj(basis1, basis2, i, j, N)

            bjbibibj_mat[ind_1, ind_2] = coeff_bibj

    return bjbibibj_mat


def get_bjbibibj_mats(basis_list, L):

    N = sum(basis_list[0])

    bjbibibj_mats = np.zeros(
        (L, L, len(basis_list), len(basis_list)),
        dtype=complex
    )

    for m in range(0, L):
        for n in range(0, L):
            bjbibibj_mats[m, n, :, :] = get_bjbibibj_mat(basis_list, m, n, N)

    return bjbibibj_mats
