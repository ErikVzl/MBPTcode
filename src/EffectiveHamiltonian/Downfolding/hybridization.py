"""The Fock coupling a rotated orbital basis creates between the active and
the environment orbitals: what Delta folds (spec 6, 6e).

A rotation of the canonical orbitals inside the occupied and inside the
virtual space (natural orbitals, `OrbitalBasis`) leaves the reference's own
operator F_s = T^T diag(eps) T non-diagonal between active and environment
orbitals of the same space. That coupling, and nothing else, is what Delta
folds:

    F_s  = T^T diag(eps) T               the reference's own operator
    F    = C^T (h0 + J[D] - K[D]/2) C     the bare Fock operator, exact J, K
           (spin s of an unrestricted reference: C^s^T (h0 + J[D_a + D_b]
           - K[D_s]) C^s, spec 6c; the coupling conserves spin, one per spin)
    lam_E, U_E = eigh F(E_v, E_v);  lam_I, U_I = eigh F(I, I)
    B_p  = F_s(A_v, E_v) U_E               an electron leaves the active space
    B_h  = F_s(A_o, I) U_I                 an electron enters it

For canonical orbitals F_s is diagonal and the coupling is identically zero
on any reference. X = F - F_s, the exchange part Sigma_x - v_xc of the bare
Fock operator on a reference that is not HF, is not folded between active
and environment orbitals (spec 6e); its norms are reported.
"""
from dataclasses import dataclass

import numpy as np
from pyscf import scf
from pyscf.scf.hf import get_jk

#: the two blocks a rotation inside the occupied and inside the virtual
#: space can create: active virtual - environment virtual (particle side) and
#: active occupied - inactive occupied (hole side)
COUPLING_BLOCKS = ('av_ev', 'ao_i')

#: the blocks Delta folds by default
DEFAULT_BLOCKS = ('av_ev',)


class DeltaOffWarning(UserWarning):
    """An active space on a rotated orbital basis built without Delta
    (`delta=False`): the coupling the rotation creates between active and
    environment orbitals is left out, and the result is the static
    Hamiltonian only."""


@dataclass(frozen=True)
class Coupling:
    """The Fock coupling of one spin channel between its active and its
    environment orbitals, in the semicanonical basis of its active space.

    b_particle: (n_act, n_ev), F_s(A, E_v) U_E: nonzero on the active
        virtual rows only. Rows follow `space.act` (occupied first).
    lam_virtual: (n_ev,), the levels of the environment virtual orbitals,
        eigenvalues of the bare Fock operator F(E_v, E_v).
    b_hole: (n_act, n_i), F_s(A, I) U_I: nonzero on the active occupied
        rows only.
    lam_occupied: (n_i,), eigenvalues of F(I, I).
    eps_active: (n_act,), diag F(A, A).
    norms: {name: Frobenius norm} of each block of COUPLING_BLOCKS, whether
        folded or not.
    left_out: {name: Frobenius norm} of X = F - F_s between active and
        environment orbitals ('av_ev', 'ao_i') and between occupied and
        virtual ones ('ao_ev', 'i_av', 'i_ev'): what Delta does not fold.
        Zero for an HF mean field with exact integrals; the size of the fit
        error on a density-fitted HF one.
    blocks: the blocks folded; the rows of a block not in `blocks` are zero.
    n_act_occ: the number of active occupied orbitals (the first rows).
    """
    b_particle: np.ndarray
    lam_virtual: np.ndarray
    b_hole: np.ndarray
    lam_occupied: np.ndarray
    eps_active: np.ndarray
    norms: dict
    left_out: dict
    blocks: tuple
    n_act_occ: int


def fock_coupling(mf, mo_rotated, T, eps, active_space,
                  blocks=COUPLING_BLOCKS, spin=None):
    """The `Coupling` of one spin channel of the reference `mf`.

    mo_rotated = mo_coeff @ T are the orbitals of the active space, T its
    semicanonical rotation (`semicanonical_rotation`) and eps the canonical
    orbital energies of the reference, all of that channel. The bare Fock
    operator is built with the exact four-centre J and K of `mf.mol`: from
    D = mf.make_rdm1() for a restricted reference (spin=None), and for spin
    s (0 alpha, 1 beta) of an unrestricted one from (D_a, D_b) =
    mf.make_rdm1() as h0 + J[D_a + D_b] - K[D_s] (spec 6c). An unrestricted
    reference without `spin`, or a restricted one with it, is refused
    (ValueError). `blocks` is a subset of COUPLING_BLOCKS; the other block
    is zeroed, its norm reported."""
    unrestricted = isinstance(mf, scf.uhf.UHF)
    if unrestricted and spin not in (0, 1):
        raise ValueError(
            f'spin={spin!r}: the Fock coupling of an unrestricted reference '
            f'is one per spin channel, spin=0 (alpha) or 1 (beta).')
    if not unrestricted and spin is not None:
        raise ValueError(
            f'spin={spin!r}: a restricted reference is one spin channel; '
            f'leave spin=None.')
    unknown = [b for b in blocks if b not in COUPLING_BLOCKS]
    if unknown:
        raise ValueError(f'unknown coupling blocks {unknown}; the blocks are '
                         f'{COUPLING_BLOCKS}.')
    space = active_space
    if unrestricted:
        j, k = get_jk(mf.mol, np.asarray(mf.make_rdm1()))
        fock = mf.get_hcore() + j[0] + j[1] - k[spin]
    else:
        j, k = get_jk(mf.mol, mf.make_rdm1())
        fock = mf.get_hcore() + j - 0.5 * k
    F = mo_rotated.T @ fock @ mo_rotated
    F_s = T.T @ (np.asarray(eps)[:, None] * T)
    X = F - F_s
    act, n_occ = space.act, space.n_act_occ
    av, ao, ev, i = (space.virt_active, space.occ_active, space.env_virt,
                     space.env_occ)

    def sub(M, rows, cols):
        return M[np.ix_(rows, cols)]

    lam_virtual, u_e = np.linalg.eigh(sub(F, ev, ev))
    lam_occupied, u_i = np.linalg.eigh(sub(F, i, i))
    b_particle = np.zeros((len(act), len(ev)))
    b_particle[n_occ:] = sub(F_s, av, ev) @ u_e
    b_hole = np.zeros((len(act), len(i)))
    b_hole[:n_occ] = sub(F_s, ao, i) @ u_i
    norms = {'av_ev': float(np.linalg.norm(b_particle)),
             'ao_i': float(np.linalg.norm(b_hole))}
    if 'av_ev' not in blocks:
        b_particle[:] = 0.0
    if 'ao_i' not in blocks:
        b_hole[:] = 0.0
    left_out = {name: float(np.linalg.norm(sub(X, r, c)))
                for name, (r, c) in (('av_ev', (av, ev)), ('ao_i', (ao, i)),
                                     ('ao_ev', (ao, ev)), ('i_av', (i, av)),
                                     ('i_ev', (i, ev)))}
    return Coupling(b_particle, lam_virtual, b_hole, lam_occupied,
                    np.diag(sub(F, act, act)).copy(), norms, left_out,
                    tuple(blocks), n_occ)
