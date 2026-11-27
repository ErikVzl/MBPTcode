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

Delta itself is energy dependent, Delta_tu(w) = sum_e b_te b_ue / (w - lam_e)
(particle side) + sum_i b_ti b_ui / (w - lam_i) (hole side), and is solved by
the dynamic backends of solvers/dynamic_backends.py. Its two static forms
live here: a one-body matrix D added to h, and a constant c added to the
energy, D_tu = 1/2 [Delta_tu(w_t) + Delta_tu(w_u)] at one energy per
orbital:

- 'static_mf' (`static_mf_delta`): w = the mean-field orbital energies
  eps_active; nothing is solved first.
- 'static_sa' (`state_averaged_delta`): w = the removal energies of the
  active virtual orbitals averaged over a declared set of states, from their
  density matrices (`removal_moment`); the record builds it once with
  `DownfoldedHamiltonian.prepare_delta` and keeps it as `Coupling.d_sa`.

The hole part of a static form is a one-body term with the sign of the
particle one plus the constant c = -g tr(D_hole), g the spin weight
(`spin_weight`): the static hole operator sum D_h a_t a+_u is a constant
minus a one-body term.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from pyscf import scf
from pyscf.scf.hf import get_jk

from src.Base.numerics.numerics import numerics_or_default

from .spin import spin_weight

#: the two blocks a rotation inside the occupied and inside the virtual
#: space can create: active virtual - environment virtual (particle side) and
#: active occupied - inactive occupied (hole side)
COUPLING_BLOCKS = ('av_ev', 'ao_i')

#: the blocks Delta folds by default
DEFAULT_BLOCKS = ('av_ev',)

#: the static forms of Delta: a one-body D added to h and a constant, solved
#: by any solver
STATIC_METHODS = ('static_sa', 'static_mf')

#: the values of delta_method that `DownfoldedHamiltonian.solve` takes
#: besides None (which is 'linearized') and False (no Delta)
DELTA_METHODS = ('linearized', 'lowdin', 'sa_remainder') + STATIC_METHODS


class DeltaOffWarning(UserWarning):
    """An active space on a rotated orbital basis built without Delta
    (`delta=False`): the coupling the rotation creates between active and
    environment orbitals is left out, and the result is the static
    Hamiltonian only."""


class StaticDeltaWarning(UserWarning):
    """delta_method='static_mf': Delta evaluated once, at the mean-field
    orbital energies of the active orbitals, instead of at the energy of
    each state. It is far from exact Loewdin, and further as the basis
    grows (the message gives the benzene benchmark of the handoff), so it
    warns on every use, with the flag 'delta_static_mf'."""


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
    d_sa: None, or (D, c), the static form 'static_sa' of Delta
        (`state_averaged_delta`): D (n_act, n_act) is added to h and c to the
        constant. Set by `DownfoldedHamiltonian.prepare_delta`, which
        returns a copy of the record with it.
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
    d_sa: Optional[tuple] = None

    def sides(self):
        """The coupling as plain data for the dynamic backends of the
        solvers package, [(kind, sigma, lam, b), ...]: kind 'particle' (an
        electron leaves the active space into an environment virtual level
        lam_virtual, b = b_particle) or 'hole' (an electron enters it from an
        inactive occupied level lam_occupied, b = b_hole); b (n_act,
        n_levels) with rows over every active orbital, zero where an orbital
        has no coupling, its column e paired with lam[e]. sigma is None: the
        Coupling of a restricted reference is one channel and its one side
        serves both spins (a per-spin coupling would name its spin, 0 or 1,
        when Delta exists for two channels). Only the blocks folded
        (`blocks`) appear, and a side whose b is identically zero is left
        out, so that no sector is built for it; no side at all means no
        Delta."""
        out = []
        for name, kind, lam, b in (('av_ev', 'particle', self.lam_virtual,
                                    self.b_particle),
                                   ('ao_i', 'hole', self.lam_occupied,
                                    self.b_hole)):
            if name in self.blocks and np.any(b):
                out.append((kind, None, lam, b))
        return out


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


def removal_moment(h1, eri, dm1, dm2):
    """The generalized Fock matrix of one state K, (norb, norb):

        M1[t, u] = sum_q h1[u, q] dm1[t, q] + sum_qrs (uq|rs) dm2[t, q, r, s],

    h1 and eri (chemist) the static active Hamiltonian K is an eigenstate
    of, and dm1, dm2 its spin-summed density matrices in pyscf's convention,
    dm1[p, q] = <q+ p>, dm2[p, q, r, s] = <p+ r+ s q>. For an eigenstate it
    is the first moment of the removal energies,
    sum_sigma <a_t K|(E_K - H^(N-1))|a_u K>, a symmetric matrix."""
    return (np.einsum('uq,tq->tu', h1, dm1)
            + np.einsum('uqrs,tqrs->tu', eri, dm2))


def _static_block(b, lam, w):
    """D_tu = 1/2 [Delta_tu(w_t) + Delta_tu(w_u)], Delta_tu(x) = sum_e b_te
    b_ue / (x - lam_e), over the rows of b; rows of b that are zero give
    zero rows and columns, and their w is not read."""
    out = np.zeros((len(b), len(b)))
    rows = np.flatnonzero(np.any(b != 0.0, axis=1))
    if len(rows) == 0:
        return out
    br = b[rows]
    at_t = (br / (w[rows, None] - lam[None, :])) @ br.T     # Delta_tu(w_t)
    out[np.ix_(rows, rows)] = 0.5 * (at_t + at_t.T)
    return out


def static_mf_delta(coupling):
    """(D, c): the static form 'static_mf' of Delta, at the mean-field
    energies eps = coupling.eps_active (the diagonal of the bare Fock
    operator on the active block, the operator the levels come from):

        D_tu = 1/2 [Delta_tu(eps_t) + Delta_tu(eps_u)],
        Delta_tu(w) = sum_e b_te b_ue / (w - lam_e)      particle side
                    + sum_i b_ti b_ui / (w - lam_i)      hole side.

    D is added to h. c = -g tr(D_hole), the hole part alone, g the spin
    weight (2 for the one channel of a restricted reference), is added to
    the energy: the static hole operator sum D_h a_t a+_u is a constant
    minus a one-body term, and that one-body term has the sign of the
    particle one. c is the same for every state, so excitation energies do
    not see it and totals and barriers do. With the particle side alone
    (the default blocks) c = 0. A plain function: the warning of
    'static_mf' (StaticDeltaWarning) is raised where the method is chosen."""
    eps = np.asarray(coupling.eps_active, dtype=float)
    particle = _static_block(np.asarray(coupling.b_particle, dtype=float),
                             np.asarray(coupling.lam_virtual, dtype=float),
                             eps)
    hole = _static_block(np.asarray(coupling.b_hole, dtype=float),
                         np.asarray(coupling.lam_occupied, dtype=float), eps)
    c = -spin_weight((coupling,)) * float(np.trace(hole))
    return particle + hole, c


def state_averaged_delta(h1, eri, coupling, rdms, numerics):
    """(D, c): the static form 'static_sa' of Delta, D_SA, from a set of
    states of the static Hamiltonian (h1, eri chemist), which enter with
    equal weight. rdms: [(dm1, dm2), ...], one spin-summed pair per state
    (pyscf's convention, `removal_moment`); numerics: a `Numerics` (None:
    the defaults).

    On the active virtual orbitals: gamma, the averaged dm1, and M1, the
    averaged `removal_moment`. The eigenvectors V of gamma whose occupation
    n is below numerics.downfolding.delta_occupation_floor are dropped: no
    state of the average removes an electron from them, so they have no
    removal energy, and D is zero along them. In the kept directions V_r,
    Omega = n^-1/2 V_r^T M1 V_r n^-1/2 = U diag(w) U^T is the state-averaged
    removal energy; in the orbitals V_r U, with b' = (V_r U)^T b,

        D'_kl = 1/2 [Delta'_kl(w_k) + Delta'_kl(w_l)],
        Delta'_kl(x) = sum_e b'_ke b'_le / (x - lam_e),

    and D = (V_r U) D' (V_r U)^T on the active virtual block, zero
    elsewhere. The particle side only: a coupling with a hole side is
    refused (NotImplementedError, Task 13), and c = 0. No state at all is
    refused (ValueError)."""
    if np.any(coupling.b_hole):
        raise NotImplementedError(
            'D_SA of a coupling with a hole side (active occupied - inactive '
            'occupied, a rotated occupied block): its hole block and '
            'constant are not built yet (Task 13).')
    if not rdms:
        raise ValueError('state_averaged_delta needs at least one state.')
    floor = numerics_or_default(numerics).downfolding.delta_occupation_floor
    n_act = len(coupling.eps_active)
    av = np.arange(coupling.n_act_occ, n_act)
    gamma = np.mean([np.asarray(dm1)[np.ix_(av, av)] for dm1, _ in rdms],
                    axis=0)
    m1 = np.mean([removal_moment(h1, eri, dm1, dm2)[np.ix_(av, av)]
                  for dm1, dm2 in rdms], axis=0)
    occupation, v = np.linalg.eigh(0.5 * (gamma + gamma.T))
    kept = occupation > floor
    d = np.zeros((n_act, n_act))
    if not np.any(kept):
        return d, 0.0
    v_r = v[:, kept]
    scale = 1.0 / np.sqrt(occupation[kept])
    omega = scale[:, None] * (v_r.T @ m1 @ v_r) * scale[None, :]
    w, u = np.linalg.eigh(0.5 * (omega + omega.T))
    orbitals = v_r @ u
    b = orbitals.T @ np.asarray(coupling.b_particle, dtype=float)[av]
    d_prime = _static_block(b, np.asarray(coupling.lam_virtual, dtype=float),
                            w)
    d[np.ix_(av, av)] = orbitals @ d_prime @ orbitals.T
    return d, 0.0
