"""Unified solver interface for (downfolded) active-space Hamiltonians.

solve() takes an effective Hamiltonian g_eff (dict body-rank -> tensor, as
produced by the builders of src/EffectiveHamiltonian: QDPT/perturbative.py
and Downfolding/effective_hamiltonian.py) and dispatches to one of the
many-body solver backends:
  'openfermion' (default) -- explicit determinant-basis matrix + exact
                             diagonalization (ExactDiagonalizationSolver)
  'pyscf_fci'             -- pyscf's FCI at fixed Sz (PyscfFCISolver); the
                             one backend that also solves a spin-resolved
                             g_eff, the per-spin blocks of an unrestricted
                             reference, with fci.direct_uhf
                             (PyscfSpinResolvedFCISolver)
  'dmrg'                  -- DMRG through block2, SU2 spin-adapted
                             (Block2DMRGSolver); reaches active spaces the
                             two exact engines above cannot. block2 must be
                             imported before numpy in the calling process (see
                             docs/installation.md) -- safe from a fresh
                             subprocess, not safe after another backend has
                             already run in the same process.
  'ccsdt', 'ccsdtq'       -- high-order coupled cluster on a Hartree-Fock
                             reference in the active space
                             (CoupledClusterSolver, order 3 or 4), energy
                             only, one state: the lowest singlet when
                             n_alpha = n_beta, else the lowest high-spin
                             state; for active spaces too large for FCI.

Any other name is refused (`SOLVERS`); `ENGINES` names the backend class
behind each, read for what it offers before anything is solved.

Every backend but 'openfermion' goes through the ActiveSpaceSolver
interface of backends.py. `static_states` is the static solve of
'pyscf_fci' and 'dmrg' as `solve` runs it, returning the states, the plain
engine and the spatial Hamiltonian instead of the energies alone, for a
caller that goes on from the states: the dynamic backends of
dynamic_backends.py, which refine them with Delta when `solve` is given the
coupling (`sides`), and the density matrices of the static form of Delta
D_SA (`DownfoldedHamiltonian.prepare_delta`). `SolveRequest` is one solve
as the backends read it.
"""
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
from pyscf import fci

from src.Base.numerical_flags import flag, flags_into_out
from src.Base.numerics.constants import (ACTIVE_SPACE_VERTEX_TOL, DMRG_BOND_DIM,
                                         DMRG_N_SWEEPS,
                                         FCI_SPIN_PENALTY_SHIFT,
                                         FCI_SPIN_SQUARE_TOL)
from src.Base.numerics.numerics import numerics_or_default
from src.EffectiveHamiltonian.solvers.backends import (
    Block2DMRGSolver, CoupledClusterSolver, ExactDiagonalizationSolver,
    PyscfFCISolver, PyscfSpinResolvedFCISolver, spin_sector,
    total_spin_square)
from src.EffectiveHamiltonian.solvers.determinants import (
    spatial_to_spin_orbitals)
from src.EffectiveHamiltonian.solvers.dynamic_backends import (
    DYNAMIC_METHODS, DeltaRootsCrossedWarning, degenerate_sets)

#: The coupled-cluster backends and their excitation order.
CC_ORDERS = {'ccsdt': 3, 'ccsdtq': 4}


class EmptyActiveSpaceWarning(UserWarning):
    """`solve` was given no active orbitals: there was nothing to diagonalize,
    and the one state's energy is the constant g_eff[0] (E_act = 0)."""


def _spatial_active_hamiltonian(g_eff, n_spin_orbs, n_elec_active, spin_free,
                                symmetrize, solver):
    """(h1_spatial, eri_spatial, norb, (nocca, noccb)) for a spatial-orbital,
    chemist-notation-eri ActiveSpaceSolver (PyscfFCISolver, Block2DMRGSolver,
    CoupledClusterSolver)
    from a downfolded g_eff (spin-orbital unless spin_free), the Hamiltonian
    unchanged: these engines take any Hermitian one- and two-body operator,
    with no 8-fold integral symmetry required.

    Refused rather than altered: a three-body (or higher) vertex they cannot
    carry; a spin-orbital g_eff that is not spin-adapted, whose same-spin
    blocks would be lost on folding to spatial orbitals; and a non-Hermitian
    g_eff unless symmetrize=True, which takes its Hermitian part, as the
    determinant-basis solver does."""
    for k in sorted(k for k in g_eff if isinstance(k, int) and k >= 3):
        biggest = float(np.max(np.abs(g_eff[k]))) if np.size(g_eff[k]) else 0.0
        if biggest > ACTIVE_SPACE_VERTEX_TOL:
            raise ValueError(
                f"solver={solver!r} takes a one- and two-body Hamiltonian, but "
                f"g_eff[{k}] has elements up to {biggest:.2e}; build g_eff with "
                f"max_vertex=2, or use solver='openfermion', which carries it.")
    if spin_free:
        h1_spatial = g_eff[1]
        eri_spatial = g_eff[2].transpose(0, 2, 1, 3)
        norb = n_spin_orbs
        nocca = n_elec_active // 2
        noccb = n_elec_active - nocca
    else:
        alpha = np.arange(0, n_spin_orbs, 2)
        beta = np.arange(1, n_spin_orbs, 2)
        norb = n_spin_orbs // 2
        h1_spatial = g_eff[1][np.ix_(alpha, alpha)]
        g2_ab = g_eff[2][np.ix_(alpha, beta, alpha, beta)]          # spatial <pq|rs>
        folded = spatial_to_spin_orbitals({1: h1_spatial, 2: g2_ab}, norb)
        residual = max(float(np.max(np.abs(folded[1] - g_eff[1]))),
                       float(np.max(np.abs(folded[2] - g_eff[2]))))
        if residual > ACTIVE_SPACE_VERTEX_TOL * max(1.0, float(np.max(np.abs(g_eff[2])))):
            raise ValueError(
                f"solver={solver!r} works on spatial orbitals, but this "
                f"spin-orbital g_eff is not spin-adapted (it differs from the "
                f"one its alpha-beta block implies by {residual:.2e}); folding "
                f"it would discard its independent spin blocks. Use "
                f"solver='openfermion', or, for the Hamiltonian of an "
                f"unrestricted reference, its per-spin blocks (built with "
                f"spin_free=True) with solver='pyscf_fci'.")
        eri_spatial = g2_ab.transpose(0, 2, 1, 3)
        nocca = sum(1 for i in range(n_elec_active) if i % 2 == 0)
        noccb = sum(1 for i in range(n_elec_active) if i % 2 == 1)

    # (pq|rs) <-> (rs|pq) relabels the same normal-ordered operator
    eri_spatial = 0.5 * (eri_spatial + eri_spatial.transpose(2, 3, 0, 1))
    if symmetrize:
        h1_spatial = 0.5 * (h1_spatial + h1_spatial.T)
        eri_spatial = 0.5 * (eri_spatial + eri_spatial.transpose(1, 0, 3, 2))
    else:
        asym = max(float(np.max(np.abs(h1_spatial - h1_spatial.T))),
                   float(np.max(np.abs(eri_spatial - eri_spatial.transpose(1, 0, 3, 2)))))
        if asym > ACTIVE_SPACE_VERTEX_TOL:
            raise ValueError(
                f"solver={solver!r} needs a Hermitian Hamiltonian, and this "
                f"g_eff is not (anti-Hermitian part up to {asym:.2e}); pass "
                f"symmetrize=True for its Hermitian part, or use "
                f"solver='openfermion' with symmetrize=False for the "
                f"non-Hermitian spectrum.")
    return h1_spatial, eri_spatial, norb, (nocca, noccb)


@dataclass(frozen=True)
class SolveRequest:
    """One static solve, as every backend reads it: the Hamiltonian dict
    g_eff ({0: constant, 1: h, 2: v}, `solve`), the active orbitals
    n_spin_orbs and electrons n_elec_active it is written on (the pair
    (N_alpha, N_beta) for a spin-resolved g_eff), spin_free, symmetrize,
    nroots, the DMRG settings dmrg_bond_dim and dmrg_n_sweeps, and spin
    (None or 2S), each as `solve` takes it and with its default."""
    g_eff: dict
    n_spin_orbs: int
    n_elec_active: object
    spin_free: bool = False
    symmetrize: bool = True
    nroots: Optional[int] = 1
    dmrg_bond_dim: int = DMRG_BOND_DIM
    dmrg_n_sweeps: int = DMRG_N_SWEEPS
    spin: Optional[int] = None

    def spatial(self, solver):
        return _spatial_active_hamiltonian(
            self.g_eff, self.n_spin_orbs, self.n_elec_active, self.spin_free,
            self.symmetrize, solver)


@dataclass(frozen=True)
class StaticStates:
    """The states of one static solve (`static_states`) and what a caller
    needs to go on from them.

    energies (nroots,): ascending, without any constant; vectors: one per
    state, as the engine's `roots` returned them; engine: the plain static
    backend of the solver (PyscfFCISolver without the spin penalty, which
    belongs to the search for the states and not to their density matrices
    or sector operations; the Block2DMRGSolver that ran, whose `rdms` read
    the states of its last `roots` call); h1, eri, norb: the spatial
    Hamiltonian the solver diagonalized (`_spatial_active_hamiltonian`),
    eri chemist with (pq|rs) = (rs|pq); with symmetrize=True, the default,
    h1 is its Hermitian part and eri also has (pq|rs) = (qp|sr), and with
    symmetrize=False a g_eff whose anti-Hermitian part exceeds
    ACTIVE_SPACE_VERTEX_TOL is refused; sector: the
    (n_alpha, n_beta) the vectors live in, for spin=2S the Ms = S sector,
    not the one of the reference."""
    energies: np.ndarray
    vectors: list
    engine: object
    h1: np.ndarray
    eri: np.ndarray
    norb: int
    sector: tuple


def _pyscf_fci_states(r):
    """The `StaticStates` of 'pyscf_fci' for a checked request."""
    h1, eri, norb, nelec = r.spatial('pyscf_fci')
    if r.spin is None:
        # every state at this Sz, the space direct_spin1 spans here
        engine = PyscfFCISolver()
        energies, vectors = engine.roots(h1, eri, norb, nelec,
                                         nroots=r.nroots, spin=None)
        return StaticStates(energies, vectors, engine, h1, eri, norb, nelec)
    # the states of total spin S: Ms = S and the spin penalty, then every
    # root's <S^2> checked, since one of another spin carries the penalty
    energies, vectors = PyscfFCISolver(
        spin_penalty=FCI_SPIN_PENALTY_SHIFT).roots(
        h1, eri, norb, nelec, nroots=r.nroots, spin=r.spin)
    want = 0.25 * r.spin * (r.spin + 2)
    sector = spin_sector(nelec, r.spin)
    got = [fci.spin_op.spin_square(v, norb, sector)[0] for v in vectors]
    wrong = [(k, s) for k, s in enumerate(got)
             if abs(s - want) > FCI_SPIN_SQUARE_TOL]
    if wrong:
        raise ValueError(
            f"spin={r.spin}: roots {[k for k, _ in wrong]} of the penalized "
            f"FCI have <S^2> = {[round(s, 6) for _, s in wrong]}, not "
            f"S(S+1) = {want:g}: a state of another spin lies within the "
            f"penalty (FCI_SPIN_PENALTY_SHIFT = {FCI_SPIN_PENALTY_SHIFT:g} "
            f"Eh times the S^2 difference) of the requested ones, and its "
            f"energy carries that penalty. Ask for fewer roots.")
    return StaticStates(energies, vectors, PyscfFCISolver(), h1, eri, norb,
                        sector)


def _dmrg_states(r):
    """The `StaticStates` of 'dmrg' for a checked request."""
    h1, eri, norb, nelec = r.spatial('dmrg')
    nocca, noccb = nelec
    # an SU2 target is a total spin, unlike pyscf_fci's spin=None: the
    # requested one, else n_alpha - n_beta, the lowest S at this Sz
    spin = nocca - noccb if r.spin is None else r.spin
    engine = Block2DMRGSolver(bond_dim=r.dmrg_bond_dim,
                              n_sweeps=r.dmrg_n_sweeps)
    energies, vectors = engine.roots(h1, eri, norb, nelec, nroots=r.nroots,
                                     spin=spin)
    return StaticStates(energies, vectors, engine, h1, eri, norb,
                        spin_sector(nelec, spin))


#: The solvers whose static solve hands on its states (`static_states`).
STATE_SOLVERS = {'pyscf_fci': _pyscf_fci_states, 'dmrg': _dmrg_states}


def static_states(solver, request):
    """The static solve of `request` (a `SolveRequest`) by `solver`, as
    `solve` runs it, up to and including its check of <S^2>: a
    `StaticStates` with the energies (no constant), the vectors, the plain
    engine, the spatial Hamiltonian and the sector.

    'pyscf_fci' and 'dmrg' only; any other solver is refused (ValueError):
    'openfermion' keeps the constant inside its matrix and gives no vectors
    through `roots`, and the coupled-cluster backends give an energy alone.
    The request is checked as `solve` checks it (the electron count, the
    spin), and refused (ValueError) for a spin-resolved g_eff and for an
    empty active space, which has no states to hand on."""
    if solver not in STATE_SOLVERS:
        raise ValueError(
            f"static_states takes solver {tuple(STATE_SOLVERS)}, the backends "
            f"whose static solve hands on its states; got {solver!r} "
            f"('openfermion' keeps the constant inside its matrix, the "
            f"coupled-cluster backends give an energy alone).")
    request = _checked(solver, request)
    if _spin_resolved(request.g_eff):
        raise ValueError(
            'static_states takes a spatial g_eff; a spin-resolved one (the '
            'per-spin blocks of an unrestricted reference) is solved by '
            "solve(..., solver='pyscf_fci').")
    if request.n_spin_orbs == 0:
        raise ValueError(
            'the active space is empty (n_spin_orbs=0): it has no states to '
            'hand on, refine or average.')
    return STATE_SOLVERS[solver](request)


def _pyscf_fci(r, out):
    st = _pyscf_fci_states(r)
    if out is not None:
        out['spin_square'] = [fci.spin_op.spin_square(v, st.norb,
                                                      st.sector)[0]
                              for v in st.vectors]
    e0 = r.g_eff.get(0, 0.0)
    return e0, st.energies, e0 + st.energies


def _dmrg(r, out):
    st = _dmrg_states(r)
    e0 = r.g_eff.get(0, 0.0)
    return e0, st.energies, e0 + st.energies


def _coupled_cluster(name):
    def run(r, out):
        h1, eri, norb, nelec = r.spatial(name)
        energies = CoupledClusterSolver(order=CC_ORDERS[name]).roots(
            h1, eri, norb, nelec, nroots=r.nroots, spin=None)[0]
        e0 = r.g_eff.get(0, 0.0)
        return e0, energies, e0 + energies
    return run


def _exact(r, out):
    g_eff, n_spin_orbs = r.g_eff, r.n_spin_orbs
    if r.spin_free:
        g_eff = spatial_to_spin_orbitals(g_eff, n_spin_orbs)
        n_spin_orbs = 2 * n_spin_orbs
    # the whole determinant-basis spectrum, a three-body vertex carried and
    # the constant inside the matrix; the lowest nroots of it
    energies = ExactDiagonalizationSolver(symmetrize=r.symmetrize).spectrum(
        g_eff, n_spin_orbs, r.n_elec_active)[0]
    energies = energies if r.nroots is None else energies[:r.nroots]
    e0 = r.g_eff.get(0, 0.0)
    return e0, energies - e0, energies


#: The backends `solve` dispatches to, by name; any other name is refused.
#: Each runner takes (request, out) and returns (constant, active energies,
#: energies).
SOLVERS = {'openfermion': _exact, 'pyscf_fci': _pyscf_fci, 'dmrg': _dmrg,
           'ccsdt': _coupled_cluster('ccsdt'),
           'ccsdtq': _coupled_cluster('ccsdtq')}

#: The static backend class behind each solver name, read for what it
#: offers (`has_rdms`, `has_sector_operations`) before anything is solved.
ENGINES = {'openfermion': ExactDiagonalizationSolver,
           'pyscf_fci': PyscfFCISolver, 'dmrg': Block2DMRGSolver,
           'ccsdt': CoupledClusterSolver, 'ccsdtq': CoupledClusterSolver}


def _spin_resolved(g_eff):
    """Whether g_eff holds per-spin blocks: g_eff[1] = (h_alpha, h_beta) and
    g_eff[2] = (aa, ab, bb), the Hamiltonian of an unrestricted reference."""
    return isinstance(g_eff.get(1), (tuple, list))


def _spin_resolved_hamiltonian(g_eff, norb, symmetrize, solver):
    """((h_a, h_b), (aa, ab, bb) in chemist notation) of a spin-resolved
    g_eff, whose two-body blocks are physicist, block[p,q,r,s] = <pq|rs> =
    (pr|qs), the opposite-spin one <p_a q_b | r_a s_b>.

    As for a spatial g_eff: a three-body vertex is refused; symmetrize takes
    the Hermitian part of every block, and without it a non-Hermitian block
    is refused. Exchanging the two pairs, (pq|rs) <-> (rs|pq), relabels the
    same operator inside one spin and is averaged there; between the spins
    it would exchange the alpha and the beta orbitals, so the opposite-spin
    block is left as it is. fci.direct_uhf packs each pair of a block, so a
    block that is not symmetric inside its pairs, (pq|rs) = (qp|rs) =
    (pq|sr), is refused."""
    for k in sorted(k for k in g_eff if isinstance(k, int) and k >= 3):
        biggest = max((float(np.max(np.abs(x))) for x in np.atleast_1d(g_eff[k])
                       if np.size(x)), default=0.0)
        if biggest > ACTIVE_SPACE_VERTEX_TOL:
            raise ValueError(
                f"solver={solver!r} takes a one- and two-body Hamiltonian, but "
                f"g_eff[{k}] has elements up to {biggest:.2e}.")
    h = [np.asarray(x) for x in g_eff[1]]
    blocks = [np.asarray(x) for x in g_eff[2]]
    if (len(h) != 2 or len(blocks) != 3
            or any(x.shape != (norb, norb) for x in h)
            or any(x.shape != (norb,) * 4 for x in blocks)):
        raise ValueError(
            f"a spin-resolved g_eff holds g_eff[1] = (h_alpha, h_beta), each "
            f"({norb}, {norb}), and g_eff[2] = (aa, ab, bb), each "
            f"{(norb,) * 4}; got shapes {[x.shape for x in h]} and "
            f"{[x.shape for x in blocks]}.")
    aa, ab, bb = (x.transpose(0, 2, 1, 3) for x in blocks)
    aa = 0.5 * (aa + aa.transpose(2, 3, 0, 1))
    bb = 0.5 * (bb + bb.transpose(2, 3, 0, 1))
    if symmetrize:
        h = [0.5 * (x + x.T) for x in h]
        aa, ab, bb = (0.5 * (x + x.transpose(1, 0, 3, 2)) for x in (aa, ab, bb))
    else:
        asym = max([float(np.max(np.abs(x - x.T))) for x in h]
                   + [float(np.max(np.abs(x - x.transpose(1, 0, 3, 2))))
                      for x in (aa, ab, bb)])
        if asym > ACTIVE_SPACE_VERTEX_TOL:
            raise ValueError(
                f"solver={solver!r} needs a Hermitian Hamiltonian, and this "
                f"spin-resolved g_eff is not (anti-Hermitian part up to "
                f"{asym:.2e}); pass symmetrize=True for its Hermitian part.")
    scale = max(1.0, max(float(np.max(np.abs(x))) for x in (aa, ab, bb)))
    packed = max(float(np.max(np.abs(x - x.transpose(1, 0, 2, 3))))
                 for x in (aa, ab, bb))
    if packed > ACTIVE_SPACE_VERTEX_TOL * scale:
        raise ValueError(
            f"solver={solver!r} solves a spin-resolved Hamiltonian with "
            f"fci.direct_uhf, which needs every two-body block symmetric "
            f"inside its pairs, (pq|rs) = (qp|rs) = (pq|sr), as integrals over "
            f"real orbitals are; this g_eff breaks that by up to {packed:.2e}.")
    return tuple(h), (aa, ab, bb)


def _pyscf_fci_uhf(r, out, spin_overlap):
    """(constant, active energies, energies) of the lowest roots of the
    spin-resolved g_eff of request `r` (its n_elec_active the pair
    (N_alpha, N_beta)); out['spin_square'], <S^2> of each root for the
    whole molecule's state, when `out` and the alpha-beta overlaps are
    given."""
    g_eff, norb, n_elec = r.g_eff, r.n_spin_orbs, r.n_elec_active
    h, eri = _spin_resolved_hamiltonian(g_eff, norb, r.symmetrize, 'pyscf_fci')
    engine = PyscfSpinResolvedFCISolver()
    energies, vectors = engine.roots(h, eri, norb, n_elec, nroots=r.nroots)
    if out is not None and spin_overlap is not None:
        out['spin_square'] = []
        for vector in vectors:
            (dm1a, dm1b), (_, dm2ab, _) = engine.rdms(vector, norb, n_elec)
            out['spin_square'].append(
                total_spin_square(dm1a, dm1b, dm2ab, spin_overlap))
    e0 = g_eff.get(0, 0.0)
    return e0, energies, e0 + energies


#: The backends that solve a spin-resolved g_eff (per-spin blocks).
SPIN_RESOLVED_SOLVERS = {'pyscf_fci': _pyscf_fci_uhf}


@flags_into_out
def solve(solver, g_eff, n_spin_orbs, n_elec_active, symmetrize=True,
          spin_free=False, nroots=1, dmrg_bond_dim=DMRG_BOND_DIM,
          dmrg_n_sweeps=DMRG_N_SWEEPS, out=None, spin_overlap=None,
          spin=None, sides=None, delta_method=None, d_sa=None,
          numerics=None):
    """
    The lowest `nroots` energies of a downfolded effective Hamiltonian g_eff,
    ascending, its constant g_eff[0] included: the same contract for every
    backend (`SOLVERS`).

    The constant: this function adds the g_eff[0] of the dict it is given,
    whatever that dict is, and applies Delta, the energy-dependent folding of
    the coupling between active and environment orbitals, only when that
    coupling is passed as `sides`; without `sides` the energies are those of
    (h, v) = (g_eff[1], g_eff[2]) as they stand, and this function does not
    know whether a Delta exists. For a `DownfoldedHamiltonian` with a
    coupling the caller is its `solve` method, which passes the coupling
    together with the constant of the model with Delta (energies.e0, not
    g_eff[0]); g_eff[0] plus a solve of (h, v) by hand is the energy of the
    model without Delta.

    n_spin_orbs: the active orbitals g_eff is written on (spin orbitals, or
    spatial orbitals when spin_free); n_elec_active: the active electrons.
    symmetrize: solve the Hermitian part of g_eff. With symmetrize=False the
    determinant-basis solver returns the (complex) spectrum of g_eff as it is,
    and 'pyscf_fci' / 'dmrg' / 'ccsdt' / 'ccsdtq' refuse a non-Hermitian g_eff.
    Those take a one- and two-body Hamiltonian only: a non-zero three-body
    vertex is an error, never dropped (build with max_vertex=2, or use
    'openfermion').
    nroots: how many states to return (None: every state the backend can
    reach, the whole determinant spectrum for 'openfermion'). The
    coupled-cluster backends produce one and refuse more: the lowest singlet
    when the active electrons are paired (RHF reference), else the lowest
    state of spin (n_alpha - n_beta) / 2 (UHF reference). 'pyscf_fci' and
    'openfermion' return the lowest states at this Sz of any spin, so where a
    triplet lies lowest they and the coupled-cluster backends differ by
    design.
    dmrg_bond_dim, dmrg_n_sweeps: Block2DMRGSolver knobs, used only when
    solver='dmrg'.

    An empty active space (n_spin_orbs = 0, as a downfolding with a (0,0)
    active space produces) has one state, of active energy 0: every backend
    returns [g_eff[0]] alone, whatever nroots asks, and warns
    (EmptyActiveSpaceWarning, named flag 'empty_active_space'). Electrons in
    no orbitals are refused.

    A spin-resolved g_eff, the Hamiltonian of an unrestricted reference:
    g_eff[1] = (h_alpha, h_beta) and g_eff[2] = (aa, ab, bb), each two-body
    block physicist like the spin-free one, on n_spin_orbs orbitals per spin,
    with n_elec_active = (N_alpha, N_beta). 'pyscf_fci' solves it with
    fci.direct_uhf: the lowest nroots states at that S_z, of any spin (in
    spin-polarized orbitals the Hamiltonian conserves S_z only), at most as
    many as the sector has. The other backends refuse it; 'openfermion'
    takes the spin-orbital form of the same Hamiltonian (spin_free=False),
    at N_alpha = N_beta only.
    spin: None (default), the lowest nroots states at the active Sz, of any
    spin, as described above for each backend; or 2S (0 singlets, 2
    triplets), the lowest nroots states of total spin S exactly. 'pyscf_fci'
    solves at Ms = S with pyscf's spin penalty (fci.addons.fix_spin_,
    FCI_SPIN_PENALTY_SHIFT), whose energies are those of the Hamiltonian
    itself on that spin, and refuses a root whose <S^2> is not S(S+1);
    'dmrg' takes it as its SU2 target. 'openfermion' and the coupled-cluster
    backends refuse it, as does a spin-resolved g_eff (Sz alone is
    conserved there; out['spin_square'] reports <S^2>). 2S must fit the
    active electrons: an integer >= 0 of their parity, (N + 2S)/2 of them
    alpha in the active orbitals.

    sides, delta_method, d_sa, numerics: Delta. sides is the coupling as
    plain data (`Coupling.sides()` of Downfolding/hybridization.py); None,
    the default, applies no Delta, and then delta_method or d_sa is refused
    (ValueError). sides == [] (a coupling with nothing to fold) is the Delta
    path with nothing folded: the static energies, Z = 1. With sides the
    static states of the request are found as without them
    (`static_states`), refined by the dynamic backend
    DYNAMIC_METHODS[delta_method] ('linearized' or 'lowdin';
    solvers/dynamic_backends.py), and g_eff[0] is added. The energies are
    returned ascending: Delta can move two roots past each other, so the
    refined roots are sorted, and out['static_index'] gives for each
    returned root the static state it came from. Two flags of
    DeltaRootsCrossedWarning report what the sorting shows:
    'delta_roots_reordered' when it changed the order of returned roots
    that are not degenerate partners, and 'delta_roots_missing' when more
    solutions of the Delta problem lie below a returned root than returned
    roots do (extra['solutions_below'] of the dynamic backend): a state the
    static solve did not compute went below it, and asking for more roots
    computes it. A root flagged 'delta_near_pole' is not checked for
    missing states: next to a pole the count is not the number of states
    below. Refused with sides: a delta_method outside DYNAMIC_METHODS
    (ValueError; the static forms change h and are solved without sides),
    d_sa (ValueError: no dynamic method reads it before the remainder
    method), a spin-resolved g_eff and spin_free=False (NotImplementedError:
    the coupling is spatial and Delta for two spin channels is not built),
    and a solver whose backend has no sector operations (ValueError naming
    the static forms 'static_sa' and 'static_mf'), all before anything is
    solved. numerics: a `Numerics` for the dynamic backend (None: the
    defaults), read only with sides.

    out: a dict, filled with what was computed; the energies do not change
    when it is given. out['e0'] is the constant added, g_eff[0], and
    out['e_active'] the energies before it was added, so the energies are
    out['e0'] + out['e_active'] (exactly; for 'openfermion', which holds
    the constant inside its matrix, to rounding). out['numerical_flags']
    lists the named flags raised during the call, each {'flag': name,
    'message': text}. out['spin_square'] is <S^2> of each returned root:
    for 'pyscf_fci' on a g_eff that is not spin-resolved, from its CI
    vector in its sector (fci.spin_op.spin_square; with sides, the refined
    vector); for a spin-resolved g_eff only when spin_overlap is given:
    with the alpha-beta overlaps of the orbitals (the blocks 'core_core',
    'core_active', 'active_core', 'active_active', as
    `build_effective_hamiltonian` puts them in space['spin_overlap']) it is
    <S^2> of the whole molecule's state (`total_spin_square`); without them
    no <S^2> is reported, since pyscf's own value assumes that alpha
    orbital p and beta orbital p are one function. With sides, also, each
    per returned root and in the returned order: 'static' (the static
    energy, no constant), 'correction' (e_active - static), 'z',
    'min_denominator', 'converged', 'static_index' and every entry of the
    dynamic backend's `DeltaResult.extra` ('solutions_below', 'near_pole',
    and for 'linearized' 'linearization_error' and 'linearization_point');
    an empty active space with sides reports static [0], correction [0],
    z [1], min_denominator [inf], converged [True], static_index [0].
    """
    if solver not in SOLVERS:
        raise ValueError(f"solver={solver!r} is not a backend of this driver; "
                         f"choose one of {tuple(SOLVERS)}.")
    if sides is None:
        if delta_method is not None or d_sa is not None:
            raise ValueError(
                'delta_method and d_sa apply Delta, which needs the coupling '
                'between active and environment orbitals as sides; without '
                'sides solve applies no Delta. A DownfoldedHamiltonian '
                'passes them from its own solve method.')
    else:
        _require_delta(solver, g_eff, spin_free, delta_method, d_sa)
    request = _checked(solver, SolveRequest(
        g_eff, n_spin_orbs, n_elec_active, spin_free, symmetrize, nroots,
        dmrg_bond_dim, dmrg_n_sweeps, spin))
    resolved = _spin_resolved(g_eff)
    if n_spin_orbs == 0:
        n_elec_active = request.n_elec_active
        if sum(n_elec_active) if resolved else n_elec_active:
            raise ValueError(f"n_elec_active={n_elec_active} electrons with no "
                             f"active orbitals (n_spin_orbs=0): there are no "
                             f"active orbitals to hold them.")
        constant = float(np.real(g_eff.get(0, 0.0)))
        # stacklevel 3: past the flags_into_out wrapper, the caller of solve
        flag('empty_active_space',
             f"the active space is empty: there is nothing to diagonalize, so "
             f"E_act = 0 and the one root returned is the constant g_eff[0] = "
             f"{constant:.10f} (for a downfolded Hamiltonian, E_0).",
             EmptyActiveSpaceWarning, stacklevel=3)
        if out is not None:
            out['e0'], out['e_active'] = constant, np.array([0.0])
            if sides is not None:
                # nothing to fold (Review Focus 1)
                out.update(static=np.array([0.0]), correction=np.array([0.0]),
                           z=np.array([1.0]), min_denominator=np.array([np.inf]),
                           converged=np.array([True]),
                           static_index=np.array([0]))
        return np.array([constant])
    if sides is not None:
        return _solve_with_delta(solver, request, sides, delta_method,
                                 numerics, out)
    if resolved:
        if solver == 'openfermion':
            raise ValueError(
                "solver='openfermion' diagonalizes the spin-orbital form of "
                "an unrestricted Hamiltonian, not its per-spin blocks: build "
                "it with spin_free=False (it then works at N_alpha = N_beta "
                "only), or solve the blocks with solver='pyscf_fci'.")
        if solver not in SPIN_RESOLVED_SOLVERS:
            raise NotImplementedError(
                f"solver={solver!r} has no path for a spin-resolved "
                f"Hamiltonian (different alpha and beta blocks, an "
                f"unrestricted reference); use solver='pyscf_fci'.")
        e0, e_active, energies = SPIN_RESOLVED_SOLVERS[solver](
            request, out, spin_overlap)
    else:
        e0, e_active, energies = SOLVERS[solver](request, out)
    if out is not None:
        out['e0'], out['e_active'] = e0, e_active
    return energies


def _checked(solver, request):
    """`request` with its electron count in the form the backends read (the
    pair (N_alpha, N_beta) for a spin-resolved g_eff, else the total), after
    the checks every solve makes: a count that fits, N_alpha = N_beta where
    the form needs it, and a spin the backend can honour (`_check_spin`)."""
    g_eff, n_spin_orbs = request.g_eff, request.n_spin_orbs
    n_elec_active = request.n_elec_active
    resolved = _spin_resolved(g_eff)
    if resolved:
        if np.ndim(n_elec_active) == 0:
            raise ValueError(
                f"a spin-resolved g_eff (per-spin blocks) takes n_elec_active "
                f"as the pair (N_alpha, N_beta); got {n_elec_active!r}.")
        n_elec_active = tuple(int(n) for n in n_elec_active)
        # an empty space with electrons has its own refusal in solve
        if min(n_elec_active) < 0 or (n_spin_orbs
                                      and max(n_elec_active) > n_spin_orbs):
            raise ValueError(
                f"n_elec_active={n_elec_active}: a space of {n_spin_orbs} "
                f"orbitals takes 0 to {n_spin_orbs} electrons per spin.")
    elif np.ndim(n_elec_active) != 0:
        n_alpha, n_beta = (int(n) for n in n_elec_active)
        if n_alpha != n_beta:
            raise NotImplementedError(
                f"solver={solver!r} on this form of g_eff works at "
                f"N_alpha = N_beta only; got {(n_alpha, n_beta)}. An "
                f"unrestricted Hamiltonian with N_alpha != N_beta is solved "
                f"from its per-spin blocks (spin_free=True) with "
                f"solver='pyscf_fci'.")
        n_elec_active = n_alpha + n_beta
    _check_spin(solver, request.spin, resolved, n_spin_orbs, n_elec_active,
                request.spin_free)
    return replace(request, n_elec_active=n_elec_active)


def _require_delta(solver, g_eff, spin_free, delta_method, d_sa):
    """Refuse, before anything is solved, a Delta solve `solve` cannot do
    (its docstring, sides)."""
    if delta_method not in DYNAMIC_METHODS:
        raise ValueError(
            f"delta_method={delta_method!r}: with sides, Delta is solved by a "
            f"dynamic backend, one of {tuple(DYNAMIC_METHODS)}. The static "
            f"forms 'static_sa' and 'static_mf' change h and are solved "
            f"without sides (DownfoldedHamiltonian.solve builds them).")
    if d_sa is not None:
        raise ValueError(
            "d_sa, the static D_SA, is read by the remainder method "
            "'sa_remainder' alone, which is not built yet (Task 11): no "
            "dynamic backend reads it.")
    if _spin_resolved(g_eff):
        raise NotImplementedError(
            'Delta on a spin-resolved g_eff (the per-spin blocks of an '
            'unrestricted reference): the dynamic backends take their sector '
            'operations from the spin-free FCI engine, and Delta for two '
            'spin channels is not built.')
    if not spin_free:
        raise NotImplementedError(
            'Delta with spin_free=False: the coupling (sides) is written on '
            'the spatial active orbitals; solve the spatial g_eff '
            '(spin_free=True).')
    engine = ENGINES[solver]
    if not engine.has_sector_operations:
        raise ValueError(
            f"solver={solver!r}: its backend {engine.__name__} has no sector "
            f"operations (the sectors with one electron less or more), which "
            f"a dynamic Delta method takes from its engine. Use "
            f"solver='pyscf_fci', or a static form of Delta, 'static_sa' or "
            f"'static_mf', which changes h and runs with any solver.")


def _solve_with_delta(solver, r, sides, method, numerics, out):
    """The energies of request `r` with Delta (`solve`, sides): the static
    states refined by DYNAMIC_METHODS[method], sorted ascending, the
    crossing flags raised, g_eff[0] added; `out` filled."""
    st = static_states(solver, r)
    res = DYNAMIC_METHODS[method](st.engine, sides, numerics).refine(
        st.h1, st.eri, st.norb, st.sector, st.energies, st.vectors,
        spin=r.spin)
    order = np.argsort(res.energies, kind='stable')
    _crossing_flags(res, order, method, numerics)
    e0 = r.g_eff.get(0, 0.0)
    e_active = res.energies[order]
    energies = e0 + e_active
    if out is not None:
        static = res.static[order]
        out.update(e0=e0, e_active=e_active, static=static,
                   correction=e_active - static, z=res.z[order],
                   min_denominator=res.min_denominator[order],
                   converged=res.converged[order], static_index=order)
        for key, value in res.extra.items():
            out[key] = np.asarray(value)[order]
        out['spin_square'] = [fci.spin_op.spin_square(
            res.vectors[k], st.norb, st.sector)[0] for k in order]
    return energies


def _crossing_flags(res, order, method, numerics):
    """'delta_roots_reordered' and 'delta_roots_missing'
    (DeltaRootsCrossedWarning) of a refined result `res` sorted by `order`
    (`solve`, sides). Degenerate partners (one set of `degenerate_sets`)
    are one state for both: their order among themselves is not a crossing,
    and none is below another. A root flagged 'delta_near_pole' is not
    checked for missing states."""
    static, energies = res.static, res.energies
    n = len(static)
    sets = degenerate_sets(static, numerics)
    label = np.empty(n, dtype=int)
    for i, members in enumerate(sets):
        label[members] = i
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(n)
    crossed = [(i, j) for i in range(n) for j in range(i + 1, n)
               if label[i] != label[j] and rank[i] > rank[j]]
    if crossed:
        # stacklevel 5: past this helper, _solve_with_delta, solve and the
        # flags_into_out wrapper, the caller of solve
        flag('delta_roots_reordered',
             f"Delta ({method}): the refined roots of the static states "
             f"{', '.join(map(str, crossed))} crossed (static states (i, j), "
             f"i below j before Delta and above it after); the energies are "
             f"returned in ascending order, and out['static_index'] gives "
             f"the static state of each returned root.",
             DeltaRootsCrossedWarning, stacklevel=5)
    margin = numerics_or_default(numerics).downfolding.delta_degenerate
    below = res.extra['solutions_below']
    for position, k in enumerate(order):
        if res.extra['near_pole'][k]:
            continue
        returned = np.count_nonzero((energies < energies[k] - margin)
                                    & (label != label[k]))
        missing = int(below[k]) - int(returned)
        if missing > 0:
            flag('delta_roots_missing',
                 f"Delta ({method}): {missing} solution(s) of the Delta "
                 f"problem that the static solve did not compute lie below "
                 f"returned root {position} (static state {k}, active energy "
                 f"{energies[k]:.8f}), so the returned roots are not the "
                 f"lowest states of the request. Ask for more roots, at least "
                 f"nroots = "
                 f"{n + missing}, more if those states come from higher "
                 f"static states.", DeltaRootsCrossedWarning, stacklevel=5)


#: The backends that give the states of one total spin (`solve`'s spin).
SPIN_SOLVERS = ('pyscf_fci', 'dmrg')


def _check_spin(solver, spin, resolved, n_spin_orbs, n_elec_active,
                spin_free):
    """Refuse a spin (2S) that `solve` cannot honour: on a backend without
    total spin, on a spin-resolved g_eff, or one the active electrons cannot
    have (not an integer >= 0 of their parity, or more alpha electrons than
    active orbitals). None passes."""
    if spin is None:
        return
    if resolved:
        raise ValueError(
            f"spin={spin!r} asks for states of one total spin, which a "
            f"spin-resolved g_eff (the per-spin blocks of an unrestricted "
            f"reference) does not conserve: it conserves Sz alone. Solve "
            f"without spin; out['spin_square'] reports <S^2> of each root.")
    if solver not in SPIN_SOLVERS:
        raise ValueError(
            f"solver={solver!r} cannot give the states of one total spin; "
            f"spin={spin!r} is taken by 'pyscf_fci' and 'dmrg'.")
    norb = n_spin_orbs if spin_free else n_spin_orbs // 2
    n = int(n_elec_active)
    if (isinstance(spin, bool) or not isinstance(spin, (int, np.integer))
            or spin < 0 or (n - spin) % 2 or (n + spin) // 2 > norb):
        raise ValueError(
            f"spin={spin!r} (2S) does not fit {n} active electrons in "
            f"{norb} orbitals: 2S is an integer >= 0 of the electron "
            f"count's parity, with (N + 2S)/2 <= {norb} alpha electrons.")
