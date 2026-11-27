"""The two-step downfolding driver: run a low-level method on the mean field
once, then build the effective Hamiltonian of any number of active spaces
from it.

    low = run_low_level(mf, method='hf')
    h = build_effective_hamiltonian(low, active_space)
    e = h.solve('pyscf_fci')          # h.g_eff, h.space, h.energies, ...

The driver knows no method: it reaches them through `LOW_LEVEL` and reads
only the `DownfoldingTerms` a method hands back. The mean-field object alone
decides the number of spin channels. A restricted reference (RHF/RKS) is ONE
channel (the orbitals, the `ActiveSpace`, one vtilde), and everything is
computed once: one exact J/K build of the inactive core, shared by the core
Fock operator and the environment energy, and one integral transformation of
the active block. An unrestricted one (UHF/UKS) is two, alpha and beta, each
with its own orbitals, occupied count and active space, also when the two
coincide:

    low = run_low_level(uhf, method='hf')
    h = build_effective_hamiltonian(low, (space_alpha, space_beta))
    e = h.solve('pyscf_fci')

The corrections are computed once each, the frozen core (core.py) and the
double counting (double_counting.py), and the three parts of H_eff only add
them up: E_0 in zero_body.py, h_eff in one_body.py, v_eff in two_body.py.
`pack` here writes them into the g_eff dictionary the active-space solvers
read (src/EffectiveHamiltonian/solvers/solve.py).
"""
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
from pyscf import scf

from src.Base.numerical_flags import (collecting, flag, flags_into_out,
                                      gathered)
from src.Base.numerics.numerics import Numerics, numerics_or_default
from src.Base.orbital_spaces import closed_shell_nocc, unrestricted_nocc
from src.Base.pyscf.pyscf_interface import (get_antisymmetrized_integrals,
                                            get_orbital_energies,
                                            require_gas_phase_ground_state,
                                            spin_orbital_chemist,
                                            spin_orbital_one_body,
                                            unrestricted_reference_diagnostics)
from src.Base.utils.parallel.mpi_grid import grid_comm
from src.Base.utils.choices import require_choice
from src.EffectiveHamiltonian.orbital_basis import (orbitals_source,
                                                    semicanonical_rotation)
from src.EffectiveHamiltonian.solvers.determinants import (
    spatial_to_spin_orbitals)
from src.EffectiveHamiltonian.solvers.solve import (ENGINES, SolveRequest,
                                                    solve, static_states)

from .core import core_terms
from .double_counting import double_counting
from .hybridization import (DEFAULT_BLOCKS, DELTA_METHODS, STATIC_METHODS,
                            Coupling, DeltaOffWarning, StaticDeltaWarning,
                            fock_coupling, state_averaged_delta,
                            static_mf_delta)
from .methods import LOW_LEVEL
from .one_body import effective_one_body
from .spin import SPINS
from .two_body import effective_interaction
from .zero_body import Energies, assemble_energies

WHO = 'Downfolding'


def _spin_channels(mf):
    """(mo, eps, nocc), each a tuple over the spin channels of `mf`: one
    channel for a restricted reference, (alpha, beta) for UHF/UKS. nocc is
    the occupied count of each channel, refused unless its occupied orbitals
    are its lowest ones: every active space takes [0, nocc) as the occupied
    block. An unrestricted reference is also refused unless every
    occupation is 0 or 1 (`unrestricted_nocc`)."""
    if not isinstance(mf, scf.uhf.UHF):
        return ((mf.mo_coeff,), (get_orbital_energies(mf),),
                (closed_shell_nocc(mf.mo_occ),))
    return (tuple(mf.mo_coeff), tuple(get_orbital_energies(mf)),
            unrestricted_nocc(mf.mo_occ))


def _active_spaces(active_space, mo, eps, nocc, numerics):
    """The `ActiveSpace` of each spin channel as a tuple: one for a
    restricted reference, (alpha, beta) for an unrestricted one, whose active
    space is a pair with the same number of active orbitals. Each space must
    be set up for the orbital and occupied counts of its own channel, its
    orbital basis, if it has one, must fit it and be built on that channel's
    orbitals `mo` (`_require_basis_of`), and it must not cut a degenerate
    set of its channel's orbitals (`ActiveSpace.require_no_degenerate_cut`,
    on the occupations of a rotated block, which warns below
    numerics.downfolding.no_cut_warn, pointing at the caller of
    build_effective_hamiltonian)."""
    paired = isinstance(active_space, (tuple, list))
    if len(eps) == 1:
        if paired:
            raise ValueError(
                'a restricted reference takes one ActiveSpace; a pair '
                '(space_alpha, space_beta) goes with a UHF/UKS reference.')
        spaces, names = (active_space,), (None,)
    else:
        if not paired or len(active_space) != 2:
            raise ValueError(
                'an unrestricted (UHF/UKS) reference takes a pair of active '
                'spaces, (space_alpha, space_beta): one ActiveSpace per spin, '
                'each built from the orbital and occupied counts of that '
                'spin.')
        spaces, names = tuple(active_space), SPINS
        if spaces[0].n_act != spaces[1].n_act:
            raise ValueError(
                f'the alpha and beta active spaces hold {spaces[0].n_act} and '
                f'{spaces[1].n_act} orbitals; both spins take the same number '
                f'of active orbitals.')
    for space, c, e, n, spin in zip(spaces, mo, eps, nocc, names):
        which, has = (('the active space', f'{n} occupied') if spin is None
                      else (f'the {spin} active space',
                            f'{n} {spin} occupied'))
        if (space.norb, space.nocc) != (len(e), n):
            raise ValueError(
                f'{which} is set up for {space.nocc} occupied of '
                f'{space.norb} orbitals; the mean field has {has} of '
                f'{len(e)}.')
        if space.basis is not None:
            _require_basis_of(space, c, which)
        # stacklevel 4: the caller of build_effective_hamiltonian
        space.require_no_degenerate_cut(e, spin, numerics=numerics,
                                        stacklevel=4)
    return spaces


def _settings(low, numerics):
    """The Numerics of a build, resolved once by the low-level record's
    `settings` (which refuses what its method cannot take); a record without
    one, a method that keeps no numerics, gets `numerics` or the defaults."""
    resolve = getattr(low, 'settings', None)
    return numerics_or_default(numerics) if resolve is None else resolve(
        numerics)


def _require_basis_of(space, mo, which):
    """Refuse (ValueError) an orbital basis that does not fit its active
    space (occupied count, shape of the rotation) or was built on other
    orbitals than `mo`, the current orbitals of its channel: another molecule
    or geometry, or a mean field converged again since."""
    basis = space.basis
    shape = np.shape(basis.rotation)
    if basis.nocc != space.nocc or shape != (space.norb, space.norb):
        raise ValueError(
            f'the orbital basis of {which} holds {basis.nocc} occupied '
            f'orbitals in a rotation of shape {shape}; the space is set up '
            f'for {space.nocc} occupied of {space.norb} orbitals.')
    if basis.source != orbitals_source(mo):
        raise ValueError(
            f'the orbital basis of {which} was built on other orbitals than '
            f'those of this mean field (another molecule or geometry, or the '
            f'mean field converged again since): build it from a low-level '
            f'run on this mean field.')


def _require_basis_route(low, spaces):
    """Refuse what the general layer does not do with an orbital basis,
    whatever the method: a basis on one spin of an unrestricted pair only
    (ValueError) and the exact four-index integrals of an RPA run
    (df=False), which have no three-index factors to rotate. Nothing for
    canonical spaces."""
    with_basis = [s.basis is not None for s in spaces]
    if not any(with_basis):
        return
    if len(spaces) == 2 and not all(with_basis):
        raise ValueError(
            'an orbital basis on one spin only: the alpha and beta active '
            'spaces of an unrestricted reference both take one, or neither.')
    if not getattr(low, 'df', True):
        raise NotImplementedError(
            'an active space on a rotated orbital basis needs the '
            'three-index factors, which a df=False low-level result (the '
            'exact four-index integrals) does not keep: use df=True.')


def _rotated_orbitals(mo, eps, spaces):
    """(mo, T): the orbitals of each channel in the basis of its active
    space, the canonical `mo` itself for a space without a basis, else
    mo @ T with T the semicanonical rotation of the basis
    (`semicanonical_rotation`); and T, None for a space without a basis."""
    rotations = tuple(None if s.basis is None
                      else semicanonical_rotation(s.basis, s, e)[0]
                      for e, s in zip(eps, spaces))
    return (tuple(c if t is None else c @ t for c, t in zip(mo, rotations)),
            rotations)


def _require_delta_route(spaces, delta):
    """Refuse, with Delta on, an unrestricted reference with an orbital
    basis: Delta for two spin channels is not built, and a rotated basis
    without it is the static Hamiltonian only, which delta=False asks for
    by name (NotImplementedError). Refuse, with Delta on, a space whose
    occupied block is rotated: the E_0 term of the hybridization would hold
    the hole side, which Delta does not fold yet (NotImplementedError,
    Task 13)."""
    if delta and len(spaces) == 2 and any(s.basis is not None
                                          for s in spaces):
        raise NotImplementedError(
            'an unrestricted (UHF/UKS) reference on a rotated orbital basis '
            'with Delta on: the unrestricted Delta, the folding of the Fock '
            'coupling of two spin channels, is not built. delta=False gives '
            'the static Hamiltonian only (E_0, the one-body and the '
            'two-body part), without the coupling the rotation creates.')
    if delta and any(s.basis is not None and s.basis.rotates_occupied
                     for s in spaces):
        raise NotImplementedError(
            'an active space whose occupied block is rotated '
            '(rotate_occupied=True) with Delta on: the hole side of Delta '
            'is not built yet (Task 13). delta=False gives the static '
            'Hamiltonian only.')


#: the options of the DMRG static solve that `prepare_delta` passes on
DMRG_OPTIONS = ('dmrg_bond_dim', 'dmrg_n_sweeps')


@dataclass(frozen=True, eq=False)
class DownfoldedHamiltonian:
    """One effective Hamiltonian, as `build_effective_hamiltonian` returns it.

    g_eff: {0: e0_without_delta, 1: h_eff, 2: v_eff}, the static Hamiltonian
        of the decoupled active space, all three parts of it, the input of
        the active-space solvers. Its constant g_eff[0] is
        energies.e0_without_delta, E_0 without the one Delta term
        e_c_hybridization (zero_body.py), and nothing in g_eff contains
        Delta. With a coupling on the record (`hybridization`, an active
        space on a rotated orbital basis) g_eff is not the whole
        Hamiltonian: the model of the active space coupled to its
        environment is g_eff together with the coupling, and its constant
        is energies.e0 (`g_eff_is_static_part`). A total energy then comes
        from `solve`, which pairs each active energy with the constant of
        its own model; a Hamiltonian for a solver outside `solve` comes from
        `static_hamiltonian`. g_eff[0] plus a solve of (h_eff, v_eff) by
        hand is the energy without Delta. Without a coupling (canonical
        orbitals, or delta=False) e0, e0_without_delta and g_eff[0] are one
        number and g_eff is the whole Hamiltonian.
    space: the active space in QDPT's form (n_orbs, n_elec_active, the
        index arrays, spin_free, active_space, and for an unrestricted
        reference spin_overlap).
    energies: the `Energies`, every term of E_0; energies.e0 is the constant
        of the model with Delta, energies.e0_without_delta that of g_eff.
    diagnostics: the low-level method's checks, keyed by name.
    numerical_flags: the named flags behind this Hamiltonian
        (src.Base.numerical_flags), each {'flag': name, 'message': text};
        empty for a clean result.
    hybridization: the Fock coupling (`Coupling`, hybridization.py) between
        the active and the environment orbitals of an active space on a
        rotated orbital basis, the term Delta folds; None for canonical
        orbitals and with delta=False. Its d_sa holds D_SA once
        `prepare_delta` has built it.
    numerics: the `Numerics` in force for this Hamiltonian, as the build
        resolved it (the build's numerics when given, else the low-level
        run's, else the defaults); `solve` hands it to the Delta solvers.
    """
    g_eff: dict
    space: dict
    energies: Energies
    diagnostics: dict
    numerical_flags: list
    hybridization: Optional[Coupling] = None
    numerics: Optional[Numerics] = None

    @property
    def g_eff_is_static_part(self):
        """True when the record has a coupling (`hybridization`): g_eff is
        then the static part of the Hamiltonian only, and its constant
        e0_without_delta is not the constant of the model with Delta."""
        return self.hybridization is not None

    @flags_into_out
    def solve(self, solver, delta_method=None, out=None, **kwargs):
        """The lowest energies of this Hamiltonian by `solver`
        (src.EffectiveHamiltonian.solvers.solve.solve), each a total: the
        active energy E^A of one model plus the constant of the same model
        (spec 6d). The record's orbital and electron counts and spin_free
        are filled in, and for an unrestricted reference its spin_overlap,
        so <S^2> can be reported; other kwargs (nroots, spin, symmetrize,
        the DMRG settings) go to that function. The coupling, D_SA and the
        numerics are the record's own: sides, d_sa and numerics in kwargs
        are refused (TypeError).

        delta_method (None, the default, is 'linearized'):

        | record, delta_method | dict solved | E^A from | constant |
        |---|---|---|---|
        | no coupling, None or False | g_eff | (h, v) | g_eff[0] (= e0 there) |
        | coupling, False | g_eff | (h, v) | e0_without_delta (= g_eff[0]); flag 'delta_off' |
        | coupling, 'static_mf' or 'static_sa' | {0: e0 + c, 1: h + D, 2: v} | (h + D, v) | e0 + c |
        | coupling, None/'linearized' or 'lowdin' | {0: e0, 1: h, 2: v} with the coupling | the static states of (h, v) refined with Delta | e0 |

        e0 is energies.e0, h and v are g_eff[1] and g_eff[2], and (D, c)
        the static form: D a one-body matrix added to h and c the constant
        of its hole part (zero without a hole side). No path adds g_eff[0]
        to an energy with Delta, and none adds e0 to one without.

        - 'linearized' (the default): linearized Loewdin, one generalized
          eigenproblem per state (LinearizedLowdinSolver). It flags a state
          whose estimated linearization error is above
          numerics.downfolding.delta_linearization_warn
          ('delta_linearization'); for such a state use 'lowdin'.
        - 'lowdin': exact Loewdin, the reference (LowdinSolver).
        - 'static_mf': D at the mean-field orbital energies
          (`static_mf_delta`), with StaticDeltaWarning and the flag
          'delta_static_mf' on every use; any solver.
        - 'static_sa': D_SA (`state_averaged_delta`), built beforehand by
          `prepare_delta` and refused (ValueError) without it; any solver.
        - 'sa_remainder': not built yet (NotImplementedError, Task 11).
        - False: no Delta, with DeltaOffWarning and the flag 'delta_off'
          when the record has a coupling.

        The two dynamic methods run on the solvers whose backend has sector
        operations ('pyscf_fci'); any other solver is refused before
        anything is solved, naming the static forms. A record without a
        coupling takes None or False and refuses a named method
        (ValueError: there is nothing to fold). A record with a coupling
        built with spin_free=False refuses every method but False
        (NotImplementedError: the coupling is spatial).

        With Delta the energies are returned in ascending order: Delta can
        move two states past each other. out['static_index'] gives for each
        returned root the static state it came from, and two flags of
        DeltaRootsCrossedWarning report it: 'delta_roots_reordered' when
        the order of the returned roots changed, and 'delta_roots_missing'
        when a state that the static solve did not compute went below a
        returned root (more roots are needed to compute it; how many is
        not known in advance). The second flag checks less than it says,
        in three ways. It counts the eigenvalues of H_eff(E_k) below the
        root's own, which is the number of solutions below E_k only below
        the lowest pole of the coupling, min(E_M + lambda) (E_M - lambda
        on the hole side); above it the count is the solutions minus the
        poles below E_k. (a) A root flagged 'delta_near_pole' is not
        checked. (b) A root above the lowest pole is not checked, and
        can have missing states below it with no flag.
        out['crossing_unchecked'], a boolean per returned root in the
        returned order, is True for the roots of (a) and (b). (c) For a
        'linearized' root flagged 'delta_linearization' the count is made,
        but its energy is off by about the linearization error, and when
        another root lies within that error the count can be wrong in
        either direction; use 'lowdin' there. Silence means that no
        missing state was found among the checked roots, not that none is
        missing.

        out, a dict, receives what the solver function writes: out['e0'],
        the constant added, and out['e_active'], the E^A of each root, so
        the energies are out['e0'] + out['e_active']; out['spin_square']
        where the solver function reports it ('pyscf_fci'; for an
        unrestricted reference from the record's spin_overlap);
        out['numerical_flags'], every flag of the call (those of Delta and
        'delta_off' and 'delta_static_mf' included); and with
        'linearized' or 'lowdin' the per-root 'static', 'correction',
        'z', 'min_denominator', 'converged', 'static_index',
        'crossing_unchecked' and the method's extra entries. For a record with a coupling, False and the
        static forms also give 'static' = out['e_active'], 'correction'
        zero and 'z' one. E_0 for reporting is energies.e0, or out['e0']
        for the constant of one solve; E^A is out['e_active'], never a
        total minus a constant chosen by hand."""
        if out is not None:
            # a dict used for an earlier solve keeps none of its Delta
            # entries: the solve below writes the ones it has
            for key in DELTA_OUT_KEYS:
                out.pop(key, None)
        supplied = sorted(k for k in ('sides', 'd_sa', 'numerics')
                          if k in kwargs)
        if supplied:
            raise TypeError(
                f'{supplied} are not options of DownfoldedHamiltonian.solve: '
                f'the record supplies its coupling (sides), D_SA (d_sa, from '
                f'prepare_delta) and its numerics itself.')
        if 'spin_overlap' in self.space:
            kwargs.setdefault('spin_overlap', self.space['spin_overlap'])
        if self.hybridization is None:
            if delta_method is not None and delta_method is not False:
                raise ValueError(_nothing_to_fold(delta_method))
            return self._solve(solver, self.g_eff, out, kwargs)
        if delta_method is False:
            # stacklevel 3: past the flags_into_out wrapper, the caller
            flag('delta_off', _DELTA_OFF, DeltaOffWarning, stacklevel=3)
            energies = self._solve(solver, self.g_eff, out, kwargs)
            _static_parts(out)
            return energies
        method = self._method(delta_method)
        if method in STATIC_METHODS:
            energies = self._solve(solver, self._static_form(method, 4), out,
                                   kwargs)
            _static_parts(out)
            return energies
        coupled = {0: self.energies.e0, 1: self.g_eff[1], 2: self.g_eff[2]}
        return self._solve(solver, coupled, out, dict(
            kwargs, sides=self.hybridization.sides(), delta_method=method,
            numerics=self.numerics))

    def static_hamiltonian(self, delta_method):
        """A static Hamiltonian in the layout of g_eff, for a solver outside
        `solve`, whose constant already matches its one-body part: the
        total is its constant plus that solver's energy.

        'static_sa' and 'static_mf': {0: e0 + c, 1: h + D, 2: v}, e0 =
        energies.e0 and (D, c) the static form (`solve`), with the same
        refusals and warning as there ('static_sa' needs `prepare_delta`;
        'static_mf' warns StaticDeltaWarning with the flag
        'delta_static_mf'; a record without a coupling has nothing to fold,
        ValueError; one built with spin_free=False, NotImplementedError).
        False: g_eff itself, {0: e0_without_delta, 1: h, 2: v}, the
        decoupled model, with DeltaOffWarning (flag 'delta_off') when the
        record has a coupling; this function has no `out`, so the warning
        is the signal. Any other method has no static form (ValueError):
        the dynamic methods are solved by `solve`."""
        if delta_method is False:
            if self.hybridization is not None:
                flag('delta_off', _DELTA_OFF, DeltaOffWarning)
            return self.g_eff
        if isinstance(delta_method, bool) or delta_method not in STATIC_METHODS:
            raise ValueError(
                f'delta_method={delta_method!r} has no static form: '
                f'static_hamiltonian takes {STATIC_METHODS} or False. The '
                f"dynamic methods ('linearized', 'lowdin', 'sa_remainder') "
                f'depend on the energy of each state and are solved by '
                f'solve.')
        if self.hybridization is None:
            raise ValueError(_nothing_to_fold(delta_method))
        self._require_spin_free()
        return self._static_form(delta_method, 3)

    def prepare_delta(self, states, solver='pyscf_fci', **solver_opts):
        """A copy of this record whose hybridization.d_sa is D_SA, the
        static form 'static_sa' of Delta (`state_averaged_delta`), built
        from a declared set of states that enter with equal weight; solve
        it with h.solve(solver, delta_method='static_sa').

        states: ((n, spin), ...), each pair the number of states and the
        spin of one request as `solve` takes them (spin None: the lowest n
        states at the active Sz, of any spin; 2S: the lowest n of total
        spin S). The requests must not overlap: a spin=None request already
        holds states of every spin and takes no other request, and
        otherwise there is at most one request per spin (ValueError). If no
        request has spin None or the spin of the reference (0 for a closed
        shell), the request (1, reference spin) is added, so the ground
        state is always in the average. For each request the static states
        of g_eff are solved by `solver` (`static_states`), and the density
        matrices of each state are taken before the next request runs (the
        DMRG backend keeps only the states of its last solve).

        solver: a backend with density matrices ('pyscf_fci', the default,
        or 'dmrg'; any other is refused, ValueError); 'dmrg' refuses a
        request with spin=None (an SU2 solve is one total spin).
        solver_opts: the DMRG settings dmrg_bond_dim and dmrg_n_sweeps, as
        `solve` takes them. Refused: a record without a coupling
        (ValueError: there is nothing to fold) and one built with
        spin_free=False (NotImplementedError). D_SA depends on the set of
        states, so two solves that are compared (another nroots, another
        spin) share one prepared record."""
        unknown = sorted(set(solver_opts) - set(DMRG_OPTIONS))
        if unknown:
            raise TypeError(f'prepare_delta takes the DMRG settings '
                            f'{DMRG_OPTIONS}; got {unknown}.')
        if self.hybridization is None:
            raise ValueError(_nothing_to_fold('static_sa'))
        self._require_spin_free()
        if solver not in ENGINES:
            raise ValueError(f'solver={solver!r} is not a backend of the '
                             f'solvers; choose one of {tuple(ENGINES)}.')
        if not ENGINES[solver].has_rdms:
            with_rdms = tuple(n for n, c in ENGINES.items() if c.has_rdms)
            raise ValueError(
                f'solver={solver!r}: D_SA is built from the one- and '
                f'two-particle density matrices of the states, which its '
                f'backend {ENGINES[solver].__name__} does not give; the '
                f'backends with density matrices are {with_rdms}.')
        requests = self._requests(states, solver)
        rdms, st = [], None
        for n, spin in requests:
            st = static_states(solver, SolveRequest(
                self.g_eff, self.space['n_orbs'], self.space['n_elec_active'],
                spin_free=True, nroots=n, spin=spin, **solver_opts))
            # now, before the next request replaces the states
            rdms.extend(st.engine.rdms(v, st.norb, st.sector, spin=None)
                        for v in st.vectors)
        d_sa = state_averaged_delta(st.h1, st.eri, self.hybridization, rdms,
                                    self.numerics)
        return replace(self, hybridization=replace(self.hybridization,
                                                   d_sa=d_sa))

    def _solve(self, solver, g_eff, out, kwargs):
        return solve(solver, g_eff, self.space['n_orbs'],
                     self.space['n_elec_active'],
                     spin_free=self.space['spin_free'], out=out, **kwargs)

    def _method(self, delta_method):
        """The Delta method of a record with a coupling: None is
        'linearized'; an unknown name is refused (ValueError), every method
        on a spin_free=False record (NotImplementedError), 'sa_remainder'
        until it is built (NotImplementedError)."""
        method = 'linearized' if delta_method is None else delta_method
        if isinstance(method, bool) or method not in DELTA_METHODS:
            raise ValueError(
                f'delta_method={delta_method!r}: one of {DELTA_METHODS}, '
                f"None (= 'linearized') or False (no Delta).")
        self._require_spin_free()
        if method == 'sa_remainder':
            raise NotImplementedError(
                "delta_method='sa_remainder', the remainder method, is not "
                "built yet (Task 11).")
        return method

    def _require_spin_free(self):
        if not self.space['spin_free']:
            raise NotImplementedError(
                'Delta on a record built with spin_free=False: the coupling '
                'is written on the spatial active orbitals; build with '
                'spin_free=True, or solve with delta_method=False for the '
                'static Hamiltonian only.')

    def _static_form(self, method, stacklevel):
        """{0: e0 + c, 1: h + D, 2: v} of the static form `method`; the
        warning of 'static_mf' points `stacklevel` frames up from here (as
        for `flag`)."""
        if method == 'static_mf':
            flag('delta_static_mf',
                 "delta_method='static_mf' evaluates Delta once, at the "
                 "mean-field energies of the active orbitals: 0.8 eV from "
                 "exact Loewdin in benzene, worse with basis size. Use "
                 "'linearized' or 'lowdin' for energies, or 'static_sa' "
                 "(prepare_delta) for a static Hamiltonian.",
                 StaticDeltaWarning, stacklevel=stacklevel)
            d, c = static_mf_delta(self.hybridization)
        else:
            if self.hybridization.d_sa is None:
                raise ValueError(
                    "delta_method='static_sa' needs D_SA, built once from a "
                    "declared set of states: h = h.prepare_delta(((nroots, "
                    "spin), ...)) first.")
            d, c = self.hybridization.d_sa
        return {0: self.energies.e0 + c, 1: self.g_eff[1] + d,
                2: self.g_eff[2]}

    def _requests(self, states, solver):
        """The requests of `prepare_delta`, checked, with the ground state's
        request added when no request holds it."""
        try:
            requests = [(n, spin) for n, spin in states]
        except (TypeError, ValueError):
            raise ValueError(f'states={states!r}: a sequence of pairs '
                             f'(n, spin).') from None
        for n, spin in requests:
            if isinstance(n, bool) or not isinstance(n, (int, np.integer)) \
                    or n < 1:
                raise ValueError(f'states: n={n!r} in ({n!r}, {spin!r}) is '
                                 f'not a positive number of states.')
        spins = [spin for _, spin in requests]
        if None in spins and len(requests) > 1:
            raise ValueError(
                f'states={states!r}: a request with spin=None holds the '
                f'lowest states of every spin and takes no other request.')
        if len(set(spins)) != len(spins):
            raise ValueError(
                f'states={states!r}: two requests of one spin overlap; ask '
                f'for their states in one request.')
        if solver == 'dmrg' and None in spins:
            raise ValueError(
                "solver='dmrg' solves one total spin per request; a request "
                "with spin=None (every spin at the active Sz) is refused: "
                "give the spin.")
        reference = int(self.space['n_elec_active']) % 2
        if None not in spins and reference not in spins:
            requests.append((1, reference))
        return requests


#: the message of 'delta_off' when a record with a coupling is solved
#: without Delta
_DELTA_OFF = (
    'solved with delta_method=False: the Fock coupling between the active '
    'and the environment orbitals of this rotated basis is left out, and the '
    'energies are those of the static Hamiltonian g_eff, with its constant '
    'e0_without_delta.')


def _nothing_to_fold(delta_method):
    return (f'delta_method={delta_method!r}: this Hamiltonian has no coupling '
            f'between its active and environment orbitals (canonical '
            f'orbitals, or built with delta=False), so there is nothing to '
            f'fold; solve it with delta_method=None or False.')


#: the entries of `out` that only a Delta solve writes (`solve`), and the
#: ones `_static_parts` writes without one: removed at the start of each
#: record solve, so that a reused `out` holds the last solve alone
DELTA_OUT_KEYS = (
    'static', 'correction', 'z', 'min_denominator', 'converged',
    'static_index', 'crossing_unchecked', 'solutions_below', 'near_pole',
    'above_pole', 'linearization_error', 'linearization_point')


def _static_parts(out):
    """The Delta entries of `out` for a solve without a dynamic method
    (False, the static forms): static = e_active, correction 0, z 1."""
    if out is not None:
        e_active = np.asarray(out['e_active'])
        out.update(static=e_active.copy(), correction=np.zeros_like(e_active),
                   z=np.ones(len(e_active)))


def run_low_level(mf, method='rpa', **method_opts):
    """The low-level result of `method` on the mean field `mf` (attributes
    `method` and `mf`, plus whatever the method keeps for its `downfold`).

    The result does not depend on an active space, so one run serves every
    active space built from it. A gas-phase, T = 0 mean field is taken,
    restricted closed-shell (RHF/RKS) or unrestricted (UHF/UKS) with the
    lowest orbitals of each spin occupied by one electron each (no
    fractional occupations), and only serially: on more than
    one MPI rank every rank would repeat the whole calculation. An open
    shell run restricted (ROHF/ROKS) is refused: its orbitals are not
    canonical for the alpha and beta Fock operators. method_opts go to the
    method, which refuses the ones it does not know.

    Every named flag raised during the run (src.Base.numerical_flags) is
    collected into `low.numerical_flags`, a list of {'flag': name, 'message':
    text}, empty for a clean run. An unrestricted reference that is
    spin-contaminated or internally unstable is taken, with a warning and
    the flags 'uhf_spin_contaminated' / 'uhf_reference_unstable' there
    (`unrestricted_reference_diagnostics`, whose numbers are kept on the
    mean field as `mf.reference_diagnostics`).
    """
    _, _, size = grid_comm()
    if size > 1:
        raise NotImplementedError(
            f'{WHO} runs serially only; this process is one of {size} MPI '
            f'ranks. Run it on a single rank.')
    require_gas_phase_ground_state(mf, WHO, unrestricted=True)
    if getattr(mf, 'with_solvent', None) is not None:
        # a pyscf solvent model is not an attached environment, which is all
        # require_gas_phase looks for
        raise NotImplementedError(
            f'{WHO} is gas phase only; mf carries a solvent model '
            f'({type(mf.with_solvent).__name__}).')
    if isinstance(mf, scf.uhf.UHF):
        # the check above does not look at the occupations of an
        # unrestricted reference: refuse, before the method runs, a spin
        # whose occupations are fractional or whose occupied orbitals are
        # not its lowest
        _spin_channels(mf)
    method = require_choice(method, tuple(LOW_LEVEL), 'method')
    flags = []
    with collecting(flags):
        unrestricted_reference_diagnostics(mf, WHO,
                                           what='effective Hamiltonians')
        low = LOW_LEVEL[method].run(mf, **method_opts)
    low.numerical_flags = gathered(flags)
    return low


def build_effective_hamiltonian(low, active_space, spin_free=True,
                                numerics=None, delta=True, _blocks=None,
                                **method_opts):
    """The `DownfoldedHamiltonian` of `active_space`, downfolded from the
    low-level result `low`: g_eff, space, energies, diagnostics,
    numerical_flags, hybridization and numerics. `h.solve(solver)` solves
    it, with Delta when it has a coupling (`DownfoldedHamiltonian.solve`).

    g_eff and space follow QDPT's build_effective_hamiltonian, so g_eff goes
    straight to src.EffectiveHamiltonian.solvers.solve.solve:
    g_eff = {0: e0_without_delta, 1: h_eff, 2: physicist v_eff}, spatial when
    spin_free, else over the 2 n_act interleaved spin orbitals; space holds
    n_orbs, n_elec_active, act_idx, act_occ_idx, act_vir_idx (canonical
    spatial indices), spin_free and active_space. energies is the `Energies`
    record of every E_0 term; diagnostics are the method's, keyed by name.

    For an unrestricted reference (UHF/UKS) `active_space` is a pair
    (space_alpha, space_beta), one ActiveSpace per spin with independent
    orbital indices and the same number of active orbitals. Then g_eff[1] =
    (h_alpha, h_beta) and g_eff[2] = (aa, ab, bb) when spin_free, each
    two-body block physicist like the restricted one, the opposite-spin
    block <p_alpha q_beta | r_alpha s_beta>; else the same 2 n_act
    interleaved spin orbitals as for a restricted reference. In space,
    n_elec_active = (N_alpha, N_beta), the index entries and active_space
    are pairs (alpha, beta), and spin_overlap holds the blocks of
    C_alpha^T S C_beta between the inactive ('core') and the active orbitals
    of the two spins, 'core_core', 'core_active' (alpha core, beta active),
    'active_core' and 'active_active', which `solve` takes to report <S^2>.
    Every term of energies is the total over both spins.

    h_eff = f - t^DC (+ Sigma_bar) with f the Fock operator of the inactive
    core, v_eff = v + vtilde with v the exact active integrals; the method
    supplies vtilde, Sigma_bar, the correlation-energy pair and the
    double-counting mode. method_opts go to the method's `downfold`.

    numerics: None, or the Numerics of this build, resolved once by the
    low-level record (`low.settings`): for method='rpa' it may differ from
    the run's in the downfolding and layout sections (refused otherwise,
    NumericsMismatch) and is handed to `downfold`; method='hf' takes none
    (TypeError). The general layer reads the cut check's
    downfolding.no_cut_warn from it.

    An ActiveSpace with an orbital basis (`OrbitalBasis`, natural orbitals)
    indexes the orbitals of that basis made semicanonical in each of its
    index sets: the Fock operator of the core, the active integrals and
    space['mo'] are those of mo @ T (`semicanonical_rotation`), and space
    also holds 'basis'. Its cut is checked on the occupations of each
    rotated block (`ActiveSpace.require_no_degenerate_cut`: refused below
    QP_DEGENERACY_TOL, NearDegenerateCutWarning and the flag
    'no_cut_near_degenerate' below no_cut_warn). Refused: a basis built on
    other orbitals than the mean field's or that does not fit its space
    (ValueError), a basis on one spin of an unrestricted pair only
    (ValueError), and the exact integrals of an RPA run (df=False,
    NotImplementedError). An unrestricted reference takes a basis on both
    spins, each built on that spin's orbitals, with delta=False only (see
    `delta`).

    delta: True | False. On a rotated orbital basis the rotation couples
    active and environment orbitals; with delta=True, the default, that
    Fock coupling is built (`fock_coupling`, hybridization.py) as
    h.hybridization, the term Delta folds, with its norms in
    diagnostics['fock_coupling'] = {'folded': the norm of each coupling
    block, 'left_out': the norms of what Delta does not fold, Sigma_x - v_xc
    on a reference that is not HF}. delta=False leaves it out: the result
    is the static Hamiltonian only, with DeltaOffWarning and the flag
    'delta_off'. Until Task 13 a space whose occupied block is rotated is
    refused with Delta on (NotImplementedError). An unrestricted reference
    with a basis is refused with Delta on, the default
    (NotImplementedError): Delta for two spin channels is not built, so
    the static Hamiltonian it gives with delta=False has to be asked for by
    name. For canonical orbitals the coupling is identically zero: nothing
    is built, hybridization is None and there is no 'fock_coupling' entry,
    whatever `delta` is. _blocks
    (tests only) replaces DEFAULT_BLOCKS, the blocks the coupling folds.
    Exactly when the coupling is built the method's `downfold` gets
    hybridization=True and E_0 gains its one Delta term,
    energies.e_c_hybridization = E_c^{A,Delta} - E_c^A (method 'rpa';
    0.0 for 'hf'): energies.e0 holds it, energies.e0_without_delta and
    g_eff[0] do not (zero_body.py). Without a coupling the term is 0.0,
    not computed, and the three are one number. `hybridization` itself is
    not an option here (TypeError): it follows from `delta` and the basis.

    `numerical_flags` lists every named flag behind this
    Hamiltonian (src.Base.numerical_flags): those of the low-level run
    (`low.numerical_flags`) and those raised while building it, each
    distinct {'flag': name, 'message': text} once; empty for a clean result.
    """
    mf = low.mf
    mol = mf.mol
    mo, eps, nocc = _spin_channels(mf)
    settings = _settings(low, numerics)
    if 'hybridization' in method_opts:
        raise TypeError(
            'hybridization is not an option of build_effective_hamiltonian: '
            'it follows from delta and the orbital basis of the space.')
    if numerics is not None:
        method_opts['numerics'] = settings
    flags = []
    with collecting(flags):
        spaces = _active_spaces(active_space, mo, eps, nocc, settings)
        _require_basis_route(low, spaces)
        _require_delta_route(spaces, delta)
        rotated = any(s.basis is not None for s in spaces)
        unrestricted = len(spaces) == 2
        method = require_choice(low.method, tuple(LOW_LEVEL), 'method')
        mo, rotations = _rotated_orbitals(mo, eps, spaces)
        # the coupling is built below exactly when this holds; the method
        # then adds the E_0 term of the hybridized space (e_c_hybridization)
        if rotated and delta:
            method_opts['hybridization'] = True
        # `downfold` is called from this function itself, not from a helper:
        # the method's warnings count the frames up to the caller of this one
        terms = LOW_LEVEL[method].downfold(
            low, spaces if unrestricted else spaces[0], **method_opts)
        _check_terms(terms, method, unrestricted)

        core = core_terms(mol, mf.get_hcore(), mo, spaces)
        dc = double_counting(terms, tuple(s.n_act_occ for s in spaces))
        h_eff = effective_one_body(core.fock, dc.hartree, dc.exchange,
                                   dc.mode, terms.sigma_bar)
        v_eff = effective_interaction(
            mol, tuple(c[:, s.act] for c, s in zip(mo, spaces)), terms.vtilde)
        energies = assemble_energies(mf.energy_nuc(), core, dc, terms)
        # g_eff is the decoupled model, all three parts: its constant is
        # e0_without_delta, never e0 (zero_body.py)
        g_eff = pack(h_eff, v_eff, energies.e0_without_delta, spin_free)
        diagnostics = dict(terms.diagnostics)
        coupling = None
        if rotated and delta:
            # one channel: an unrestricted reference with a basis and Delta
            # on is refused above (_require_delta_route)
            coupling = fock_coupling(
                mf, mo[0], rotations[0], eps[0], spaces[0],
                blocks=DEFAULT_BLOCKS if _blocks is None else _blocks)
            diagnostics['fock_coupling'] = {'folded': coupling.norms,
                                            'left_out': coupling.left_out}
        elif rotated:
            # stacklevel 2: the caller of build_effective_hamiltonian
            flag('delta_off',
                 'an active space on a rotated orbital basis built with '
                 'delta=False: the Fock coupling the rotation creates '
                 'between the active and the environment orbitals is left '
                 'out, and the result is the static Hamiltonian only.',
                 DeltaOffWarning)
    inherited = list(getattr(low, 'numerical_flags', ()))
    return DownfoldedHamiltonian(
        g_eff=g_eff, space=_space(mf, mo, spaces, spin_free),
        energies=energies, diagnostics=diagnostics,
        numerical_flags=gathered(inherited + flags),
        hybridization=coupling, numerics=settings)


def _check_terms(terms, method, unrestricted):
    """Refuse a `DownfoldingTerms` whose spin blocks do not fit the reference:
    vtilde = (v,) and sigma_bar = (s,) for a restricted one, (aa, ab, bb)
    and (a, b) for an unrestricted one (spin.py)."""
    blocks = [len(terms.vtilde)]
    if terms.sigma_bar is not None:
        blocks.append(len(terms.sigma_bar))
    if blocks != ([3, 2] if unrestricted else [1, 1])[:len(blocks)]:
        raise ValueError(
            f"method={method!r} returned {len(terms.vtilde)} vtilde blocks"
            + (f" and {len(terms.sigma_bar)} Sigma_bar blocks"
               if terms.sigma_bar is not None else "")
            + ("; an unrestricted reference is two spin channels: "
               "vtilde = (aa, ab, bb), sigma_bar = (a, b)."
               if unrestricted else
               "; a restricted reference is one spin channel: "
               "vtilde = (v,), sigma_bar = (s,)."))


def _space(mf, mo, spaces, spin_free):
    """The `space` record of `build_effective_hamiltonian`: the active space
    in the form QDPT's builder gives it, each index entry a pair (alpha, beta)
    for an unrestricted reference, which also gets spin_overlap. `mo` are the
    orbitals of the spaces (rotated where a space has a basis); a space with
    a basis adds 'basis' and 'mo', the rotated coefficients, for viewers."""
    restricted = len(spaces) == 1

    def per_channel(view):
        return view(spaces[0]) if restricted else tuple(map(view, spaces))

    n_act = spaces[0].n_act
    space = {
        'n_orbs': n_act if spin_free else 2 * n_act,
        'n_elec_active': (spaces[0].nelec_active if restricted
                          else tuple(s.n_act_occ for s in spaces)),
        'act_idx': per_channel(lambda s: s.act),
        'act_occ_idx': per_channel(lambda s: s.occ_active),
        'act_vir_idx': per_channel(lambda s: s.virt_active),
        'spin_free': spin_free,
        'active_space': spaces[0] if restricted else spaces,
    }
    if not restricted:
        space['spin_overlap'] = _spin_overlap(mf, mo, spaces)
    if any(s.basis is not None for s in spaces):
        space['basis'] = per_channel(lambda s: s.basis)
        space['mo'] = mo[0] if restricted else mo
    return space


def _spin_overlap(mf, mo, spaces):
    """The blocks of O = C_alpha^T S C_beta between the inactive ('core') and
    the active orbitals of the two spins, rows alpha and columns beta, the
    active orbitals occupied first as in H_eff: what <S^2> of an active-space
    state with its core determinant needs, since alpha orbital p and beta
    orbital p of an unrestricted reference are different functions."""
    (c_a, c_b), (space_a, space_b) = mo, spaces
    overlap = c_a.T @ mf.get_ovlp() @ c_b
    rows = {'core': space_a.env_occ, 'active': space_a.act}
    cols = {'core': space_b.env_occ, 'active': space_b.act}
    return {f'{left}_{right}': overlap[np.ix_(i, j)]
            for left, i in rows.items() for right, j in cols.items()}


def pack(h_eff, v_eff, e0, spin_free):
    """The g_eff dictionary {0: e0, 1: one-body, 2: two-body} of the solvers.

    h_eff and v_eff hold the one-body term and the chemist active interaction
    (bare plus vtilde) per spin channel, (h,) and (v,) or (h_a, h_b) and
    (aa, ab, bb) (spin.py). The number of channels alone decides the form;
    two channels are never collapsed to one, also when they are equal.

    One channel, spin-free: g_eff[1] is the spatial one-body and g_eff[2] the
    spatial physicist <pq|rs> = (pr|qs); otherwise both go through
    spatial_to_spin_orbitals and key 0 is added here.

    Two channels, spin-free: the per-spin blocks, g_eff[1] = (h_a, h_b) and
    g_eff[2] = (aa, ab, bb), each two-body block physicist like the
    one-channel one, block[p,q,r,s] = <pq|rs> = (pr|qs), so the
    opposite-spin block is <p_a q_b | r_a s_b>. Otherwise the same
    interleaved spin orbitals as for one channel (alpha at 2p, beta at
    2p + 1) with the antisymmetrized physicist two-body term.

    The arrays of g_eff are copies, never views of the inputs; an empty
    active space packs zero-size arrays.
    """
    if len(h_eff) == 1:
        (h_a,), (aa,) = h_eff, v_eff
        spatial = {1: np.array(h_a), 2: np.array(aa.transpose(0, 2, 1, 3))}
        if spin_free:
            return {0: e0, **spatial}
        g = spatial_to_spin_orbitals(spatial, h_a.shape[0])
        g[0] = e0
        return g
    if spin_free:
        return {0: e0, 1: tuple(np.array(h) for h in h_eff),
                2: tuple(np.array(v.transpose(0, 2, 1, 3)) for v in v_eff)}
    physicist = spin_orbital_chemist(*v_eff).transpose(0, 2, 1, 3)
    return {0: e0, 1: spin_orbital_one_body(*h_eff),
            2: get_antisymmetrized_integrals(physicist)}
