"""Delta applied by `solve`, and the two static forms of Delta.

The solver function (`solvers.solve.solve`): `static_states` is the static
solve of today, returning the states, the plain engine and the sector;
with `sides` the states are refined by a dynamic backend and returned in
ascending order, `out['static_index']` mapping each back to its static
state and every per-state array of `out` permuted with it; two refined
roots that cross are flagged 'delta_roots_reordered', a state that was not
computed lying below a returned root 'delta_roots_missing'
(`DeltaRootsCrossedWarning`), from the count of solutions below each root
(`extra['solutions_below']`, checked against the explicit P + Q matrix,
only the requested spin counted). `out` reports the constant added and the
active energies. The record (`DownfoldedHamiltonian`): `solve` with each
`delta_method` and its constant (spec 6d), `static_hamiltonian`,
`prepare_delta` and its requests, `numerics` from the build, the
refusals. The static forms (`hybridization.py`): `static_mf_delta` against
its formula, `removal_moment` against the (N-1)-electron sector,
`state_averaged_delta` dropping the empty directions of gamma.

Water/6-31G RHF on the NO CAS(4,4) of Task 8, whose excited states are
coupled strongly enough to cross; N2/6-31G on an NO CAS(6,6), coupled
weakly, where nothing crosses. `prepare_delta` with solver='dmrg' is
checked by tests/dmrg_prepare_delta_check.py in its own interpreter
(block2 must be imported before numpy), which `test_dmrg_prepare_delta`
runs, and skips without block2."""
import math
import os
import subprocess
import sys
import warnings
from dataclasses import replace

import numpy as np
import pytest
from pyscf import fci

from src.Base.numerical_flags import collecting
from src.Base.numerics.numerics import Numerics
from src.EffectiveHamiltonian.active_space import ActiveSpace
from src.EffectiveHamiltonian.Downfolding import (build_effective_hamiltonian,
                                                  run_low_level)
from src.EffectiveHamiltonian.Downfolding.hybridization import (
    DELTA_METHODS, STATIC_METHODS, DeltaOffWarning, StaticDeltaWarning,
    removal_moment, state_averaged_delta, static_mf_delta)
from src.EffectiveHamiltonian.solvers import solve as solve_module
from src.EffectiveHamiltonian.solvers.backends import PyscfFCISolver
from src.EffectiveHamiltonian.solvers.dynamic_backends import (
    DeltaRootsCrossedWarning, LinearizedLowdinSolver, LowdinSolver)
from src.EffectiveHamiltonian.solvers.solve import (
    ENGINES, EmptyActiveSpaceWarning, SolveRequest, StaticStates, solve,
    static_states)
from tests.no_downfolding import (PIN_TOL, nitrogen, no_space, pq_spectrum,
                                  water)

CROSSING = ('delta_roots_reordered', 'delta_roots_missing')


def _record(call):
    """(result, flag names, warning classes, warnings) of call(): every
    warning recorded, so the output stays clean and each test asserts what
    it expects."""
    flags = []
    with collecting(flags), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = call()
    return (result, sorted({f['flag'] for f in flags}),
            {w.category for w in caught}, caught)


def _names(out):
    return sorted({f['flag'] for f in out['numerical_flags']})


def _request(h, nroots, spin=None):
    return SolveRequest(h.g_eff, h.space['n_orbs'], h.space['n_elec_active'],
                        spin_free=True, nroots=nroots, spin=spin)


def _spin_square(vector, norb, sector):
    return fci.spin_op.spin_square(vector, norb, sector)[0]


@pytest.fixture(scope='module')
def low():
    return run_low_level(water('hf'), method='rpa', mode='casida',
                         natural_orbitals=True)


@pytest.fixture(scope='module')
def h(low):
    """The water NO CAS(4,4) of Task 8, with its coupling."""
    return build_effective_hamiltonian(low, no_space(low, 2, 2),
                                       screening='crpa')


@pytest.fixture(scope='module')
def canonical(low):
    """Water CAS(4,4) on canonical orbitals: no coupling."""
    norb = low.mf.mo_coeff.shape[1]
    return build_effective_hamiltonian(
        low, ActiveSpace.from_indices(norb, 5, [3, 4], [5, 6]),
        screening='crpa')


def _refined(h, nroots, method, spin=None):
    """(st, DeltaResult) of the states of one request refined directly with
    the dynamic backend, the reference `solve` must reproduce."""
    st = static_states('pyscf_fci', _request(h, nroots, spin))
    cls = {'lowdin': LowdinSolver, 'linearized': LinearizedLowdinSolver}[method]
    res, _, _, _ = _record(lambda: cls(st.engine, h.hybridization.sides(),
                                       h.numerics).refine(
        st.h1, st.eri, st.norb, st.sector, st.energies, st.vectors,
        spin=spin))
    return st, res


# ------------------------------------------------------------ static_states
def test_static_states_are_today_s_solve(h):
    for spin, nroots, sector in ((None, 3, (2, 2)), (0, 2, (2, 2)),
                                 (2, 1, (3, 1))):
        st = static_states('pyscf_fci', _request(h, nroots, spin))
        assert isinstance(st, StaticStates)
        e = solve('pyscf_fci', h.g_eff, 4, 4, spin_free=True, nroots=nroots,
                  spin=spin)
        assert np.array_equal(st.energies + h.g_eff[0], e)
        assert st.sector == sector and st.norb == 4
        assert st.engine.spin_penalty is None
        assert len(st.vectors) == nroots
        # the arithmetic of today: the energies of the engine itself
        if spin is None:
            direct = PyscfFCISolver().roots(st.h1, st.eri, 4, (2, 2),
                                            nroots=nroots, spin=None)[0]
            assert np.array_equal(st.energies, direct)
        # h1 Hermitian, eri pair symmetric (rs|pq) = (pq|rs)
        assert np.array_equal(st.h1, st.h1.T)
        assert np.array_equal(st.eri, st.eri.transpose(2, 3, 0, 1))
        for v in st.vectors:
            want = 0.25 * spin * (spin + 2) if spin is not None else None
            if want is not None:
                assert abs(_spin_square(v, 4, sector) - want) < 1e-8


def test_static_states_refusals(h):
    with pytest.raises(ValueError, match='pyscf_fci'):
        static_states('ccsdt', _request(h, 1))
    with pytest.raises(ValueError, match='pyscf_fci'):
        static_states('openfermion', _request(h, 1))
    with pytest.raises(ValueError, match='spin'):
        static_states('pyscf_fci', _request(h, 1, spin=1))
    empty = SolveRequest({0: 1.0, 1: np.zeros((0, 0)),
                          2: np.zeros((0, 0, 0, 0))}, 0, 0, spin_free=True)
    with pytest.raises(ValueError, match='empty'):
        static_states('pyscf_fci', empty)


def test_engines_name_every_solver():
    assert set(ENGINES) == set(solve_module.SOLVERS)
    assert ENGINES['pyscf_fci'].has_sector_operations
    assert ENGINES['pyscf_fci'].has_rdms and ENGINES['dmrg'].has_rdms


# --------------------------------------------------- Delta through .solve
def test_solve_default_is_linearized(h):
    e, names, _, _ = _record(lambda: h.solve('pyscf_fci', nroots=2))
    st, ref = _refined(h, 2, 'linearized')
    order = np.argsort(ref.energies, kind='stable')
    diff = np.abs(e - (h.energies.e0 + ref.energies[order])).max()
    print(f'default solve against LinearizedLowdinSolver: {diff:.1e}')
    assert diff < 1e-10
    # root 1, the lowest triplet, is coupled strongly (Z 0.72) and the
    # second triplet, not computed, goes below it
    assert names == ['delta_linearization', 'delta_roots_missing']


def test_a_triplet_with_delta(h):
    out = {}
    e, names, caught, _ = _record(lambda: h.solve(
        'pyscf_fci', nroots=1, spin=2, delta_method='lowdin', out=out))
    st = static_states('pyscf_fci', _request(h, 1, 2))
    pq, weight = pq_spectrum(st.h1, st.eri, 4, (3, 1),
                             h.hybridization.sides())
    k = np.argmin(abs(pq - out['e_active'][0]))
    print(f'triplet: lowdin - P + Q eigenvalue {pq[k] - out["e_active"][0]:.1e}'
          f', P weight {weight[k]:.3f}')
    assert abs(pq[k] - out['e_active'][0]) < 1e-8 and weight[k] > 0.5
    assert abs(out['spin_square'][0] - 2.0) < 1e-8
    # the second triplet, not computed, lies below the one returned
    assert names == ['delta_roots_missing'] == _names(out)
    assert DeltaRootsCrossedWarning in caught
    more, more_names, _, _ = _record(lambda: h.solve(
        'pyscf_fci', nroots=2, spin=2, delta_method='lowdin'))
    assert more[0] < e[0] - 0.01
    assert more_names == ['delta_roots_reordered']


def test_out_reports_the_constant_and_the_parts(h, canonical):
    out = {}
    e, names, _, _ = _record(lambda: h.solve('pyscf_fci', out=out))
    assert names == []
    assert np.array_equal(e, out['e0'] + out['e_active'])
    assert out['e0'] == h.energies.e0
    assert np.abs(out['e_active'] - (out['static'] + out['correction'])
                  ).max() < 1e-12
    for key in ('static', 'correction', 'z', 'min_denominator', 'converged',
                'static_index', 'linearization_error', 'linearization_point',
                'solutions_below', 'spin_square'):
        assert len(out[key]) == len(e), key
    off = {}
    e_off, _, _, _ = _record(lambda: h.solve('pyscf_fci', delta_method=False,
                                             out=off))
    assert off['e0'] == h.energies.e0_without_delta == h.g_eff[0]
    assert np.array_equal(e_off, off['e0'] + off['e_active'])
    plain = {}
    e_c = canonical.solve('pyscf_fci', nroots=2, out=plain)
    assert plain['e0'] == canonical.energies.e0 == canonical.g_eff[0]
    assert np.array_equal(e_c, plain['e0'] + plain['e_active'])
    assert plain['numerical_flags'] == []


def test_delta_off_is_flagged(low, h):
    with pytest.warns(DeltaOffWarning):
        h_off = build_effective_hamiltonian(low, no_space(low, 2, 2),
                                            screening='crpa', delta=False)
    out = {}
    e, names, caught, records = _record(lambda: h.solve(
        'pyscf_fci', nroots=2, delta_method=False, out=out))
    assert np.array_equal(e, h_off.solve('pyscf_fci', nroots=2))
    assert names == ['delta_off'] == _names(out)
    assert caught == {DeltaOffWarning}
    # the warning points at the line that called h.solve
    assert all(w.filename == __file__ for w in records)
    assert np.array_equal(out['correction'], np.zeros(2))
    assert np.array_equal(out['z'], np.ones(2))
    assert np.array_equal(out['static'], out['e_active'])


def test_canonical_solve_is_unchanged(canonical):
    g, n = canonical.g_eff, canonical.space['n_orbs']
    want = solve('pyscf_fci', g, n, 4, spin_free=True, nroots=3)
    assert np.array_equal(canonical.solve('pyscf_fci', nroots=3), want)
    out = {}
    assert np.array_equal(canonical.solve('pyscf_fci', nroots=3, out=out),
                          want)
    assert np.array_equal(canonical.solve('pyscf_fci', nroots=3,
                                          delta_method=False), want)
    # today's arithmetic: the constant plus the engine's energies
    st = static_states('pyscf_fci', _request(canonical, 3))
    assert np.array_equal(want, g.get(0, 0.0) + st.energies)
    assert 'static' not in out and 'static_index' not in out


def test_a_method_without_a_coupling_is_refused(canonical):
    assert not canonical.g_eff_is_static_part
    for method in ('lowdin', 'linearized', 'static_mf', 'static_sa'):
        with pytest.raises(ValueError, match='nothing to fold'):
            canonical.solve('pyscf_fci', delta_method=method)
    with pytest.raises(ValueError, match='nothing to fold'):
        canonical.prepare_delta(((1, 0),))


def test_spin_square_for_a_restricted_reference(canonical):
    out = {}
    canonical.solve('pyscf_fci', nroots=2, out=out)
    assert np.abs(np.array(out['spin_square']) - [0.0, 2.0]).max() < 1e-8
    for spin, want in ((0, 0.0), (2, 2.0)):
        out = {}
        canonical.solve('pyscf_fci', nroots=1, spin=spin, out=out)
        assert abs(out['spin_square'][0] - want) < 1e-8


def test_record_takes_its_coupling_and_numerics_itself(h):
    assert h.g_eff_is_static_part
    assert isinstance(h.numerics, Numerics)
    for key, value in (('sides', []), ('d_sa', None), ('numerics', None)):
        with pytest.raises(TypeError, match=key):
            h.solve('pyscf_fci', **{key: value})
    with pytest.raises(ValueError, match='linearized'):
        h.solve('pyscf_fci', delta_method='exact')
    with pytest.raises(ValueError, match='prepare_delta'):
        h.solve('pyscf_fci', delta_method='sa_remainder')
    assert set(DELTA_METHODS) == {'linearized', 'lowdin', 'sa_remainder',
                                  'static_sa', 'static_mf'}
    assert STATIC_METHODS == ('static_sa', 'static_mf')


def test_solver_function_refusals(h):
    args = (h.g_eff, 4, 4)
    sides = h.hybridization.sides()
    with pytest.raises(ValueError, match='sides'):
        solve('pyscf_fci', *args, spin_free=True, delta_method='lowdin')
    with pytest.raises(ValueError, match='sides'):
        solve('pyscf_fci', *args, spin_free=True, d_sa=(0, 0))
    with pytest.raises(ValueError, match='d_sa'):
        solve('pyscf_fci', *args, spin_free=True, sides=sides,
              delta_method='lowdin', d_sa=(np.zeros((4, 4)), 0.0))
    for method in (None, 'static_mf', 'sa_remainder'):
        with pytest.raises(ValueError, match='lowdin'):
            solve('pyscf_fci', *args, spin_free=True, sides=sides,
                  delta_method=method)
    with pytest.raises(NotImplementedError, match='spatial'):
        solve('pyscf_fci', *args, spin_free=False, sides=sides,
              delta_method='lowdin')
    resolved = {0: 0.0, 1: (np.eye(2), np.eye(2)),
                2: (np.zeros((2,) * 4),) * 3}
    with pytest.raises(NotImplementedError, match='spin-resolved'):
        solve('pyscf_fci', resolved, 2, (1, 1), spin_free=True, sides=[],
              delta_method='lowdin')


def test_empty_space_with_sides():
    out = {}
    g = {0: 1.5, 1: np.zeros((0, 0)), 2: np.zeros((0, 0, 0, 0))}
    with pytest.warns(EmptyActiveSpaceWarning) as caught:
        e = solve('pyscf_fci', g, 0, 0, spin_free=True, sides=[],
                  delta_method='linearized', out=out)
    assert caught[0].filename == __file__
    assert np.array_equal(e, [1.5])
    assert out['e0'] == 1.5 and np.array_equal(out['e_active'], [0.0])
    assert np.array_equal(out['static'], [0.0])
    assert np.array_equal(out['correction'], [0.0])
    assert np.array_equal(out['z'], [1.0])
    assert np.array_equal(out['min_denominator'], [np.inf])
    assert np.array_equal(out['converged'], [True])
    assert np.array_equal(out['static_index'], [0])
    assert _names(out) == ['empty_active_space']


def test_no_active_virtual_means_no_delta(low):
    h0 = build_effective_hamiltonian(low, no_space(low, 2, 0),
                                     screening='crpa')
    assert h0.hybridization.sides() == []
    out = {}
    e, names, _, _ = _record(lambda: h0.solve('pyscf_fci', out=out))
    st = static_states('pyscf_fci', _request(h0, 1))
    assert np.array_equal(e, h0.energies.e0 + st.energies)
    assert np.array_equal(out['z'], [1.0])
    assert np.array_equal(out['correction'], [0.0])
    assert names == []


# ------------------------------------------------------- order and crossing
def test_crossed_roots_are_sorted(h):
    """Water, the lowest three states of any spin with exact Loewdin: the
    second triplet (static 2) goes below the first (static 1). The result
    is ascending, static_index maps back, and every per-state array of out
    is permuted with it."""
    out = {}
    e, names, caught, records = _record(lambda: h.solve(
        'pyscf_fci', nroots=3, delta_method='lowdin', out=out))
    st, ref = _refined(h, 3, 'lowdin')
    order = np.argsort(ref.energies, kind='stable')
    print(f'static {np.round(ref.static, 5)} -> refined '
          f'{np.round(ref.energies, 5)}, order {order}')
    assert order.tolist() == [0, 2, 1]
    assert np.array_equal(out['static_index'], order)
    assert np.all(np.diff(e) > 0)
    assert np.array_equal(e, h.energies.e0 + ref.energies[order])
    assert np.array_equal(out['e_active'], ref.energies[order])
    for key in ('static', 'z', 'min_denominator', 'converged'):
        assert np.array_equal(out[key], getattr(ref, key)[order]), key
    assert ref.extra
    for key, value in ref.extra.items():
        assert np.array_equal(out[key], np.asarray(value)[order]), key
    want = [_spin_square(ref.vectors[k], 4, (2, 2)) for k in order]
    assert np.abs(np.array(out['spin_square']) - want).max() < 1e-10
    assert np.abs(np.array(want) - [0.0, 2.0, 2.0]).max() < 1e-8
    # a singlet that was not computed (static 5) also goes below static 1
    assert names == sorted(CROSSING) == _names(out)
    assert caught == {DeltaRootsCrossedWarning}
    crossed = [str(w.message) for w in records
               if 'crossed' in str(w.message)]
    assert len(crossed) == 1 and 'static states (1, 2)' in crossed[0]


def test_solutions_below_are_those_of_the_pq_matrix(h):
    """extra['solutions_below'] of each root against the explicit P + Q
    matrix: below the lowest pole every eigenvalue of P + Q below E_k is a
    solution of the Delta problem, and the requested spin is that of the
    P part of its eigenvector."""
    st = static_states('pyscf_fci', _request(h, 1))
    sides = h.hybridization.sides()
    pq, _, p_parts = pq_spectrum(st.h1, st.eri, 4, (2, 2), sides,
                                 with_vectors=True)
    shape = (math.comb(4, 2),) * 2
    spins = np.array([_spin_square((p / np.linalg.norm(p)).reshape(shape),
                                   4, (2, 2)) for p in p_parts.T])
    engine = PyscfFCISolver()
    first_pole = min(np.linalg.eigvalsh(engine.sector_matrix(
        st.h1, st.eri, 4, sector))[0] for sector in ((1, 2), (2, 1))) + min(
        sides[0][2])
    for spin, nroots in ((None, 3), (0, 2)):
        out = {}
        _record(lambda: h.solve('pyscf_fci', nroots=nroots, spin=spin,
                                delta_method='lowdin', out=out))
        for k, energy in enumerate(out['e_active']):
            assert energy < first_pole
            keep = pq < energy - 1e-6
            if spin is not None:
                keep &= np.abs(spins - 0.25 * spin * (spin + 2)) < 1e-6
            print(f'spin {spin} root {k}: solutions below {out["solutions_below"][k]}, '
                  f'P + Q {keep.sum()}')
            assert out['solutions_below'][k] == keep.sum()
    # spin 0: the singlet static 5 lies below root 1, the triplets do not
    # count
    assert _names(out) == ['delta_roots_missing']


def _shifted(h, shift, **kwargs):
    """`solve` of the water NO record with the poles of its coupling moved
    down by `shift` (lam - shift): a model with a low level, the way the
    Task 8 near-pole test moves one. Returns (energies, out, caught,
    first pole); the first pole is the lowest state of the (1, 2) sector
    plus the lowest level."""
    sides = [(kind, sigma, lam - shift, b)
             for kind, sigma, lam, b in h.hybridization.sides()]
    coupled = {0: h.energies.e0, 1: h.g_eff[1], 2: h.g_eff[2]}
    out = {}
    e, _, _, caught = _record(lambda: solve(
        'pyscf_fci', coupled, 4, 4, spin_free=True, nroots=3, sides=sides,
        delta_method='lowdin', out=out, **kwargs))
    st = static_states('pyscf_fci', _request(h, 1))
    first_pole = np.linalg.eigvalsh(PyscfFCISolver().sector_matrix(
        st.h1, st.eri, 4, (1, 2)))[0] + min(sides[0][2])
    return e, out, caught, first_pole


@pytest.mark.parametrize('shift', [0.8, 1.0])
def test_roots_above_the_lowest_pole_are_not_checked(h, shift):
    """With a level moved down so that the first pole lies between the
    ground state and the other roots, a returned root above the pole is
    marked in out['crossing_unchecked'] (the count of solutions below it
    is solutions minus poles below) and is not flagged
    'delta_roots_missing'; the ground state, below the pole, is checked.
    A near-pole root is unchecked too."""
    e, out, caught, first_pole = _shifted(h, shift)
    above = out['e_active'] > first_pole
    print(f'shift {shift}: first pole {first_pole:.5f}, roots '
          f'{np.round(out["e_active"], 5)}, near pole {out["near_pole"]}, '
          f'solutions below {out["solutions_below"]}, unchecked '
          f'{out["crossing_unchecked"]}')
    assert above.tolist() == [False, True, True]
    assert out['crossing_unchecked'].dtype == bool
    assert np.array_equal(out['crossing_unchecked'],
                          above | out['near_pole'])
    assert not out['crossing_unchecked'][0]
    # solutions_below of the roots above the pole is large (it counts the
    # solutions below the pole too); no message names them
    assert out['solutions_below'][1] > 0
    said = [str(w.message) for w in caught
            if 'solution(s) of the Delta' in str(w.message)]
    assert said and all('returned root 0 ' in m for m in said), said
    assert 'delta_roots_missing' in _names(out)
    if shift == 0.8:
        assert out['near_pole'].any() and out['crossing_unchecked'][
            out['near_pole']].all()


def test_per_root_flag_messages_name_the_static_state(h):
    """'delta_near_pole', 'delta_satellite' and 'delta_linearization' name
    the static state, and say that out['static_index'] maps static states
    to returned roots (the returned order is the sorted one)."""
    _, out, caught, _ = _shifted(h, 0.8)
    texts = [str(w.message) for w in caught
             if str(w.message).startswith('Delta (lowdin): ')
             and ' state ' in str(w.message)[:40]]
    assert texts
    for text in texts:
        assert 'static state ' in text and "out['static_index']" in text, text
        assert not text.startswith('Delta (lowdin): state')


def test_a_reused_out_loses_the_delta_keys(h):
    """One dict given to a Delta solve and then to one without Delta keeps
    none of the first solve's Delta-only entries."""
    out = {}
    _record(lambda: h.solve('pyscf_fci', nroots=2, out=out))
    first = set(out)
    assert {'static_index', 'solutions_below', 'near_pole',
            'linearization_error'} <= first
    _record(lambda: h.solve('pyscf_fci', nroots=2, delta_method=False,
                            out=out))
    assert set(out) == {'e0', 'e_active', 'spin_square', 'numerical_flags',
                        'static', 'correction', 'z'}, sorted(out)
    assert np.array_equal(out['static'], out['e_active'])
    out = {}
    _record(lambda: h.solve('pyscf_fci', nroots=2, out=out))
    _record(lambda: h.solve('pyscf_fci', nroots=2, delta_method='static_mf',
                            out=out))
    for key in ('static_index', 'solutions_below', 'near_pole',
                'crossing_unchecked', 'linearization_error',
                'linearization_point', 'min_denominator', 'converged'):
        assert key not in out, key


#: energies of solves without sides taken from the code before Task 9
#: (f8244a2; solvers/solve.py is the file of ebb53c7), written with
#: float.hex(): the water CAS(4,4) on canonical orbitals through
#: 'pyscf_fci', and a 4-site Hubbard ring (t = -1, U = 2, g_eff[0] = 0.5)
PINNED = {
    (1, None): ['-0x1.307cebea56a76p+6'],
    (3, None): ['-0x1.307cebea56a76p+6', '-0x1.2f0c56752a470p+6',
                '-0x1.2ef44d0c8ccebp+6'],
    (1, 2): ['-0x1.2f0c56752a470p+6'],
    (3, 2): ['-0x1.2f0c56752a470p+6', '-0x1.2ec6b15e9648ep+6',
             '-0x1.2ea9937fe0a8bp+6'],
}
PINNED_HUBBARD = ['-0x1.2a09e667f3bbap+1', '-0x1.17c9cea19249ep+1',
                  '-0x1.7ffffffffffccp+0']


def test_solve_without_sides_matches_the_pre_task_9_energies(canonical):
    """Comparison (to PIN_TOL) with energies computed by the code before
    Task 9, where `static_states` did not exist: test_static_states_are_
    today_s_solve compares `solve` with a function `solve` now calls."""
    for (nroots, spin), want in PINNED.items():
        e = solve('pyscf_fci', canonical.g_eff, 4, 4, spin_free=True,
                  nroots=nroots, spin=spin)
        assert all(abs(a - float.fromhex(b)) < PIN_TOL for a, b in zip(e, want)), (
            nroots, spin)
        assert len(e) == len(want)
    t = np.zeros((4, 4))
    for i in range(4):
        t[i, (i + 1) % 4] = t[(i + 1) % 4, i] = -1.0
    v = np.zeros((4,) * 4)
    for i in range(4):
        v[i, i, i, i] = 2.0
    e = solve('pyscf_fci', {0: 0.5, 1: t, 2: v}, 4, 4, spin_free=True,
              nroots=3)
    assert len(e) == 3
    assert all(abs(a - float.fromhex(b)) < PIN_TOL
               for a, b in zip(e, PINNED_HUBBARD))


def test_spin_square_from_the_sector_operations():
    """`EnergyDependentSolver.spin_square`, which tells the states of the
    requested spin apart for extra['solutions_below'], from the engine's
    creation and annihilation alone: pyscf's spin_square on random vectors
    of sectors with more alpha, more beta and no beta electrons."""
    backend = LowdinSolver(PyscfFCISolver(), [])
    rng = np.random.default_rng(0)
    for norb, sector in ((4, (2, 2)), (4, (3, 1)), (4, (1, 3)), (5, (3, 2)),
                         (6, (3, 3)), (4, (4, 0))):
        v = rng.normal(size=(math.comb(norb, sector[0]),
                             math.comb(norb, sector[1])))
        want = _spin_square(v / np.linalg.norm(v), norb, sector)
        assert abs(backend.spin_square(v, norb, sector) - want) < 1e-12


def test_weak_coupling_raises_no_crossing_flag():
    """N2 NO CAS(6,6), six states of any spin (two degenerate pairs),
    Z above 0.9: nothing crosses and nothing is missing."""
    low = run_low_level(nitrogen(), method='rpa', mode='casida',
                        natural_orbitals=True)
    h = build_effective_hamiltonian(low, no_space(low, 3, 3),
                                    screening='crpa')
    out = {}
    e, names, _, _ = _record(lambda: h.solve('pyscf_fci', nroots=6, out=out))
    print(f'N2: static index {out["static_index"]}, Z {np.round(out["z"], 3)}'
          f', solutions below {out["solutions_below"]}')
    assert np.all(np.diff(e) >= 0)
    assert names == []


# ------------------------------------------------------- the static forms
def test_static_mf_matches_its_formula_and_warns(h):
    cp = h.hybridization
    D, c = static_mf_delta(cp)
    n = len(cp.eps_active)

    def delta(t, u, w, hole):
        b, lam = ((cp.b_hole, cp.lam_occupied) if hole
                  else (cp.b_particle, cp.lam_virtual))
        return sum(b[t, x] * b[u, x] / (w - lam[x]) for x in range(len(lam)))

    want = np.zeros((n, n))
    for t in range(n):
        for u in range(n):
            for w in (cp.eps_active[t], cp.eps_active[u]):
                want[t, u] += 0.5 * (delta(t, u, w, False)
                                     + delta(t, u, w, True))
    print(f'static_mf against its formula: {np.abs(D - want).max():.1e}')
    assert np.abs(D - want).max() < 1e-12
    assert c == 0.0               # canonical occupied orbitals: no hole side
    # the constant of a hole side: -g tr(D_hole), g = 2
    rng = np.random.default_rng(2)
    b_hole = np.zeros((n, 3))
    b_hole[:2] = 0.1 * rng.normal(size=(2, 3))
    lam_hole = np.array([-2.0, -1.5, -1.2])
    holed = replace(cp, b_hole=b_hole, lam_occupied=lam_hole)
    D_h, c_h = static_mf_delta(holed)

    def hole_delta(t, u, w):
        return sum(b_hole[t, x] * b_hole[u, x] / (w - lam_hole[x])
                   for x in range(3))

    eps = cp.eps_active
    hole = np.array([[0.5 * (hole_delta(t, u, eps[t])
                             + hole_delta(t, u, eps[u]))
                      for u in range(n)] for t in range(n)])
    assert abs(c_h - (-2.0 * np.trace(hole))) < 1e-12
    assert np.abs(D_h - (D + hole)).max() < 1e-12
    out = {}
    with pytest.warns(StaticDeltaWarning) as caught:
        e = h.solve('pyscf_fci', delta_method='static_mf', nroots=2, out=out)
    assert _names(out) == ['delta_static_mf']
    assert caught[0].filename == __file__
    by_hand = solve('pyscf_fci', {0: h.energies.e0 + c, 1: h.g_eff[1] + D,
                                  2: h.g_eff[2]}, 4, 4, spin_free=True,
                    nroots=2)
    assert np.array_equal(e, by_hand)
    assert out['e0'] == h.energies.e0 + c
    assert np.array_equal(out['correction'], np.zeros(2))
    assert np.array_equal(out['z'], np.ones(2))
    assert np.array_equal(out['static'], out['e_active'])


def test_removal_moment_from_density_matrices(h):
    """removal_moment is sum_sigma <a_t K|(E_K - H^(N-1))|a_u K>, from the
    vectors and the (N-1)-electron sector matrix of the engine."""
    st = static_states('pyscf_fci', _request(h, 3))
    engine, worst = st.engine, 0.0
    sectors = {0: (1, 2), 1: (2, 1)}
    h_m = {s: engine.sector_matrix(st.h1, st.eri, 4, m)
           for s, m in sectors.items()}
    for e_k, k in zip(st.energies, st.vectors):
        dm1, dm2 = engine.rdms(k, 4, st.sector, spin=None)
        got = removal_moment(st.h1, st.eri, dm1, dm2)
        want = np.zeros((4, 4))
        for s, m in sectors.items():
            a = [engine.annihilate(k, 4, st.sector, u, s).ravel()
                 for u in range(4)]
            for t in range(4):
                for u in range(4):
                    want[t, u] += a[t] @ (e_k * a[u] - h_m[s] @ a[u])
        worst = max(worst, np.abs(got - want).max())
    print(f'removal moment against the sector: {worst:.1e}')
    assert worst < 1e-9


def test_static_sa_needs_prepare_delta(h):
    with pytest.raises(ValueError, match='prepare_delta'):
        h.solve('pyscf_fci', delta_method='static_sa')
    with pytest.raises(ValueError, match='prepare_delta'):
        h.static_hamiltonian('static_sa')


def test_prepare_delta_requests(h):
    h3 = h.prepare_delta(((3, None),))
    assert h3.hybridization.d_sa is not None and h.hybridization.d_sa is None
    assert h3.g_eff is h.g_eff and h3.numerics is h.numerics
    h.prepare_delta(((2, 0), (1, 2)))
    for bad in (((3, None), (1, 2)), ((1, 0), (2, 0)), ((0, 0),)):
        with pytest.raises(ValueError):
            h.prepare_delta(bad)
    with pytest.raises(ValueError, match='spin=None'):
        h.prepare_delta(((1, None),), solver='dmrg')
    # the ground state is added when no request holds it
    alone = h.prepare_delta(((1, 2),))
    rdms = []
    for n, spin in ((1, 2), (1, 0)):
        st = static_states('pyscf_fci', _request(h, n, spin))
        rdms += [st.engine.rdms(v, st.norb, st.sector, spin=None)
                 for v in st.vectors]
    D, c = state_averaged_delta(st.h1, st.eri, h.hybridization, rdms,
                                h.numerics)
    D_triplet, _ = state_averaged_delta(st.h1, st.eri, h.hybridization,
                                        rdms[:1], h.numerics)
    got_D, got_c = alone.hybridization.d_sa
    print(f'prepare_delta((1, 2)) against (1, 2) + (1, 0) by hand: '
          f'{np.abs(got_D - D).max():.1e}; without the ground state '
          f'{np.abs(got_D - D_triplet).max():.1e}')
    assert np.abs(got_D - D).max() < 1e-12 and got_c == c == 0.0
    assert np.abs(got_D - D_triplet).max() > 1e-4


def test_empty_virtual_directions_are_dropped(h):
    """A state whose second active virtual (active index 3) is exactly
    empty: its integrals with every other orbital are zeroed, so gamma has
    an exact null direction there. D is finite, zero along that direction,
    and along the other one the formula of one kept direction."""
    st = static_states('pyscf_fci', _request(h, 1, 0))
    h1, eri = st.h1.copy(), st.eri.copy()
    k = 3
    h1[k, :] = h1[:, k] = 0.0
    h1[k, k] = 5.0
    eri[k], eri[:, k], eri[:, :, k], eri[:, :, :, k] = 0.0, 0.0, 0.0, 0.0
    engine = PyscfFCISolver()
    _, V = engine.roots(h1, eri, 4, (2, 2), nroots=1, spin=0)
    dm1, dm2 = engine.rdms(V[0], 4, (2, 2), spin=None)
    assert abs(dm1[k, k]) < 1e-14 and dm1[2, 2] > 1e-4
    cp = h.hybridization
    D, c = state_averaged_delta(h1, eri, cp, [(dm1, dm2)], h.numerics)
    assert np.all(np.isfinite(D)) and c == 0.0
    assert np.abs(D[k]).max() < 1e-14 and np.abs(D[:, k]).max() < 1e-14
    w = removal_moment(h1, eri, dm1, dm2)[2, 2] / dm1[2, 2]
    want = np.sum(cp.b_particle[2] ** 2 / (w - cp.lam_virtual))
    assert abs(D[2, 2] - want) < 1e-12 and abs(want) > 1e-3
    # a determinant: every active virtual empty, nothing to fold
    det = np.zeros((math.comb(4, 2),) * 2)
    det[0, 0] = 1.0
    D0, _ = state_averaged_delta(h1, eri, cp, [engine.rdms(det, 4, (2, 2),
                                                           spin=None)],
                                 h.numerics)
    assert np.array_equal(D0, np.zeros((4, 4)))


def test_static_hamiltonian_forms(h, canonical):
    h2 = h.prepare_delta(((2, None),))
    g = h2.static_hamiltonian('static_sa')
    D, c = h2.hybridization.d_sa
    assert g[0] == h.energies.e0 + c
    assert np.array_equal(g[1], h.g_eff[1] + D) and g[2] is h.g_eff[2]
    h1 = 0.5 * (g[1] + g[1].T)
    eri = g[2].transpose(0, 2, 1, 3)
    eri = 0.5 * (eri + eri.transpose(2, 3, 0, 1))
    eri = 0.5 * (eri + eri.transpose(1, 0, 3, 2))
    by_hand = g[0] + PyscfFCISolver().roots(h1, eri, 4, (2, 2), nroots=1,
                                            spin=None)[0]
    e = h2.solve('pyscf_fci', delta_method='static_sa')
    print(f"static_sa by hand: {np.abs(e - by_hand).max():.1e}")
    assert np.abs(e - by_hand).max() < 1e-10
    with pytest.warns(StaticDeltaWarning) as caught:
        g_mf = h.static_hamiltonian('static_mf')
    assert caught[0].filename == __file__
    D_mf, c_mf = static_mf_delta(h.hybridization)
    assert g_mf[0] == h.energies.e0 + c_mf
    assert np.array_equal(g_mf[1], h.g_eff[1] + D_mf)
    with pytest.warns(DeltaOffWarning):
        assert h.static_hamiltonian(False) is h.g_eff
    for method in ('linearized', 'lowdin', 'sa_remainder', None):
        with pytest.raises(ValueError):
            h.static_hamiltonian(method)
    assert canonical.static_hamiltonian(False) is canonical.g_eff
    with pytest.raises(ValueError, match='nothing to fold'):
        canonical.static_hamiltonian('static_mf')


def test_backends_without_what_a_method_needs(h, monkeypatch):
    def no_solve(*args, **kwargs):
        raise AssertionError('a static solve ran before the refusal')
    monkeypatch.setattr(solve_module, 'static_states', no_solve)
    for solver in ('ccsdt', 'ccsdtq', 'openfermion', 'dmrg'):
        for method in (None, 'lowdin'):
            with pytest.raises(ValueError, match='static_sa') as err:
                h.solve(solver, delta_method=method)
            assert 'static_mf' in str(err.value)
    for solver in ('ccsdt', 'openfermion'):
        with pytest.raises(ValueError, match='density matrices') as err:
            h.prepare_delta(((1, 0),), solver=solver)
        assert 'pyscf_fci' in str(err.value) and 'dmrg' in str(err.value)
    with pytest.raises(ValueError, match='solver'):
        h.prepare_delta(((1, 0),), solver='nonsense')


def test_spin_orbital_record_is_refused(low):
    hs = build_effective_hamiltonian(low, no_space(low, 2, 2),
                                     screening='crpa', spin_free=False)
    assert hs.g_eff_is_static_part
    for method in (None, 'lowdin', 'static_mf'):
        with pytest.raises(NotImplementedError, match='spin_free'):
            hs.solve('pyscf_fci', delta_method=method)
    with pytest.raises(NotImplementedError, match='spin_free'):
        hs.prepare_delta(((1, 0),))
    with pytest.raises(NotImplementedError, match='spin_free'):
        hs.static_hamiltonian('static_mf')
    with pytest.warns(DeltaOffWarning):
        e = hs.solve('pyscf_fci', delta_method=False)
    assert np.isfinite(e[0])


def test_build_numerics_reach_the_solver(low):
    tiny = Numerics(layout={'delta_dense_gb': 1e-6})
    ht = build_effective_hamiltonian(low, no_space(low, 2, 2),
                                     screening='crpa', numerics=tiny)
    assert ht.numerics.layout.delta_dense_gb == 1e-6
    with pytest.raises(ValueError, match='delta_dense_gb'):
        ht.solve('pyscf_fci')


def test_delta_occupation_floor_is_a_setting():
    assert Numerics().downfolding.delta_occupation_floor == 1e-6
    for bad in (0.0, -1e-6, float('nan'), float('inf')):
        with pytest.raises(ValueError, match='delta_occupation_floor'):
            Numerics(downfolding={'delta_occupation_floor': bad})


# ------------------------------------------------------- D_SA built by DMRG
def test_dmrg_prepare_delta():
    """prepare_delta(solver='dmrg') against solver='pyscf_fci', with
    'static_sa' (tests/dmrg_prepare_delta_check.py: D_SA and the static
    energies within 1e-6 Ha on water/6-31G NO CAS(4,4), both blocks
    rotated). The script runs in its own interpreter, since block2 must be
    imported before numpy; skipped when block2 is not installed."""
    probe = subprocess.run([sys.executable, '-c', 'import block2'],
                           capture_output=True)
    if probe.returncode != 0:
        pytest.skip('block2 is not installed: the DMRG path of '
                    'prepare_delta is not checked here')
    script = os.path.join(os.path.dirname(__file__),
                          'dmrg_prepare_delta_check.py')
    result = subprocess.run([sys.executable, script], capture_output=True,
                            text=True)
    assert result.returncode == 0 and (
        'DMRG PREPARE_DELTA VERIFIED' in result.stdout), (
        result.stdout[-4000:] + result.stderr[-4000:])
