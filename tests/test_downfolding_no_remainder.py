"""The remainder method 'sa_remainder' (`RemainderSolver` of
`solvers/dynamic_backends.py`): the static D_SA in h, and one scalar per
state, the expectation value of what D_SA leaves out,

    r_K(E) = <K|Delta(E)|K> - (tr(D dm1_K) + c),
    dE_K   = r_K(E_K) / (1 - r'_K(E_K)),

with |K>, E_K the states of h + D and their energies plus c, and the
(N-1)- and (N+1)-electron spectra of h by block Lanczos.

Water/6-31G on the NO CAS(4,4) of Task 8, D_SA from the states
((3, 0), (2, 2)) declared together: the excitation energies against exact
Loewdin, with canonical occupied orbitals (particle side) and with both
blocks rotated (both sides) on HF and on PBE0, and the total ground-state
energy with both sides, which sees the constant c of D_SA, all at the
spec's bar of 0.01 eV; at the default depth the change of dE from one
block less below 1e-5. These miss their bars and are xfail(strict=True),
with the measured numbers in their docstrings (WATER_MISSES), for Erik to
decide. At the full Lanczos depth the remainder equals the one built from
the dense amplitudes of the dense methods, and E_K + r_K equals
<K|H_N + Delta(E_K)|K>; lanczos_change is the difference from one block
less of the same run; a Lanczos pole moved next to and onto a state gives
a finite energy and 'delta_near_pole'; every root is marked unchecked for
missing states; the density matrix taken from the sector operations equals
the engine's. N2/6-31G on the NO CAS(6,6) of Task 8, coupled weakly: the
ten states of D_SA against exact Loewdin within the bar, and the two
partners of a degenerate state with one energy. A record without D_SA, a
solve without d_sa, and a solver without sector operations are refused
before anything is solved."""
import warnings

import numpy as np
import pytest

from src.Base.numerical_flags import collecting
from src.Base.numerics.numerics import Numerics
from src.EffectiveHamiltonian.Downfolding import (build_effective_hamiltonian,
                                                  run_low_level)
from src.EffectiveHamiltonian.solvers import solve as solve_module
from src.EffectiveHamiltonian.solvers.backends import PyscfFCISolver
from src.EffectiveHamiltonian.solvers.dynamic_backends import (
    DYNAMIC_METHODS, DeltaNearPoleWarning, LowdinSolver, RemainderSolver,
    removal_spectrum)
from src.EffectiveHamiltonian.solvers.solve import solve
from tests.no_downfolding import nitrogen, no_space, water

#: the states D_SA is built from, declared together: three singlets and
#: two triplets
STATES = ((3, 0), (2, 2))

#: the spec's bar for 'sa_remainder' against exact Loewdin, 0.01 eV in Ha
BAR = 3.7e-4

#: a depth above every sector dimension of the CAS(4,4) (24 for one
#: electron less or more): the Krylov space is exhausted first
FULL = Numerics(downfolding={'delta_lanczos_depth': 100})


def _integrals(h):
    """(h1, eri) of a build, as pyscf's FCI takes them."""
    h1 = 0.5 * (h.g_eff[1] + h.g_eff[1].T)
    eri = h.g_eff[2].transpose(0, 2, 1, 3)
    eri = 0.5 * (eri + eri.transpose(1, 0, 2, 3))
    eri = 0.5 * (eri + eri.transpose(0, 1, 3, 2))
    eri = 0.5 * (eri + eri.transpose(2, 3, 0, 1))
    return h1, eri


def _record(call):
    """(result, flag names, warning classes) of call(): every warning
    recorded, so the output stays clean and each test asserts what it
    expects."""
    flags = []
    with collecting(flags), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = call()
    return (result, sorted({f['flag'] for f in flags}),
            {w.category for w in caught})


def _close_chkfile(mf):
    """Close pyscf's temporary chkfile of a mean field. The xfail tests
    below raise, and the last traceback keeps the fixtures alive past the
    module; the chkfile is then finalized by the last garbage collection,
    which warns (ResourceWarning) unless it was closed here."""
    chkfile = getattr(mf, '_chkfile', None)
    if chkfile is not None:
        chkfile.close()


@pytest.fixture(scope='module')
def lows():
    """The NO RPA runs of water: RHF, and PBE0."""
    out = {'hf': run_low_level(water('hf'), method='rpa', mode='casida',
                               natural_orbitals=True),
           'pbe0': run_low_level(water('pbe0'), method='rpa', mode='casida',
                                 natural_orbitals=True)}
    yield out
    for low in out.values():
        _close_chkfile(low.mf)


@pytest.fixture(scope='module')
def records(lows):
    """{(xc, rotate_occupied): the water NO CAS(4,4) with D_SA of STATES}:
    HF with canonical occupied orbitals (particle side only), HF and PBE0
    with both blocks rotated (both sides)."""
    out = {}
    for xc, rotate in (('hf', False), ('hf', True), ('pbe0', True)):
        low = lows[xc]
        h = build_effective_hamiltonian(
            low, no_space(low, 2, 2, rotate_occupied=rotate),
            screening='crpa')
        out[xc, rotate] = h.prepare_delta(STATES)
    return out


def _totals(h, method):
    """{(spin, k): total energy} of the requests of STATES by `method`, and
    the flags raised."""
    energies, names = {}, set()
    for n, spin in STATES:
        e, flags, _ = _record(lambda: h.solve('pyscf_fci', delta_method=method,
                                              nroots=n, spin=spin))
        names.update(flags)
        energies.update({(spin, k): float(x) for k, x in enumerate(e)})
    return energies, sorted(names)


def _against_lowdin(h, label):
    """(max |difference| of the excitation energies, of the totals) of
    'sa_remainder' against 'lowdin' over the states of STATES, the
    excitation energies taken from each method's own singlet ground
    state."""
    got, got_flags = _totals(h, 'sa_remainder')
    want, want_flags = _totals(h, 'lowdin')
    print(f'{label}: flags sa_remainder {got_flags}, lowdin {want_flags}')
    worst = 0.0
    for key in sorted(got):
        exc_got = got[key] - got[0, 0]
        exc_want = want[key] - want[0, 0]
        worst = max(worst, abs(exc_got - exc_want))
        print(f'  spin {key[0]} root {key[1]}: total {got[key]:.8f} vs '
              f'{want[key]:.8f} ({got[key] - want[key]:+.2e}), excitation '
              f'{exc_got:.6f} vs {exc_want:.6f} '
              f'({exc_got - exc_want:+.2e})')
    return worst, abs(got[0, 0] - want[0, 0])


def _report(h):
    """Z and lanczos_change of each state of STATES by 'sa_remainder'."""
    for n, spin in STATES:
        out = {}
        _record(lambda: h.solve('pyscf_fci', delta_method='sa_remainder',
                                nroots=n, spin=spin, out=out))
        print(f'  spin {spin}: Z {np.round(out["z"], 4).tolist()}, '
              f'correction {np.round(out["correction"], 6).tolist()}, '
              f'lanczos_change '
              f'{[f"{x:.1e}" for x in out["lanczos_change"]]}')


#: why the comparisons with exact Loewdin on water fail the bar, for Erik
#: to decide (Task 11, ruling 5: the bar is not loosened)
WATER_MISSES = (
    "sa_remainder misses the spec's bar of 3.7e-4 Ha on the water NO "
    "CAS(4,4), HF: the diagonal of D_SA of ((3, 0), (2, 2)) on the active "
    "virtuals is -0.51 and -0.86 Ha (particle side; -0.43 and -0.64 with "
    "both sides), because the excited singlets' removal energies from "
    "their singly occupied active virtual, +0.33 and +0.36 Ha, lie 0.24 to "
    "0.27 Ha below the lowest environment level, 0.60 Ha; the states of "
    "h + D_SA are then not those of Loewdin. See the measured numbers in "
    "the docstring")


# ------------------------------------------------------ against Loewdin
@pytest.mark.xfail(strict=True, reason=WATER_MISSES)
def test_remainder_against_lowdin(records):
    """The excitation energies of the states of STATES, canonical
    occupied orbitals (the particle side), against exact Loewdin: within
    the spec's bar, 0.01 eV.

    Measured (2026-11-27), sa_remainder - lowdin, excitation energies from
    each method's singlet ground state: singlets 1, 2 +1.12e-2, +8.14e-1
    Ha; triplets 0, 1 +8.69e-3, +1.85e-2 Ha; the ground-state total
    +4.92e-2 Ha. Z (sa_remainder) 0.995, 0.834, 0.880 (singlets), 0.847,
    0.847 (triplets); lanczos_change 1.5e-5, 7.4e-5, 1.4e-3, 9.5e-5,
    1.7e-5. The third singlet of h + D_SA is another state than Loewdin's
    (active virtual 3 nearly doubly occupied, occupation 1.93), so its
    remainder is 1.25 Ha. With D_SA, the ground state of h + D_SA has an
    active virtual occupation of 0.081 where that of h has 0.002. On N2,
    coupled weakly, the method meets the bar
    (`test_remainder_against_lowdin_on_n2`)."""
    h = records['hf', False]
    worst, _ = _against_lowdin(h, 'HF, particle side')
    _report(h)
    print(f'max |sa_remainder - lowdin| on excitation energies: {worst:.2e}')
    assert worst < BAR


@pytest.mark.xfail(strict=True, reason=WATER_MISSES)
@pytest.mark.parametrize('xc', ['hf', 'pbe0'])
def test_remainder_with_the_hole_side(records, xc):
    """Both blocks rotated, so both sides are folded: the excitation
    energies against exact Loewdin within the spec's bar.

    Measured (2026-11-27), sa_remainder - lowdin on the excitation
    energies, singlets 1, 2 and triplets 0, 1:
    HF +7.68e-3, +5.12e-2, +6.29e-3, +2.78e-2 Ha (Z 0.995, 0.752, 0.782,
    0.818, 0.818; lanczos_change 2.9e-5, 1.0e-3, 2.1e-3, 2.2e-4, 6.2e-4);
    PBE0 +2.14e-3, +1.56e-2, +3.90e-3, +1.10e-2 Ha (Z 0.998, 0.738,
    0.782, 0.812, 0.824; lanczos_change 8.7e-6, 4.2e-4, 9.0e-4, 1.4e-4,
    2.3e-4)."""
    h = records[xc, True]
    assert [kind for kind, _, _, _ in h.hybridization.sides()] == [
        'particle', 'hole']
    worst, _ = _against_lowdin(h, f'{xc}, both sides')
    _report(h)
    print(f'max |sa_remainder - lowdin| on excitation energies: {worst:.2e}')
    assert worst < BAR


@pytest.mark.xfail(strict=True, reason=WATER_MISSES)
@pytest.mark.parametrize('xc', ['hf', 'pbe0'])
def test_total_energy_with_the_hole_side(records, xc):
    """The total of the ground state with both sides against exact
    Loewdin: the excitation energies do not see the constant c of D_SA,
    the total does (it enters through the static energies E_K and is
    taken out again by the remainder).

    Measured (2026-11-27), sa_remainder - lowdin on the ground-state
    total: HF +2.87e-2 Ha (c = -0.125153, Z 0.995, lanczos_change
    2.9e-5); PBE0 +4.47e-3 Ha (c = -0.089374, Z 0.998, lanczos_change
    8.7e-6). That the constant is counted once is checked to 1e-9 by
    `test_lanczos_equals_dense_spectrum` (E_K + r_K against
    <K|H_N + Delta(E_K)|K>)."""
    h = records[xc, True]
    c = h.hybridization.d_sa[1]
    assert abs(c) > 1e-4
    got, _, _ = _record(lambda: h.solve('pyscf_fci',
                                        delta_method='sa_remainder', spin=0))
    want, _, _ = _record(lambda: h.solve('pyscf_fci', delta_method='lowdin',
                                         spin=0))
    print(f'{xc}: c = {c:.6f}, total {got[0]:.8f} vs Loewdin {want[0]:.8f}: '
          f'{got[0] - want[0]:+.2e}')
    assert abs(got[0] - want[0]) < BAR


# ------------------------------------------------------- the Lanczos part
def _dense_remainder(h1, eri, sides, d_sa, engine, E, K):
    """(dE, Z) of one state from the dense amplitudes of Task 8
    (`EnergyDependentSolver.sector`), both sides and both spins."""
    dense = LowdinSolver(engine, sides)
    g = dg = 0.0
    for plan in dense.planned_sectors(4, (2, 2)):
        kind, e_m, amp, lam, b = dense.sector(h1, eri, 4, (2, 2), *plan)
        sign = 1.0 if kind == 'particle' else -1.0
        w = np.zeros((len(e_m), len(lam)))
        for u in np.flatnonzero(np.any(b != 0.0, axis=1)):
            w += np.outer(amp(u) @ K.ravel(), b[u])
        den = (E - e_m)[:, None] - sign * lam[None, :]
        g += float(np.sum(w ** 2 / den))
        dg -= float(np.sum(w ** 2 / den ** 2))
    D, c = d_sa
    dm1 = engine.rdms(K, 4, (2, 2), spin=None)[0]
    r = g - (float(np.sum(D * dm1)) + c)
    return r / (1.0 - dg), 1.0 / (1.0 - dg), r, g


@pytest.mark.parametrize('rotate', [False, True])
def test_lanczos_equals_dense_spectrum(records, rotate):
    """At the full depth the remainder, its Z and the energy equal those
    from the dense amplitudes, 1e-9; the Krylov space is exhausted, so the
    change from one block less is exactly zero. And E_K + r_K is
    <K|H_N|K> + <K|Delta(E_K)|K>, H_N the dense sector matrix of h without
    D: the stand-in tr(D dm1) + c takes out exactly what E_K holds of
    D_SA, constant included."""
    h = records['hf', rotate]
    h1, eri = _integrals(h)
    sides, d_sa = h.hybridization.sides(), h.hybridization.d_sa
    D, c = d_sa
    engine = PyscfFCISolver()
    E, V = engine.roots(h1 + D, eri, 4, (2, 2), nroots=3, spin=None)
    res, _, _ = _record(lambda: RemainderSolver(
        engine, sides, FULL, d_sa=d_sa).refine(h1, eri, 4, (2, 2), E + c, V))
    assert np.array_equal(res.static, E + c)
    h_n = engine.sector_matrix(h1, eri, 4, (2, 2))
    worst = first_order = 0.0
    for k in range(3):
        de, z, r, g = _dense_remainder(h1, eri, sides, d_sa, engine,
                                       E[k] + c, V[k])
        worst = max(worst, abs(res.energies[k] - (E[k] + c + de)),
                    abs(res.z[k] - z))
        K = V[k].ravel()
        first_order = max(first_order,
                          abs(E[k] + c + r - (K @ h_n @ K + g)))
        print(f'state {k}: dE {res.energies[k] - res.static[k]:+.8f} vs '
              f'dense {de:+.8f}, Z {res.z[k]:.6f} vs {z:.6f}')
    print(f'full-depth Lanczos against the dense amplitudes: {worst:.1e}; '
          f'E_K + r_K against <K|H_N + Delta(E_K)|K>: {first_order:.1e}')
    assert worst < 1e-9 and first_order < 1e-9
    assert np.array_equal(res.extra['lanczos_change'], np.zeros(3))


def test_lanczos_change_is_one_block_less(records):
    """extra['lanczos_change'] is |dE(depth) - dE(depth - 1)| from one
    Lanczos run: it equals the difference of two runs at depth 3 and 2,
    1e-12."""
    h = records['hf', True]
    h1, eri = _integrals(h)
    sides, d_sa = h.hybridization.sides(), h.hybridization.d_sa
    D, c = d_sa
    engine = PyscfFCISolver()
    E, V = engine.roots(h1 + D, eri, 4, (2, 2), nroots=3, spin=None)

    def at(depth):
        numerics = Numerics(downfolding={'delta_lanczos_depth': depth})
        return _record(lambda: RemainderSolver(
            engine, sides, numerics, d_sa=d_sa).refine(
            h1, eri, 4, (2, 2), E + c, V))[0]
    three, two = at(3), at(2)
    by_hand = np.abs(three.energies - two.energies)
    print(f'lanczos_change {three.extra["lanczos_change"]}, two runs '
          f'{by_hand}')
    assert np.abs(three.extra['lanczos_change'] - by_hand).max() < 1e-12


@pytest.mark.xfail(strict=True, reason=(
    'lanczos_change at the default depth 3 is above 1e-5 on water and on '
    'N2; see the measured numbers in the docstring'))
def test_depth_convergence_is_reported(records):
    """out['lanczos_change'] below 1e-5 at the default depth (3), water
    NO CAS(4,4) with both sides, the three lowest singlets.

    Measured (2026-11-27): 2.9e-5, 1.0e-3, 2.1e-3 Ha. With depth the
    corrections converge (singlets of the same case, change from one
    block less: depth 4 3.5e-6, 2.7e-5, 6.9e-5; depth 5 3.1e-7, 5.9e-7,
    2.9e-6). On N2 (`test_remainder_against_lowdin_on_n2`, weak coupling)
    the change at depth 3 is up to 5.6e-5 Ha, below the bar of 3.7e-4 Ha
    but above 1e-5."""
    h = records['hf', True]
    out = {}
    _record(lambda: h.solve('pyscf_fci', delta_method='sa_remainder',
                            nroots=3, spin=0, out=out))
    print(f"default depth: lanczos_change {out['lanczos_change']}")
    assert out['lanczos_change'].shape == (3,)
    assert np.all(out['lanczos_change'] < 1e-5)


def _moved_pole(records, distance):
    """(result, flag names, flagged states, warning classes) of the
    particle-side water states of h + D_SA with the level e of the Lanczos
    pole (m, e) the ground state weighs most (alpha removal, default depth)
    moved to `distance` Ha below the ground state's static energy E_K."""
    h = records['hf', False]
    h1, eri = _integrals(h)
    (kind, sigma, lam, b), = h.hybridization.sides()
    D, c = h.hybridization.d_sa
    engine = PyscfFCISolver()
    E, V = engine.roots(h1 + D, eri, 4, (2, 2), nroots=3, spin=None)
    rows = np.flatnonzero(np.any(b != 0.0, axis=1))
    poles, weights = removal_spectrum(engine, V[0], h1, eri, 4, (2, 2),
                                      rows, 0, None)
    strength = (b[rows].T @ weights) ** 2                # (e, m)
    e, m = np.unravel_index(np.argmax(strength), strength.shape)
    moved = lam.copy()
    moved[e] = (E[0] + c) - poles[m] - distance
    assert abs((E[0] + c) - poles[m] - moved[e] - distance) < 1e-12
    if distance == 0.0:
        assert (E[0] + c) - poles[m] - moved[e] == 0.0
    flags = []
    with collecting(flags), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        res = RemainderSolver(engine, [(kind, sigma, moved, b)],
                              d_sa=(D, c)).refine(h1, eri, 4, (2, 2),
                                                  E + c, V)
    names = sorted({f['flag'] for f in flags})
    states = sorted({int(f['message'].split('state ')[1].split()[0])
                     for f in flags if f['flag'] == 'delta_near_pole'})
    print(f'pole moved to {distance} Ha: weight {strength[e, m]:.1e}, Z '
          f'{np.round(res.z, 3)}, smallest denominator '
          f'{np.round(res.min_denominator, 4)}, flags {names}, near-pole '
          f'states {states}')
    return res, names, states, {w.category for w in caught}


@pytest.mark.parametrize('distance', [0.005, 0.0])
def test_near_pole_is_flagged_not_nan(records, distance):
    """A Lanczos pole moved to 0.005 Ha below the ground state's static
    energy, and exactly onto it (where r and r' are evaluated): a finite
    energy, Z and smallest denominator, and 'delta_near_pole' on the
    ground state, never inf or NaN."""
    res, names, states, caught = _moved_pole(records, distance)
    assert 'delta_near_pole' in names and 0 in states
    assert set(names) <= {'delta_near_pole', 'delta_satellite'}
    assert caught == {DeltaNearPoleWarning}
    assert res.extra['near_pole'][0]
    for x in (res.energies, res.z, res.min_denominator,
              res.extra['lanczos_change']):
        assert np.all(np.isfinite(x))


class _NoRdms(PyscfFCISolver):
    """The FCI engine without density matrices: the remainder takes dm1
    from the sector operations."""
    has_rdms = False

    def rdms(self, *args, **kwargs):
        raise AssertionError('rdms called on an engine without them')


def test_density_from_the_sector_operations(records):
    """dm1 of each state from <K|E_tu|K> by the sector operations, on an
    engine without density matrices, gives the energies of the engine's
    rdms, 1e-12."""
    h = records['hf', True]
    h1, eri = _integrals(h)
    sides, d_sa = h.hybridization.sides(), h.hybridization.d_sa
    D, c = d_sa
    E, V = PyscfFCISolver().roots(h1 + D, eri, 4, (2, 2), nroots=3,
                                  spin=None)
    res = [_record(lambda e=e: RemainderSolver(e, sides, d_sa=d_sa).refine(
        h1, eri, 4, (2, 2), E + c, V))[0] for e in (PyscfFCISolver(),
                                                   _NoRdms())]
    diff = np.abs(res[0].energies - res[1].energies).max()
    print(f'dm1 from the sector operations against rdms: {diff:.1e}')
    assert diff < 1e-12


# --------------------------------------------------------- through solve
def test_out_and_the_crossing_check(records):
    """The record's solve: out['static'] holds E_K, the energies of
    h + D_SA plus c (those of 'static_sa' without e0), out['e0'] is e0,
    the totals are out['e0'] + out['e_active']; no count of solutions is
    made, so every root is unchecked and 'solutions_below' is absent; a
    reused out keeps no key of an earlier Delta solve."""
    h = records['hf', True]
    out = {}
    e, flags, _ = _record(lambda: h.solve('pyscf_fci',
                                          delta_method='sa_remainder',
                                          nroots=3, spin=0, out=out))
    assert 'delta_roots_missing' not in flags
    static_sa, _, _ = _record(lambda: h.solve(
        'pyscf_fci', delta_method='static_sa', nroots=3, spin=0))
    assert out['e0'] == h.energies.e0
    assert np.array_equal(e, out['e0'] + out['e_active'])
    assert np.all(np.diff(e) >= 0)
    by_static = np.argsort(out['static_index'])
    diff = np.abs(out['static'][by_static] + h.energies.e0 - static_sa).max()
    print(f"out['static'] against static_sa: {diff:.1e}")
    assert diff < 1e-10
    assert np.array_equal(out['crossing_unchecked'], np.ones(3, bool))
    assert 'solutions_below' not in out and 'above_pole' not in out
    assert set(out) >= {'z', 'min_denominator', 'converged', 'near_pole',
                        'lanczos_change', 'correction'}
    assert np.all(out['converged'])
    _record(lambda: h.solve('pyscf_fci', nroots=3, spin=0, out=out))
    assert 'lanczos_change' not in out and 'solutions_below' in out
    _record(lambda: h.solve('pyscf_fci', delta_method='sa_remainder',
                            nroots=3, spin=0, out=out))
    assert 'solutions_below' not in out and 'linearization_error' not in out


# ------------------------------------------------------- degenerate states
@pytest.fixture(scope='module')
def n2():
    """N2 NO CAS(6,6) of Task 8 (particle side), without and with D_SA of
    its lowest ten states of any spin (a set that ends between two
    pairs)."""
    low = run_low_level(nitrogen(), method='rpa', mode='casida',
                        natural_orbitals=True)
    h = build_effective_hamiltonian(low, no_space(low, 3, 3),
                                    screening='crpa')
    yield h, h.prepare_delta(((10, None),))
    _close_chkfile(low.mf)


def test_remainder_against_lowdin_on_n2(n2):
    """N2, coupled weakly (Z above 0.9): the ten states D_SA is built from,
    totals and excitation energies against exact Loewdin within the spec's
    bar."""
    h, hs = n2
    got, _, _ = _record(lambda: hs.solve('pyscf_fci', nroots=10,
                                         delta_method='sa_remainder'))
    want, _, _ = _record(lambda: h.solve('pyscf_fci', nroots=10,
                                         delta_method='lowdin'))
    static, _, _ = _record(lambda: hs.solve('pyscf_fci', nroots=10,
                                            delta_method='static_sa'))
    total = np.abs(got - want).max()
    excitation = np.abs((got - got[0]) - (want - want[0])).max()
    print(f'N2: sa_remainder - lowdin {np.round(got - want, 6).tolist()}; '
          f'static_sa - lowdin {np.round(static - want, 6).tolist()}')
    print(f'max |sa_remainder - lowdin|: totals {total:.2e}, excitation '
          f'energies {excitation:.2e}')
    assert total < BAR and excitation < BAR


def test_degenerate_partners(n2):
    """The pairs of degenerate states among the lowest eight of h + D_SA
    of N2 CAS(6,6) at (3, 3): one energy after the remainder, 1e-8 Ha (a
    per-state scalar, each partner refined on its own), and two different
    states."""
    _, hs = n2
    h1, eri = _integrals(hs)
    sides, d_sa = hs.hybridization.sides(), hs.hybridization.d_sa
    D, c = d_sa
    engine = PyscfFCISolver()
    E, V = engine.roots(h1 + D, eri, 6, (3, 3), nroots=11, spin=None)
    print(f'static gaps {np.round(np.diff(E), 8).tolist()}')
    assert E[10] - E[9] > 1e-4                 # D_SA's set holds whole pairs
    E, V = E[:8], V[:8]
    res, flags, _ = _record(lambda: RemainderSolver(
        engine, sides, d_sa=d_sa).refine(h1, eri, 6, (3, 3), E + c, V))
    pairs = [(k, k + 1) for k in range(len(E) - 1) if E[k + 1] - E[k] < 1e-8]
    print(f'pairs {pairs}, flags {flags}')
    assert len(pairs) >= 2
    for k, l in pairs:
        de = abs(res.energies[k] - res.energies[l])
        ov = abs(np.vdot(res.vectors[k], res.vectors[l]))
        print(f'  {k},{l}: E {res.energies[k]:.10f} {res.energies[l]:.10f} '
              f'|dE| {de:.1e}, |<c1|c2>| {ov:.1e}, '
              f'correction {res.energies[k] - res.static[k]:.6f}, '
              f'lanczos_change {res.extra["lanczos_change"][k]:.1e}')
        assert de < 1e-8 and ov < 1e-6


# ---------------------------------------------------------------- refusals
def test_needs_prepare_delta(records, lows, monkeypatch):
    """Without D_SA the record refuses 'sa_remainder' naming
    prepare_delta, as do the solver function without d_sa and the backend
    without it; d_sa with another method is refused; a solver without
    sector operations is refused before any static solve."""
    assert DYNAMIC_METHODS['sa_remainder'] is RemainderSolver
    low = lows['hf']
    bare = build_effective_hamiltonian(low, no_space(low, 2, 2),
                                       screening='crpa')
    with pytest.raises(ValueError, match='prepare_delta'):
        bare.solve('pyscf_fci', delta_method='sa_remainder')
    h = records['hf', False]
    sides, d_sa = h.hybridization.sides(), h.hybridization.d_sa
    args = (h.g_eff, 4, 4)
    with pytest.raises(ValueError, match='prepare_delta'):
        solve('pyscf_fci', *args, spin_free=True, sides=sides,
              delta_method='sa_remainder')
    for method in ('lowdin', 'linearized'):
        with pytest.raises(ValueError, match='d_sa'):
            solve('pyscf_fci', *args, spin_free=True, sides=sides,
                  delta_method=method, d_sa=d_sa)
    with pytest.raises(ValueError, match='prepare_delta'):
        RemainderSolver(PyscfFCISolver(), sides)
    with pytest.raises(ValueError, match='d_sa'):
        RemainderSolver(PyscfFCISolver(), sides, d_sa=(np.zeros((3, 3)), 0.0)
                        ).refine(*_integrals(h), 4, (2, 2), [0.0],
                                 [np.zeros((6, 6))])

    def no_solve(*args, **kwargs):
        raise AssertionError('a static solve ran before the refusal')
    monkeypatch.setattr(solve_module, 'static_states', no_solve)
    for solver in ('ccsdt', 'openfermion', 'dmrg'):
        with pytest.raises(ValueError, match='sector operations'):
            h.solve(solver, delta_method='sa_remainder')
