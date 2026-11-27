"""The remainder method 'sa_remainder' (`RemainderSolver` of
`solvers/dynamic_backends.py`): the static D_SA in h, and one scalar per
state, the expectation value of what D_SA leaves out,

    r_K(E) = <K|Delta(E)|K> - (tr(D dm1_K) + c),
    dE_K   = r_K(E_K) / (1 - r'_K(E_K)),

with |K>, E_K the states of h + D and their energies plus c, and the
(N-1)- and (N+1)-electron spectra of h by block Lanczos.

The weakly coupled cases meet the spec's bar of 0.01 eV against exact
Loewdin, on totals and excitation energies, without the flag
'delta_remainder_large': benzene/cc-pVDZ pi NO CAS(6,6) with virtual NOs,
N2/6-31G NO CAS(6,6) of Task 8 with the particle side and with both sides,
and the ground-state total of N2 with both sides, which sees the constant
c of D_SA. Water/6-31G NO CAS(4,4) of Task 8, D_SA of ((3, 0), (2, 2)), is
coupled strongly and misses the bar (HF with the particle side, HF and PBE0
with both sides): 'delta_remainder_large' names exactly the states whose
correction exceeds delta_remainder_warn, every state that misses among
them. At the full Lanczos depth the remainder equals the one built from
the dense amplitudes of the dense methods, and E_K + r_K equals
<K|H_N + Delta(E_K)|K>; lanczos_change is the difference from one block
less of the same run and below 1e-5 at the default depth on water; a
Lanczos pole moved next to and onto a state gives a finite energy and
'delta_near_pole'; a zero coupling and a space without active virtual
orbitals give the static energies; every root is marked unchecked for
missing states; the density matrix taken from the sector operations equals
the engine's; the two partners of a degenerate state of N2 come out with
one energy. A record without D_SA, a solve without d_sa, and a solver
without sector operations are refused before anything is solved."""
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
    DYNAMIC_METHODS, DeltaNearPoleWarning, DeltaRemainderWarning,
    LowdinSolver, RemainderSolver, removal_spectrum)
from src.EffectiveHamiltonian.solvers.solve import solve
from tests.no_downfolding import (benzene, benzene_pi_space, nitrogen,
                                  no_space, water)

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
    """Close pyscf's temporary chkfile of a mean field at the teardown of
    a fixture. A traceback that outlives the module (a failing test) keeps
    the fixtures alive; the chkfile is then finalized by the last garbage
    collection, which warns (ResourceWarning) unless it was closed here."""
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


#: the states of D_SA on N2 (a set that ends between two pairs) and on
#: benzene, of any spin
N2_STATES = ((10, None),)
BENZENE_STATES = ((7, None),)


@pytest.fixture(scope='module')
def weak():
    """{(molecule, rotate_occupied): (record, record with D_SA)}, the
    weakly coupled cases: N2/6-31G NO CAS(6,6) (Task 8) with canonical
    occupied orbitals (particle side) and with both blocks rotated (both
    sides; its occupied cut runs with NearDegenerateCutWarning), D_SA of
    N2_STATES; benzene/cc-pVDZ pi NO CAS(6,6), virtual NOs, D_SA of
    BENZENE_STATES."""
    out, lows = {}, []
    low = run_low_level(nitrogen(), method='rpa', mode='casida',
                        natural_orbitals=True)
    lows.append(low)
    for rotate in (False, True):
        h = _record(lambda: build_effective_hamiltonian(
            low, no_space(low, 3, 3, rotate_occupied=rotate),
            screening='crpa'))[0]
        out['n2', rotate] = h, h.prepare_delta(N2_STATES)
    low = run_low_level(benzene(), method='rpa', mode='casida',
                        natural_orbitals=True)
    lows.append(low)
    h = build_effective_hamiltonian(low, benzene_pi_space(low),
                                    screening='crpa')
    out['benzene', False] = h, h.prepare_delta(BENZENE_STATES)
    yield out
    for low in lows:
        _close_chkfile(low.mf)


def _compare(h, hs, requests, label, messages=None):
    """'sa_remainder' (record hs, with D_SA) against 'lowdin' (record h)
    over the requests: (rows, flagged, names), rows {(spin, k): (total
    sa_remainder, total lowdin, |dE_K|)} and flagged the (spin, k) that
    'delta_remainder_large' names, per request through
    out['static_index'], and names those of every flag 'sa_remainder'
    raised. messages, a list, receives (message, |dE| of the root it
    names) for every 'delta_remainder_large'. Printed per state: |dE|, the
    total and the excitation difference (from the first request's lowest
    state)."""
    rows, flagged, names = {}, set(), set()
    for n, spin in requests:
        out, flags = {}, []
        with collecting(flags), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            got = hs.solve('pyscf_fci', delta_method='sa_remainder',
                           nroots=n, spin=spin, out=out)
        want, _, _ = _record(lambda: h.solve('pyscf_fci',
                                             delta_method='lowdin',
                                             nroots=n, spin=spin))
        names.update(f['flag'] for f in flags)
        returned = {int(k): i for i, k in enumerate(out['static_index'])}
        flagged.update((spin, returned[int(
            f['message'].split('static state ')[1].split()[0])])
            for f in flags if f['flag'] == 'delta_remainder_large')
        if messages is not None:
            messages.extend(
                (f['message'], abs(float(out['correction'][returned[int(
                    f['message'].split('static state ')[1].split()[0])]])))
                for f in flags if f['flag'] == 'delta_remainder_large')
        for k in range(len(got)):
            rows[spin, k] = (float(got[k]), float(want[k]),
                             abs(float(out['correction'][k])))
    first = (requests[0][1], 0)
    print(f'{label}:')
    for key, (got, want, de) in rows.items():
        exc = (got - rows[first][0]) - (want - rows[first][1])
        print(f'  spin {key[0]} root {key[1]}: |dE| {de:.2e}, total '
              f'{got - want:+.2e}, excitation {exc:+.2e}'
              f'{"  flagged" if key in flagged else ""}')
    print(f'  flags {sorted(names)}')
    return rows, flagged, names


def _worst(rows):
    """(max |difference| of the totals, of the excitation energies)."""
    first = next(iter(rows))
    totals = max(abs(got - want) for got, want, _ in rows.values())
    excitation = max(abs((got - rows[first][0]) - (want - rows[first][1]))
                     for got, want, _ in rows.values())
    return totals, excitation


# ------------------------------------------------------ against Loewdin
@pytest.mark.parametrize('molecule', ['benzene', 'n2'])
def test_remainder_against_lowdin(weak, molecule):
    """The particle side (canonical occupied orbitals on N2, virtual NOs on
    benzene): the totals and the excitation energies of the states D_SA is
    built from against exact Loewdin within the spec's bar, 0.01 eV, and no
    'delta_remainder_large'.

    Measured (2026-11-27), max |sa_remainder - lowdin|: benzene totals
    1.09e-4 Ha, excitation energies 8.5e-5 Ha, |dE_K| 1.7e-4 to 4.2e-3 Ha;
    N2 totals 1.73e-4 Ha, excitation energies 1.29e-4 Ha, |dE_K| 5.5e-5 to
    6.1e-3 Ha."""
    h, hs = weak[molecule, False]
    rows, flagged, _ = _compare(h, hs, BENZENE_STATES if molecule == 'benzene'
                             else N2_STATES, molecule)
    totals, excitation = _worst(rows)
    print(f'max |sa_remainder - lowdin|: totals {totals:.2e}, excitation '
          f'energies {excitation:.2e}')
    assert totals < BAR and excitation < BAR
    assert flagged == set()


def test_remainder_with_the_hole_side(weak):
    """N2 with both blocks rotated, so both sides are folded: the
    excitation energies against exact Loewdin within the spec's bar, and
    no 'delta_remainder_large'. (Benzene's pi CAS(6,6) holds every pi
    occupied orbital, so its hole coupling to the inactive sigma orbitals
    vanishes by symmetry and adds no case.)

    Measured (2026-11-27): max |sa_remainder - lowdin| on the excitation
    energies 1.61e-4 Ha, on the totals 1.92e-4 Ha; |dE_K| 7.0e-4 to
    7.0e-3 Ha."""
    h, hs = weak['n2', True]
    assert [kind for kind, _, _, _ in h.hybridization.sides()] == [
        'particle', 'hole']
    rows, flagged, _ = _compare(h, hs, N2_STATES, 'n2, both sides')
    totals, excitation = _worst(rows)
    print(f'max |sa_remainder - lowdin|: totals {totals:.2e}, excitation '
          f'energies {excitation:.2e}')
    assert excitation < BAR and flagged == set()


def test_total_energy_with_the_hole_side(weak):
    """The total of the ground state of N2 with both sides against exact
    Loewdin: the excitation energies do not see the constant c of D_SA,
    the total does (it enters through the static energies E_K and is
    taken out again by the remainder). That it is counted once is checked
    to 1e-9 by `test_lanczos_equals_dense_spectrum`.

    Measured (2026-11-27): +1.92e-4 Ha."""
    h, hs = weak['n2', True]
    c = hs.hybridization.d_sa[1]
    assert abs(c) > 1e-4
    got, _, _ = _record(lambda: hs.solve('pyscf_fci',
                                         delta_method='sa_remainder'))
    want, _, _ = _record(lambda: h.solve('pyscf_fci', delta_method='lowdin'))
    print(f'N2: c = {c:.6f}, total {got[0]:.8f} vs Loewdin {want[0]:.8f}: '
          f'{got[0] - want[0]:+.2e}')
    assert abs(got[0] - want[0]) < BAR


@pytest.mark.parametrize('case', [('hf', False), ('hf', True),
                                  ('pbe0', True)])
def test_strong_coupling_is_flagged(records, case):
    """Water NO CAS(4,4), D_SA of STATES: coupled strongly, and
    sa_remainder misses the bar. 'delta_remainder_large' names exactly the
    states whose |dE_K| exceeds delta_remainder_warn, every state that
    misses the bar on its total is among them, and each message names the
    |dE| of its state and the threshold.

    Measured (2026-11-27), per case |dE_K| (singlets 0, 1, 2; triplets 0,
    1) and sa_remainder - lowdin on the totals, all flagged:
    HF particle side |dE| 6.1e-2, 5.3e-1, 1.25, 5.5e-1, 5.5e-1 Ha, totals
    +4.9e-2, +6.0e-2, +8.6e-1, +5.8e-2, +6.8e-2 Ha (the third singlet of
    h + D_SA is another state than Loewdin's); HF both sides |dE| 5.7e-2,
    2.9e-1, 3.1e-1, 3.7e-1, 3.2e-1 Ha, totals +2.9e-2, +3.6e-2, +8.0e-2,
    +3.5e-2, +5.7e-2 Ha; PBE0 both sides |dE| 1.5e-2, 1.1e-1, 1.1e-1,
    1.7e-1, 1.4e-1 Ha, totals +4.5e-3, +6.6e-3, +2.0e-2, +8.4e-3,
    +1.6e-2 Ha. The ground state, whose |dE| is smallest, is above the
    threshold in all three. Z of these states 0.74 to 1.0; nothing lies
    next to a pole ('delta_near_pole' does not fire)."""
    hs = records[case]
    threshold = hs.numerics.downfolding.delta_remainder_warn
    messages = []
    rows, flagged, names = _compare(hs, hs, STATES, f'water {case}',
                                    messages)
    large = {key for key, (_, _, de) in rows.items() if de > threshold}
    missed = {key for key, (got, want, _) in rows.items()
              if abs(got - want) > BAR}
    assert flagged == large
    assert missed and missed <= flagged
    assert 'delta_near_pole' not in names
    # each message names the |dE| of its state and the threshold
    assert len(messages) == len(flagged)
    for message, de in messages:
        assert f'|dE| = {de:.2e} Ha' in message
        assert f'delta_remainder_warn = {threshold} Ha' in message


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


def test_depth_convergence_is_reported(records):
    """out['lanczos_change'] below 1e-5 at the default depth (5), water
    NO CAS(4,4) with both sides, the three lowest singlets, the most
    strongly coupled case.

    Measured (2026-11-27): 3.1e-7, 5.9e-7, 2.9e-6 Ha at depth 5 (2.9e-5,
    1.0e-3, 2.1e-3 Ha at depth 3, the default before). The change at depth
    d follows the error at depth d - 1 against the full depth (depth 3:
    3.8e-6, 2.8e-5, 7.2e-5 Ha; depth 5: 7.5e-10, 3.4e-8, 1.2e-7 Ha)."""
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
    assert set(names) <= {'delta_near_pole', 'delta_satellite',
                          'delta_remainder_large'}
    assert DeltaNearPoleWarning in caught
    assert caught <= {DeltaNearPoleWarning, DeltaRemainderWarning}
    assert res.extra['near_pole'][0]
    for x in (res.energies, res.z, res.min_denominator,
              res.extra['lanczos_change']):
        assert np.all(np.isfinite(x))


def test_zero_coupling_and_no_active_virtual(records, lows):
    """Review Focus 1 for 'sa_remainder' (the dense methods have theirs in
    tests/test_downfolding_no_delta_dense.py): a coupling whose b is zero,
    with a zero D_SA, gives the static energies exactly, Z = 1 and no
    Lanczos change; an active space with no active virtual orbital has no
    side and a zero D_SA, and the record's solve gives the energies of
    'lowdin' (the static ones) exactly, with zero corrections."""
    h = records['hf', False]
    h1, eri = _integrals(h)
    zero = [(k, s, lam, np.zeros_like(b))
            for k, s, lam, b in h.hybridization.sides()]
    engine = PyscfFCISolver()
    E, V = engine.roots(h1, eri, 4, (2, 2), nroots=3, spin=None)
    res, names, _ = _record(lambda: RemainderSolver(
        engine, zero, d_sa=(np.zeros((4, 4)), 0.0)).refine(
        h1, eri, 4, (2, 2), E, V))
    assert np.array_equal(res.energies, E) and np.all(res.z == 1.0)
    assert np.array_equal(res.extra['lanczos_change'], np.zeros(3))
    assert names == []
    low = lows['hf']
    bare = build_effective_hamiltonian(low, no_space(low, 2, 0),
                                       screening='crpa')
    assert bare.hybridization.sides() == []
    prepared = bare.prepare_delta(((1, 0),))
    D, c = prepared.hybridization.d_sa
    assert np.array_equal(D, np.zeros((2, 2))) and c == 0.0
    out = {}
    got, names, _ = _record(lambda: prepared.solve(
        'pyscf_fci', delta_method='sa_remainder', out=out))
    want, _, _ = _record(lambda: bare.solve('pyscf_fci',
                                            delta_method='lowdin'))
    assert np.array_equal(got, want) and names == []
    assert np.array_equal(out['correction'], [0.0]) and out['z'][0] == 1.0


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


def test_degenerate_partners(weak):
    """The pairs of degenerate states among the lowest eight of h + D_SA
    of N2 CAS(6,6) at (3, 3): one energy after the remainder, 1e-8 Ha (a
    per-state scalar, each partner refined on its own), and two different
    states."""
    _, hs = weak['n2', False]
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
