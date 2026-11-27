"""The natural-orbital route on references that are not HF (PBE, PBE0).
With the coupling of the rotation only (spec decision 7, Delta folds
F_s = T^T diag(eps) T) such a reference has the structure of the HF case: no
occupied-virtual block, no singles term, the E_0 term as derived. The
exchange part of the coupling, <t|Sigma_x - v_xc|e>, is left out and only
its norms are reported (`diagnostics['fock_coupling']['left_out']`). No
method is added here; these are tests of what Tasks 6 to 13 do for any
reference. Water/6-31G, the mean field density fitted (the exact-integral
HF case of the left-out norms is named where it is used)."""
import inspect
import warnings

import numpy as np
import pytest

from src.Base.numerical_flags import collecting
from src.EffectiveHamiltonian.active_space import ActiveSpace
from src.EffectiveHamiltonian.Downfolding import (build_effective_hamiltonian,
                                                  run_low_level)
from src.EffectiveHamiltonian.solvers.backends import PyscfFCISolver
from src.EffectiveHamiltonian.solvers.dynamic_backends import (
    DYNAMIC_METHODS, DeltaLinearizationWarning)
from src.EffectiveHamiltonian.solvers.solve import SolveRequest
from tests.no_downfolding import no_space, pq_spectrum, water

LEFT_OUT = ('av_ev', 'ao_i', 'ao_ev', 'i_av', 'i_ev')


def _integrals(h):
    """(h1, eri) of a build, as pyscf's FCI takes them."""
    h1 = 0.5 * (h.g_eff[1] + h.g_eff[1].T)
    eri = h.g_eff[2].transpose(0, 2, 1, 3)
    eri = 0.5 * (eri + eri.transpose(1, 0, 2, 3))
    eri = 0.5 * (eri + eri.transpose(0, 1, 3, 2))
    eri = 0.5 * (eri + eri.transpose(2, 3, 0, 1))
    return h1, eri


def _record(call):
    """(result, flags [(name, message)], warning classes) of call()."""
    flags = []
    with collecting(flags), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = call()
    return (result, [(f['flag'], f['message']) for f in flags],
            [w.category for w in caught])


def _state(text):
    return int(text.split('state ')[1].split()[0])


@pytest.fixture(scope='module')
def lows():
    return {xc: run_low_level(water(xc, df=True), method='rpa',
                              mode='casida', natural_orbitals=True)
            for xc in ('pbe', 'pbe0')}


@pytest.fixture(scope='module')
def hs(lows):
    """The water NO CAS(4,4) with both blocks rotated, per functional."""
    return {xc: build_effective_hamiltonian(
        low, no_space(low, 2, 2, rotate_occupied=True), screening='crpa')
        for xc, low in lows.items()}


def _lowdin_and_linearized(h):
    h1, eri = _integrals(h)
    engine, sides = PyscfFCISolver(), h.hybridization.sides()
    E, V = engine.roots(h1, eri, 4, (2, 2), nroots=3, spin=None)
    res = {m: _record(lambda m=m: DYNAMIC_METHODS[m](engine, sides).refine(
        h1, eri, 4, (2, 2), E, V)) for m in ('lowdin', 'linearized')}
    return h1, eri, sides, res


@pytest.mark.parametrize('xc', ['pbe', 'pbe0'])
def test_lowdin_is_the_pq_spectrum(xc, hs):
    h = hs[xc]
    h1, eri, sides, res = _lowdin_and_linearized(h)
    assert [kind for kind, _, _, _ in sides] == ['particle', 'hole']
    assert np.abs(h.hybridization.b_hole[:2]).max() > 1e-3
    low, flags, caught = res['lowdin']
    e, w = pq_spectrum(h1, eri, 4, (2, 2), sides)
    worst = 0.0
    for x in low.energies:
        k = np.argmin(abs(e - x))
        worst = max(worst, abs(e[k] - x))
        assert abs(e[k] - x) < 1e-8 and w[k] > 0.5
    print(f'{xc}: Loewdin with both sides against P + Q: {worst:.1e}')
    assert np.all(low.converged)
    assert flags == [] and caught == []


def test_linearized_follows_lowdin_on_pbe0(hs):
    """The bar of 1e-6 Ha holds for the ground state, which raises no
    'delta_linearization'. The two excited states of water, coupled
    strongly (Z about 0.75-0.79), miss it, as in Tasks 8 and 13: the
    property of the linearized method, not of the reference. Every state
    above the bar has an estimate Z |<c|H_eff(E)|c> - E| within 10% of
    |linearized - lowdin|, and is flagged 'delta_linearization' exactly
    when that estimate exceeds numerics.downfolding.delta_linearization_warn
    (1e-3 Ha): here both excited states lie between the bar and that
    threshold, so neither is flagged. Measured on 2026-11-27
    (|linearized - lowdin|, estimate, Z): state 0 1.17e-11, 1.01e-11,
    0.999; state 1 6.25e-04, 6.25e-04, 0.750; state 2 1.36e-04, 1.36e-04,
    0.788."""
    h = hs['pbe0']
    _, _, _, res = _lowdin_and_linearized(h)
    lin, flags, caught = res['linearized']
    low = res['lowdin'][0]
    gap = np.abs(lin.energies - low.energies)
    estimate = lin.extra['linearization_error']
    for k in range(len(gap)):
        print(f'pbe0 state {k}: |linearized - lowdin| {gap[k]:.2e}, '
              f'estimate {estimate[k]:.2e}, Z {lin.z[k]:.3f}')
    bar = 1e-6
    assert gap[0] < bar
    off = [k for k in range(len(gap)) if gap[k] > bar]
    assert off == [1, 2]
    ratio = estimate[off] / gap[off]
    assert np.all(ratio > 0.9) and np.all(ratio < 1.1)
    warn = h.numerics.downfolding.delta_linearization_warn
    expected = [k for k in range(len(gap)) if estimate[k] > warn]
    flagged = sorted(_state(t) for n, t in flags if n == 'delta_linearization')
    assert flagged == expected
    assert {n for n, _ in flags} == ({'delta_linearization'} if expected
                                     else set())
    assert set(caught) == ({DeltaLinearizationWarning} if expected else set())


def test_canonical_on_pbe_is_untouched(lows):
    """basis=None on PBE: no coupling, g_eff and the solve equal for
    delta=True and delta=False, exactly."""
    low = lows['pbe']
    nocc, norb = low.nocc, len(low.eps)
    space = ActiveSpace.from_indices(norb, nocc, [nocc - 2, nocc - 1],
                                     [nocc, nocc + 1])
    on = build_effective_hamiltonian(low, space, screening='crpa')
    off = build_effective_hamiltonian(low, space, screening='crpa',
                                      delta=False)
    for h in (on, off):
        assert h.hybridization is None
        assert 'fock_coupling' not in h.diagnostics
        assert 'delta_off' not in [f['flag'] for f in h.numerical_flags]
    assert on.g_eff[0] == off.g_eff[0]
    for k in (1, 2):
        assert np.array_equal(on.g_eff[k], off.g_eff[k])
    assert on.energies.e_c_hybridization == 0.0
    e_on, _, c1 = _record(lambda: on.solve('pyscf_fci', nroots=3))
    e_off, _, c2 = _record(lambda: off.solve('pyscf_fci', nroots=3))
    assert np.array_equal(e_on, e_off)
    assert c1 == c2 == []


def test_e_c_terms_cancel_with_every_orbital_active(lows):
    low = lows['pbe']
    nocc, norb = low.nocc, len(low.eps)
    h = build_effective_hamiltonian(
        low, no_space(low, nocc, norb - nocc, rotate_occupied=True))
    en = h.energies
    rest = en.e_c_full - en.e_c_active - en.e_c_hybridization
    print(f'PBE all active: e_c_full {en.e_c_full:.10f}, e_c_hybridization '
          f'{en.e_c_hybridization:.6e}, e_c_full - e_c_active - '
          f'e_c_hybridization {rest:.1e}')
    assert h.hybridization is not None
    assert abs(rest) < 1e-10


def test_left_out_norms_are_reported(lows, hs):
    """On PBE the five norms are of the size of Sigma_x - v_xc (> 1e-3); on
    HF with exact integrals they are below 1e-7 (Task 6's bar), so the test
    tells the two cases apart."""
    left = hs['pbe'].diagnostics['fock_coupling']['left_out']
    print('PBE left_out:', {k: f'{v:.3e}' for k, v in left.items()})
    assert set(left) == set(LEFT_OUT)
    assert all(v > 1e-3 for v in left.values())
    mf = water('hf', df=False)
    low = run_low_level(mf, method='rpa', mode='casida',
                        natural_orbitals=True)
    h = build_effective_hamiltonian(
        low, no_space(low, 2, 2, rotate_occupied=True), screening='crpa')
    exact = h.diagnostics['fock_coupling']['left_out']
    print('HF exact left_out:', {k: f'{v:.1e}' for k, v in exact.items()})
    assert set(exact) == set(LEFT_OUT)
    assert max(exact.values()) < 1e-7


def test_no_public_block_or_coupling_option():
    params = inspect.signature(build_effective_hamiltonian).parameters
    assert 'blocks' not in params
    assert '_blocks' in params   # the private test hook
    assert not [p for p in params
                if p != '_blocks' and any(w in p for w in
                                          ('coupling', 'operator', 'fock'))]
