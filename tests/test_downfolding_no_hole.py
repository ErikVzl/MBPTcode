"""The hole side of Delta: an active space whose occupied block is rotated
(`rotate_occupied=True`) couples its active occupied orbitals to the
inactive occupied ones, and Delta folds that block beside the particle one
(spec 6; `Coupling.sides`, `static_mf_delta`, `state_averaged_delta`,
`attachment_moment` in Downfolding/hybridization.py).

Water/6-31G RHF on an NO CAS(4,4) with both blocks rotated: the exact
Loewdin roots with both sides are eigenvalues of the explicit P + Q matrix
(`pq_spectrum`); linearized Loewdin against Loewdin; `attachment_moment`
against the (N+1)-electron sector; the hole part of D_SA against its
formula; the static hole term with its constant, -g tr(D_h), against
Loewdin as the levels move away (its error falls as 1/lam^2, without the
constant as 1/lam); canonical occupied orbitals fold no hole term and give
the energies of the particle side alone; no active occupied orbital, no
hole term; `e_c_hybridization` and Delta cover the same blocks. N2/6-31G
on an NO CAS(6,6) with both blocks rotated: the partners of a degenerate
state come out with one energy and as two different states."""
import warnings
from dataclasses import replace

import numpy as np
import pytest

from src.Base.numerical_flags import collecting
from src.EffectiveHamiltonian.active_space import (
    ActiveSpace, NearDegenerateCutWarning)
from src.EffectiveHamiltonian.Downfolding import (build_effective_hamiltonian,
                                                  run_low_level)
from src.EffectiveHamiltonian.Downfolding.hybridization import (
    COUPLING_BLOCKS, DEFAULT_BLOCKS, StaticDeltaWarning, attachment_moment,
    state_averaged_delta, static_mf_delta)
from src.EffectiveHamiltonian.solvers.backends import PyscfFCISolver
from src.EffectiveHamiltonian.solvers.dynamic_backends import (
    DYNAMIC_METHODS, DeltaLinearizationWarning, LowdinSolver)
from src.EffectiveHamiltonian.solvers.solve import SolveRequest, static_states
from tests.no_downfolding import nitrogen, no_space, pq_spectrum, water


def _integrals(h):
    """(h1, eri) of a build, h1 symmetric and eri (chemist) symmetric in
    each pair and under the pair swap, as pyscf's FCI assumes."""
    h1 = 0.5 * (h.g_eff[1] + h.g_eff[1].T)
    eri = h.g_eff[2].transpose(0, 2, 1, 3)
    eri = 0.5 * (eri + eri.transpose(1, 0, 2, 3))
    eri = 0.5 * (eri + eri.transpose(0, 1, 3, 2))
    eri = 0.5 * (eri + eri.transpose(2, 3, 0, 1))
    return h1, eri


def _record(call):
    """(result, flags, warning classes) of call(): the flags as [(name,
    message)], every warning recorded, so that the output stays clean and
    each test asserts what it expects."""
    flags = []
    with collecting(flags), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = call()
    return (result, [(f['flag'], f['message']) for f in flags],
            [w.category for w in caught])


def _request(h, nroots, spin=None):
    return SolveRequest(h.g_eff, h.space['n_orbs'], h.space['n_elec_active'],
                        spin_free=True, nroots=nroots, spin=spin)


def _state(text):
    """The state a flag message names ('... state k ...')."""
    return int(text.split('state ')[1].split()[0])


@pytest.fixture(scope='module')
def low():
    return run_low_level(water('hf'), method='rpa', mode='casida',
                         natural_orbitals=True)


@pytest.fixture(scope='module')
def h(low):
    """Water NO CAS(4,4), both the occupied and the virtual block rotated:
    a coupling with both sides."""
    return build_effective_hamiltonian(
        low, no_space(low, 2, 2, rotate_occupied=True), screening='crpa')


@pytest.fixture(scope='module')
def h_virt(low):
    """The water NO CAS(4,4) of Tasks 8 and 9: canonical occupied
    orbitals, the virtual block rotated."""
    return build_effective_hamiltonian(low, no_space(low, 2, 2),
                                       screening='crpa')


@pytest.fixture(scope='module')
def case(h):
    """(h1, eri, sides, engine, E, V) of the rotated space, E, V the
    lowest three states at (2, 2) of any spin."""
    h1, eri = _integrals(h)
    engine = PyscfFCISolver()
    E, V = engine.roots(h1, eri, 4, (2, 2), nroots=3, spin=None)
    return h1, eri, h.hybridization.sides(), engine, E, V


@pytest.fixture(scope='module')
def solved(case):
    """{method: (DeltaResult, flags, warning classes)} of the water states
    with both sides."""
    h1, eri, sides, engine, E, V = case
    return {m: _record(lambda m=m: DYNAMIC_METHODS[m](engine, sides).refine(
        h1, eri, 4, (2, 2), E, V)) for m in DYNAMIC_METHODS}


# ---------------------------------------------------------- the dense side
def test_lowdin_with_both_sides_is_the_pq_spectrum(h, case, solved):
    h1, eri, sides, _, E, _ = case
    assert [kind for kind, _, _, _ in sides] == ['particle', 'hole']
    assert np.all(h.hybridization.b_hole[2:] == 0.0)
    assert np.abs(h.hybridization.b_hole[:2]).max() > 1e-3
    res, flags, caught = solved['lowdin']
    e, w = pq_spectrum(h1, eri, 4, (2, 2), sides)
    worst = 0.0
    for x in res.energies:
        k = np.argmin(abs(e - x))
        worst = max(worst, abs(e[k] - x))
        assert abs(e[k] - x) < 1e-8 and w[k] > 0.5
    particle = [s for s in sides if s[0] == 'particle']
    alone = LowdinSolver(PyscfFCISolver(), particle).refine(
        h1, eri, 4, (2, 2), E, case[5])
    print(f'Loewdin with both sides against P + Q: {worst:.1e}; Delta shift '
          f'{np.round(res.energies - E, 6)}; hole side alone shifts by '
          f'{np.round(res.energies - alone.energies, 6)}')
    assert np.all(res.converged)
    assert flags == [] and caught == []


def test_linearized_follows_lowdin_with_both_sides(solved):
    """The bar of 1e-6 Ha holds for the ground state, which raises no
    'delta_linearization'. The two excited states, coupled strongly (Z
    about 0.7), miss it, as with the particle side alone in Task 8: this
    is the method, not the hole side (controller's ruling, Task 13). Each
    state off by more than the bar is flagged 'delta_linearization', and
    its estimate Z |<c|H_eff(E)|c> - E| is within 10% of the gap to
    Loewdin. Measured on 2026-11-27 (|linearized - lowdin|, estimate, Z):
    state 0 6.8e-11, 5.9e-11, 0.999; state 1 3.85e-3, 3.84e-3, 0.694;
    state 2 1.50e-3, 1.50e-3, 0.729."""
    lin, flags, caught = solved['linearized']
    low = solved['lowdin'][0]
    gap = np.abs(lin.energies - low.energies)
    estimate = lin.extra['linearization_error']
    for k in range(len(gap)):
        print(f'water state {k}: |linearized - lowdin| {gap[k]:.2e}, '
              f'estimate {estimate[k]:.2e}, Z {lin.z[k]:.3f}')
    bar = 1e-6
    assert gap[0] < bar
    off = [k for k in range(len(gap)) if gap[k] > bar]
    flagged = sorted(_state(text) for name, text in flags
                     if name == 'delta_linearization')
    assert flagged == off == [1, 2]
    assert {name for name, _ in flags} == {'delta_linearization'}
    assert set(caught) == {DeltaLinearizationWarning}
    ratio = estimate[off] / gap[off]
    assert np.all(ratio > 0.9) and np.all(ratio < 1.1)


def test_canonical_occupied_has_no_hole_term(low, h_virt):
    """rotate_occupied=False: the hole block is exactly zero, so sides()
    has no hole entry and the energies are those of Tasks 8 and 9, pinned
    from the code before this task (7c4bf1b, float.hex) and equal bit for
    bit to a build that folds the particle block alone (the test hook)."""
    cp = h_virt.hybridization
    assert cp.blocks == COUPLING_BLOCKS == DEFAULT_BLOCKS
    assert np.all(cp.b_hole == 0.0) and cp.norms['ao_i'] == 0.0
    assert [kind for kind, _, _, _ in cp.sides()] == ['particle']
    particle = build_effective_hamiltonian(low, no_space(low, 2, 2),
                                           screening='crpa',
                                           _blocks=('av_ev',))
    assert particle.energies.e0 == h_virt.energies.e0
    for method, want in PINNED_TASK_8.items():
        e, _, _ = _record(lambda: h_virt.solve('pyscf_fci', nroots=3,
                                               delta_method=method))
        again, _, _ = _record(lambda: particle.solve(
            'pyscf_fci', nroots=3, delta_method=method))
        assert np.array_equal(e, again), method
        assert all(abs(a - float.fromhex(b)) < 1e-10
                   for a, b in zip(e, want)), method
    assert static_mf_delta(cp)[1] == 0.0


#: the energies of the water NO CAS(4,4) with canonical occupied orbitals,
#: h.solve('pyscf_fci', nroots=3, delta_method=m), from the code before
#: this task (7c4bf1b), written with float.hex()
PINNED_TASK_8 = {
    'lowdin': ['-0x1.307bb472d1721p+6', '-0x1.2eb9ec88045d3p+6',
               '-0x1.2e85c8c34fe0fp+6'],
    'linearized': ['-0x1.307bb472d1714p+6', '-0x1.2ea8b74e86977p+6',
                   '-0x1.2e84304c9c878p+6'],
}


# ----------------------------------------------------------- static forms
def test_attachment_moment_from_density_matrices(h_virt):
    """attachment_moment is sum_sigma <a+_t K|(H^(N+1) - E_K)|a+_u K>, from
    the vectors and the (N+1)-electron sector matrix of the engine."""
    st = static_states('pyscf_fci', _request(h_virt, 3))
    engine, worst = st.engine, 0.0
    sectors = {0: (3, 2), 1: (2, 3)}
    h_m = {s: engine.sector_matrix(st.h1, st.eri, 4, m)
           for s, m in sectors.items()}
    for e_k, k in zip(st.energies, st.vectors):
        dm1, dm2 = engine.rdms(k, 4, st.sector, spin=None)
        got = attachment_moment(st.h1, st.eri, dm1, dm2)
        want = np.zeros((4, 4))
        for s, m in sectors.items():
            a = [engine.create(k, 4, st.sector, u, s).ravel()
                 for u in range(4)]
            for t in range(4):
                for u in range(4):
                    want[t, u] += a[t] @ (h_m[s] @ a[u] - e_k * a[u])
        worst = max(worst, np.abs(got - want).max())
    print(f'attachment moment against the (N+1) sector: {worst:.1e}')
    assert worst < 1e-9


def test_state_averaged_hole_block_matches_its_formula(h):
    """The hole block of D_SA from the hole density gamma_h = 2 - dm1 and
    the attachment moment of the active occupied orbitals, as the particle
    block is from gamma and the removal moment; the constant -g tr(D_h),
    g = 2; the particle block is that of the coupling without its hole
    side. prepare_delta stores it and 'static_sa' adds e0 + c."""
    cp = h.hybridization
    st = static_states('pyscf_fci', _request(h, 2))
    rdms = [st.engine.rdms(v, 4, st.sector, spin=None) for v in st.vectors]
    D, c = state_averaged_delta(st.h1, st.eri, cp, rdms, h.numerics)
    ao = np.arange(cp.n_act_occ)
    gamma = np.mean([2.0 * np.eye(2) - dm1[np.ix_(ao, ao)]
                     for dm1, _ in rdms], axis=0)
    m1 = np.mean([attachment_moment(st.h1, st.eri, dm1, dm2)[np.ix_(ao, ao)]
                  for dm1, dm2 in rdms], axis=0)
    n, v = np.linalg.eigh(gamma)
    assert np.all(n > 1e-3)                 # both directions kept here
    s = v / np.sqrt(n)                      # V n^-1/2
    w, u = np.linalg.eigh(s.T @ m1 @ s)     # the attachment energies
    orbitals = v @ u
    b = orbitals.T @ cp.b_hole[ao]

    def delta(x):
        return (b / (x - cp.lam_occupied)) @ b.T

    d_prime = 0.5 * np.array([[delta(w[k])[k, l] + delta(w[l])[k, l]
                               for l in range(2)] for k in range(2)])
    want = orbitals @ d_prime @ orbitals.T
    particle, c_particle = state_averaged_delta(
        st.h1, st.eri, replace(cp, b_hole=np.zeros_like(cp.b_hole)), rdms,
        h.numerics)
    print(f'D_SA hole block against its formula: '
          f'{np.abs(D[:2, :2] - want).max():.1e}; c = {c:.6e}; '
          f'attachment energies {np.round(w, 4)}')
    assert np.abs(D[:2, :2] - want).max() < 1e-12
    assert np.abs(want).max() > 1e-4
    assert c == pytest.approx(-2.0 * np.trace(want), abs=1e-14) and c != 0.0
    assert np.array_equal(D[2:, 2:], particle[2:, 2:]) and c_particle == 0.0
    assert np.all(D[:2, 2:] == 0.0) and np.all(D[2:, :2] == 0.0)
    # through the record
    h2 = h.prepare_delta(((2, None),))
    got_D, got_c = h2.hybridization.d_sa
    assert np.abs(got_D - D).max() < 1e-12 and abs(got_c - c) < 1e-14
    out = {}
    h2.solve('pyscf_fci', delta_method='static_sa', out=out)
    assert out['e0'] == h.energies.e0 + got_c


#: the shifts, Ha, by which the inactive occupied levels are moved down in
#: test_static_hole_term_has_its_constant
SHIFTS = (10.0, 20.0, 40.0, 80.0)


def test_static_hole_term_has_its_constant(h, case):
    """The hole side alone, its levels lam_i moved down by L: the error of
    each static form against Loewdin falls as 1/L^2 with the constant
    -g tr(D_h), and only as 1/L without it (the constant is the 1/L part
    of <K|a_t a+_u|K> Delta_tu). 'static_mf' at the mean-field energies,
    'static_sa' from the three states themselves."""
    h1, eri, sides, engine, E, V = case
    cp = h.hybridization
    rdms = [engine.rdms(v, 4, (2, 2), spin=None) for v in V]
    errors = {form: ([], []) for form in ('static_mf', 'static_sa')}
    for L in SHIFTS:
        lam = cp.lam_occupied - L
        holed = replace(cp, b_particle=np.zeros_like(cp.b_particle),
                        lam_occupied=lam)
        exact = LowdinSolver(engine, [('hole', None, lam, cp.b_hole)]
                             ).refine(h1, eri, 4, (2, 2), E, V).energies
        for form in errors:
            D, c = (static_mf_delta(holed) if form == 'static_mf' else
                    state_averaged_delta(h1, eri, holed, rdms, h.numerics))
            assert c == pytest.approx(-2.0 * np.trace(D), abs=1e-14)
            static = engine.roots(h1 + D, eri, 4, (2, 2), nroots=3,
                                  spin=None)[0]
            errors[form][0].append(np.abs(exact - (static + c)).max())
            errors[form][1].append(np.abs(exact - static).max())
    for form, (with_c, without_c) in errors.items():
        with_c, without_c = np.array(with_c), np.array(without_c)
        print(f'{form}: shifts {SHIFTS}; |lowdin - static| with the '
              f'constant {with_c}, x L^2 {with_c * np.square(SHIFTS)}; '
              f'without {without_c}, x L {without_c * np.array(SHIFTS)}')
        # doubling L divides the error by 4 with the constant, by 2 without
        assert np.all(with_c[:-1] / with_c[1:] > 3.5)
        assert np.all(without_c[:-1] / without_c[1:] < 2.2)
        assert np.all(without_c > 10.0 * with_c)


# --------------------------------------------------- degenerate states
@pytest.fixture(scope='module')
def n2():
    """N2 NO CAS(6,6) with both blocks rotated: (h1, eri, sides, engine).
    Its occupied cut lies between two sigma NOs whose occupations differ by
    less than no_cut_warn: it runs, with NearDegenerateCutWarning."""
    low = run_low_level(nitrogen(), method='rpa', mode='casida',
                        natural_orbitals=True)
    with pytest.warns(NearDegenerateCutWarning, match='occupation'):
        h = build_effective_hamiltonian(
            low, no_space(low, 3, 3, rotate_occupied=True), screening='crpa')
    assert 'no_cut_near_degenerate' in [f['flag'] for f in h.numerical_flags]
    h1, eri = _integrals(h)
    return h1, eri, h.hybridization.sides(), PyscfFCISolver()


@pytest.mark.parametrize('method', sorted(DYNAMIC_METHODS))
def test_degenerate_partners_with_both_sides(n2, method):
    """The pairs of degenerate states among the lowest eight of N2 CAS(6,6)
    at (3, 3), both sides folded: one energy, 1e-8 Ha, and two different
    states."""
    h1, eri, sides, engine = n2
    assert [kind for kind, _, _, _ in sides] == ['particle', 'hole']
    E, V = engine.roots(h1, eri, 6, (3, 3), nroots=8, spin=None)
    res, flags, caught = _record(lambda: DYNAMIC_METHODS[method](
        engine, sides).refine(h1, eri, 6, (3, 3), E, V))
    pairs = [(k, k + 1) for k in range(len(E) - 1) if E[k + 1] - E[k] < 1e-8]
    print(f'{method}: pairs {pairs}, flags {sorted({f for f, _ in flags})}')
    assert len(pairs) >= 2
    for k, l in pairs:
        de = abs(res.energies[k] - res.energies[l])
        ov = abs(np.vdot(res.vectors[k], res.vectors[l]))
        print(f'  {k},{l}: E {res.energies[k]:.10f} {res.energies[l]:.10f} '
              f'|dE| {de:.1e}, |<c1|c2>| {ov:.1e}, '
              f'shift {res.energies[k] - E[k]:.6f}')
        assert de < 1e-8 and ov < 1e-6
    assert flags == [] and caught == []


# ------------------------------------------------------------ the blocks
def test_no_active_occupied_means_no_hole_term(low):
    """A rotated occupied block with no active occupied orbital: no row of
    the hole block, so no hole side, no constant, and no active ph pair for
    e_c_hybridization."""
    h0 = build_effective_hamiltonian(
        low, no_space(low, 0, 2, rotate_occupied=True), screening='crpa')
    cp = h0.hybridization
    assert cp.n_act_occ == 0 and cp.b_hole.shape == (2, low.nocc)
    assert np.all(cp.b_hole == 0.0) and cp.norms['ao_i'] == 0.0
    assert 'hole' not in [kind for kind, _, _, _ in cp.sides()]
    assert static_mf_delta(cp)[1] == 0.0
    assert h0.energies.e_c_hybridization == 0.0
    st = static_states('pyscf_fci', _request(h0, 1))
    rdms = [st.engine.rdms(v, st.norb, st.sector, spin=None)
            for v in st.vectors]
    D, c = state_averaged_delta(st.h1, st.eri, cp, rdms, h0.numerics)
    assert c == 0.0 and np.array_equal(D, np.zeros((2, 2)))


def test_e_c_hybridization_and_delta_use_the_same_blocks(low, h, h_virt):
    """Spec 6b: e_c_hybridization dresses the active orbitals with every
    block the rotation couples, the occupied one included, so Delta folds
    the same blocks. A record has a coupling exactly when it has the E_0
    term, the coupling folds every block, and its sides are the blocks
    whose coupling is nonzero."""
    norb = low.mf.mo_coeff.shape[1]
    canonical = build_effective_hamiltonian(
        low, ActiveSpace.from_indices(norb, 5, [3, 4], [5, 6]),
        screening='crpa')
    assert canonical.hybridization is None
    assert canonical.energies.e_c_hybridization == 0.0
    for record, kinds in ((h, ['particle', 'hole']), (h_virt, ['particle'])):
        cp = record.hybridization
        assert cp.blocks == COUPLING_BLOCKS
        assert [kind for kind, _, _, _ in cp.sides()] == kinds
        folded = record.diagnostics['fock_coupling']['folded']
        assert [name for name in COUPLING_BLOCKS if folded[name] > 0.0] == \
            [{'particle': 'av_ev', 'hole': 'ao_i'}[k] for k in kinds]
        assert record.energies.e_c_hybridization != 0.0
    print(f'e_c_hybridization: both blocks rotated '
          f'{h.energies.e_c_hybridization:.6e}, the virtual block alone '
          f'{h_virt.energies.e_c_hybridization:.6e}')
    # the default solve folds both sides and adds e0
    out = {}
    e, flags, caught = _record(lambda: h.solve('pyscf_fci', out=out))
    assert out['e0'] == h.energies.e0
    assert np.array_equal(e, out['e0'] + out['e_active'])
    # the static form 'static_mf' carries the constant of its hole part
    _, c = static_mf_delta(h.hybridization)
    with pytest.warns(StaticDeltaWarning):
        h.solve('pyscf_fci', delta_method='static_mf', out=out)
    assert c != 0.0 and out['e0'] == h.energies.e0 + c
