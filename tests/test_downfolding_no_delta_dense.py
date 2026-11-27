"""Delta solved densely: the sector operations of the FCI engine
(`PyscfFCISolver.sector_matrix`, `sector_product`, `annihilate`, `create`)
and the two dense dynamic backends of `solvers/dynamic_backends.py`, exact
Loewdin (`LowdinSolver`) and linearized Loewdin (`LinearizedLowdinSolver`).

Water/6-31G RHF on an NO CAS(4,4): the Loewdin roots are eigenvalues of the
explicit P + Q matrix (`pq_spectrum` in `tests/no_downfolding.py`); the
linearized ground state follows them, the strongly coupled excited states
are flagged 'delta_linearization' with an estimate that measures their gap;
a zero coupling gives the static spectrum and Z = 1, 0 < Z <= 1; the
strongest pole next to the ground state gives exactly the flags of each
method ('delta_near_pole', for linearized also at its linearization point,
'delta_satellite'), a root exactly on a pole stays finite; no active virtual
means no Delta. N2/6-31G on an NO CAS(6,6): linearized follows Loewdin, the
two partners of a degenerate state come out with one energy and as two
different states, and the memory estimate bounds the tracemalloc
peak. The kernel `delta_roots` on a model with two spin channels and both
kinds of side; the pairing of the columns of b with their levels against the
Fock blocks on PBE; the capability markers, the refusal of an engine without
sector operations and of a solve above the memory limit, and the default
limit."""
import ast
import math
import pathlib
import tracemalloc
import warnings

import numpy as np
import pytest
from pyscf import fci
from pyscf.scf.hf import get_jk

from src.Base.numerical_flags import collecting
from src.Base.numerics.numerics import Numerics
from src.EffectiveHamiltonian.Downfolding import (build_effective_hamiltonian,
                                                  run_low_level)
from src.EffectiveHamiltonian.orbital_basis import semicanonical_rotation
from src.EffectiveHamiltonian.solvers.backends import (
    Block2DMRGSolver, CoupledClusterSolver, ExactDiagonalizationSolver,
    PyscfFCISolver, PyscfSpinResolvedFCISolver)
from src.EffectiveHamiltonian.solvers.dynamic_backends import (
    DYNAMIC_METHODS, DeltaLinearizationWarning, DeltaNearPoleWarning,
    DeltaNotConverged, LinearizedLowdinSolver, LowdinSolver, delta_roots,
    dense_memory_gb, dense_memory_limit_gb)
from tests.no_downfolding import nitrogen, no_space, pq_spectrum, water

REPO = pathlib.Path(__file__).resolve().parent.parent


def _integrals(h):
    """(h1, eri) of a build, h1 symmetric and eri (chemist) symmetric in
    each pair and under the pair swap, as pyscf's FCI assumes."""
    h1 = 0.5 * (h.g_eff[1] + h.g_eff[1].T)
    eri = h.g_eff[2].transpose(0, 2, 1, 3)
    eri = 0.5 * (eri + eri.transpose(1, 0, 2, 3))
    eri = 0.5 * (eri + eri.transpose(0, 1, 3, 2))
    eri = 0.5 * (eri + eri.transpose(2, 3, 0, 1))
    return h1, eri


@pytest.fixture(scope='module')
def low():
    return run_low_level(water('hf'), method='rpa', mode='casida',
                         natural_orbitals=True)


@pytest.fixture(scope='module')
def case(low):
    """The water NO CAS(4,4): (h1, eri, sides, engine, E, V), E, V the
    lowest three states at (2, 2) of any spin."""
    h = build_effective_hamiltonian(low, no_space(low, 2, 2),
                                    screening='crpa')
    h1, eri = _integrals(h)
    engine = PyscfFCISolver()
    E, V = engine.roots(h1, eri, 4, (2, 2), nroots=3, spin=None)
    return h1, eri, h.hybridization.sides(), engine, E, V


def _refine(method, engine, sides, h1, eri, norb, nelec, E, V):
    """(result, flags, warnings) of one refine: the flags it raised, as
    [(name, message)], and the classes of the warnings it issued, every one
    recorded here so that the output stays clean and each test asserts
    exactly what it expects."""
    flags = []
    with collecting(flags), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        res = DYNAMIC_METHODS[method](engine, sides).refine(h1, eri, norb,
                                                            nelec, E, V)
    return (res, [(f['flag'], f['message']) for f in flags],
            [w.category for w in caught])


@pytest.fixture(scope='module')
def solved(case):
    """{method: (result, flags, warnings)} of the water states."""
    h1, eri, sides, engine, E, V = case
    return {m: _refine(m, engine, sides, h1, eri, 4, (2, 2), E, V)
            for m in DYNAMIC_METHODS}


@pytest.fixture(scope='module')
def results(solved):
    return {m: out[0] for m, out in solved.items()}


@pytest.fixture(scope='module')
def n2():
    """N2 NO CAS(6,6), both pi and both pi* orbitals active: (h1, eri,
    sides, engine)."""
    low = run_low_level(nitrogen(), method='rpa', mode='casida',
                        natural_orbitals=True)
    h = build_effective_hamiltonian(low, no_space(low, 3, 3),
                                    screening='crpa')
    h1, eri = _integrals(h)
    return h1, eri, h.hybridization.sides(), PyscfFCISolver()


# ------------------------------------------------------- sector operations
def test_sector_matrix_spectrum_is_fci(case):
    h1, eri, _, engine, _, _ = case
    for nelec in ((2, 2), (1, 2), (3, 2)):
        m = engine.sector_matrix(h1, eri, 4, nelec)
        assert np.abs(m - m.T).max() < 1e-12
        solver = fci.direct_spin1.FCI()
        solver.conv_tol, solver.verbose = 1e-13, 0
        e0, _ = solver.kernel(h1, eri, 4, nelec, ecore=0.0)
        assert abs(np.linalg.eigvalsh(m)[0] - e0) < 1e-10, nelec


def test_sector_product_is_the_matrix(case):
    h1, eri, _, engine, _, _ = case
    rng = np.random.default_rng(3)
    for nelec in ((2, 2), (1, 2), (2, 1), (3, 2), (2, 3)):
        shape = (math.comb(4, nelec[0]), math.comb(4, nelec[1]))
        v = rng.normal(size=shape)
        hv = engine.sector_product(h1, eri, 4, nelec)(v)
        m = engine.sector_matrix(h1, eri, 4, nelec)
        assert np.abs(hv.ravel() - m @ v.ravel()).max() < 1e-12, nelec


def test_annihilate_and_create_shapes(case):
    _, _, _, engine, _, V = case
    k = V[0]
    for u in range(4):
        assert engine.annihilate(k, 4, (2, 2), u, 0).shape == (4, 6)
        assert engine.annihilate(k, 4, (2, 2), u, 1).shape == (6, 4)
        assert engine.create(k, 4, (2, 2), u, 0).shape == (4, 6)
        assert engine.create(k, 4, (2, 2), u, 1).shape == (6, 4)
        # the values are pyscf's
        assert np.array_equal(engine.annihilate(k, 4, (2, 2), u, 1),
                              fci.addons.des_b(k, 4, (2, 2), u))
        assert np.array_equal(engine.create(k, 4, (2, 2), u, 0),
                              fci.addons.cre_a(k, 4, (2, 2), u))
    # <M|a_u|K> = <K|a+_u|M>
    rng = np.random.default_rng(1)
    for u in range(4):
        for sigma, sector in ((0, (1, 2)), (1, (2, 1))):
            m = rng.normal(size=(math.comb(4, sector[0]),
                                 math.comb(4, sector[1])))
            left = np.vdot(m, engine.annihilate(k, 4, (2, 2), u, sigma))
            right = np.vdot(k, engine.create(m, 4, sector, u, sigma))
            assert abs(left - right) < 1e-14
    # an emptied or a filled spin: a sector of size zero, no error
    empty = np.ones((1, 6))                              # (0, 2) in 4 orbitals
    assert engine.annihilate(empty, 4, (0, 2), 1, 0).shape == (0, 6)
    full = np.ones((1, 6))                               # (4, 2)
    assert engine.create(full, 4, (4, 2), 1, 0).shape == (0, 6)
    assert engine.create(np.ones((6, 1)), 4, (2, 4), 0, 1).shape == (6, 0)


# ------------------------------------------------------------ the methods
def test_lowdin_roots_are_the_pq_eigenvalues(case, results):
    h1, eri, sides, _, E, _ = case
    res = results['lowdin']
    e, w = pq_spectrum(h1, eri, 4, (2, 2), sides)
    worst = 0.0
    for x in res.energies:
        k = np.argmin(abs(e - x))
        worst = max(worst, abs(e[k] - x))
        assert abs(e[k] - x) < 1e-8 and w[k] > 0.5
    print(f'Loewdin against P + Q: {worst:.1e}; Delta shift '
          f'{np.round(res.energies - E, 6)}')
    assert np.all(res.converged)
    assert np.array_equal(res.static, E)


def test_lowdin_with_a_hole_side_is_a_pq_eigenvalue(case):
    """Both kinds through the engine: a made-up hole side (levels below the
    active space, couplings on the occupied rows) beside the particle
    side."""
    h1, eri, sides, engine, E, V = case
    rng = np.random.default_rng(5)
    b = np.zeros((4, 3))
    b[:2] = 0.2 * rng.normal(size=(2, 3))
    both = list(sides) + [('hole', None, np.array([-2.5, -1.9, -1.4]), b)]
    res = LowdinSolver(engine, both).refine(h1, eri, 4, (2, 2), E, V)
    e, w = pq_spectrum(h1, eri, 4, (2, 2), both)
    for x in res.energies:
        k = np.argmin(abs(e - x))
        assert abs(e[k] - x) < 1e-8 and w[k] > 0.5


def test_linearized_ground_state_follows_lowdin(solved):
    """Water: the ground state (Z close to 1) agrees with Loewdin to 1e-6
    and raises no 'delta_linearization'. Measured on 2026-11-27:
    |linearized - lowdin| 1.8e-13 Ha, estimate 1.8e-13 Ha (Z 0.9998)."""
    lin, flags, _ = solved['linearized']
    low = solved['lowdin'][0]
    diff = abs(lin.energies[0] - low.energies[0])
    print(f'water ground state: linearized - lowdin {diff:.1e} Ha, '
          f'estimate {lin.extra["linearization_error"][0]:.1e}')
    assert diff < 1e-6
    assert not any(name == 'delta_linearization' and 'state 0 ' in text
                   for name, text in flags)


def test_linearized_excited_states_are_flagged(solved):
    """Water NO CAS(4,4) couples its active virtuals strongly: the two
    excited states (Z about 0.7 and 0.64) are flagged 'delta_linearization',
    and the estimate Z |<c|H_eff(E)|c> - E| measures the gap to Loewdin.
    Measured on 2026-11-27 (|linearized - lowdin|, estimate, Z): state 1
    1.56e-3, 1.56e-3, 0.72; state 2 1.68e-2, 1.66e-2, 0.63. The estimate
    is within 1% of the gap; asserted within 10%."""
    lin, flags, caught = solved['linearized']
    low = solved['lowdin'][0]
    gap = np.abs(lin.energies - low.energies)
    estimate = lin.extra['linearization_error']
    for k in range(3):
        print(f'water state {k}: |linearized - lowdin| {gap[k]:.2e}, '
              f'estimate {estimate[k]:.2e}, Z {lin.z[k]:.3f}')
    flagged = sorted(int(text.split('state ')[1].split()[0])
                     for name, text in flags if name == 'delta_linearization')
    assert flagged == [1, 2]
    assert DeltaLinearizationWarning in caught
    assert {name for name, _ in flags} == {'delta_linearization'}
    warn = Numerics().downfolding.delta_linearization_warn
    assert np.all(estimate[1:] > warn) and estimate[0] < warn
    ratio = estimate[1:] / gap[1:]
    assert np.all(ratio > 0.9) and np.all(ratio < 1.1)
    # Loewdin is the reference and raises nothing here
    assert solved['lowdin'][1] == [] and solved['lowdin'][2] == []


def test_linearized_follows_lowdin_on_n2(n2):
    """N2 NO CAS(6,6), the lowest eight states at (3, 3), Z 0.94 to 0.998.
    Measured on 2026-11-27: max |linearized - lowdin| = 7.1e-9 Ha, the
    linearization error estimates 2e-11 to 6.9e-9 Ha, none flagged."""
    h1, eri, sides, engine = n2
    E, V = engine.roots(h1, eri, 6, (3, 3), nroots=8, spin=None)
    out = [DYNAMIC_METHODS[m](engine, sides).refine(h1, eri, 6, (3, 3), E, V)
           for m in ('lowdin', 'linearized')]
    diff = np.abs(out[1].energies - out[0].energies).max()
    print(f'N2 linearized - lowdin: {diff:.1e} Ha')
    assert diff < 1e-6
    assert np.all(out[1].extra['linearization_error']
                  < Numerics().downfolding.delta_linearization_warn)


def test_zero_coupling_is_the_static_spectrum(case):
    h1, eri, sides, engine, E, V = case
    zero = [(k, s, lam, np.zeros_like(b)) for k, s, lam, b in sides]
    for cls in DYNAMIC_METHODS.values():
        res = cls(engine, zero).refine(h1, eri, 4, (2, 2), E, V)
        assert np.abs(res.energies - E).max() < 1e-12
        assert np.all(res.z == 1.0)


def test_z_is_between_zero_and_one(results):
    for res in results.values():
        assert np.all(res.z > 0.0) and np.all(res.z <= 1.0)
        assert np.all(res.z < 1.0)


def test_vectors_are_the_sector_and_unit_norm(case, results):
    _, _, _, _, E, V = case
    for res in results.values():
        assert len(res.vectors) == len(E)
        for c, k in zip(res.vectors, V):
            assert c.shape == k.shape
            assert abs(np.linalg.norm(c) - 1.0) < 1e-12
            assert abs(np.vdot(c, k)) > 0.9


#: the flags, (name, state), each method raises with the strongest pole of
#: the ground state 0.005 Ha below it (test_near_pole_is_flagged_not_nan).
#: Loewdin stays on the active state (Z 0.991), 0.007 Ha from the pole.
#: Linearized, from E0 0.0098 Ha from the pole, lands on the satellite
#: (Z 0.023, 0.013 Ha from Loewdin): near the pole at its linearization
#: point, a satellite, its error estimate 2.2e-2 Ha; state 1 (Z 0.71) is
#: flagged for its linearization error as without the moved pole.
NEAR_POLE_FLAGS = {
    'lowdin': [('delta_near_pole', 0)],
    'linearized': [('delta_linearization', 0), ('delta_linearization', 1),
                   ('delta_near_pole', 0), ('delta_satellite', 0)]}


def _state(text):
    """The state a flag message names ('... state k at ...')."""
    return int(text.split('state ')[1].split()[0])


@pytest.mark.parametrize('method', sorted(DYNAMIC_METHODS))
def test_near_pole_is_flagged_not_nan(case, method):
    """The pole (M, e) the ground state weighs most (alpha sector (1, 2))
    moved to 0.005 Ha below the static ground state: a finite result and
    exactly the flags of NEAR_POLE_FLAGS."""
    h1, eri, sides, engine, E, V = case
    (kind, sigma, lam, b), = sides
    e_m, v_m = np.linalg.eigh(engine.sector_matrix(h1, eri, 4, (1, 2)))
    amp = np.array([v_m.T @ engine.annihilate(V[0], 4, (2, 2), u, 0).ravel()
                    for u in range(4)])                 # (orbital, M)
    weight = (amp.T @ b) ** 2                            # (M, e)
    m, e = np.unravel_index(np.argmax(weight), weight.shape)
    moved = lam.copy()
    moved[e] = E[0] - e_m[m] - 0.005
    res, flags, caught = _refine(method, engine, [(kind, sigma, moved, b)],
                                 h1, eri, 4, (2, 2), E, V)
    print(f'{method}: weight {weight[m, e]:.1e}, Z {np.round(res.z, 3)}, '
          f'smallest denominator {np.round(res.min_denominator, 4)}, '
          f'flags {sorted(flags)}')
    assert sorted((name, _state(text)) for name, text in flags) == \
        NEAR_POLE_FLAGS[method]
    assert DeltaNearPoleWarning in caught
    assert np.all(np.isfinite(res.energies)) and np.all(np.isfinite(res.z))
    for c in res.vectors:
        assert np.all(np.isfinite(c))


@pytest.mark.parametrize('method', sorted(DYNAMIC_METHODS))
def test_a_root_on_a_pole_is_finite(case, method):
    """The same pole moved exactly onto the static ground state, where the
    first step evaluates Delta: E_K - E_M - lam = 0 in floating point, with
    the solver's own spectrum of the sector. A finite energy, Z and vector,
    and 'delta_near_pole', never inf or NaN."""
    h1, eri, sides, engine, E, V = case
    (kind, sigma, lam, b), = sides
    _, e_m, amp, _, _ = LowdinSolver(engine, sides).sector(
        h1, eri, 4, (2, 2), kind, 0, (1, 2), lam, b)
    rows = np.flatnonzero(np.any(b != 0.0, axis=1))   # amp(u) exists for these
    weight = np.array([amp(u) @ V[0].ravel() for u in rows]).T @ b[rows]
    m, e = np.unravel_index(np.argmax(weight ** 2), weight.shape)
    moved = lam.copy()
    moved[e] = E[0] - e_m[m]
    assert (E[0] - e_m[m]) - moved[e] == 0.0
    res, flags, caught = _refine(method, engine, [(kind, sigma, moved, b)],
                                 h1, eri, 4, (2, 2), E, V)
    names = [name for name, _ in flags]
    print(f'{method} on the pole: E - E_static {res.energies[0] - E[0]:.3e}, '
          f'Z {res.z[0]:.3e}, flags {sorted(set(names))}')
    assert 'delta_near_pole' in names
    # no division by zero or overflow from numpy, only the solver's verdicts
    assert all(issubclass(c, (DeltaNearPoleWarning, DeltaNotConverged,
                              DeltaLinearizationWarning)) for c in caught)
    assert np.all(np.isfinite(res.energies)) and np.all(np.isfinite(res.z))
    assert np.all(np.isfinite(res.min_denominator))
    for c in res.vectors:
        assert np.all(np.isfinite(c))


def test_no_active_virtual_means_no_delta(low):
    h = build_effective_hamiltonian(low, no_space(low, 2, 0),
                                    screening='crpa')
    assert h.hybridization.sides() == []
    h1, eri = _integrals(h)
    engine = PyscfFCISolver()
    E, V = engine.roots(h1, eri, 2, (2, 2), nroots=1, spin=None)
    for cls in DYNAMIC_METHODS.values():
        res = cls(engine, h.hybridization.sides()).refine(h1, eri, 2, (2, 2),
                                                          E, V)
        assert np.array_equal(res.energies, E) and np.all(res.z == 1.0)
        assert np.all(np.isinf(res.min_denominator))


def test_no_sector_linearized_reports_zero_error_at_the_static_energy(low):
    h = build_effective_hamiltonian(low, no_space(low, 2, 0),
                                    screening='crpa')
    h1, eri = _integrals(h)
    engine = PyscfFCISolver()
    E, V = engine.roots(h1, eri, 2, (2, 2), nroots=1, spin=None)
    res = DYNAMIC_METHODS['linearized'](engine, h.hybridization.sides()
                                        ).refine(h1, eri, 2, (2, 2), E, V)
    assert np.array_equal(res.extra['linearization_error'],
                          np.zeros(len(E)))
    assert np.array_equal(res.extra['linearization_point'], E)


def test_columns_of_b_pair_with_their_levels():
    """Sum_e b_te b_ue / (w - lam_e) from `Coupling.sides()` is
    F_s(A_v, E_v) (w - F(E_v, E_v))^-1 F_s(E_v, A_v) on a PBE reference,
    where the bare F and F_s differ: a wrong U_E or a permuted lam shows
    here, not in b b^T."""
    mf = water('pbe')
    low = run_low_level(mf, method='rpa', mode='casida',
                        natural_orbitals=True)
    space = no_space(low, 2, 2)
    h = build_effective_hamiltonian(low, space, screening='crpa')
    (kind, sigma, lam, b), = h.hybridization.sides()
    assert kind == 'particle' and sigma is None
    T, _ = semicanonical_rotation(space.basis, space, mf.mo_energy)
    mo = mf.mo_coeff @ T
    j, k = get_jk(mf.mol, mf.make_rdm1())
    F = mo.T @ (mf.get_hcore() + j - 0.5 * k) @ mo
    F_s = T.T @ np.diag(mf.mo_energy) @ T
    av, ev = space.virt_active, space.env_virt
    assert np.abs(F - F_s)[np.ix_(ev, ev)].max() > 1e-3
    w = 0.3
    want = np.zeros((len(space.act),) * 2)
    n = space.n_act_occ
    want[n:, n:] = F_s[np.ix_(av, ev)] @ np.linalg.solve(
        w * np.eye(len(ev)) - F[np.ix_(ev, ev)], F_s[np.ix_(ev, av)])
    got = (b / (w - lam)) @ b.T
    print(f'b / (w - lam) b^T against the Fock blocks: '
          f'{np.abs(got - want).max():.1e}')
    assert np.abs(got - want).max() < 1e-10


# ------------------------------------------------------- degenerate states
@pytest.mark.parametrize('method', sorted(DYNAMIC_METHODS))
def test_degenerate_partners(n2, method):
    """The pairs of degenerate states among the lowest eight of N2 CAS(6,6)
    at (3, 3): one energy, 1e-8 Ha, and two different states. The overlap is
    Euclidean: inside a degenerate set of a symmetric Hamiltonian the
    metric (1 - D1) of the linearized method is a multiple of the identity
    (Schur), so the two vanish together."""
    h1, eri, sides, engine = n2
    E, V = engine.roots(h1, eri, 6, (3, 3), nroots=8, spin=None)
    res = DYNAMIC_METHODS[method](engine, sides).refine(h1, eri, 6, (3, 3),
                                                        E, V)
    pairs = [(k, k + 1) for k in range(len(E) - 1) if E[k + 1] - E[k] < 1e-8]
    # at least one pair is pi -> pi*: the sigma_g orbital (active 0, the
    # lowest canonical one of the three occupied) stays doubly occupied
    sigma_g = [engine.rdms(V[k], 6, (3, 3), spin=None)[0][0, 0]
               for k, _ in pairs]
    print(f'{method}: pairs {pairs}, sigma_g occupation {np.round(sigma_g, 3)}')
    assert len(pairs) >= 2 and max(sigma_g) > 1.9
    for k, l in pairs:
        de = abs(res.energies[k] - res.energies[l])
        ov = abs(np.vdot(res.vectors[k], res.vectors[l]))
        print(f'  {k},{l}: E {res.energies[k]:.10f} {res.energies[l]:.10f} '
              f'|dE| {de:.1e}, |<c1|c2>| {ov:.1e}, '
              f'shift {res.energies[k] - E[k]:.6f}')
        assert de < 1e-8 and ov < 1e-6


# ------------------------------------------------------------- the kernel
def test_kernel_with_two_spin_channels_and_both_kinds():
    """A model with different alpha and beta integrals (pyscf direct_uhf),
    one coupling and one set of levels per spin and kind, sector (2, 1): every
    Loewdin root started from an eigenstate of H_N is an eigenvalue of the
    explicit P + Q matrix that lies mostly in P."""
    rng = np.random.default_rng(21)
    norb, nelec = 4, (2, 1)

    def pair_symmetric(a):
        return a + a.transpose(0, 2, 1)
    fa = pair_symmetric(rng.normal(size=(5, norb, norb))) * 0.25
    fb = pair_symmetric(rng.normal(size=(5, norb, norb))) * 0.25
    eri = tuple(np.einsum('Qpq,Qrs->pqrs', x, y)
                for x, y in ((fa, fa), (fa, fb), (fb, fb)))
    ha, hb = rng.normal(size=(norb, norb)), rng.normal(size=(norb, norb))
    h1 = (0.5 * (ha + ha.T), 0.5 * (hb + hb.T))

    def shape(ne):
        return math.comb(norb, ne[0]), math.comb(norb, ne[1])

    def matrix(ne):
        h2e = fci.direct_uhf.absorb_h1e(h1, eri, norb, ne, 0.5)
        n = np.prod(shape(ne))
        return np.array([fci.direct_uhf.contract_2e(
            h2e, np.eye(n)[k].reshape(shape(ne)), norb, ne).ravel()
            for k in range(n)]).T

    def operator(op, ne, u):
        n = np.prod(shape(nelec))
        return np.array([op(np.eye(n)[k].reshape(shape(nelec)), norb, nelec,
                            u).ravel() for k in range(n)]).T

    sides = {('particle', 0): ((1, 1), fci.addons.des_a, 3, (4.0, 6.0)),
             ('particle', 1): ((2, 0), fci.addons.des_b, 2, (4.0, 6.0)),
             ('hole', 0): ((3, 1), fci.addons.cre_a, 2, (-6.0, -4.0)),
             ('hole', 1): ((2, 2), fci.addons.cre_b, 2, (-6.0, -4.0))}
    h_n = matrix(nelec)
    sectors, couplings, blocks = [], [], []
    for (kind, _), (ne, op, n_lev, window) in sides.items():
        lam = rng.uniform(*window, n_lev)
        b = rng.normal(size=(norb, n_lev)) * 0.4
        h_m = matrix(ne)
        e_m, v_m = np.linalg.eigh(h_m)
        ops = [operator(op, ne, u) for u in range(norb)]
        sectors.append((kind, e_m, lambda u, v=v_m, o=ops: v.T @ o[u], lam, b))
        sign = 1.0 if kind == 'particle' else -1.0
        for x in range(n_lev):
            couplings.append(sum(b[u, x] * ops[u] for u in range(norb)))
            blocks.append(h_m + sign * lam[x] * np.eye(len(h_m)))
    c = np.vstack(couplings)
    full = np.block([[h_n, c.T],
                     [c, np.block([[blocks[i] if i == j else
                                    np.zeros((len(blocks[i]), len(blocks[j])))
                                    for j in range(len(blocks))]
                                   for i in range(len(blocks))])]])
    e, vec = np.linalg.eigh(full)
    w = (vec[:len(h_n)] ** 2).sum(0)
    e_n, v_n = np.linalg.eigh(h_n)
    n = 6
    res = delta_roots(h_n, sectors, e_n[:n], [v_n[:, k] for k in range(n)],
                      'lowdin', Numerics())
    worst = 0.0
    for x in res.energies:
        k = np.argmin(abs(e - x))
        worst = max(worst, abs(e[k] - x))
        assert abs(e[k] - x) < 1e-8 and w[k] > 0.5
    print(f'two channels, both kinds: P dim {len(h_n)}, Q dim '
          f'{len(full) - len(h_n)}, worst {worst:.1e}, Delta shift '
          f'{np.round(res.energies - e_n[:n], 4)}')
    assert np.all(res.converged)


# ------------------------------------------------------------ the contract
def test_capability_markers():
    assert PyscfFCISolver.has_rdms and PyscfFCISolver.has_sector_operations
    assert Block2DMRGSolver.has_rdms
    assert not Block2DMRGSolver.has_sector_operations
    for cls in (PyscfSpinResolvedFCISolver, ExactDiagonalizationSolver,
                CoupledClusterSolver):
        assert not cls.has_rdms and not cls.has_sector_operations, cls


def test_an_engine_without_sector_operations_is_refused(case):
    sides = case[2]
    for engine in (CoupledClusterSolver(), ExactDiagonalizationSolver()):
        with pytest.raises(ValueError, match='static_sa') as err:
            LowdinSolver(engine, sides)
        assert 'static_mf' in str(err.value)


def test_static_backends_do_not_import_the_dynamic_ones():
    tree = ast.parse((REPO / 'src/EffectiveHamiltonian/solvers/backends.py')
                     .read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names = [node.module or ''] + [a.name for a in node.names]
        elif isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        else:
            continue
        assert not any('dynamic_backends' in n for n in names)


class _Untouched(PyscfFCISolver):
    """An engine that fails as soon as a sector is built."""

    def sector_matrix(self, *args):
        raise AssertionError('a sector was built')

    def sector_product(self, *args):
        raise AssertionError('a sector was built')


def test_too_large_is_refused(case):
    h1, eri, sides, _, E, V = case
    tiny = Numerics(layout={'delta_dense_gb': 1e-6})
    for cls in DYNAMIC_METHODS.values():
        with pytest.raises(ValueError, match='sa_remainder') as err:
            cls(_Untouched(), sides, tiny).refine(h1, eri, 4, (2, 2), E, V)
        assert 'delta_dense_gb' in str(err.value)


def test_memory_estimate_bounds_the_peak(n2):
    """The tracemalloc peak of one Loewdin solve is below the estimate plus
    100 MB (the estimate counts the arrays, not the interpreter and pyscf's
    own buffers)."""
    h1, eri, sides, engine = n2
    E, V = engine.roots(h1, eri, 6, (3, 3), nroots=3, spin=None)
    solver = LowdinSolver(engine, sides)
    tracemalloc.start()
    try:
        solver.refine(h1, eri, 6, (3, 3), E, V)
        peak = tracemalloc.get_traced_memory()[1] / 1e9
    finally:
        tracemalloc.stop()
    estimate = dense_memory_gb(400, [300, 300], 'lowdin')
    print(f'memory: estimate {estimate * 1e3:.2f} MB, tracemalloc peak '
          f'{peak * 1e3:.2f} MB')
    assert peak <= estimate + 0.1


def test_default_memory_limit(monkeypatch):
    for name in ('SLURM_MEM_PER_NODE', 'SLURM_MEM_PER_CPU', 'SLURM_JOB_ID',
                 'SLURM_CPUS_PER_TASK', 'SLURM_NTASKS_PER_NODE',
                 'SLURM_TASKS_PER_NODE'):
        monkeypatch.delenv(name, raising=False)
    assert dense_memory_limit_gb(Numerics()) == 8.0
    assert dense_memory_limit_gb(Numerics(layout={'delta_dense_gb': 3.0})) == 3.0
    monkeypatch.setenv('SLURM_JOB_ID', '1')
    monkeypatch.setenv('SLURM_MEM_PER_NODE', '64000')
    # the share of the allocation the BSE and Davidson solvers also take
    assert dense_memory_limit_gb(Numerics()) == pytest.approx(64.0 * 0.6)
    assert dense_memory_limit_gb(Numerics(layout={'delta_dense_gb': 3.0})) == 3.0


def test_the_settings_are_validated():
    for bad in ({'delta_pole_warn': -1.0}, {'delta_conv': 0.0},
                {'delta_max_iter': 0}, {'delta_weight_floor': -1e-9},
                {'delta_degenerate': -1.0},
                {'delta_linearization_warn': 0.0}):
        with pytest.raises(ValueError):
            Numerics(downfolding=bad)
    # the guard of the zero denominator must stay below the pole flag
    with pytest.raises(ValueError, match='delta_conv.*delta_pole_warn'):
        Numerics(downfolding={'delta_conv': 1e-3, 'delta_pole_warn': 1e-3})
    with pytest.raises(ValueError):
        Numerics(layout={'delta_dense_gb': 0.0})
