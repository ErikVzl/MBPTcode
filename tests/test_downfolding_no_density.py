"""The oo and vv blocks of the GW density, stored by the low-level RPA run
with natural_orbitals=True: the blocks of the unrelaxed density of
`compute_gw_density_matrix` on the same route, in the canonical MO basis.
casida takes the run's own eigenvectors (no second eigensolve); the two
imaginary-axis routes agree with it to their quadrature error. A run on
the exact four-centre integrals (df=False) is refused: a build on natural
orbitals needs three-index factors."""
import numpy as np
import pytest
from pyscf import scf

from src.EffectiveHamiltonian.Downfolding import run_low_level
from src.EffectiveHamiltonian.Downfolding.methods import rpa
from src.SingleReference.GW.density_matrix import compute_gw_density_matrix
from tests.no_downfolding import G1, ammonia, water


@pytest.mark.parametrize('xc', ['hf', 'pbe0'])
def test_blocks_are_those_of_the_gw_density(xc):
    mf = water(xc, df=True)
    low = run_low_level(mf, method='rpa', mode='casida', natural_orbitals=True)
    dm = compute_gw_density_matrix(mf, mode='casida', relaxation='none')
    c, s, nocc = mf.mo_coeff, mf.get_ovlp(), low.nocc
    g = c.T @ s @ dm @ s @ c
    assert np.abs(low.natural.gamma_oo - g[:nocc, :nocc]).max() < 1e-10
    assert np.abs(low.natural.gamma_vv - g[nocc:, nocc:]).max() < 1e-10
    assert abs(low.natural.ov_max - np.abs(g[:nocc, nocc:]).max()) < 1e-10
    assert abs(low.natural.ov_fro - np.linalg.norm(g[:nocc, nocc:])) < 1e-10


def test_exact_integrals_are_refused(monkeypatch):
    """df=False with natural_orbitals=True is refused by the run, before
    the RPA is solved (NotImplementedError naming the three-index factors):
    every build on its natural orbitals would be refused for the lack of
    them."""
    def ran(*args, **kwargs):
        raise AssertionError('the RPA ran before the refusal')
    monkeypatch.setattr(rpa, '_casida_fields', ran)
    with pytest.raises(NotImplementedError,
                       match='natural_orbitals=True.*df=False.*three-index'):
        run_low_level(water('hf'), method='rpa', mode='casida', df=False,
                      natural_orbitals=True)


def test_signs_and_traces():
    nat = run_low_level(water('pbe0', df=True), method='rpa', mode='casida',
                        natural_orbitals=True).natural
    n_oo, n_vv = np.linalg.eigvalsh(nat.gamma_oo), np.linalg.eigvalsh(nat.gamma_vv)
    assert n_oo.max() <= 2 + 1e-10 and n_vv.min() >= -1e-10
    assert abs((2 * len(n_oo) - n_oo.sum()) - n_vv.sum()) < 1e-9


def _blocks_differ(a, b):
    return max(np.abs(a.gamma_oo - b.gamma_oo).max(),
               np.abs(a.gamma_vv - b.gamma_vv).max())


def test_routes_agree():
    """imagfrequency 1e-6 on water; space-time 1e-4 on ammonia with G1."""
    mf = water('hf', df=True)
    ref = run_low_level(mf, method='rpa', mode='casida', natural_orbitals=True)
    imag = run_low_level(mf, method='rpa', mode='imagfrequency',
                         natural_orbitals=True)
    assert _blocks_differ(ref.natural, imag.natural) < 1e-6

    mf = ammonia('hf')
    ref = run_low_level(mf, method='rpa', mode='casida', natural_orbitals=True)
    st = run_low_level(mf, method='rpa', mode='space-time', numerics=G1,
                       natural_orbitals=True)
    assert _blocks_differ(ref.natural, st.natural) < 1e-4


def test_off_by_default():
    low = run_low_level(water('hf', df=True), method='rpa', mode='casida')
    assert low.natural is None


def test_unrestricted_refused():
    mf = scf.UHF(water().mol).run()
    with pytest.raises(NotImplementedError, match='unrestricted'):
        run_low_level(mf, method='rpa', mode='casida', natural_orbitals=True)


def test_hf_method_refuses_the_option():
    with pytest.raises(TypeError, match='natural_orbitals'):
        run_low_level(water(), method='hf', natural_orbitals=True)
