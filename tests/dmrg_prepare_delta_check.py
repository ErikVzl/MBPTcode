"""Standalone check of `DownfoldedHamiltonian.prepare_delta(solver='dmrg')`,
run in its own interpreter for the same reason as dmrg_adapter_check.py:
block2 must be imported before numpy/pyscf in this process, or its OpenMP
runtime clashes with numpy's MKL one and aborts the process -- not a
catchable ImportError, so this cannot run inside the main test process via
pytest.importorskip and is instead invoked as a subprocess from
test_downfolding_no_delta_solve.py (test_dmrg_prepare_delta), which skips
when block2 is not installed. It is in no tools/ci list.

D_SA, the static form 'static_sa' of Delta, is built from the one- and
two-particle density matrices of a declared set of states, so any backend
that has them can build it. Water/6-31G RHF on the natural-orbital CAS(4,4)
of the Delta tests, both blocks rotated (rotate_occupied=True, so D_SA has a
particle and a hole part and the hole constant c is not zero), with the
states ((2, 0), (1, 2)): the two lowest singlets and the lowest triplet. The
check: D_SA = (D, c) built with solver='dmrg' (bond dimension 200, 20
sweeps: exact for four orbitals) against the one built with
solver='pyscf_fci', and the 'static_sa' energies of the two records, each
solved with 'pyscf_fci', all within TOL.
"""
import block2  # noqa: F401  -- MUST precede numpy

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from src.EffectiveHamiltonian.Downfolding import (build_effective_hamiltonian,
                                                  run_low_level)
from tests.no_downfolding import no_space, water

#: the agreement asked of D, c and the 'static_sa' energies, Hartree
TOL = 1e-6
STATES = ((2, 0), (1, 2))


def main():
    low = run_low_level(water('hf'), method='rpa', mode='casida',
                        natural_orbitals=True)
    h = build_effective_hamiltonian(
        low, no_space(low, 2, 2, rotate_occupied=True), screening='crpa')
    fci = h.prepare_delta(STATES, solver='pyscf_fci')
    dmrg = h.prepare_delta(STATES, solver='dmrg', dmrg_bond_dim=200,
                           dmrg_n_sweeps=20)
    (d_fci, c_fci), (d_dmrg, c_dmrg) = (fci.hybridization.d_sa,
                                        dmrg.hybridization.d_sa)
    ok = True
    dd, dc = float(np.abs(d_dmrg - d_fci).max()), abs(c_dmrg - c_fci)
    this = dd < TOL and dc < TOL
    ok &= this
    print(f"  D_SA: max |D_dmrg - D_fci| {dd:.2e}, |c_dmrg - c_fci| {dc:.2e} "
          f"(c = {c_fci:.8f}): {'OK' if this else 'FAIL'}")
    for nroots, spin in STATES:
        e_fci = fci.solve('pyscf_fci', delta_method='static_sa',
                          nroots=nroots, spin=spin)
        e_dmrg = dmrg.solve('pyscf_fci', delta_method='static_sa',
                            nroots=nroots, spin=spin)
        de = float(np.abs(np.asarray(e_dmrg) - np.asarray(e_fci)).max())
        this = de < TOL
        ok &= this
        print(f"  'static_sa', spin {spin}: {np.round(e_fci, 8)} (D_SA by "
              f"FCI), max difference with D_SA by DMRG {de:.2e}: "
              f"{'OK' if this else 'FAIL'}")
    if ok:
        print("DMRG PREPARE_DELTA VERIFIED")
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
