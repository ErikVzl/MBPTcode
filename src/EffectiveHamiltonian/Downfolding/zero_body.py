"""The constant E_0 of the downfolded Hamiltonian, term by term.

E_0 = E_nuc + E_core + E_Hartree + E_exchange        (the environment)
      + E_Hartree,dc + E_exchange,dc                  (E_HF,ee[vtilde])
      + E_c[v] - E_c^A[v_eff] - dE_c^A
      (+ the exchange energy term - Tr[Sigma_bar gamma^A,0] with Sigma^E),
with dE_c^A = E_c^{A,Delta}[v_eff] - E_c^A[v_eff].
The `Energies` record keeps every one of these on its own; `eh`, `ex` and
`ec` are the Hartree, exchange and correlation sums of each line pair.
The HF terms of the environment are those of the inactive core alone (the
active-environment cross terms are in the core Fock operator), from core.py;
the double-counting terms come from double_counting.py, which says why
E_HF,ee[vtilde] enters with a plus sign. This file only assembles them.

E_0 subtracts the RPA correlation energy of the model whose active energy
is added to it, so that nothing is counted twice, and there are two models
(spec 6b, 6d):

- the decoupled active space, (h, v) of g_eff alone: its RPA correlation
  energy is E_c^A[v_eff], on the semicanonical active pairs. Its constant
  is `Energies.e0_without_delta`, the E_0 above without dE_c^A: the ten
  terms of a downfolding without Delta. It is g_eff[0], so that g_eff,
  all three parts, is this one model and contains no Delta.
- the active space coupled to the environment, (h, v) and the coupling
  Delta folds (the record's `hybridization`): its RPA correlation energy
  is E_c^{A,Delta}[v_eff], the same ring sum with the active orbitals
  expanded in the canonical ones, so E_0 subtracts dE_c^A as well. Its
  constant is `Energies.e0`, every term. It belongs to an active energy
  computed with Delta.

dE_c^A is `Energies.e_c_hybridization`. It is 0.0, not computed, whenever
no coupling is built (canonical orbitals, delta=False), and 0.0 for method
'hf', which has no correlation energy; then the two constants are one
number, to the bit the E_0 of a downfolding without an orbital basis. A total energy adds `e0` to an active energy with Delta and
`e0_without_delta` (= g_eff[0]) to one without; any other pairing is the
energy of no model.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Energies:
    """Every term of E_0 on its own (the paper's Fig. 4 breakdown).

    Stored, each with the sign it has before entering E_0:
    e_nuc: the nuclear repulsion.
    e_core: the one-electron energy of the inactive core.
    e_hartree, e_exchange: the Hartree and exchange energies of the inactive
        core (exchange is negative).
    e_hartree_dc, e_exchange_dc: the Hartree and exchange halves of
        E_HF,ee[vtilde], the HF energy of the screening correction in the
        active reference determinant: the double-counting correction of the
        mean-field energy. Both zero when vtilde has no occupied-occupied
        element ('bare', 'crpa_ph', 'mrpa', method 'hf').
    e_c_full: the correlation energy of the full space, E_c[v].
    e_c_active: the correlation energy of the active space alone,
        E_c^A[v_eff], the double-counting correction of the correlation
        energy; E_0 subtracts it.
    e_exchange_term: the exchange energy term that comes with Sigma^E,
        -2 e_exchange_dc; nonzero only with dc='hartree'.
    e_sigma_trace: Tr[Sigma_bar gamma^A,0]; E_0 subtracts it; nonzero only
        with a Sigma_bar.
    e_c_hybridization: dE_c^A = E_c^{A,Delta}[v_eff] - E_c^A[v_eff], the
        change of the active space's RPA correlation energy when its
        orbitals are dressed by the coupling to the environment that Delta
        folds; `e0` subtracts it, `e0_without_delta` does not. Nonzero only
        for an active space on a rotated orbital basis with Delta on (a
        coupling on the record, h.hybridization), and with method 'rpa';
        0.0 exactly, not computed, otherwise. Its sign is not fixed by
        construction; on the spaces measured so far it is negative (the
        hybridized space correlates more: water and NH3 natural orbitals,
        -0.45 to -2.4 mEh), so there `e0` lies above `e0_without_delta`.

    Sums (properties): `eh`, `ex`, `ec` are the Hartree, exchange and
    correlation energies of E_0 with their corrections,
    ec = e_c_full - e_c_active - e_c_hybridization.

    The two constants (zero_body.py says why there are two):
    e0 = e_nuc + e_core + eh + ex + ec - e_sigma_trace, every term: the
        constant of the active space coupled to the environment, to which
        an active energy computed with Delta is added.
    e0_without_delta: the same without e_c_hybridization, the ten terms of
        a downfolding without Delta: the constant of the decoupled active
        space, (h, v) alone, to which an active energy without Delta is
        added. g_eff[0] holds e0_without_delta, so that g_eff = {0, 1, 2}
        is one model, all three parts the decoupled one: nothing in g_eff
        contains Delta, and a solve of g_eff alone is a consistent total.
    With Delta off (no coupling: canonical orbitals, delta=False), and for
    method 'hf', e_c_hybridization is 0.0 and e0, e0_without_delta and
    g_eff[0] are one number, to the bit.
    """
    e_nuc: float
    e_core: float
    e_hartree: float
    e_exchange: float
    e_hartree_dc: float
    e_exchange_dc: float
    e_c_full: float
    e_c_active: float
    e_exchange_term: float
    e_sigma_trace: float
    e_c_hybridization: float = 0.0

    @property
    def eh(self):
        """The Hartree energy in E_0: environment plus double counting."""
        return self.e_hartree + self.e_hartree_dc

    @property
    def ex(self):
        """The exchange energy in E_0: environment, double counting and,
        with Sigma^E, the exchange energy term."""
        return self.e_exchange + self.e_exchange_dc + self.e_exchange_term

    @property
    def ec(self):
        """The correlation energy in E_0,
        E_c[v] - E_c^A[v_eff] - dE_c^A."""
        return self.e_c_full - self.e_c_active - self.e_c_hybridization

    @property
    def e0(self):
        """E_0 with every term: the constant of the active space coupled to
        the environment (with Delta). Equal to e0_without_delta, and so to
        g_eff[0], when e_c_hybridization is zero."""
        return (self.e_nuc + self.e_core + self.eh + self.ex + self.ec
                - self.e_sigma_trace)

    @property
    def e0_without_delta(self):
        """E_0 without e_c_hybridization: the constant of the decoupled
        active space, g_eff[0]. The expression of a downfolding without
        Delta, term for term in the same order, so that it is that number
        to the bit."""
        return (self.e_nuc + self.e_core + self.eh + self.ex
                + (self.e_c_full - self.e_c_active) - self.e_sigma_trace)


def assemble_energies(e_nuc, core, dc, terms):
    """The `Energies` of one Hamiltonian: the nuclear repulsion, the frozen
    core's three energies (`Core`), the four energies of the double counting
    (`DoubleCounting`) and the method's three correlation energies
    (`DownfoldingTerms`)."""
    return Energies(e_nuc=e_nuc, e_core=core.e_core,
                    e_hartree=core.e_hartree, e_exchange=core.e_exchange,
                    e_hartree_dc=dc.e_hartree_dc,
                    e_exchange_dc=dc.e_exchange_dc,
                    e_c_full=terms.e_c_full, e_c_active=terms.e_c_active,
                    e_exchange_term=dc.e_exchange_term,
                    e_sigma_trace=dc.e_sigma_trace,
                    e_c_hybridization=terms.e_c_hybridization)
