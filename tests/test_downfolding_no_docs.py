"""The documentation of the natural-orbital downfolding names what a user
needs to pair a constant with an active energy, and every setting of Delta.

In the manner of test_warnings_doc_complete.py: the names are looked up as
text, so a renamed field, or a new one without its documentation, fails
here.

- The constants (spec 6d): `e0_without_delta` (the constant of the
  decoupled active space, g_eff[0]), `static_hamiltonian` (the Hamiltonian
  for a solver outside `solve`, its constant matching its one-body part),
  `g_eff_is_static_part` (True when g_eff is the static part only) and
  `e_active` (out['e_active'], the active energy a solve added its constant
  to) appear in docs/methods/downfolding.md, and in the docstrings of
  `Energies`, `DownfoldedHamiltonian` and `DownfoldedHamiltonian.solve`
  where they apply (APPLIES).
- Every field of `DownfoldingNumerics`, and `LayoutNumerics.delta_dense_gb`,
  appears in docs/methods/downfolding.md and in the docstring of its class.
"""
import dataclasses
import inspect
import os
import re

from src.Base.numerics.numerics import DownfoldingNumerics, LayoutNumerics
from src.EffectiveHamiltonian.Downfolding.effective_hamiltonian import (
    DownfoldedHamiltonian)
from src.EffectiveHamiltonian.Downfolding.zero_body import Energies

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DOC = os.path.join(REPO, 'docs', 'methods', 'downfolding.md')

#: the names of the constants and the docstrings each one applies to
CONSTANT_NAMES = ('e0_without_delta', 'static_hamiltonian',
                  'g_eff_is_static_part', 'e_active')
APPLIES = {
    # the record of the E_0 terms: its second constant, and where a total
    # and an active energy come from
    'Energies': (Energies, ('e0_without_delta', 'static_hamiltonian',
                            'e_active')),
    'DownfoldedHamiltonian': (DownfoldedHamiltonian, CONSTANT_NAMES),
    'DownfoldedHamiltonian.solve': (DownfoldedHamiltonian.solve,
                                    CONSTANT_NAMES),
}


def _page():
    with open(DOC) as f:
        return f.read()


def _named(name, text):
    """`name` as a word of `text` (not a part of a longer name)."""
    return re.search(r'(?<![A-Za-z0-9_])' + re.escape(name)
                     + r'(?![A-Za-z0-9_])', text) is not None


def _numerics_names():
    """{class name: (class, its documented fields)}: every field of
    DownfoldingNumerics, and delta_dense_gb of LayoutNumerics."""
    return {
        'DownfoldingNumerics': (DownfoldingNumerics, tuple(
            f.name for f in dataclasses.fields(DownfoldingNumerics))),
        'LayoutNumerics': (LayoutNumerics, ('delta_dense_gb',)),
    }


def test_constants_in_the_page():
    page = _page()
    missing = [n for n in CONSTANT_NAMES if not _named(n, page)]
    assert not missing, (
        f'docs/methods/downfolding.md does not name {missing}: the page '
        f'has to say which constant goes with which energy (spec 6d).')


def test_constants_in_the_docstrings():
    missing = {where: [n for n in names
                       if not _named(n, inspect.getdoc(obj) or '')]
               for where, (obj, names) in APPLIES.items()}
    missing = {k: v for k, v in missing.items() if v}
    assert not missing, (
        f'docstrings that do not name the constants that apply to them '
        f'(docstring: names): {missing}')


def test_numerics_in_the_page():
    page = _page()
    missing = {cls: [n for n in names if not _named(n, page)]
               for cls, (_, names) in _numerics_names().items()}
    missing = {k: v for k, v in missing.items() if v}
    assert not missing, (
        f'docs/methods/downfolding.md does not name these settings '
        f'(class: fields): {missing}')


def test_numerics_in_their_class_docstring():
    missing = {cls: [n for n in names
                     if not _named(n, inspect.getdoc(obj) or '')]
               for cls, (obj, names) in _numerics_names().items()}
    missing = {k: v for k, v in missing.items() if v}
    assert not missing, (
        f'settings their class docstring does not name (class: fields): '
        f'{missing}')


def test_the_lookup_sees_a_missing_name():
    """The word match fails on a name that is only part of a longer one."""
    assert _named('e0', 'the constant e0 of')
    assert not _named('e0', 'e0_without_delta')
    assert not _named('delta_conv', 'no delta_converged here')
