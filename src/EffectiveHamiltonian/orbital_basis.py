"""Orbital bases of the downfolding that are not the canonical orbitals: for
now the blocks of the GW density that natural orbitals are built from.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NaturalDensity:
    """The occupied-occupied and virtual-virtual blocks of the spatial RPA
    (GW) density in the canonical MO basis of the mean field, and the size of
    the occupied-virtual block, which natural orbitals leave out.

    gamma_oo: (nocc, nocc), with the reference's 2 on the diagonal.
    gamma_vv: (nvirt, nvirt).
    ov_max, ov_fro: the largest element and the Frobenius norm of the
        occupied-virtual block.
    """
    gamma_oo: np.ndarray
    gamma_vv: np.ndarray
    ov_max: float
    ov_fro: float
