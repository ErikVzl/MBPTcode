"""The many-body solvers of an active-space Hamiltonian g_eff, as the two
builders of src/EffectiveHamiltonian produce it: solve.py dispatches by name
(`solve`, `SOLVERS`, `ENGINES`) over the engines of backends.py, the static
backends (FCI, the determinant basis, DMRG, CCSDT/CCSDTQ behind the
ActiveSpaceSolver interface), hands on the states of a static solve
(`static_states`) and, given the coupling (`sides`), refines them with a
dynamic backend; dynamic_backends.py holds the dynamic backends, which refine the
states of a static backend with the energy-dependent term Delta of a
downfolding on a rotated basis (exact and linearized Loewdin, and the
remainder method on the static D_SA, `DYNAMIC_METHODS`, on the FCI
engine's sector operations), and the
removal and attachment Green's functions of a CI state by block Lanczos
(`block_lanczos`, `removal_spectrum`, `attachment_spectrum`), and import
backends.py, never the reverse; determinants.py builds the explicit determinant-basis matrix and the
spin-orbital form of a spatial g_eff. Each entry point is imported from the
module that defines it; this package re-exports nothing.
"""
