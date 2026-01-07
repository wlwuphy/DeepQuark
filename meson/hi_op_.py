import netket as nk
from netket.utils.types import Array
from netket.utils import struct

import jax
import jax.numpy as jnp
from jax.numpy import pi, exp, sqrt

from config_ import potential, quarks, MASS, masses, PARA, S, I
from model_ import compute_rij

"""Hilbert Space"""
_nd, _nparticles = 3, len(quarks)
hi = nk.hilbert.Particle(N=_nparticles, L=(jnp.inf, jnp.inf, jnp.inf))

"""Operator"""


@struct.dataclass
class CQM_PotentialPara:
    # AL1/AP1/AL2/AP2
    k: float = struct.field(pytree_node=False)
    k_: float = struct.field(pytree_node=False)
    l: float = struct.field(pytree_node=False)
    L: float = struct.field(pytree_node=False)
    A: float = struct.field(pytree_node=False)
    B: float = struct.field(pytree_node=False)
    rc: float = struct.field(pytree_node=False)
    p: float = struct.field(pytree_node=False)


@struct.dataclass
class ChQM_PotentialPara:
    # ChQM
    mpi: float = struct.field(pytree_node=False)
    msigma: float = struct.field(pytree_node=False)
    mk: float = struct.field(pytree_node=False)
    meta: float = struct.field(pytree_node=False)
    Lambda_pi: float = struct.field(pytree_node=False)
    Lambda_sigma: float = struct.field(pytree_node=False)
    Lambda_k: float = struct.field(pytree_node=False)
    Lambda_eta: float = struct.field(pytree_node=False)
    gch2: float = struct.field(pytree_node=False)
    thetap: float = struct.field(pytree_node=False)
    alpha0: float = struct.field(pytree_node=False)
    Lambda0: float = struct.field(pytree_node=False)
    mu0: float = struct.field(pytree_node=False)
    r0: float = struct.field(pytree_node=False)
    ac: float = struct.field(pytree_node=False)
    muc: float = struct.field(pytree_node=False)
    delta: float = struct.field(pytree_node=False)


_mass = []
for m in MASS:
    _mass += [m] * _nd
kin = nk.operator.KineticEnergy(hi, mass=_mass)
mass = jnp.array(MASS)
if potential == 'ChQM':
    para = ChQM_PotentialPara(**PARA)
else:
    para = CQM_PotentialPara(**PARA)
sme = -3 / 4 if S == 0 else 1 / 4


def oge_vij(x: Array):
    """Two-body potential vij with one-gluon-exchange and confinement interaction
        Args:
            x: spatial coordinates (shape=(2*nd,))
    """

    cme = -16 / 3
    rij = compute_rij(x).squeeze()

    mi, mj = mass[0], mass[1]
    r0 = para.A * (2 * mi * mj / (mi + mj)) ** (-para.B)
    k = jnp.where(para.rc == 0, para.k, para.k * (1 - jnp.exp(-rij / para.rc)))
    k_ = jnp.where(para.rc == 0, para.k_, para.k_ * (1 - jnp.exp(-rij / para.rc)))
    v0 = -3 / 16 * cme * (-k / rij + para.l * rij ** para.p - para.L)
    v1 = -3 / 16 * cme * (8 * pi * k_ / (3 * mi * mj)
                          * exp(-rij ** 2 / r0 ** 2) / (pi ** (3 / 2) * r0 ** 3) * sme)
    vme = v0 + v1 + mi + mj

    return vme


def ch_vij(x: Array):
    """Two-body potential vij with one-gluon-exchange, one-boson-exchange (for non-strange system) and confinement interaction
        Args:
            x: spatial coordinates (shape=(2*nd,))
    """

    cme = -16 / 3
    rij = compute_rij(x).squeeze()

    mi, mj = mass[0], mass[1]
    mu = mi * mj / (mi + mj)
    r0 = para.r0 * masses['q'] / 2 / mu
    alphas = para.alpha0 / jnp.log((mu ** 2 + para.mu0 ** 2) / para.Lambda0 ** 2)
    voge0 = cme * (alphas / (4 * rij) - para.ac * (1 - jnp.exp(-para.muc * rij)) + para.delta)
    voge1 = -cme * sme * alphas / (6 * mi * mj) * jnp.exp(-rij / r0) / (r0 ** 2 * rij)

    def Y(x):
        return jnp.exp(-x) / x

    def vpseudo(mch, Lch):
        return para.gch2 / jnp.pi * mch ** 2 / (12 * mi * mj) * Lch ** 2 / (Lch ** 2 - mch ** 2) * mch * sme * (
                Y(mch * rij) - Lch ** 3 / mch ** 3 * Y(Lch * rij))

    pifme = -1
    etafme = jnp.cos(para.thetap) * 1 / 3 - jnp.sin(para.thetap)
    vpi = vpseudo(para.mpi, para.Lambda_pi) * pifme
    veta = vpseudo(para.meta, para.Lambda_eta) * etafme
    vsigma = -para.gch2 / (4 * jnp.pi) * para.Lambda_sigma ** 2 / (
            para.Lambda_sigma ** 2 - para.msigma ** 2) * para.msigma * (
                     Y(para.msigma * rij) - para.Lambda_sigma / para.msigma * Y(para.Lambda_sigma * rij))
    vobe = jnp.where((mi == masses['q']) & (mj == masses['q']), vpi + veta + vsigma, 0)
    vme = voge0 + voge1 + vobe + mi + mj

    return vme


if 'A' in potential:
    pot = nk.operator.PotentialEnergy(hi, oge_vij)
elif 'ChQM' in potential:
    pot = nk.operator.PotentialEnergy(hi, ch_vij)

ha = pot + kin

