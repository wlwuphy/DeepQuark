import netket as nk
from netket.utils.types import Array
from netket.utils import struct

import jax
import jax.numpy as jnp
from jax.numpy import pi, exp, sqrt

from config_ import potential, quarks, MASS, masses, PARA, S, I
from operator_ import MultiPotentialEnergy

"""Hilbert Space"""
_nd, _nparticles = 3, len(quarks)
dof_color = {3: 1, 4: 2, 5: 3}[_nparticles]
dof_spin = {0: 2, 1: 3, 2: 1, 0.5: 5, 1.5: 4, 2.5: 1}[S]
hi_con = nk.hilbert.Particle(N=_nparticles, L=(jnp.inf, jnp.inf, jnp.inf))
hi_color = nk.hilbert.Fock(N=dof_color, n_particles=1)
hi_spin = nk.hilbert.Fock(N=dof_spin, n_particles=1)
hi_dis = hi_color * hi_spin
hi = hi_con * hi_dis
n_discrete = hi_dis.size

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
_mass += [jnp.inf] * n_discrete
kin = nk.operator.KineticEnergy(hi, mass=_mass)
mass = jnp.array(MASS)
if potential == 'ChQM':
    para = ChQM_PotentialPara(**PARA)
else:
    para = CQM_PotentialPara(**PARA)


def multiquark_potential(vij, a, a_, r):
    """Multiquark potential
        Args:
            vij: two-body potential function
            a: color-spin configuration of bra state
            a_: color-spin configuration of ket state
            r: n-body coordinates
    """

    r = r.reshape(_nparticles, _nd)
    i, j = jnp.triu_indices(_nparticles, k=1)
    c, s = a[:dof_color], a[dof_color:]
    c_, s_ = a_[:dof_color], a_[dof_color:]
    ri, rj = r[i], r[j]

    def _vij(_i: Array, _j: Array, xi: Array, xj: Array):
        return vij(_i, _j, xi, xj, c, c_, s, s_)

    vij_inputs = (i, j, ri, rj)
    pot = jax.vmap(_vij)(*vij_inputs)
    diag = jnp.all(a == a_)
    massterm = jnp.where(diag, jnp.sum(jnp.array(MASS)), 0)
    return jnp.sum(pot) + massterm


def oge_vij(i: Array, j: Array, xi: Array, xj: Array, c: Array, c_: Array, s: Array, s_: Array):
    """Two-body potential vij with one-gluon-exchange and confinement interaction
        Args:
            i: particle indices
            j: particle indices (i<j)
            xi: spatial coordinates of i (shape=(nd,))
            xj: spatial coordinates of j (shape=(nd,))
            c: color configuration of bra state
            c_: color configuration of ket state
            s: spin configuration of bra state
            s_: spin configuration of ket state
    """

    cme = colorfactor(i, j, c, c_)
    sme = spinfactor(i, j, s, s_)
    rij = jnp.linalg.norm(xi - xj, axis=-1)

    mi, mj = mass[i], mass[j]
    r0 = para.A * (2 * mi * mj / (mi + mj)) ** (-para.B)
    k = jnp.where(para.rc == 0, para.k, para.k * (1 - jnp.exp(-rij / para.rc)))
    k_ = jnp.where(para.rc == 0, para.k_, para.k_ * (1 - jnp.exp(-rij / para.rc)))
    v0 = jnp.where(jnp.all(s == s_), -3 / 16 * cme * (-k / rij + para.l * rij ** para.p - para.L), 0)
    v1 = -3 / 16 * cme * (8 * pi * k_ / (3 * mi * mj)
                          * exp(-rij ** 2 / r0 ** 2) / (pi ** (3 / 2) * r0 ** 3) * sme)
    vme = v0 + v1

    return vme


def ch_vij(i: Array, j: Array, xi: Array, xj: Array, c: Array, c_: Array, s: Array, s_: Array):
    """Two-body potential vij with one-gluon-exchange, one-boson-exchange (for non-strange system) and confinement interaction
        Args:
            i: particle indices
            j: particle indices (i<j)
            xi: spatial coordinates of i (shape=(nd,))
            xj: spatial coordinates of j (shape=(nd,))
            c: color configuration of bra state
            c_: color configuration of ket state
            s: spin configuration of bra state
            s_: spin configuration of ket state
    """

    cme = colorfactor(i, j, c, c_)
    sme = spinfactor(i, j, s, s_)
    rij = jnp.linalg.norm(xi - xj, axis=-1)

    mi, mj = mass[i], mass[j]
    mu = mi * mj / (mi + mj)
    r0 = para.r0 * masses['q'] / 2 / mu
    alphas = para.alpha0 / jnp.log((mu ** 2 + para.mu0 ** 2) / para.Lambda0 ** 2)
    voge0 = jnp.where(jnp.all(s == s_),
                      cme * (alphas / (4 * rij) - para.ac * (1 - jnp.exp(-para.muc * rij)) + para.delta),
                      0)
    voge1 = -cme * sme * alphas / (6 * mi * mj) * jnp.exp(-rij / r0) / (r0 ** 2 * rij)

    def Y(x):
        return jnp.exp(-x) / x

    def vpseudo(mch, Lch):
        return para.gch2 / jnp.pi * mch ** 2 / (12 * mi * mj) * Lch ** 2 / (Lch ** 2 - mch ** 2) * mch * sme * (
                Y(mch * rij) - Lch ** 3 / mch ** 3 * Y(Lch * rij))

    pifme = jnp.select([I == 0, I == 1], [-3, 1], 0)
    etafme = jnp.cos(para.thetap) * 1 / 3 - jnp.sin(para.thetap)
    vpi = vpseudo(para.mpi, para.Lambda_pi) * pifme
    veta = vpseudo(para.meta, para.Lambda_eta) * etafme
    vsigma = -para.gch2 / (4 * jnp.pi) * para.Lambda_sigma ** 2 / (
            para.Lambda_sigma ** 2 - para.msigma ** 2) * para.msigma * (
                     Y(para.msigma * rij) - para.Lambda_sigma / para.msigma * Y(para.Lambda_sigma * rij))
    vobe = jnp.where((mi == masses['q']) & (mj == masses['q']), vpi + veta + vsigma, 0)
    vme = voge0 + voge1 + vobe

    return vme


if 'A' in potential:
    multiquark_pot = lambda a, a_, r: multiquark_potential(oge_vij, a, a_, r)
elif 'ChQM' in potential:
    multiquark_pot = lambda a, a_, r: multiquark_potential(ch_vij, a, a_, r)

mulpot = MultiPotentialEnergy(hi, hi_dis, multiquark_pot)
ha = mulpot + kin

"""Color"""
if _nparticles == 4:
    def colorfactor(i: Array, j: Array, c: Array, c_: Array):
        """color factor <a|λiλj|a_>
            :param
                i,j: i<j, particle indices
                c,c_: color configuration (0,1)->3*3; (1,0)->6*6
        """

        ij12 = jnp.array([0, 1], dtype=int)
        ij34 = jnp.array([2, 3], dtype=int)
        ij13 = jnp.array([0, 2], dtype=int)
        ij24 = jnp.array([1, 3], dtype=int)
        ij = jnp.array([i, j])

        pair12_34 = jnp.all(ij == ij12) | jnp.all(ij == ij34)
        pair13_24 = jnp.all(ij == ij13) | jnp.all(ij == ij24)

        color3 = jnp.array([0, 1], dtype=c.dtype)
        color6 = jnp.array([1, 0], dtype=c.dtype)

        cond33 = jnp.all(c == color3, axis=-1) & jnp.all(c_ == color3, axis=-1)
        cond66 = jnp.all(c == color6, axis=-1) & jnp.all(c_ == color6, axis=-1)

        cme33 = jnp.array([-8 / 3, -4 / 3, -4 / 3], dtype=jnp.float32)  # 12/34, 13/24, 14/23
        cme66 = jnp.array([4 / 3, -10 / 3, -10 / 3], dtype=jnp.float32)
        cme36 = jnp.array([0., -2 * jnp.sqrt(2), 2 * jnp.sqrt(2)], dtype=jnp.float32)

        cmes = jnp.select([cond33, cond66], [cme33, cme66], cme36)
        cme = jnp.select([pair12_34, pair13_24], cmes[:2], cmes[-1])

        return cme


    def color_proportions(arr):
        color33 = jnp.array([0, 1], dtype=arr.dtype)
        color66 = jnp.array([1, 0], dtype=arr.dtype)

        mask1 = jnp.all(arr == color33, axis=-1)
        mask2 = jnp.all(arr == color66, axis=-1)

        n1 = jnp.sum(mask1)
        n2 = jnp.sum(mask2)

        ntotal = arr.shape[0] * arr.shape[1]

        return n1 / ntotal, n2 / ntotal
elif _nparticles == 5:
    def colorfactor(i: Array, j: Array, c: Array, c_: Array):
        """color factor <a|λiλj|a_>
            :param
                i,j: i<j, particle indices
                c,c_: color configuration (0,0,1)->3bar*3bar*3bar; (0,1,0)->3bar*6*3bar; (1,0,0)->6*3bar*3bar;
        """

        ijpairs = jnp.array([[0, 1], [2, 3], [0, 2], [1, 3], [0, 3], [1, 2], [0, 4], [1, 4], [2, 4], [3, 4]], dtype=int)
        ij = jnp.array([i, j])
        pair = jnp.all(ijpairs == ij, axis=1)

        color33 = jnp.array([0, 0, 1], dtype=c.dtype)
        color36 = jnp.array([0, 1, 0], dtype=c.dtype)
        color63 = jnp.array([1, 0, 0], dtype=c.dtype)

        cond33_33 = jnp.all(c == color33, axis=-1) & jnp.all(c_ == color33, axis=-1)
        cond33_36 = ((jnp.all(c == color33, axis=-1) & jnp.all(c_ == color36, axis=-1)) |
                     (jnp.all(c == color36, axis=-1) & jnp.all(c_ == color33, axis=-1)))
        cond33_63 = ((jnp.all(c == color33, axis=-1) & jnp.all(c_ == color63, axis=-1)) |
                     (jnp.all(c == color63, axis=-1) & jnp.all(c_ == color33, axis=-1)))
        cond36_36 = jnp.all(c == color36, axis=-1) & jnp.all(c_ == color36, axis=-1)
        cond36_63 = ((jnp.all(c == color36, axis=-1) & jnp.all(c_ == color63, axis=-1)) |
                     (jnp.all(c == color63, axis=-1) & jnp.all(c_ == color36, axis=-1)))
        cond63_63 = jnp.all(c == color63, axis=-1) & jnp.all(c_ == color63, axis=-1)

        cme33_33 = jnp.array([-8 / 3, -8 / 3, -2 / 3, -2 / 3, -2 / 3, -2 / 3, -4 / 3, -4 / 3, -4 / 3, -4 / 3],
                             dtype=jnp.float32)  # 12,34,13,24,14,23,15,25,35,45
        cme36_36 = jnp.array([-8 / 3, 4 / 3, -5 / 3, -5 / 3, -5 / 3, -5 / 3, 2 / 3, 2 / 3, -10 / 3, -10 / 3],
                             dtype=jnp.float32)
        cme63_63 = jnp.array([4 / 3, -8 / 3, -5 / 3, -5 / 3, -5 / 3, -5 / 3, -10 / 3, -10 / 3, 2 / 3, 2 / 3],
                             dtype=jnp.float32)
        cme33_36 = jnp.array([0, 0, sqrt(2), -sqrt(2), -sqrt(2), sqrt(2), 0, 0, -2 * sqrt(2), 2 * sqrt(2)],
                             dtype=jnp.float32)
        cme33_63 = jnp.array([0, 0, -sqrt(2), sqrt(2), -sqrt(2), sqrt(2), 2 * sqrt(2), -2 * sqrt(2), 0, 0],
                             dtype=jnp.float32)
        cme36_63 = jnp.array([0, 0, -1, -1, 1, 1, 0, 0, 0, 0], dtype=jnp.float32)

        cmes = jnp.select([cond33_33, cond36_36, cond63_63, cond33_36, cond33_63, cond36_63],
                          [cme33_33, cme36_36, cme63_63, cme33_36, cme33_63, cme36_63])
        cme = jnp.select(pair, cmes, jnp.nan)

        return cme


    def color_proportions(arr):
        color33 = jnp.array([0, 0, 1], dtype=arr.dtype)
        color36 = jnp.array([0, 1, 0], dtype=arr.dtype)
        color63 = jnp.array([1, 0, 0], dtype=arr.dtype)

        mask1 = jnp.all(arr == color33, axis=-1)
        mask2 = jnp.all(arr == color36, axis=-1)
        mask3 = jnp.all(arr == color63, axis=-1)

        n1 = jnp.sum(mask1)
        n2 = jnp.sum(mask2)
        n3 = jnp.sum(mask3)

        ntotal = arr.shape[0] * arr.shape[1]

        return n1 / ntotal, n2 / ntotal, n3 / ntotal

"""Spin"""
if S == 0:
    def compute_spin_sign(spin):
        spin1_12 = jnp.all(spin == jnp.array([1, 0], dtype=spin.dtype), axis=-1)
        spin1_34 = jnp.all(spin == jnp.array([1, 0], dtype=spin.dtype), axis=-1)
        spin_sign12 = jnp.expand_dims(jnp.where(spin1_12, 1, -1), axis=0)
        spin_sign34 = jnp.expand_dims(jnp.where(spin1_34, 1, -1), axis=0)
        return spin_sign12, spin_sign34


    def spinfactor(i: Array, j: Array, s: Array, s_: Array):
        """spin factor <a|sisj|a_> with total spin 0
            :param
                i,j: i<j, particle indices
                s,s_: spin configuration,
                (0,1)->{[q1q2]^0[q3q4]^0}^0; (1,0)->{[q1q2]^1[q3q4]^1}^0
        """

        ij12 = jnp.array([0, 1], dtype=int)
        ij34 = jnp.array([2, 3], dtype=int)
        ij13 = jnp.array([0, 2], dtype=int)
        ij24 = jnp.array([1, 3], dtype=int)
        ij = jnp.array([i, j])

        pair12_34 = jnp.all(ij == ij12) | jnp.all(ij == ij34)
        pair13_24 = jnp.all(ij == ij13) | jnp.all(ij == ij24)

        spin00 = jnp.array([0, 1], dtype=s.dtype)
        spin11 = jnp.array([1, 0], dtype=s.dtype)

        cond00_00 = jnp.all(s == spin00, axis=-1) & jnp.all(s_ == spin00, axis=-1)
        cond11_11 = jnp.all(s == spin11, axis=-1) & jnp.all(s_ == spin11, axis=-1)

        sme00_00 = jnp.array([-3 / 4, 0, 0], dtype=jnp.float32)  # 12/34, 13/24, 14/23
        sme11_11 = jnp.array([1 / 4, -1 / 2, -1 / 2], dtype=jnp.float32)
        sme00_11 = jnp.array([0, -jnp.sqrt(3) / 4, jnp.sqrt(3) / 4], dtype=jnp.float32)

        smes = jnp.select([cond00_00, cond11_11], [sme00_00, sme11_11], sme00_11)
        sme = jnp.select([pair12_34, pair13_24], smes[:2], smes[-1])

        return sme


    def spin_proportions(arr):
        spin00 = jnp.array([0, 1], dtype=arr.dtype)
        spin11 = jnp.array([1, 0], dtype=arr.dtype)

        mask1 = jnp.all(arr == spin00, axis=-1)
        mask2 = jnp.all(arr == spin11, axis=-1)

        n1 = jnp.sum(mask1)
        n2 = jnp.sum(mask2)

        ntotal = arr.shape[0] * arr.shape[1]

        return n1 / ntotal, n2 / ntotal
elif S == 1 / 2:
    def compute_spin_sign(spin):
        spin0_12 = (jnp.all(spin == jnp.array([0, 0, 0, 0, 1], dtype=spin.dtype), axis=-1) |
                    jnp.all(spin == jnp.array([0, 0, 1, 0, 0], dtype=spin.dtype), axis=-1))

        spin0_34 = (jnp.all(spin == jnp.array([0, 0, 0, 0, 1], dtype=spin.dtype), axis=-1) |
                    jnp.all(spin == jnp.array([0, 1, 0, 0, 0], dtype=spin.dtype), axis=-1))
        spin_sign12 = jnp.expand_dims(jnp.where(spin0_12, -1, 1), axis=0)
        spin_sign34 = jnp.expand_dims(jnp.where(spin0_34, -1, 1), axis=0)
        return spin_sign12, spin_sign34


    def spinfactor(i: Array, j: Array, s: Array, s_: Array):
        """spin factor <a|sisj|a_> with total spin 0
            :param
                i,j: i<j, particle indices
                s,s_: spin configuration,
                (0,0,0,0,1)->{[q1q2]^0[q3q4]^0}^0; (0,0,0,1,0)->{[q1q2]^1[q3q4]^1}^0;
                (0,0,1,0,0)->{[q1q2]^0[q3q4]^1}^1; (0,1,0,0,0)->{[q1q2]^1[q3q4]^0}^1;
                (1,0,0,0,0)->{[q1q2]^1[q3q4]^1}^1
        """

        ijpairs = jnp.array([[0, 1], [2, 3], [0, 2], [1, 3], [0, 3], [1, 2], [0, 4], [1, 4], [2, 4], [3, 4]], dtype=int)
        ij = jnp.array([i, j])
        pair = jnp.all(ijpairs == ij, axis=1)

        spin000 = jnp.array([0, 0, 0, 0, 1], dtype=s.dtype)
        spin110 = jnp.array([0, 0, 0, 1, 0], dtype=s.dtype)
        spin011 = jnp.array([0, 0, 1, 0, 0], dtype=s.dtype)
        spin101 = jnp.array([0, 1, 0, 0, 0], dtype=s.dtype)
        spin111 = jnp.array([1, 0, 0, 0, 0], dtype=s.dtype)

        cond000_000 = jnp.all(s == spin000, axis=-1) & jnp.all(s_ == spin000, axis=-1)
        cond110_110 = jnp.all(s == spin110, axis=-1) & jnp.all(s_ == spin110, axis=-1)
        cond011_011 = jnp.all(s == spin011, axis=-1) & jnp.all(s_ == spin011, axis=-1)
        cond101_101 = jnp.all(s == spin101, axis=-1) & jnp.all(s_ == spin101, axis=-1)
        cond111_111 = jnp.all(s == spin111, axis=-1) & jnp.all(s_ == spin111, axis=-1)

        cond000_110 = (jnp.all(s == spin000, axis=-1) & jnp.all(s_ == spin110, axis=-1) |
                       jnp.all(s == spin110, axis=-1) & jnp.all(s_ == spin000, axis=-1))
        cond000_011 = (jnp.all(s == spin000, axis=-1) & jnp.all(s_ == spin011, axis=-1) |
                       jnp.all(s == spin011, axis=-1) & jnp.all(s_ == spin000, axis=-1))
        cond000_101 = (jnp.all(s == spin000, axis=-1) & jnp.all(s_ == spin101, axis=-1) |
                       jnp.all(s == spin101, axis=-1) & jnp.all(s_ == spin000, axis=-1))
        cond000_111 = (jnp.all(s == spin000, axis=-1) & jnp.all(s_ == spin111, axis=-1) |
                       jnp.all(s == spin111, axis=-1) & jnp.all(s_ == spin000, axis=-1))

        cond110_011 = (jnp.all(s == spin110, axis=-1) & jnp.all(s_ == spin011, axis=-1) |
                       jnp.all(s == spin011, axis=-1) & jnp.all(s_ == spin110, axis=-1))
        cond110_101 = (jnp.all(s == spin110, axis=-1) & jnp.all(s_ == spin101, axis=-1) |
                       jnp.all(s == spin101, axis=-1) & jnp.all(s_ == spin110, axis=-1))
        cond110_111 = (jnp.all(s == spin110, axis=-1) & jnp.all(s_ == spin111, axis=-1) |
                       jnp.all(s == spin111, axis=-1) & jnp.all(s_ == spin110, axis=-1))

        cond011_101 = (jnp.all(s == spin011, axis=-1) & jnp.all(s_ == spin101, axis=-1) |
                       jnp.all(s == spin101, axis=-1) & jnp.all(s_ == spin011, axis=-1))
        cond011_111 = (jnp.all(s == spin011, axis=-1) & jnp.all(s_ == spin111, axis=-1) |
                       jnp.all(s == spin111, axis=-1) & jnp.all(s_ == spin011, axis=-1))
        cond101_111 = (jnp.all(s == spin101, axis=-1) & jnp.all(s_ == spin111, axis=-1) |
                       jnp.all(s == spin111, axis=-1) & jnp.all(s_ == spin101, axis=-1))

        sme000_000 = jnp.array([-3 / 4, -3 / 4, 0, 0, 0, 0] + [0] * 4, dtype=jnp.float32)
        sme110_110 = jnp.array([1 / 4, 1 / 4, -1 / 2, -1 / 2, -1 / 2, -1 / 2] + [0] * 4, dtype=jnp.float32)
        sme011_011 = jnp.array([-3 / 4, 1 / 4, 0, 0, 0, 0, 0, 0, -1 / 2, -1 / 2], dtype=jnp.float32)
        sme101_101 = jnp.array([1 / 4, -3 / 4, 0, 0, 0, 0, -1 / 2, -1 / 2, 0, 0], dtype=jnp.float32)
        sme111_111 = jnp.array([1 / 4, 1 / 4, -1 / 4, -1 / 4, -1 / 4, -1 / 4, -1 / 4, -1 / 4, -1 / 4, -1 / 4],
                               dtype=jnp.float32)
        sme000_110 = jnp.array([0, 0, -sqrt(3) / 4, -sqrt(3) / 4, sqrt(3) / 4, sqrt(3) / 4] + [0] * 4,
                               dtype=jnp.float32)
        sme000_011 = jnp.array([0] * 6 + [0, 0, -sqrt(3) / 4, sqrt(3) / 4], dtype=jnp.float32)
        sme000_101 = jnp.array([0] * 6 + [-sqrt(3) / 4, sqrt(3) / 4, 0, 0], dtype=jnp.float32)
        sme000_111 = jnp.array([0] * 6 + [0] * 4, dtype=jnp.float32)
        sme110_011 = jnp.array([0] * 6 + [-1 / 4, 1 / 4, 0, 0], dtype=jnp.float32)
        sme110_101 = jnp.array([0] * 6 + [0, 0, -1 / 4, 1 / 4], dtype=jnp.float32)
        sme110_111 = jnp.array([0] * 6 + [-sqrt(2) / 4, -sqrt(2) / 4, sqrt(2) / 4, sqrt(2) / 4], dtype=jnp.float32)
        sme011_101 = jnp.array([0, 0, 1 / 4, 1 / 4, -1 / 4, -1 / 4] + [0] * 4, dtype=jnp.float32)
        sme011_111 = jnp.array([0, 0, -1 / (2 * sqrt(2)), 1 / (2 * sqrt(2)), -1 / (2 * sqrt(2)), 1 / (2 * sqrt(2)),
                                -sqrt(2) / 4, sqrt(2) / 4, 0, 0], dtype=jnp.float32)
        sme101_111 = jnp.array([0, 0, 1 / (2 * sqrt(2)), -1 / (2 * sqrt(2)), -1 / (2 * sqrt(2)), 1 / (2 * sqrt(2)),
                                0, 0, sqrt(2) / 4, -sqrt(2) / 4], dtype=jnp.float32)

        smes = jnp.select(
            [cond000_000, cond110_110, cond011_011, cond101_101, cond111_111, cond000_110, cond000_011, cond000_101,
             cond000_111, cond110_011, cond110_101, cond110_111, cond011_101, cond011_111, cond101_111],
            [sme000_000, sme110_110, sme011_011, sme101_101, sme111_111, sme000_110, sme000_011, sme000_101,
             sme000_111, sme110_011, sme110_101, sme110_111, sme011_101, sme011_111, sme101_111])
        sme = jnp.select(pair, smes, jnp.nan)

        return sme


    def spin_proportions(arr):
        spin000 = jnp.array([0, 0, 0, 0, 1], dtype=arr.dtype)
        spin110 = jnp.array([0, 0, 0, 1, 0], dtype=arr.dtype)
        spin011 = jnp.array([0, 0, 1, 0, 0], dtype=arr.dtype)
        spin101 = jnp.array([0, 1, 0, 0, 0], dtype=arr.dtype)
        spin111 = jnp.array([1, 0, 0, 0, 0], dtype=arr.dtype)

        mask1 = jnp.all(arr == spin000, axis=-1)
        mask2 = jnp.all(arr == spin110, axis=-1)
        mask3 = jnp.all(arr == spin011, axis=-1)
        mask4 = jnp.all(arr == spin101, axis=-1)
        mask5 = jnp.all(arr == spin111, axis=-1)

        n1 = jnp.sum(mask1)
        n2 = jnp.sum(mask2)
        n3 = jnp.sum(mask3)
        n4 = jnp.sum(mask4)
        n5 = jnp.sum(mask5)

        ntotal = arr.shape[0] * arr.shape[1]

        return n1 / ntotal, n2 / ntotal, n3 / ntotal, n4 / ntotal, n5 / ntotal
elif S == 1:
    def compute_spin_sign(spin):
        spin0_12 = jnp.all(spin == jnp.array([0, 0, 1], dtype=spin.dtype), axis=-1)
        spin0_34 = jnp.all(spin == jnp.array([0, 1, 0], dtype=spin.dtype), axis=-1)
        spin_sign12 = jnp.expand_dims(jnp.where(spin0_12, -1, 1), axis=0)
        spin_sign34 = jnp.expand_dims(jnp.where(spin0_34, -1, 1), axis=0)
        return spin_sign12, spin_sign34


    def spinfactor(i: Array, j: Array, s: Array, s_: Array):
        """spin factor <a|sisj|a_> with total spin 1
            :param
                i,j: i<j, particle indices
                s,s_: spin configuration,
                (0,0,1)->{[q1q2]^0[q3q4]^1}^1; (0,1,0)->{[q1q2]^1[q3q4]^0}^1; (1,0,0)->{[q1q2]^1[q3q4]^1}^1
        """

        ij12 = jnp.array([0, 1], dtype=int)
        ij34 = jnp.array([2, 3], dtype=int)
        ij13 = jnp.array([0, 2], dtype=int)
        ij24 = jnp.array([1, 3], dtype=int)
        ij14 = jnp.array([0, 3], dtype=int)
        ij = jnp.array([i, j])

        pair12 = jnp.all(ij == ij12)
        pair34 = jnp.all(ij == ij34)
        pair13 = jnp.all(ij == ij13)
        pair24 = jnp.all(ij == ij24)
        pair14 = jnp.all(ij == ij14)

        spin01 = jnp.array([0, 0, 1], dtype=s.dtype)
        spin10 = jnp.array([0, 1, 0], dtype=s.dtype)
        spin11 = jnp.array([1, 0, 0], dtype=s.dtype)

        cond01_01 = jnp.all(s == spin01, axis=-1) & jnp.all(s_ == spin01, axis=-1)
        cond10_10 = jnp.all(s == spin10, axis=-1) & jnp.all(s_ == spin10, axis=-1)
        cond11_11 = jnp.all(s == spin11, axis=-1) & jnp.all(s_ == spin11, axis=-1)
        cond11_10 = (jnp.all(s == spin11, axis=-1) & jnp.all(s_ == spin10, axis=-1)) | \
                    (jnp.all(s == spin10, axis=-1) & jnp.all(s_ == spin11, axis=-1))
        cond11_01 = (jnp.all(s == spin11, axis=-1) & jnp.all(s_ == spin01, axis=-1)) | \
                    (jnp.all(s == spin01, axis=-1) & jnp.all(s_ == spin11, axis=-1))

        sme01_01 = jnp.array([-3 / 4, 1 / 4, 0, 0, 0, 0], dtype=jnp.float32)  # 12,34,13,24,14,23
        sme10_10 = jnp.array([1 / 4, -3 / 4, 0, 0, 0, 0], dtype=jnp.float32)
        sme11_11 = jnp.array([1 / 4, 1 / 4, -1 / 4, -1 / 4, -1 / 4, -1 / 4], dtype=jnp.float32)
        sme11_10 = jnp.array(
            [0, 0, 1 / (2 * jnp.sqrt(2)), -1 / (2 * jnp.sqrt(2)), -1 / (2 * jnp.sqrt(2)), 1 / (2 * jnp.sqrt(2))],
            dtype=jnp.float32)
        sme11_01 = jnp.array(
            [0, 0, -1 / (2 * jnp.sqrt(2)), 1 / (2 * jnp.sqrt(2)), -1 / (2 * jnp.sqrt(2)), 1 / (2 * jnp.sqrt(2))],
            dtype=jnp.float32)
        sme10_01 = jnp.array([0, 0, 1 / 4, 1 / 4, -1 / 4, -1 / 4], dtype=jnp.float32)

        smes = jnp.select([cond01_01, cond10_10, cond11_11, cond11_10, cond11_01],
                          [sme01_01, sme10_10, sme11_11, sme11_10, sme11_01], sme10_01)
        sme = jnp.select([pair12, pair34, pair13, pair24, pair14], smes[:5], smes[-1])

        return sme


    def spin_proportions(arr):
        spin01 = jnp.array([0, 0, 1], dtype=arr.dtype)
        spin10 = jnp.array([0, 1, 0], dtype=arr.dtype)
        spin11 = jnp.array([1, 0, 0], dtype=arr.dtype)

        mask1 = jnp.all(arr == spin01, axis=-1)
        mask2 = jnp.all(arr == spin10, axis=-1)
        mask3 = jnp.all(arr == spin11, axis=-1)

        n1 = jnp.sum(mask1)
        n2 = jnp.sum(mask2)
        n3 = jnp.sum(mask3)

        ntotal = arr.shape[0] * arr.shape[1]

        return n1 / ntotal, n2 / ntotal, n3 / ntotal
elif S == 3 / 2:
    def compute_spin_sign(spin):
        spin0_12 = jnp.all(spin == jnp.array([0, 0, 0, 1], dtype=spin.dtype), axis=-1)
        spin0_34 = jnp.all(spin == jnp.array([0, 0, 1, 0], dtype=spin.dtype), axis=-1)
        spin_sign12 = jnp.expand_dims(jnp.where(spin0_12, -1, 1), axis=0)
        spin_sign34 = jnp.expand_dims(jnp.where(spin0_34, -1, 1), axis=0)
        return spin_sign12, spin_sign34


    def spinfactor(i: Array, j: Array, s: Array, s_: Array):
        """spin factor <a|sisj|a_> with total spin 1
            :param
                i,j: i<j, particle indices
                s,s_: spin configuration,
                (0,0,0,1)->{[q1q2]^0[q3q4]^1}^1; (0,0,1,0)->{[q1q2]^1[q3q4]^0}^1;
                (0,1,0,0)->{[q1q2]^1[q3q4]^1}^1; (1,0,0,0)->{[q1q2]^1[q3q4]^1}^2;
        """

        ijpairs = jnp.array([[0, 1], [2, 3], [0, 2], [1, 3], [0, 3], [1, 2], [0, 4], [1, 4], [2, 4], [3, 4]], dtype=int)
        ij = jnp.array([i, j])
        pair = jnp.all(ijpairs == ij, axis=1)

        spin011 = jnp.array([0, 0, 0, 1], dtype=s.dtype)
        spin101 = jnp.array([0, 0, 1, 0], dtype=s.dtype)
        spin111 = jnp.array([0, 1, 0, 0], dtype=s.dtype)
        spin112 = jnp.array([1, 0, 0, 0], dtype=s.dtype)

        cond011_011 = jnp.all(s == spin011, axis=-1) & jnp.all(s_ == spin011, axis=-1)
        cond101_101 = jnp.all(s == spin101, axis=-1) & jnp.all(s_ == spin101, axis=-1)
        cond111_111 = jnp.all(s == spin111, axis=-1) & jnp.all(s_ == spin111, axis=-1)
        cond112_112 = jnp.all(s == spin112, axis=-1) & jnp.all(s_ == spin112, axis=-1)

        cond011_101 = (jnp.all(s == spin011, axis=-1) & jnp.all(s_ == spin101, axis=-1)) | \
                      (jnp.all(s == spin101, axis=-1) & jnp.all(s_ == spin011, axis=-1))
        cond011_111 = (jnp.all(s == spin011, axis=-1) & jnp.all(s_ == spin111, axis=-1)) | \
                      (jnp.all(s == spin111, axis=-1) & jnp.all(s_ == spin011, axis=-1))
        cond011_112 = (jnp.all(s == spin011, axis=-1) & jnp.all(s_ == spin112, axis=-1)) | \
                      (jnp.all(s == spin112, axis=-1) & jnp.all(s_ == spin011, axis=-1))

        cond101_111 = (jnp.all(s == spin101, axis=-1) & jnp.all(s_ == spin111, axis=-1)) | \
                      (jnp.all(s == spin111, axis=-1) & jnp.all(s_ == spin101, axis=-1))
        cond101_112 = (jnp.all(s == spin101, axis=-1) & jnp.all(s_ == spin112, axis=-1)) | \
                      (jnp.all(s == spin112, axis=-1) & jnp.all(s_ == spin101, axis=-1))

        cond111_112 = (jnp.all(s == spin111, axis=-1) & jnp.all(s_ == spin112, axis=-1)) | \
                      (jnp.all(s == spin112, axis=-1) & jnp.all(s_ == spin111, axis=-1))

        sme011_011 = jnp.array([-3 / 4, 1 / 4, 0, 0, 0, 0, 0, 0, 1 / 4, 1 / 4],
                               dtype=jnp.float32)  # 12,34,13,24,14,23,15,25,35,45
        sme101_101 = jnp.array([1 / 4, -3 / 4, 0, 0, 0, 0, 1 / 4, 1 / 4, 0, 0], dtype=jnp.float32)
        sme111_111 = jnp.array([1 / 4, 1 / 4, -1 / 4, -1 / 4, -1 / 4, -1 / 4, 1 / 8, 1 / 8, 1 / 8, 1 / 8],
                               dtype=jnp.float32)
        sme112_112 = jnp.array([1 / 4] * 6 + [-3 / 8] * 4, dtype=jnp.float32)
        sme011_101 = jnp.array([0, 0, 1 / 4, 1 / 4, -1 / 4, -1 / 4] + [0] * 4, dtype=jnp.float32)
        sme011_111 = jnp.array([0, 0, -1 / (2 * sqrt(2)), 1 / (2 * sqrt(2)), -1 / (2 * sqrt(2)), 1 / (2 * sqrt(2)),
                                -sqrt(2) / 8, sqrt(2) / 8, 0, 0], dtype=jnp.float32)
        sme011_112 = jnp.array([0] * 6 + [-sqrt(10) / 8, sqrt(10) / 8, 0, 0], dtype=jnp.float32)
        sme101_111 = jnp.array([0, 0, 1 / (2 * sqrt(2)), -1 / (2 * sqrt(2)), -1 / (2 * sqrt(2)), 1 / (2 * sqrt(2)),
                                0, 0, sqrt(2) / 8, -sqrt(2) / 8], dtype=jnp.float32)
        sme101_112 = jnp.array([0] * 6 + [0, 0, -sqrt(10) / 8, sqrt(10) / 8], dtype=jnp.float32)
        sme111_112 = jnp.array([0] * 6 + [-sqrt(5) / 8, -sqrt(5) / 8, sqrt(5) / 8, sqrt(5) / 8],
                               dtype=jnp.float32)

        smes = jnp.select(
            [cond011_011, cond101_101, cond111_111, cond112_112, cond011_101, cond011_111, cond011_112, cond101_111,
             cond101_112, cond111_112],
            [sme011_011, sme101_101, sme111_111, sme112_112, sme011_101, sme011_111, sme011_112, sme101_111,
             sme101_112, sme111_112])
        sme = jnp.select(pair, smes, jnp.nan)

        return sme


    def spin_proportions(arr):
        spin011 = jnp.array([0, 0, 0, 1], dtype=arr.dtype)
        spin101 = jnp.array([0, 0, 1, 0], dtype=arr.dtype)
        spin111 = jnp.array([0, 1, 0, 0], dtype=arr.dtype)
        spin112 = jnp.array([1, 0, 0, 0], dtype=arr.dtype)

        mask1 = jnp.all(arr == spin011, axis=-1)
        mask2 = jnp.all(arr == spin101, axis=-1)
        mask3 = jnp.all(arr == spin111, axis=-1)
        mask4 = jnp.all(arr == spin112, axis=-1)

        n1 = jnp.sum(mask1)
        n2 = jnp.sum(mask2)
        n3 = jnp.sum(mask3)
        n4 = jnp.sum(mask4)

        ntotal = arr.shape[0] * arr.shape[1]

        return n1 / ntotal, n2 / ntotal, n3 / ntotal, n4 / ntotal
elif S == 2 or S == 5 / 2:
    def compute_spin_sign(spin):
        spin1_12 = jnp.all(spin == jnp.array([1], dtype=spin.dtype), axis=-1)
        spin1_34 = jnp.all(spin == jnp.array([1], dtype=spin.dtype), axis=-1)
        spin_sign12 = jnp.expand_dims(jnp.where(spin1_12, 1, -1), axis=0)
        spin_sign34 = jnp.expand_dims(jnp.where(spin1_34, 1, -1), axis=0)
        return spin_sign12, spin_sign34


    def spinfactor(i: Array, j: Array, s: Array, s_: Array):
        """spin factor <a|sisj|a_> with total spin 2
            :param
                i,j: i<j, particle indices
                s,s_: spin configuration, (1)->{[q1q2]^1[q3q4]^1}^2
        """

        return 1 / 4


    def spin_proportions(arr):

        return 1


def ms_radius(i, j, a, a_, r):
    """mean-square radius <rij^2>
        :param
            i,j: int, particle indices
            a, a_: Array, color-spin configuration
            r: four-body coordinates
    """

    nd = 3
    n_particles = 4
    r = r.reshape(n_particles, nd)
    ri, rj = r[i], r[j]
    rij = jnp.linalg.norm(ri - rj, axis=-1) * 0.197  # unit:fm

    return jnp.where(jnp.all(a == a_), rij ** 2, 0)
