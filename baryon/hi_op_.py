import netket as nk
import jax
import jax.numpy as jnp
from jax.numpy import pi, exp
from netket.utils.types import Array
from netket.utils import struct

from config_ import quarks, MASS, PARA, S, I
from operator_ import MultiPotentialEnergy

"""Hilbert Space"""
_nd, _nparticles = 3, 3
dof_spin = {1 / 2: 2, 3 / 2: 1}[S]

if quarks[0] == quarks[1] == quarks[2] == 'q':
    dof_fla = {1 / 2: 2, 3 / 2: 1}[I]
    hi_spin = nk.hilbert.Fock(N=dof_spin, n_particles=1)
    hi_flav = nk.hilbert.Fock(N=dof_fla, n_particles=1)
    hi_dis = hi_spin * hi_flav
else:
    hi_spin = nk.hilbert.Fock(N=dof_spin, n_particles=1)
    hi_dis = hi_spin

hi_con = nk.hilbert.Particle(N=_nparticles, L=(jnp.inf, jnp.inf, jnp.inf))
hi = hi_con * hi_dis
n_discrete = hi_dis.size

"""Operator"""
_mass = []
for m in MASS:
    _mass += [m] * _nd
_mass += [jnp.inf] * n_discrete
kin = nk.operator.KineticEnergy(hi, mass=_mass)


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



_mass = []
for m in MASS:
    _mass += [m] * _nd
_mass += [jnp.inf] * n_discrete
kin = nk.operator.KineticEnergy(hi, mass=_mass)
mass = jnp.array(MASS)
para = CQM_PotentialPara(**PARA)


def baryon_potential(a, a_, r):
    """Baryon potential in AL1/flux-tube model
        Args:
            a: color-spin configuration of bra state
            a_: color-spin configuration of ket state
            r: 3-body coordinates
    """
    nd = 3
    n_particles = 3
    r = r.reshape(n_particles, nd)
    i, j = jnp.triu_indices(n_particles, k=1)
    s, s_ = a[:2], a_[:2]
    f, f_ = a[2:], a_[2:]
    ri, rj = r[i], r[j]

    def _vij(_i: Array, _j: Array, xi: Array, xj: Array):
        return vij(_i, _j, xi, xj, s, s_, f, f_)

    vij_inputs = (i, j, ri, rj)
    pot = jax.vmap(_vij)(*vij_inputs)

    diag = jnp.all(a == a_)
    vconf = jnp.where(diag, para.l / 2 * jnp.sum(jnp.linalg.norm(ri - rj, axis=-1)), 0)  # AL1 delta-type
    # vconf = jnp.where(diag, para.l * 0.9204 * compute_Lmin(r), 0)  # Y-type
    massterm = jnp.where(diag, jnp.sum(jnp.array(MASS)), 0)
    return jnp.sum(pot) + massterm + vconf


def vij(i: Array, j: Array, xi: Array, xj: Array, s: Array, s_: Array, f: Array, f_: Array):
    """Two-body AL1 potential vij
        :param
            i,j: int, i<j, particle indices
            xi,xj: Array, shape=(nd,), spatial coordinates of i and j
            s,s_: Array, spin configuration
            para: PyTree, parameters of potential
    """

    cme = -8 / 3
    sme = spinfactor(i, j, s, s_)
    rij = jnp.linalg.norm(xi - xj, axis=-1)

    mi, mj = mass[i], mass[j]
    r0 = para.A * (2 * mi * mj / (mi + mj)) ** (-para.B)

    v0 = jnp.where(jnp.all(s == s_), -3 / 16 * cme * (-para.k / rij - para.L), 0)
    v1 = -3 / 16 * cme * (8 * pi * para.k_ / (3 * mi * mj)
                          * exp(-rij ** 2 / r0 ** 2) / (pi ** (3 / 2) * r0 ** 3) * sme)
    vme = jnp.where(jnp.all(f == f_), v0 + v1, 0)

    return vme


mulpot = MultiPotentialEnergy(hi, hi_dis, baryon_potential)
ha = mulpot + kin


def compute_Lmin(points):
    """
    :param points: shape=(3,nd)
    :return: Lmin: float, minimal length of flux tubes
    """

    A, B, C = points

    a = jnp.linalg.norm(B - C)
    b = jnp.linalg.norm(A - C)
    c = jnp.linalg.norm(A - B)

    cos_alpha = (b ** 2 + c ** 2 - a ** 2) / (2 * b * c + 1e-8)
    cos_beta = (a ** 2 + c ** 2 - b ** 2) / (2 * a * c + 1e-8)
    cos_gamma = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b + 1e-8)

    alpha = jnp.arccos(jnp.clip(cos_alpha, -1.0, 1.0))
    beta = jnp.arccos(jnp.clip(cos_beta, -1.0, 1.0))
    gamma = jnp.arccos(jnp.clip(cos_gamma, -1.0, 1.0))

    has_large_angle = jnp.any(jnp.array([alpha, beta, gamma]) > 2 * jnp.pi / 3)

    def compute_without_large_angle():
        s = a + b + c
        inner = 0.5 * (a ** 2 + b ** 2 + c ** 2) + (jnp.sqrt(3) / 2) * jnp.sqrt(
            s * (s - 2 * a) * (s - 2 * b) * (s - 2 * c))
        return jnp.sqrt(jnp.maximum(inner, 0))  # 确保非负

    def compute_with_large_angle():
        return a + b + c - jnp.max(jnp.array([a, b, c]))

    return jax.lax.cond(
        has_large_angle,
        compute_with_large_angle,
        compute_without_large_angle
    )


if S == 3 / 2:
    def spinfactor(i: Array, j: Array, s: Array, s_: Array):
        """spin factor <a|sisj|a_> with total spin 2
            :param
                i,j: int, i<j, particle indices
                s,s_: Array, spin configuration, (1)->{[q1q2]^1[q3q4]^1}^2
        """

        return 1 / 4


    def spin_proportions(arr):

        return 1


elif S == 1 / 2:
    def spinfactor(i: Array, j: Array, s: Array, s_: Array):
        """spin factor <a|sisj|a_> with total spin 0
            :param
                i,j: int, i<j, particle indices
                s,s_: Array, spin configuration,
                (0,1)->{[q1q2]^0q3}^1/2; (1,0)->{[q1q2]^1q3}^1/2
        """

        ijpairs = jnp.array([[0, 1], [0, 2], [1, 2]], dtype=int)
        ij = jnp.array([i, j])
        pair = jnp.all(ij == ijpairs, axis=1)

        spin0 = jnp.array([0, 1], dtype=s.dtype)
        spin1 = jnp.array([1, 0], dtype=s.dtype)

        cond00 = jnp.all(s == spin0, axis=-1) & jnp.all(s_ == spin0, axis=-1)
        cond11 = jnp.all(s == spin1, axis=-1) & jnp.all(s_ == spin1, axis=-1)

        sme00 = jnp.array([-3 / 4, 0, 0], dtype=jnp.float32)  # 12,13,23
        sme11 = jnp.array([1 / 4, -1 / 2, -1 / 2], dtype=jnp.float32)
        sme10 = jnp.array([0, -jnp.sqrt(3) / 4, jnp.sqrt(3) / 4], dtype=jnp.float32)

        smes = jnp.select([cond00, cond11], [sme00, sme11], sme10)
        sme = jnp.select(pair, smes, jnp.nan)

        return sme


    def spin_proportions(arr):
        spin0 = jnp.array([0, 1], dtype=arr.dtype)
        spin1 = jnp.array([1, 0], dtype=arr.dtype)

        mask1 = jnp.all(arr == spin0, axis=-1)
        mask2 = jnp.all(arr == spin1, axis=-1)

        n1 = jnp.sum(mask1)
        n2 = jnp.sum(mask2)

        ntotal = arr.shape[0] * arr.shape[1]

        return n1 / ntotal, n2 / ntotal
