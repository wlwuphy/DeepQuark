from flax import linen as nn
import jax.numpy as jnp
from jax.numpy import sqrt
import jax
from functools import partial
from itertools import permutations
from netket.utils.types import PyTree


class QQQ_FCN(nn.Module):
    """Wavefunction for baryons with three identical quarks, I=1/2, S=1/2, P=+
    input (x1,y1,z1,...,xn,yn,zn,rij) -> FCN -> output (logpsi)

    Arguments：
        nd: dimension
        nparticles: particle number
        mass: particle mass
        nhid: number of nodes
        nlayers: number of layers
        bound: boundary condition
        hi: discrete hilbert space
        indices: (Anti)symmetriazation exchange indices
        signs: (Anti)symmetriazation exchange signs
    """
    nd: int
    nparticles: int
    mass: tuple
    nhid: int
    nlayers: int
    bound: float
    hi: PyTree
    indices: tuple
    signs: tuple

    def setup(self):
        """Setup layers"""

        self.layers = [nn.Dense(self.nhid)]
        self.layers += tuple([nn.Dense(self.nhid) for _ in range(self.nlayers - 1)])
        self.layers += tuple([nn.Dense(1)])

    @partial(jax.vmap, in_axes=(None, None, 0, 0))
    def process_term(self, spatialx, indice, discretex):
        spatialxi = spatialx[:, indice]
        spatialxi_flat = spatialxi.reshape(spatialxi.shape[0], -1)
        rij, bc = self._boundary_condition(spatialxi_flat)  # rij: (batch, npairs), bc: (batch,)

        features = jnp.concatenate([spatialxi_flat, rij, discretex], axis=-1)

        for layer in self.layers[:-1]:
            features = nn.tanh(layer(features))
        output = self.layers[-1](features).squeeze()  # (batch,)

        return bc * output

    def psi(self, x: jnp.ndarray):
        """
        x.shape = (batch, nparticles*nd+ndiscrete)
        Output: logpsi.shape = (batch,)
        """
        hilbert = self.hi.hilb
        n_dis = hilbert.size

        indices, signs = jnp.array(self.indices), jnp.array(self.signs)
        n_perm, n_id = indices.shape[0], indices.shape[1]
        indices = jnp.broadcast_to(jnp.expand_dims(indices, axis=1),
                                   (n_perm, hilbert.n_states, n_id))
        dis_states = jnp.broadcast_to(jnp.expand_dims(hilbert.all_states(), axis=(0, 2)),
                                      (n_perm, hilbert.n_states, x.shape[0], n_dis))
        indices = jnp.reshape(indices, (n_perm * hilbert.n_states, n_id))
        dis_states = jnp.reshape(dis_states, (n_perm * hilbert.n_states, x.shape[0], n_dis))

        spatialx = jnp.reshape(x[:, :-n_dis], (-1, self.nparticles, self.nd))  # spatial dof

        outputs = self.process_term(spatialx, indices, dis_states)  # outputs: (n_per*n_states, batch)
        outputs = jnp.reshape(outputs, (n_perm, hilbert.n_states, x.shape[0]))

        # # spin
        # P12 = jnp.array([[-1, 0], [0, 1]])
        # Pn = jnp.array([[-1 / 2, sqrt(3) / 2], [-sqrt(3) / 2, -1 / 2]])
        # Iden = jnp.array([[1, 0], [0, 1]])

        # spin-flavor
        P12 = jnp.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        Pn = jnp.array([[1 / 4, -sqrt(3) / 4, -sqrt(3) / 4, 3 / 4],
                        [sqrt(3) / 4, 1 / 4, -3 / 4, -sqrt(3) / 4],
                        [sqrt(3) / 4, -3 / 4, 1 / 4, -sqrt(3) / 4],
                        [3 / 4, sqrt(3) / 4, sqrt(3) / 4, 1 / 4]])
        Iden = jnp.eye(4)
        recouple_matrices = jnp.array([Iden, P12 @ Pn @ Pn, P12, Pn, Pn @ Pn, P12 @ Pn])

        # change notations of discretex to (1,0,0,0),(0,1,0,0),..., so that the recoupling coefficients are in order
        discretex = x[:, -n_dis:]
        matches = jnp.all(discretex[:, None, :] == hilbert.all_states(), axis=-1)  # (batch, nstates)
        discretex = jnp.where(matches.any(axis=1, keepdims=True),
                              jax.nn.one_hot(jnp.argmax(matches, axis=1), hilbert.n_states), jnp.nan)

        recouple_coeff = jnp.einsum('jki,ni->jkn', recouple_matrices, discretex)  # (nper,nstates,batch)

        psi = jnp.sum(outputs * recouple_coeff * jnp.expand_dims(signs, axis=(1, 2)), axis=(0, 1))  # (batch,)

        return psi

    def _boundary_condition(self, x: jnp.ndarray):
        """Exponetial boundary condition"""

        rij = compute_rij(x, self.nd)  # shape=(batch, n_pairs)
        bc = jnp.exp(-jnp.sum(rij ** 2 / self.bound ** 2, axis=-1))

        return rij, bc

    def __call__(self, x: jnp.ndarray):
        n_dis = self.hi.hilb.size
        spatialx = x[:, :-n_dis]
        discretex = x[:, -n_dis:]
        xcm = center_of_mass(spatialx, jnp.array(self.mass))
        spatialx -= xcm

        x = jnp.concatenate([spatialx, discretex], axis=-1)
        Px = jnp.concatenate([-spatialx, discretex], axis=-1)

        inputs = jnp.stack([x, Px], axis=0)
        Psi = jnp.sum(jax.vmap(self.psi)(inputs), axis=0)
        logpsi = jnp.log(Psi.astype(complex))
        return logpsi


class QQq_FCN(nn.Module):
    """Wavefunction for baryons with two identical quarks QQq
    input (x1,y1,z1,...,xn,yn,zn,rij) -> FCN -> output (logpsi)

    Arguments：
        nd: dimension
        nparticles: particle number
        mass: particle mass
        nhid: number of nodes
        nlayers: number of layers
        bound: boundary condition
        indices: (Anti)symmetriazation exchange indices
        signs: (Anti)symmetriazation exchange signs
    """
    nd: int
    nparticles: int
    mass: tuple
    ndiscrete: int
    nhid: int
    nlayers: int
    bound: float
    indices: tuple
    signs: tuple

    def setup(self):
        """Setup layers"""

        self.layers = [nn.Dense(self.nhid)]
        self.layers += tuple([nn.Dense(self.nhid) for _ in range(self.nlayers - 1)])
        self.layers += tuple([nn.Dense(1)])

    @partial(jax.vmap, in_axes=(None, None, None, 0))
    def process_term(self, spatialx, discretex, indice):
        spatialxi = spatialx[:, indice]
        spatialxi_flat = spatialxi.reshape(spatialxi.shape[0], -1)
        rij, bc = self._boundary_condition(spatialxi_flat)  # rij: (batch, npairs), bc: (batch,)

        features = jnp.concatenate([spatialxi_flat, rij, discretex], axis=-1)

        for layer in self.layers[:-1]:
            features = nn.tanh(layer(features))
        output = self.layers[-1](features).squeeze()  # (batch,)

        return bc * output

    def psi(self, x: jnp.ndarray):
        """
        x.shape = (batch, nparticles*nd+ndiscrete)
        Output: logpsi.shape = (batch,)
        """
        discretex = x[:, -self.ndiscrete:]  # discrete dof
        spin, flav = discretex[:, :2], discretex[:, 2:]


        # spin1 = jnp.all(spin == jnp.array([1,0], dtype=spin.dtype), axis=-1)  # S=1/2
        spin1 = jnp.all(spin == jnp.array([1], dtype=spin.dtype), axis=-1)  # S=3/2
        spin_sign = jnp.expand_dims(jnp.where(spin1, 1, -1), axis=0)

        identity = jnp.ones_like(spin_sign)
        spin_sign = jnp.concatenate([identity, -spin_sign], axis=0)  # color antisymmetry

        spatialx = jnp.reshape(x[:, :-self.ndiscrete], (-1, self.nparticles, self.nd))  # spatial dof
        indices, signs = jnp.array(self.indices), jnp.array(self.signs)
        signs = jnp.broadcast_to(jnp.expand_dims(signs, axis=1),
                                 (indices.shape[0], x.shape[0]))
        outputs = self.process_term(spatialx, discretex, indices)  # outputs: (exchange_terms, batch)
        spatial_signs = signs * spin_sign

        psi = jnp.sum(outputs * spatial_signs, axis=0)  # (batch,)
        return psi

    def _boundary_condition(self, x: jnp.ndarray):
        """Exponetial boundary condition"""

        rij = compute_rij(x, self.nd)  # shape=(batch, n_pairs)
        bc = jnp.exp(-jnp.sum(rij ** 2 / self.bound ** 2, axis=-1))

        return rij, bc

    def __call__(self, x: jnp.ndarray):
        spatialx = x[:, :-self.ndiscrete]
        discretex = x[:, -self.ndiscrete:]
        xcm = center_of_mass(spatialx, jnp.array(self.mass))
        spatialx -= xcm

        x = jnp.concatenate([spatialx, discretex], axis=-1)
        Px = jnp.concatenate([-spatialx, discretex], axis=-1)

        inputs = jnp.stack([x, Px], axis=0)
        psi = jnp.sum(jax.vmap(self.psi)(inputs), axis=0)
        logpsi = jnp.log(psi.astype(complex))

        return logpsi


class FCN(nn.Module):
    """FewBody Wavefunction
    input (x1,y1,z1,...,xn,yn,zn,rij) -> FCN -> output (logpsi)

    Arguments：
        nd: dimension
        nparticles: particle number
        ndiscrete: discrete d.o.f
        nhid: number of nodes
        nlayers: number of layers
        bound: boundary condition
    """
    nd: int
    nparticles: int
    mass: tuple
    ndiscrete: int
    nhid: int
    nlayers: int
    bound: float

    def setup(self):
        """Setup layers"""

        self.layers = [nn.Dense(self.nhid) for _ in range(self.nlayers)]
        self.layers += tuple([nn.Dense(1)])

    def psi(self, x: jnp.ndarray):
        """
        x.shape = (batch, nparticles*nd+ndiscrete)
        Output: logpsi.shape = (batch,)
        """

        spatialx = x[:, :-self.ndiscrete]  # spatial dof
        rij, bc = self._boundary_condition(spatialx)
        features = jnp.concatenate([x, rij], axis=-1)

        for layer in self.layers[:-1]:
            features = nn.tanh(layer(features))
        output = self.layers[-1](features).squeeze()

        return output * bc

    def __call__(self, x: jnp.ndarray):
        n_dis = self.ndiscrete
        spatialx = x[:, :-n_dis]
        discretex = x[:, -n_dis:]
        xcm = center_of_mass(spatialx, jnp.array(self.mass))
        spatialx -= xcm

        x = jnp.concatenate([spatialx, discretex], axis=-1)
        Px = jnp.concatenate([-spatialx, discretex], axis=-1)

        inputs = jnp.stack([x, Px], axis=0)
        Psi = jnp.sum(jax.vmap(self.psi)(inputs), axis=0)
        logpsi = jnp.log(Psi.astype(complex))
        return logpsi

    def _boundary_condition(self, x: jnp.ndarray):
        """Exponetial boundary condition"""

        rij = compute_rij(x, self.nd)  # shape=(batch, n_pairs)
        bc = jnp.exp(-jnp.sum(rij ** 2 / self.bound ** 2, axis=-1))

        return rij, bc


def compute_rij(x, nd):
    """计算粒子间欧氏距离的上三角部分
    输入形状: (batch, nparticles*nd)
    输出形状: (batch, n_pairs)
    """

    if x.ndim == 1:  # input size (nparticles*nd)
        x = jnp.expand_dims(x, axis=0)

    n_particles = x.shape[-1] // nd
    x = x.reshape(-1, n_particles, nd)

    i, j = jnp.triu_indices(n_particles, k=1)
    delta = x[:, i] - x[:, j]  # shape=(batch, n_pairs, nd)
    rij = jnp.linalg.norm(delta, axis=-1)  # shape=(batch, n_pairs)

    return rij


def center_of_mass(x: jnp.ndarray, m: jnp.ndarray) -> jnp.ndarray:
    """
    Args:
        x: particle coordinates, shape=(nsamples, n*3)
        m: particle mass，shape=(n,)
    Returns:
        Xcm， shape=(nsamples, n*3)
    """

    x = x.reshape(-1, x.shape[1] // 3, 3)  # shape=(nsamples, n, 3)
    M = jnp.sum(m)
    m_expanded = m[None, :, None]  # shape=(1, n, 1)
    xcm = jnp.sum(x * m_expanded, axis=1, keepdims=True) / M  # shape=(nsamples, 1, 3)
    xcm = jnp.broadcast_to(xcm, x.shape)

    return xcm.reshape(x.shape[0], x.shape[1] * 3)


def generate_permutations(nparticles: int, identical_particles: list):
    group = identical_particles
    original = list(range(nparticles))
    all_perms = list(permutations(group))

    indices = []
    signs = []

    for p in all_perms:
        # Create a mapping from original group element to permutation element
        mapping = {g: p[i] for i, g in enumerate(group)}
        new_indices = [mapping[i] if i in mapping else i for i in original]
        indices.append(tuple(new_indices))

        # Calculate the sign based on the number of inversions in the permutation p
        inv_count = 0
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                if p[i] > p[j]:
                    inv_count += 1
        signs.append((-1) ** inv_count)


    return tuple(indices), tuple(signs)


def exchange_indices(nparticles: int, identical_particles: list):
    """(Anti)symmetrized indices"""

    def _create_swap_indices(indices_: list, i0: int, i1: int) -> list:
        """Exchange i0 and i1"""
        copy_indices_ = indices_[:]
        copy_indices_[i0], copy_indices_[i1] = indices_[i1], indices_[i0]
        return copy_indices_

    indices = list(range(nparticles))

    n_swaps = len(identical_particles)
    swap_indices_ = [indices]
    signs = [1]

    for swap in identical_particles:
        i0, i1, sign = swap
        swap_indices_.append(_create_swap_indices(indices, i0, i1))
        signs.append(sign)

    # 处理两个交换对的组合情况
    if n_swaps == 2:
        (i0a, i1a, signa), (i0b, i1b, signb) = identical_particles
        indices_a = _create_swap_indices(indices, i0a, i1a)
        indices_ab = _create_swap_indices(indices_a, i0b, i1b)
        swap_indices_.append(indices_ab)
        signs.append(signa * signb)

    swap_indices_ = tuple(tuple(i) for i in swap_indices_)
    signs = tuple(signs)

    return swap_indices_, signs
