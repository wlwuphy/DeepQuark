from flax import linen as nn
import jax.numpy as jnp
import jax
from functools import partial
from hi_op_ import compute_spin_sign


class QQqq_FCN(nn.Module):
    """QQq-q- Tetraquark Wavefunction, P=+
    input (x1,y1,z1,...,xn,yn,zn,rij) -> FCN -> output (logpsi)

    Arguments：
        nd: dimension
        nparticles: particle number
        mass: particle mass
        ndiscrete: discrete d.o.f
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

        self.layers = [nn.Dense(self.nhid * 2)]
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
        color, spin = discretex[:, :2], discretex[:, 2:]

        color66 = jnp.all(color == jnp.array([1, 0], dtype=color.dtype), axis=-1)
        color_sign = jnp.expand_dims(jnp.where(color66, 1, -1), axis=0)  # (1,batch)

        spin_sign12, spin_sign34 = compute_spin_sign(spin)

        identity = jnp.ones_like(color_sign)
        color_sign = jnp.concatenate([identity, color_sign, color_sign, identity], axis=0)  # (exchange_terms, batch)
        spin_sign = jnp.concatenate([identity, spin_sign12, spin_sign34, spin_sign12 * spin_sign34], axis=0)

        spatialx = jnp.reshape(x[:, :-self.ndiscrete], (-1, self.nparticles, self.nd))  # spatial dof
        indices, signs = jnp.array(self.indices), jnp.array(self.signs)
        signs = jnp.broadcast_to(jnp.expand_dims(signs, axis=1),
                                 (indices.shape[0], x.shape[0]))
        outputs = self.process_term(spatialx, discretex, indices)  # outputs: (exchange_terms, batch)
        spatial_signs = signs * color_sign * spin_sign  # (exchange_terms, batch)

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


class Q1Q2qq_FCN(nn.Module):
    """Q1Q2q-q- Tetraquark Wavefunction, P=+
    input (x1,y1,z1,...,xn,yn,zn,rij) -> FCN -> output (logpsi)

    Arguments：
        nd: dimension
        nparticles: particle number
        mass: particle mass
        ndiscrete: discrete d.o.f
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

        self.layers = [nn.Dense(self.nhid * 2)]
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
        color, spin = discretex[:, :2], discretex[:, 2:]

        color66 = jnp.all(color == jnp.array([1, 0], dtype=color.dtype), axis=-1)
        color_sign = jnp.expand_dims(jnp.where(color66, 1, -1), axis=0)  # (1,batch)

        spin_sign = compute_spin_sign(spin)[1]

        identity = jnp.ones_like(color_sign)
        color_sign = jnp.concatenate([identity, color_sign], axis=0)  # (exchange_terms, batch)
        spin_sign = jnp.concatenate([identity, spin_sign], axis=0)

        spatialx = jnp.reshape(x[:, :-self.ndiscrete], (-1, self.nparticles, self.nd))  # spatial dof
        indices, signs = jnp.array(self.indices), jnp.array(self.signs)
        signs = jnp.broadcast_to(jnp.expand_dims(signs, axis=1),
                                 (indices.shape[0], x.shape[0]))
        outputs = self.process_term(spatialx, discretex, indices)  # outputs: (exchange_terms, batch)
        spatial_signs = signs * color_sign * spin_sign  # (exchange_terms, batch)

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


class Penta_FCN(nn.Module):
    """QQqqQ- Pentaquark Wavefunction, P=-
    input (x1,y1,z1,...,xn,yn,zn,rij) -> FCN -> output (logpsi)

    Arguments：
        nd: dimension
        nparticles: particle number
        mass: particle mass
        ndiscrete: discrete d.o.f
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

        self.layers = [nn.Dense(self.nhid * 2)]
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
        color, spin = discretex[:, :3], discretex[:, 3:]

        color6_12 = jnp.all(color == jnp.array([1, 0, 0], dtype=color.dtype), axis=-1)
        color6_34 = jnp.all(color == jnp.array([0, 1, 0], dtype=color.dtype), axis=-1)
        color_sign12 = jnp.expand_dims(jnp.where(color6_12, 1, -1), axis=0)  # (1,batch)
        color_sign34 = jnp.expand_dims(jnp.where(color6_34, 1, -1), axis=0)  # (1,batch)

        spin_sign12, spin_sign34 = compute_spin_sign(spin)

        identity = jnp.ones_like(color_sign12)
        color_sign = jnp.concatenate([identity, color_sign12, color_sign34, color_sign12 * color_sign34], axis=0)
        spin_sign = jnp.concatenate([identity, spin_sign12, spin_sign34, spin_sign12 * spin_sign34], axis=0)

        spatialx = jnp.reshape(x[:, :-self.ndiscrete], (-1, self.nparticles, self.nd))  # spatial dof
        indices, signs = jnp.array(self.indices), jnp.array(self.signs)
        signs = jnp.broadcast_to(jnp.expand_dims(signs, axis=1),
                                 (indices.shape[0], x.shape[0]))
        outputs = self.process_term(spatialx, discretex, indices)  # outputs: (exchange_terms, batch)
        spatial_signs = signs * color_sign * spin_sign  # (exchange_terms, batch)

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



def compute_rij(x, nd):
    """Interparticle distances"""

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

    if n_swaps == 2:
        (i0a, i1a, signa), (i0b, i1b, signb) = identical_particles
        indices_a = _create_swap_indices(indices, i0a, i1a)
        indices_ab = _create_swap_indices(indices_a, i0b, i1b)
        swap_indices_.append(indices_ab)
        signs.append(signa * signb)

    swap_indices_ = tuple(tuple(i) for i in swap_indices_)
    signs = tuple(signs)

    return swap_indices_, signs
