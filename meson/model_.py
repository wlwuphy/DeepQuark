from flax import linen as nn
import jax.numpy as jnp
import jax



class FCN(nn.Module):
    """Fully connected neural network wavefunction
    input (x1,y1,z1,...,xn,yn,zn,rij) -> FCN -> output (logpsi)

    Arguments：
        nd: dimension
        nparticles: particle number
        nhid: number of nodes
        nlayers: number of layers
        bound: boundary condition
    """
    nd: int
    nparticles: int
    mass: tuple
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

        spatialx = x  # spatial dof
        rij, bc = self._boundary_condition(spatialx)
        features = jnp.concatenate([x, rij], axis=-1)

        for layer in self.layers[:-1]:
            features = nn.tanh(layer(features))
        output = self.layers[-1](features).squeeze()

        return output * bc

    def __call__(self, x: jnp.ndarray):
        spatialx = x
        xcm = center_of_mass(spatialx, jnp.array(self.mass))
        spatialx -= xcm

        inputs = jnp.stack([spatialx, -spatialx], axis=0)
        Psi = jnp.sum(jax.vmap(self.psi)(inputs), axis=0)
        logpsi = jnp.log(Psi.astype(complex))
        return logpsi

    def _boundary_condition(self, x: jnp.ndarray):
        """Exponetial boundary condition"""

        rij = compute_rij(x, self.nd)  # shape=(batch, n_pairs)
        bc = jnp.exp(-jnp.sum(rij ** 2 / self.bound ** 2, axis=-1))

        return rij, bc


def compute_rij(x, nd=3):
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

