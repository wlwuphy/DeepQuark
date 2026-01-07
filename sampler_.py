from netket.sampler import MetropolisRule
from netket.hilbert import DiscreteHilbert
from netket.utils import struct
import jax
import jax.numpy as jnp


class GaussianFlipRule(MetropolisRule):
    """
    Transition rule consisting of a Gaussian kick in positions and random flips in discrete d.o.f.
    """
    sigma: float
    dis_hilb: DiscreteHilbert = struct.field(pytree_node=False)

    def __init__(self, hi_dis: DiscreteHilbert, sigma):
        """
        Args:
            hi_dis: discrete Hilbert space
            sigma: The variance of the gaussian distribution centered around the current
                configuration, used to propose new configurations.
        """
        self.sigma = sigma
        self.dis_hilb = hi_dis

    def transition(rule, sampler, machine, parameters, state, key, rs):
        key1, key2 = jax.random.split(key, 2)
        n_discrete = rule.dis_hilb.size
        r = rs[:, :-n_discrete]

        if jnp.issubdtype(r.dtype, jnp.complexfloating):
            raise TypeError(
                "Gaussian Rule does not work with complex " "basis elements.")

        n_chains = r.shape[0]

        rprop = jax.random.normal(
            key1, shape=(n_chains, r.shape[1]), dtype=r.dtype) * jnp.asarray(rule.sigma, dtype=r.dtype)
        rp = r + rprop

        indices = jax.random.randint(key2, shape=(n_chains,), minval=0, maxval=rule.dis_hilb.n_states)
        sp = rule.dis_hilb.all_states()[indices]

        rsp = jnp.concatenate([rp, sp], axis=1)

        return rsp, None

    def __repr__(self):
        return f"GaussianFlipRule(sigma={self.sigma})"
