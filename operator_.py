from collections.abc import Callable
from collections.abc import Hashable

from netket.utils.types import DType, PyTree, Array

from netket.hilbert import AbstractHilbert, DiscreteHilbert
from netket.operator import ContinuousOperator
from netket.utils import struct, HashableArray

import jax
import jax.numpy as jnp



@struct.dataclass
class MultiPotentialOperatorPyTree:
    """Internal class used to pass data from the operator to the jax kernel."""

    potential_fun: Callable = struct.field(pytree_node=False)
    dis_hilb: DiscreteHilbert = struct.field(pytree_node=False)
    coefficient: Array


class MultiPotentialEnergy(ContinuousOperator):
    """Local energy of multichannel potential"""

    def __init__(
            self,
            hilbert: AbstractHilbert,
            hi_dis: DiscreteHilbert,
            afun: Callable,
            coefficient: float = 1.0,
            dtype: DType | None = None,
    ):
        """
        Args:
            hilbert: The underlying Hilbert space on which the operator is defined
            hi_dis: discrete Hilbert space
            afun: The potential function
            coefficient: A coefficient for the ContinuousOperator object
            dtype: Data type of the coefficient
        """

        self._afun = afun
        self._coefficient = jnp.array(coefficient, dtype=dtype)
        self._dis_hilb = hi_dis

        self.__attrs = None

        super().__init__(hilbert, self._coefficient.dtype)

    @property
    def coefficient(self) -> Array:
        return self._coefficient

    @property
    def is_hermitian(self) -> bool:
        return True

    @staticmethod
    def _expect_kernel(logpsi: Callable, params: PyTree, x: Array, data: PyTree | None) -> Array:
        def logpsi_x(x):
            return logpsi(params, x)

        nsamples = x.shape[0]
        n_discrete = data.dis_hilb.size  # discrete dof
        nstates = data.dis_hilb.n_states  # number of discrete states
        dis_states = data.dis_hilb.all_states()
        r = x[:, :-n_discrete]
        s = x[:, -n_discrete:]

        r = jnp.broadcast_to(jnp.expand_dims(r, axis=1),
                             (nsamples, nstates, r.shape[1]))  # (n_samples, nstates, r_dim)
        s = jnp.broadcast_to(jnp.expand_dims(s, axis=1), (nsamples, nstates, n_discrete))
        s_ = jnp.broadcast_to(jnp.expand_dims(dis_states, axis=0),
                              (nsamples, nstates, n_discrete))
        rs_ = jnp.concatenate([r, s_], axis=2)  # (n_s, nstates, totaldim)

        psirs = jnp.exp(jax.vmap(logpsi_x, in_axes=(0,))(x))  # (n_s,)
        psirs_ = jnp.exp(jax.vmap(logpsi_x, in_axes=(0,))(rs_))  # (n_s, nstates,)

        batch_size = nsamples * nstates
        V_inputs = (
            s.reshape(batch_size, -1),  # (n_s*nstates, dis_dim)
            s_.reshape(batch_size, -1),  # (n_s*nstates, dis_dim)
            r.reshape(batch_size, -1)  # (n_s*nstates, r_dim)
        )
        V_flat = jax.vmap(data.potential_fun)(*V_inputs)  # (n_s*nstates,)
        V_matrix = V_flat.reshape(nsamples, nstates)  # (n_s, nstates)

        psi_ratio = psirs_ / psirs[:, None]

        return data.coefficient * jnp.sum(V_matrix * psi_ratio, axis=1)  # sum over nstates

    def _pack_arguments(self) -> MultiPotentialOperatorPyTree:
        return MultiPotentialOperatorPyTree(self._afun, self._dis_hilb, self.coefficient)

    @property
    def _attrs(self) -> tuple[Hashable, ...]:
        if self.__attrs is None:
            self.__attrs = (
                self.hilbert,
                self._dis_hilb,
                self._afun,
                self.dtype,
                HashableArray(self.coefficient),
            )
        return self.__attrs

    def __repr__(self):
        return f"Potential(coefficient={self.coefficient}, function={self._afun})"

