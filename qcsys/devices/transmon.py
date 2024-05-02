""" Transmon."""

from flax import struct
from jax import config
import jaxquantum as jqt
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit

from qcsys.devices.base import CompactPhaseDevice

config.update("jax_enable_x64", True)


@struct.dataclass
class Transmon(CompactPhaseDevice):
    """
    Transmon Device.
    """

    def common_ops(self):
        """ Written in the linear basis. """
        
        ops = {}

        N = self.N_pre_diag
        ops["id"] = jqt.identity(N)
        ops["a"] = jqt.destroy(N)
        ops["a_dag"] = jqt.create(N)
        ops["phi"] = self.phi_zpf() * (ops["a"] + ops["a_dag"])
        ops["n"] = 1j * self.n_zpf() * (ops["a_dag"] - ops["a"])

        ops["cos(φ)"] = jqt.cosm(ops["phi"])

        return ops

    @property
    def Ej(self):
        return self.params["Ej"]

    def phi_zpf(self):
        """Return Phase ZPF."""
        return (2 * self.params["Ec"] / self.Ej) ** (0.25)

    def n_zpf(self):
        """Return Charge ZPF."""
        return (self.Ej / (32 * self.params["Ec"])) ** (0.25)

    def get_linear_ω(self):
        """Get frequency of linear terms."""
        return jnp.sqrt(8 * self.params["Ec"] * self.Ej)

    def get_H_linear(self):
        """Return linear terms in H."""
        w = self.get_linear_ω()
        return w * self.linear_ops["a_dag"] @ self.linear_ops["a"]

    def get_H_full(self):
        """Return full H in linear basis."""
        # cos_phi_op = (
        #     jsp.linalg.expm(1j * self.linear_ops["phi"])
        #     + jsp.linalg.expm(-1j * self.linear_ops["phi"])
        # ) / 2

        cos_phi_op = self.linear_ops["cos(φ)"]

        H_nl = -self.Ej * cos_phi_op - self.Ej / 2 * self.linear_ops["phi"] @ self.linear_ops["phi"]
        return self.get_H_linear() + H_nl
    
        # n_op = self.linear_ops["n"]
        # return 4*self.params["Ec"]*n_op@n_op - self.Ej * cos_phi_op

    def potential(self, phi):
        """Return potential energy for a given phi."""
        return - self.Ej * jnp.cos(2 * jnp.pi * phi)

