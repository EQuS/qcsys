""" Kerr Nonlinear Oscillator """
from flax import struct
from jax import config
import jaxquantum as jqt
import jax.numpy as jnp

from qcsys.devices.base import Device

config.update("jax_enable_x64", True)


@struct.dataclass
class KNO(Device):
    """
    Kerr Nonlinear Oscillator Device.
    """

    @classmethod
    def create(cls, N, params, label=0, use_linear=False):
        return cls(N, params, label, use_linear)

    def common_ops(self):
        ops = {}

        N = self.N
        ops["id"] = jqt.identity(N)
        ops["a"] = jqt.destroy(N)
        ops["a_dag"] = jqt.create(N)
        ops["phi"] = (ops["a"] + ops["a_dag"]) / jnp.sqrt(2)
        ops["n"] = 1j * (ops["a_dag"] - ops["a"]) / jnp.sqrt(2)
        return ops

    def get_linear_ω(self):
        """Get frequency of linear terms."""
        return self.params["ω"]

    def get_anharm(self):
        """Get anharmonicity."""
        return self.params["α"]

    def get_H_linear(self):
        """Return linear terms in H."""
        w = self.get_linear_ω()
        return w * self.linear_ops["a_dag"] @ self.linear_ops["a"]

    def get_H_full(self):
        """Return full H in linear basis."""
        α = self.get_anharm()

        return self.get_H_linear() + (α / 2) * (
            self.linear_ops["a_dag"]
            @ self.linear_ops["a_dag"]
            @ self.linear_ops["a"]
            @ self.linear_ops["a"]
        )
