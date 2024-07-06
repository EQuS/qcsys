""" IdealQubit."""

from flax import struct
from jax import config
import jaxquantum as jqt

from .base import Device


config.update("jax_enable_x64", True)


@struct.dataclass
class IdealQubit(Device):
    """
    Ideal qubit Device.
    """

    def common_ops(self):
        """Written in the linear basis."""
        ops = {}

        assert self.N_pre_diag == 2
        assert self.N == 2

        N = self.N_pre_diag
        ops["id"] = jqt.identity(N)
        ops["sigma_z"] = jqt.sigmaz()
        ops["sigma_x"] = jqt.sigmax()

        return ops

    def get_linear_ω(self):
        """Get frequency of linear terms."""
        return self.params["frequency"]

    def get_H_linear(self):
        """Return linear terms in H."""
        w = self.get_linear_ω()
        return (w / 2) * self.linear_ops["sigma_z"]

    def get_H_full(self):
        """Return full H in linear basis."""
        H = self.get_H_linear()
        return H
