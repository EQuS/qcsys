""" ATS."""

from flax import struct
from jax import config
import jaxquantum as jqt
import jax.numpy as jnp

from qcsys.devices.base import FluxDevice

config.update("jax_enable_x64", True)

@struct.dataclass
class ATS(FluxDevice):        
    """
    ATS Device.
    """
    
    def common_ops(self):
        """ Written in the linear basis. """
        ops = {}
        
        N = self.N_pre_diag
        ops["id"] = jqt.identity(N)
        ops["a"] = jqt.destroy(N)
        ops["a_dag"] = jqt.create(N)
        ops["phi"] = self.phi_zpf()*(ops["a"] + ops["a_dag"])  
        return ops

    def phi_zpf(self):
        """Return Phase ZPF."""
        return (2*self.params["Ec"]/self.params["El"])**(.25)
    
    def get_linear_ω(self):
        """Get frequency of linear terms."""
        return jnp.sqrt(8*self.params["El"]*self.params["Ec"])
    
    def get_H_linear(self):
        """Return linear terms in H."""
        w = self.get_linear_ω()
        return w*(self.linear_ops["a_dag"]@self.linear_ops["a"] + 0.5 * self.linear_ops["id"])

    def get_H_full(self):
        """Return full H in linear basis."""
        # raise NotImplemented("This has not yet been implemented.")

        # extra ATS terms
        Ej = self.params["Ej"]
        dEj = self.params["dEj"]
        Ej2 = self.params["Ej2"]

        phi_b = self.linear_ops["phi"]
        id = self.linear_ops["id"]

        phi_delta_ext_op = self.params["phi_delta_ext"] * id

        H_nl = - 2 * Ej * jqt.cosm(phi_b + 2 * jnp.pi * phi_delta_ext_op) * jnp.cos(2 * jnp.pi * self.params["phi_sum_ext"])
        H_nl += 2 * dEj * jqt.sinm(phi_b + 2 * jnp.pi * phi_delta_ext_op) * jnp.sin(2 * jnp.pi * self.params["phi_sum_ext"]) 
        H_nl += 2 * Ej2 * jqt.cosm(2*phi_b + 2 * 2 * jnp.pi * phi_delta_ext_op) * jnp.cos(2 * 2 * jnp.pi * self.params["phi_sum_ext"])

        H = self.get_H_linear() + H_nl
        return H

    def potential(self, phi):
        """Return potential energy for a given phi."""

        phi_delta_ext = self.params["phi_delta_ext"]
        phi_sum_ext = self.params["phi_sum_ext"]

        V = 0.5 * self.params["El"] * (2 * jnp.pi * phi) ** 2
        V += - 2 * self.params["Ej"] * jnp.cos(2 * jnp.pi * (phi + phi_delta_ext)) * jnp.cos(2 * jnp.pi * phi_sum_ext)
        V += 2 * self.params["dEj"] * jnp.sin(2 * jnp.pi * (phi + phi_delta_ext)) * jnp.sin(2 * jnp.pi * phi_sum_ext)
        V += 2 * self.params["Ej2"] * jnp.cos(2 * 2 * jnp.pi * (phi + phi_delta_ext)) * jnp.cos(2 * 2 * jnp.pi * phi_sum_ext)

        return V