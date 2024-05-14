""" Resonator."""
from flax import struct
from jax import config
import jaxquantum as jqt
import jax.numpy as jnp

from qcsys.devices.base import FluxDevice

config.update("jax_enable_x64", True)

@struct.dataclass
class Resonator(FluxDevice):        
    """
    Resonator Device.
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
        return w*(self.linear_ops["a_dag"]@self.linear_ops["a"] + 1/2)
  
    def get_H_full(self):
        """Return full H in linear basis."""
        return self.get_H_linear()
    
    def potential(self, phi):
        """Return potential energy for a given phi."""
        return 0.5 * self.params["El"] * (2 * jnp.pi * phi) ** 2