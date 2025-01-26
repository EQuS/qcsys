from qcsys.devices.base import FluxDevice
from qcsys.common.profiling import profile_function, line_profile_function

from flax import struct
from jax import config
import jaxquantum as jqt
import jax.numpy as jnp

config.update("jax_enable_x64", True)

@struct.dataclass
class Flux(FluxDevice):
    """
    Flux Device
    """
    
    def common_ops(self):
        ops = {}

        N = self.N_pre_diag
        ops['id'] = jqt.identity(N)
        ops['a'] = jqt.destroy(N)
        ops['a_dag'] = jqt.create(N)
        ops["phi"] = self.phi_zpf() * (ops["a"] + ops["a_dag"])
        ops["n"] = 1j * self.n_zpf() * (ops["a_dag"] - ops["a"])

        return ops

    def n_zpf(self):
        n_zpf = (self.params["El"] / (32.0 * self.params["Ec"])) ** (0.25)
        return n_zpf

    def phi_zpf(self):
        """Return Phase ZPF."""
        return (2 * self.params["Ec"] / self.params["El"]) ** (0.25)

    def get_linear_ω(self):
        return jnp.sqrt(8 * self.params['Ec'] * self.params['Ej'])

    @profile_function
    def get_H_linear(self):
        phi_ext = self.prams['phi_ext']
        return 0.25 * self.get_linear_ω() * (self.linear_ops['n'] * jnp.transpose(self.linear_ops['n']).conjugate() + self.linear_ops['phi'] * jnp.transpose(self.linear_ops['phi']).conjugate()) + 0.5 * slef.linear_ops['Ec'] * (phi_ext ** 2) - self.linear_ops['Ec'] * self.phi_zpf() * self.linear_ops['phi'] * phi_ext

    @profile_function
    def get_H_nonlinear(self):
        return -1 * self.params['Ej'] * jnp.cosm(self.phi_zpf * self.linear_ops['phi']) 
    
    @profile_function
    def get_H_full(self):
        return self.get_H_linear + self.get_H_nonlinear