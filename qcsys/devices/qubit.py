from qcsys.devices.base import FluxDevice
from flax import struct
from jax import config
import jaxquantum as jqt
import jax.numpy as jnp

config.update("jax_enable_x64", True)

@struct.dataclass
class SuperconductingQubit(FluxDevice):
    """
    Superconducting Qubit
    Base class for superconducting qubits like Flux Qubits, Transmons, etc.
    """

    qubit_type: str
    
    def common_ops(self):
        """
        Common operations for any type of superconducting qubit in fock basis.
        """
        ops = {}

        N = self.N_pre_diag
        ops['id'] = jqt.identity(N)
        ops['a'] = jqt.destroy(N)
        ops['a_dag'] = jqt.create(N)
        
        # Define phase and charge operators based on ZPF (Zero Point Fluctuations)
        ops["phi"] = self.phi_zpf() * (ops["a"] + ops["a_dag"])
        ops["n"] = 1j * self.n_zpf() * (ops["a_dag"] - ops["a"])

        return ops

    def n_zpf(self):
        """Zero point fluctuation for the charge operator"""
        n_zpf = (self.params["El"] / (32.0 * self.params["Ec"])) ** (0.25)
        return n_zpf

    def phi_zpf(self):
        """Zero point fluctuation for the phase operator"""
        return (2 * self.params["Ec"] / self.params["El"]) ** (0.25)

    def get_linear_ω(self):
        """Linear frequency of the qubit"""
        return jnp.sqrt(8 * self.params['Ec'] * self.params['Ej'])

    def get_H_linear(self):
        """Linear Hamiltonian of the qubit"""
        phi_ext = self.params['phi_ext']
        if self.qubit_type == "flux":
            phi_ext = self.prams['phi_ext']
            return 0.25 * self.get_linear_ω() * (self.linear_ops['n'] * jnp.transpose(self.linear_ops['n']).conjugate() + self.linear_ops['phi'] * jnp.transpose(self.linear_ops['phi']).conjugate()) + 0.5 * slef.linear_ops['Ec'] * (phi_ext ** 2) - self.linear_ops['Ec'] * self.phi_zpf() * self.linear_ops['phi'] * phi_ext
        if self.qubit_type == 'transmon':
            return self.get_linear_ω() * self.original_ops["a_dag"] @ self.original_ops["a"]
        if self.qubit_type == 'fluxonium':
            return self.get_linear_ω * (self.linear_ops["a_dag"] @ self.linear_ops["a"]+ 0.5 * self.linear_ops["id"])
        
    def get_H_nonlinear(self):
        """Nonlinear Hamiltonian of the qubit"""
        if self.qubit_type == 'flux':
            return -self.params['Ej'] * jnp.cos(self.phi_zpf() * self.linear_ops['phi'])
        elif self.qubit_type == 'transmon':
            return -self.params['Ej'] * jnp.cos(self.linear_ops['phi'])
        elif self.qubit_type == 'fluxonium':
            op_cos_phi = jqt.cosm(self.linear_ops['phi'])
            op_sin_phi = jqt.sinm(self.linear_ops['phi'])

            phi_ext = self.params["phi_ext"]
            Hcos = op_cos_phi * jnp.cos(2.0 * jnp.pi * phi_ext) + op_sin_phi * jnp.sin(2.0 * jnp.pi * phi_ext)
            H_nl = - self.params["Ej"] * Hcos
            return H_nl
        return 0

    def get_H_full(self):
        """Complete Hamiltonian (linear + nonlinear)"""
        return self.get_H_linear() + self.get_H_nonlinear()