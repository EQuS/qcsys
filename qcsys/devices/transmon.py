""" Transmon."""

from flax import struct
from jax import config
import jaxquantum as jqt
import jax.numpy as jnp
import jax.scipy as jsp

from qcsys.devices.base import BasisTypes, FluxDevice, HamiltonianTypes

config.update("jax_enable_x64", True)


@struct.dataclass
class Transmon(FluxDevice):
    """
    Transmon Device.
    """

    @classmethod
    def param_validation(cls, N, N_pre_diag, params, hamiltonian, basis):
        """ This can be overridden by subclasses."""
        if hamiltonian == HamiltonianTypes.linear:
            assert basis == BasisTypes.fock, "Linear Hamiltonian only works with Fock basis."
        elif hamiltonian == HamiltonianTypes.truncated:
            assert basis == BasisTypes.fock, "Truncated Hamiltonian only works with Fock basis."
        elif hamiltonian == HamiltonianTypes.full:
            assert basis == BasisTypes.charge, "Full Hamiltonian only works with charge basis."
        
        assert (N_pre_diag - 1) % 2 == 0, "N_pre_diag must be odd."

    def common_ops(self):
        """ Written in the specified basis. """
        
        ops = {}

        N = self.N_pre_diag

        if self.basis == BasisTypes.fock:
            ops["id"] = jqt.identity(N)
            ops["a"] = jqt.destroy(N)
            ops["a_dag"] = jqt.create(N)
            ops["phi"] = self.phi_zpf() * (ops["a"] + ops["a_dag"])
            ops["n"] = 1j * self.n_zpf() * (ops["a_dag"] - ops["a"])

        elif self.basis == BasisTypes.charge:
            ops["id"] = jqt.identity(N)
            ops["cos(φ)"] = 0.5*(jqt.jnp2jqt(jnp.eye(N,k=1) + jnp.eye(N,k=-1)))
            n_max = (N - 1) // 2
            ops["n"] = jqt.jnp2jqt(jnp.diag(jnp.arange(-n_max, n_max + 1)))

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
        return w * self.original_ops["a_dag"] @ self.original_ops["a"]

    def get_H_full(self):
        """Return full H in specified basis."""
        
        cos_phi_op = self.original_ops["cos(φ)"]
        n_op = self.original_ops["n"]
        return 4*self.params["Ec"]*n_op@n_op - self.Ej * cos_phi_op
    
    def get_H_truncated(self):
        """Return truncated H in specified basis."""
        phi_op = self.original_ops["phi"]  
        fourth_order_term =  - (1/24) * self.Ej * phi_op @ phi_op @ phi_op @ phi_op 
        sixth_order_term = (1/720) * self.Ej * phi_op @ phi_op @ phi_op @ phi_op @ phi_op @ phi_op
        return self.get_H_linear() + fourth_order_term + sixth_order_term
    
    def _get_H_in_original_basis(self):
        """ This returns the Hamiltonian in the original specified basis. This can be overridden by subclasses."""

        if self.hamiltonian == HamiltonianTypes.linear:
            return self.get_H_linear()
        elif self.hamiltonian == HamiltonianTypes.full:
            return self.get_H_full()
        elif self.hamiltonian == HamiltonianTypes.truncated:
            return self.get_H_truncated()
    

    def potential(self, phi):
        """Return potential energy for a given phi."""
        if self.hamiltonian == HamiltonianTypes.linear:
            return 0.5 * self.Ej * (2 * jnp.pi * phi) ** 2
        elif self.hamiltonian == HamiltonianTypes.full:
            return - self.Ej * jnp.cos(2 * jnp.pi * phi)
        elif self.hamiltonian == HamiltonianTypes.truncated:
            phi_scaled = 2 * jnp.pi * phi
            second_order = 0.5 * self.Ej * phi_scaled ** 2
            fourth_order =  - (1/24) * self.Ej * phi_scaled ** 4
            sixth_order = (1/720) * self.Ej * phi_scaled ** 6
            return second_order + fourth_order + sixth_order

    def calculate_wavefunctions(self, phi_vals):
        """Calculate wavefunctions at phi_exts."""

        if self.basis == BasisTypes.fock:
            return super().calculate_wavefunctions(phi_vals)
        elif self.basis == BasisTypes.charge:
            phi_vals = jnp.array(phi_vals)

            n_labels = jnp.diag(self.original_ops["n"].data)

            wavefunctions = []
            for nj in range(self.N_pre_diag):
                wavefunction = []
                for phi in phi_vals:
                    wavefunction.append(
                        (1j ** nj / jnp.sqrt(2*jnp.pi)) * jnp.sum(
                            self.eig_systems["vecs"][:,nj] * jnp.exp(1j * phi * n_labels)
                        )
                    )
                wavefunctions.append(jnp.array(wavefunction))
            return jnp.array(wavefunctions)