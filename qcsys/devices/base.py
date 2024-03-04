""" Base device."""

from abc import abstractmethod, ABC
from typing import Dict, Any

from flax import struct
from jax import config, Array
import jax.numpy as jnp
import matplotlib.pyplot as plt

from qcsys.common.utils import harm_osc_wavefunction
import jaxquantum as jqt

config.update("jax_enable_x64", True)


@struct.dataclass
class Device(ABC):
    N: int = struct.field(pytree_node=False)
    N_pre_diag: int = struct.field(pytree_node=False)
    params: Dict[str, Any]
    _label: int = struct.field(pytree_node=False)
    _use_linear: bool = struct.field(pytree_node=False)

    @classmethod
    def create(cls, N, params, label=0, use_linear=True, N_pre_diag=None):
        if N_pre_diag is None:
            N_pre_diag = N
        return cls(N, N_pre_diag, params, label, use_linear)

    @property
    def label(self):
        return self.__class__.__name__ + str(self._label)

    @property
    def linear_ops(self):
        return self.common_ops()

    @property
    def ops(self):
        return self.full_ops()

    @abstractmethod
    def common_ops(self) -> Dict[str, jqt.Qarray]:
        """Set up common ops in the linear basis."""

    @abstractmethod
    def get_linear_ω(self):
        """Get frequency of linear terms."""

    @abstractmethod
    def get_H_linear(self):
        """Return linear terms in H."""

    @abstractmethod
    def get_H_full(self):
        """Return full H."""

    def get_H(self):
        """
        Return diagonalized H. Explicitly keep only diagonal elements of matrix.
        """
        return jnp.diag(
            jnp.diag(self.get_op_in_H_eigenbasis(self._get_H_in_linear_basis()))
        )

    def _get_H_in_linear_basis(self):
        return self.get_H_linear() if self._use_linear else self.get_H_full()

    def _calculate_eig_systems(self):
        evs, evecs = jnp.linalg.eigh(self._get_H_in_linear_basis().data)  # Hermitian
        idxs_sorted = jnp.argsort(evs)
        return evs[idxs_sorted], evecs[:, idxs_sorted]

    @property
    def eig_systems(self):
        eig_systems = {}
        eig_systems["vals"], eig_systems["vecs"] = self._calculate_eig_systems()

        eig_systems["vecs"] = eig_systems["vecs"]
        eig_systems["vals"] = eig_systems["vals"]
        return eig_systems

    def get_op_in_H_eigenbasis(self, op: jqt.Qarray):
        evecs = self.eig_systems["vecs"][:, : self.N]
        return get_op_in_new_basis(op, evecs)
    
    def get_op_data_in_H_eigenbasis(self, op: Array):
        evecs = self.eig_systems["vecs"][:, : self.N]
        return get_op_data_in_new_basis(op, evecs)

    def get_vec_in_H_eigenbasis(self, vec: jqt.Qarray):
        evecs = self.eig_systems["vecs"][:, : self.N]
        return get_vec_in_new_basis(vec, evecs)
    
    def get_vec_data_in_H_eigenbasis(self, vec: Array):
        evecs = self.eig_systems["vecs"][:, : self.N]
        return get_vec_data_in_new_basis(vec, evecs)

    def full_ops(self):
        # TODO: use JAX vmap here

        linear_ops = self.linear_ops
        ops = {}
        for name, op in linear_ops.items():
            ops[name] = self.get_op_in_H_eigenbasis(op)

        return ops


def get_op_in_new_basis(op: jqt.Qarray, evecs: Array) -> jqt.Qarray:
    dims = op.dims
    return jqt.Qarray.create(get_op_data_in_new_basis(op.data, evecs), dims=dims)

def get_op_data_in_new_basis(op_data: Array, evecs: Array) -> Array:
    return jnp.dot(jnp.conjugate(evecs.transpose()), jnp.dot(op_data, evecs))

def get_vec_in_new_basis(vec: jqt.Qarray, evecs: Array) -> jqt.Qarray:
    dims = vec.dims
    return jqt.Qarray.create(get_vec_data_in_new_basis(vec.data, evecs), dims=dims)

def get_vec_data_in_new_basis(vec_data: Array, evecs: Array) -> Array:
    return jnp.dot(jnp.conjugate(evecs.transpose()), vec_data)


@struct.dataclass
class FluxDevice(Device):
    @abstractmethod
    def phi_zpf(self):
        """Return Phase ZPF."""

    def calculate_wavefunctions(self, phi_vals):
        """Calculate wavefunctions at phi_exts."""
        phi_osc = self.phi_zpf() * jnp.sqrt(2) # length of oscillator
        phi_vals = jnp.array(phi_vals)

        # calculate basis functions
        basis_functions = []
        for n in range(self.N_pre_diag):
            basis_functions.append(
                harm_osc_wavefunction(n, phi_vals, phi_osc)
            )
        basis_functions = jnp.array(basis_functions)

        # transform to better diagonal basis
        basis_functions_in_H_eigenbasis = self.get_vec_data_in_H_eigenbasis(basis_functions)
        
        # the below is equivalent to evecs_in_H_eigenbasis @ basis_functions_in_H_eigenbasis
        # since evecs in H_eigenbasis is diagonal, i.e. the identity matrix
        wavefunctions = basis_functions_in_H_eigenbasis 
        return wavefunctions
    
    @abstractmethod
    def potential(self, phi):
        """Return potential energy as a funciton of phi."""


    def plot_wavefunctions(self, phi_vals):
        """Plot wavefunctions at phi_exts."""
        wavefunctions = self.calculate_wavefunctions(phi_vals)
        energy_levels = self.eig_systems["vals"][:self.N]

        potential = self.potential(phi_vals)

        fig, axs = plt.subplots(1,1, figsize=(6,5), dpi=200, squeeze=False)

        min_val = None
        max_val = None

        ax = axs[0][0]
        for n in range(self.N):
            wf_vals = wavefunctions[n, :].real + energy_levels[n]
            curr_min_val = min(wf_vals)
            curr_max_val = max(wf_vals)

            if min_val is None or curr_min_val < min_val:
                min_val = curr_min_val

            if max_val is None or curr_max_val > max_val:
                max_val = curr_max_val

            ax.plot(phi_vals, wf_vals, label=f"{n}")
        
        ax.plot(phi_vals, potential, label="potential", color="black", linestyle="--")

        ax.set_ylim([min_val-1, max_val+1])

        ax.set_xlabel(r"$\varphi/\Phi_0$")
        ax.set_ylabel(r"Energy [GHz]")

        plt.legend(fontsize=6)
        fig.tight_layout()

        return axs

