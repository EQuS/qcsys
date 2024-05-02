
""" Base device."""

from abc import abstractmethod, ABC
from enum import Enum
from typing import Dict, Any, List

from flax import struct
from jax import config, Array
import jax.numpy as jnp
import matplotlib.pyplot as plt

from qcsys.common.utils import harm_osc_wavefunction
import jaxquantum as jqt

config.update("jax_enable_x64", True)


class BasisTypes(str, Enum):
    fock = "fock"
    charge = "charge"

    @classmethod
    def from_str(cls, string: str):
        return cls(string)

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.value == other.value

    def __ne__(self, other):
        return self.value != other.value

    def __hash__(self):
        return hash(self.value)

class HamiltonianTypes(str, Enum):
    linear = "linear"
    truncated = "truncated"
    full = "full"

    @classmethod
    def from_str(cls, string: str):
        return cls(string)

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.value == other.value

    def __ne__(self, other):
        return self.value != other.value

    def __hash__(self):
        return hash(self.value)

@struct.dataclass
class Device(ABC):
    N: int = struct.field(pytree_node=False)
    N_pre_diag: int = struct.field(pytree_node=False)
    params: Dict[str, Any]
    _label: int = struct.field(pytree_node=False)
    _basis: BasisTypes = struct.field(pytree_node=False)
    _hamiltonian: HamiltonianTypes = struct.field(pytree_node=False)

    @classmethod
    def param_validation(cls, N, N_pre_diag, params, hamiltonian, basis):
        """ This can be overridden by subclasses."""
        pass

    @classmethod
    def create(cls, N, params, label=0, use_linear=None, N_pre_diag=None, hamiltonian: HamiltonianTypes = None, basis: BasisTypes = None):
        if N_pre_diag is None:
            N_pre_diag = N

        _basis = basis if basis is not None else BasisTypes.fock
        _hamiltonian = hamiltonian if hamiltonian is not None else HamiltonianTypes.full
        if use_linear is not None and use_linear:
            _hamiltonian = HamiltonianTypes.linear
        
        cls.param_validation(N, N_pre_diag, params, _hamiltonian, _basis)

        return cls(N, N_pre_diag, params, label, _basis, _hamiltonian)
    
    @property
    def basis(self):
        return self._basis
    
    @property
    def hamiltonian(self):
        return self._hamiltonian

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
        """Set up common ops in the specified basis."""

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
        return self.get_op_in_H_eigenbasis(self._get_H_in_original_basis()).keep_only_diag_elements()

    def _get_H_in_original_basis(self):
        """ This returns the Hamiltonian in the original specified basis. This can be overridden by subclasses."""

        if self.hamiltonian == HamiltonianTypes.linear:
            return self.get_H_linear()
        elif self.hamiltonian == HamiltonianTypes.full:
            return self.get_H_full()
        
    def _calculate_eig_systems(self):
        evs, evecs = jnp.linalg.eigh(self._get_H_in_original_basis().data)  # Hermitian
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
        dims = [[self.N], [self.N]]
        return get_op_in_new_basis(op, evecs, dims)
    
    def get_op_data_in_H_eigenbasis(self, op: Array):
        evecs = self.eig_systems["vecs"][:, : self.N]
        return get_op_data_in_new_basis(op, evecs)

    def get_vec_in_H_eigenbasis(self, vec: jqt.Qarray):
        evecs = self.eig_systems["vecs"][:, : self.N]
        if vec.qtype == jqt.Qtypes.ket:
            dims = [[self.N],[1]]
        else:
            dims = [[1], [self.N]]
        return get_vec_in_new_basis(vec, evecs, dims)
    
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


def get_op_in_new_basis(op: jqt.Qarray, evecs: Array, dims: List[List[int]]) -> jqt.Qarray:
    return jqt.Qarray.create(get_op_data_in_new_basis(op.data, evecs), dims=dims)

def get_op_data_in_new_basis(op_data: Array, evecs: Array) -> Array:
    return jnp.dot(jnp.conjugate(evecs.transpose()), jnp.dot(op_data, evecs))

def get_vec_in_new_basis(vec: jqt.Qarray, evecs: Array, dims: List[List[int]]) -> jqt.Qarray:
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
                harm_osc_wavefunction(n, phi_vals, jnp.real(phi_osc))
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
        """Return potential energy as a function of phi."""


    def plot_wavefunctions(self, phi_vals, max_n=None, which=None, ax=None, mode="abs"):
        """Plot wavefunctions at phi_exts."""
        wavefunctions = self.calculate_wavefunctions(phi_vals)
        energy_levels = self.eig_systems["vals"][:self.N]

        potential = self.potential(phi_vals)

        if ax is None:
            fig, ax = plt.subplots(1,1, figsize=(6,5), dpi=200)
        else:
            fig = ax.get_figure()

        min_val = None
        max_val = None

        assert max_n is None or which is None, "Can't specify both max_n and which"

        max_n = self.N if max_n is None else max_n
        levels = range(max_n) if which is None else which

        for n in levels:
            if mode == "abs":
                wf_vals = jnp.abs(wavefunctions[n, :])**2
            elif mode == "real":
                wf_vals = wavefunctions[n, :].real
            elif mode == "imag":
                wf_vals = wavefunctions[n, :].imag

            wf_vals += energy_levels[n]
            curr_min_val = min(wf_vals)
            curr_max_val = max(wf_vals)

            if min_val is None or curr_min_val < min_val:
                min_val = curr_min_val

            if max_val is None or curr_max_val > max_val:
                max_val = curr_max_val

            ax.plot(phi_vals, wf_vals, label=f"{n}")
            ax.fill_between(phi_vals, energy_levels[n], wf_vals, alpha=0.5)
        
        ax.plot(phi_vals, potential, label="potential", color="black", linestyle="--")

        ax.set_ylim([min_val-1, max_val+1])

        ax.set_xlabel(r"$\Phi/\Phi_0$")
        ax.set_ylabel(r"Energy [GHz]")

        if mode == "abs":
            title_str = r"$|\psi_n(\Phi)|^2$"
        elif mode == "real":
            title_str = r"Re($\psi_n(\Phi)$)"
        elif mode == "imag":
            title_str = r"Im($\psi_n(\Phi)$)"

        ax.set_title(f"{title_str}")

        plt.legend(fontsize=6)
        fig.tight_layout()

        return ax

