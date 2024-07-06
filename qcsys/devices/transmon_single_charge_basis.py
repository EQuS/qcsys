""" 
Single-charge basis transmon.
"""

from flax import struct
from jax import config
import jaxquantum as jqt
import jax.numpy as jnp
from jax import jit

from qcsys.devices.base import Device, HamiltonianTypes, BasisTypes

config.update("jax_enable_x64", True)


@struct.dataclass
class SingleChargeTransmon(Device):
    """
    Offset-Charge Sensitive Transmon Device, written in single-charge basis.

    Required params:
    - Ec: Charging Energy
    - Ej: Josephson Energy
    - ng: Gate offset charge
    - N_max_charge: Maximum number of charge levels to consider

    """

    N_max_charge: int = struct.field(pytree_node=False)

    @classmethod
    def param_validation(cls, N, N_max_charge, params, hamiltonian, basis):
        assert N <= 2 * N_max_charge + 1

    @classmethod
    def create(
        cls,
        N,
        N_max_charge,
        params,
        label=0,
        use_linear=None,
        N_pre_diag=None,
        hamiltonian: HamiltonianTypes = None,
        basis: BasisTypes = None,
    ):

        if N_pre_diag is not None and N_pre_diag != N_max_charge:
            print("Warning: N_max_charge will be used as N_pre_diag!")

        _basis = basis if basis is not None else BasisTypes.fock
        _hamiltonian = hamiltonian if hamiltonian is not None else HamiltonianTypes.full
        if use_linear is not None and use_linear:
            _hamiltonian = HamiltonianTypes.linear

        cls.param_validation(N, N_max_charge, params, _hamiltonian, _basis)

        return cls(
            N,
            N_max_charge,
            params,
            label,
            _basis,
            _hamiltonian,
            N_max_charge,
        )

    def common_ops(self):
        """
        Operators defined in the single charge basis.
        """
        ops = {}

        ops["n"] = jqt.Qarray.create(self.build_n_op())
        ops["cos(φ)"] = jqt.Qarray.create(self.build_cos_phi_op())
        ops["cos(φ/2)"] = jqt.Qarray.create(self.build_cos_phi_2_op())
        ops["sin(φ/2)"] = jqt.Qarray.create(self.build_sin_phi_2_op())
        ops["H_charge"] = jqt.Qarray.create(self.build_H_charge_op())
        return ops

    def build_n_op(self):
        # We define n = ∑ₙ n|n⟩⟨n| in the single charge basis. Here n counts electrons, not Cooper pairs.
        return jnp.diag(jnp.arange(-self.N_max_charge, self.N_max_charge + 1))

    def build_cos_phi_op(self):
        # We define cos(φ) = 1/2 * ∑ₙ|n⟩⟨n+2| + h.c. in the single charge basis
        return 0.5 * (
            jnp.eye(2 * self.N_max_charge + 1, k=2)
            + jnp.eye(2 * self.N_max_charge + 1, k=-2)
        )

    def build_cos_phi_2_op(self):
        # We define cos(φ/2) = 1/2 * ∑ₙ|n⟩⟨n+1| + h.c. in the single charge basis
        return 0.5 * (
            jnp.eye(2 * self.N_max_charge + 1, k=1)
            + jnp.eye(2 * self.N_max_charge + 1, k=-1)
        )

    def build_sin_phi_2_op(self):
        # We define sin(φ/2) = i/2 * ∑ₙ|n⟩⟨n+1| + h.c. in the single charge basis
        return 0.5j * (
            jnp.eye(2 * self.N_max_charge + 1, k=1)
            - jnp.eye(2 * self.N_max_charge + 1, k=-1)
        )

    def build_H_charge_op(self):
        """
        Construct the "charge" (i.e. the "Ec" part) of the Hamiltonian in the single charge basis.
        Defined as H = Ec (n - 2ng)² where n counts the number of electrons, not Cooper pairs.
        """

        # (n - 2*ng)
        n_minus_ng_array = jnp.arange(
            -self.N_max_charge, self.N_max_charge + 1
        ) - 2 * self.params["ng"] * jnp.ones(2 * self.N_max_charge + 1)

        return jnp.diag(self.params["Ec"] * n_minus_ng_array**2)

    @property
    def phi_zpf(self):
        """Return Phase ZPF"""
        return (2 * self.params["Ec"] / self.params["Ej"]) ** (0.25)

    @property
    def n_zpf(self):
        """Return charge ZPF"""
        return (self.params["Ej"] / (32.0 * self.params["Ec"])) ** (0.25)

    def get_linear_ω(self):
        """Get frequency of linear terms"""
        return jnp.sqrt(8 * self.params["Ec"] * self.params["Ej"])

    def get_H_linear(self):
        raise NotImplemented("No linear oscillator basis for single charge transmon.")

    def get_H_full(self):
        """
        Return full Hamiltonian H = Ec (n - 2ng)² - Ej cos(φ) in the single charge basis. Using Eq. (5.36)
        of Kyle Serniak's thesis, we have H = Ec ∑ₙ(n - 2*ng) |n⟩⟨n| - Ej/2 * ∑ₙ|n⟩⟨n+2| + h.c where now n
        counts the number of electrons, not Cooper pairs.
        """
        return (
            self.original_ops["H_charge"]
            - self.params["Ej"] * self.linear_ops["cos(φ)"]
        )


@struct.dataclass
class SymmetricSQUIDTransmon(SingleChargeTransmon):
    """
    Flux-Tunable Symmetric SQUID Transmon Device, written in single-charge
    basis. Here, the two junctions Ej1 = Ej2 = Ej are assumed to be identical.

    Required params:
    - Ec: Charging Energy
    - Ej: Josephson Energy
    - ng: Gate offset charge
    - N_max_charge: Maximum number of charge levels to consider

    """

    @property
    def Ej_eff(self):
        return 2 * self.params["Ej"] * jnp.cos(self.params["phi_ext"] / 2)

    @property
    def phi_zpf(self):
        """Return Phase ZPF"""
        return (2 * self.params["Ec"] / self.Ej_eff) ** (0.25)

    @property
    def n_zpf(self):
        """Return charge ZPF"""
        return (self.Ej_eff / (32.0 * self.params["Ec"])) ** (0.25)

    def get_linear_ω(self):
        """Get frequency of linear terms"""
        return jnp.sqrt(8 * self.params["Ec"] * self.Ej_eff)

    def get_H_full(self):
        """
        Return full Hamiltonian H = Ec (n - 2ng)² - 2Ej cos(φext/2) cos(φ) in the single charge basis. Using
        Eq. (5.36) of Kyle Serniak's thesis, we have H = Ec ∑ₙ(n - 2*ng) |n⟩⟨n| - Ej * cos(φext/2) * ∑ₙ|n⟩⟨n+2| + h.c
        where now n counts the number of electrons, not Cooper pairs.
        """
        phi_ext = self.params["phi_ext"]
        return (
            self.original_ops["H_charge"]
            - 2 * self.params["Ej"] * jnp.cos(phi_ext / 2) * self.original_ops["cos(φ)"]
        )
