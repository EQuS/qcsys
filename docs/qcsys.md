# Introduction

Quantum computing systems like QCSYS require robust and versatile support for different qubit types to cater to diverse computational needs. This project introduces a comprehensive framework to integrate superconducting qubits into QCSYS. The implementation adds support for flux qubits, transmons, and other types of superconducting qubits, each modeled with their distinct characteristics using a custom `Flux` class. The goal is to provide a scalable, realistic, and efficient system for simulating superconducting qubits with precise energy constraints and error resistance.

This implementation builds upon previous qubit models by making critical adjustments to support a broader range of qubit types while maintaining efficient Hamiltonian modeling and operator calculations.

## Key Changes in `Flux` Compared to the Original Approach

- **Generalized Qubit Support**:  
  The `Flux` class expands functionality to handle multiple qubit types (flux, transmon, etc.) using a unified structure, rather than being specific to a single qubit type.
  
- **Unified Hamiltonian Calculations**:  
  Linear and nonlinear Hamiltonians are constructed with conditional logic based on the qubit type, allowing accurate modeling for each qubit's unique behavior.

- **Simplified Operator Definitions**:  
  Specific operators like `cos(φ/2)` and `sin(φ/2)` are excluded in favor of generalized operators (`phi` and `n`), making the class more versatile.

- **Custom Qubit-Type Logic**:  
  The zero-point fluctuation (ZPF) calculations, operator scaling, and Hamiltonian terms are tailored dynamically based on the selected qubit type, enhancing flexibility for diverse superconducting qubit simulations.

By incorporating these changes, the `Flux` class provides a versatile foundation for simulating a variety of qubit types, while retaining detailed physical modeling and performance.

---

## Detailed Design and Implementation

### 1. Superconducting Qubit Framework

The foundation of this implementation is the `Flux` class, derived from `FluxDevice`. This class encapsulates the physical properties and mathematical models for superconducting qubits, including:

#### (a) Hamiltonian Construction

1. **Linear Hamiltonian**: Models the basic energy dynamics of qubits.
   - For flux qubits, includes phase and charge terms influenced by external flux.
   - For transmons, uses a simple harmonic oscillator model.
   
2. **Nonlinear Hamiltonian**: Captures the periodic potential specific to each qubit type.
   - Flux qubits: Use a cosine term with Josephson energy (`EJ`).
   - Transmons: Utilize a cosine potential without additional scaling.

#### (b) Quantum Operators

The class defines standard quantum operators, which are essential for simulating qubits:

- **Annihilation (`a`) and creation (`a†`) operators**: Represent qubit energy levels.
- **Phase (`ϕ`) and charge (`n`) operators**: Derived using zero-point fluctuations (ZPF).
- **Zero-point fluctuations**: Calculated using circuit parameters (`EL`, `EC`), ensuring realistic scaling of operators for superconducting circuits.

---

### Code Example: `Flux` Class

Below is a simplified example of the `Flux` class, which includes the updates without profiling decorators:

```python
from qcsys.devices.base import FluxDevice
from flax import struct
import jaxquantum as jqt
import jax.numpy as jnp

config.update("jax_enable_x64", True)

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

    def get_H_linear(self):
        phi_ext = self.prams['phi_ext']
        return 0.25 * self.get_linear_ω() * (self.linear_ops['n'] * jnp.transpose(self.linear_ops['n']).conjugate() + self.linear_ops['phi'] * jnp.transpose(self.linear_ops['phi']).conjugate()) + 0.5 * self.linear_ops['Ec'] * (phi_ext ** 2) - self.linear_ops['Ec'] * self.phi_zpf() * self.linear_ops['phi'] * phi_ext

    def get_H_nonlinear(self):
        return -1 * self.params['Ej'] * jnp.cosm(self.phi_zpf() * self.linear_ops['phi']) 
    
    def get_H_full(self):
        return self.get_H_linear() + self.get_H_nonlinear()

---

## Key Features of the QCSYS Integration

- **Multi-Qubit Support**:  
  The `Flux` class allows seamless integration of multiple superconducting qubits, each with unique Hamiltonians and configurations.

- **Custom Hamiltonians**:  
  Supports both linear and nonlinear terms for flux qubits and transmons, ensuring accurate simulation of their behaviors.

---

## Conclusion

The addition of superconducting qubits to QCSYS represents a significant step toward creating a versatile and scalable quantum computing framework. By generalizing the `Flux` class to support multiple qubit types, the system ensures flexibility while retaining the precise modeling and efficiency of previous qubit models. This comprehensive framework lays a strong foundation for future advancements in quantum systems, including expanded qubit support, optimized multi-qubit interactions, and advanced error-correction mechanisms.
