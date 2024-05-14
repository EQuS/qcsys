# qcsys

[![License](https://img.shields.io/github/license/EQuS/qcsys.svg?style=popout-square)](https://opensource.org/license/apache-2-0) [![](https://img.shields.io/github/release/EQuS/qcsys.svg?style=popout-square)](https://github.com/EQuS/qcsys/releases) [![](https://img.shields.io/pypi/dm/qcsys.svg?style=popout-square)](https://pypi.org/project/qcsys/)

[S. R. Jha](https://github.com/Phionx), [S. Chowdhury](https://github.com/shoumikdc), [M. Hays](https://scholar.google.com/citations?user=06z0MjwAAAAJ), [J. A. Grover](https://scholar.google.com/citations?user=igewch8AAAAJ), [W. D. Oliver](https://scholar.google.com/citations?user=4vNbnqcAAAAJ&hl=en)


**Docs:** [https://equs.github.io/qcsys](https://equs.github.io/qcsys)

Built on JAX,  `qcsys` presents a scalable way to assemble and simulate systems of quantum circuits. 

## Installation

`qcsys` is published on PyPI. Simply run the following code to install the package:


```bash
pip install qcsys
```

For more details, please visit the getting started > installation section of our [docs](https://equs.github.io/qcsys/getting_started/installation.html).

## An Example

Here's an example on how to use `qcsys`:

```python
import qcsys as qs


# Devices ----


_, Ec_a, El_a = qs.calculate_lambda_over_four_resonator_zpf(3, 50)

resonator = qs.Resonator.create(
    10,
    {"Ec": Ec_a, "El": El_a},
    N_pre_diag=10,
)


Ec_q = 1
El_q = 0.5
Ej_q = 8

qubit = qs.Fluxonium.create(
    25,
    {"Ec": Ec_q, "El": El_q, "Ej": Ej_q, "phi_ext": 0.47},
    use_linear=False,
    N_pre_diag=100,
)

# System ----

g_rq = 0.3

devices = [resonator, qubit]
r_indx = 0
q_indx = 1
Ns = [device.N for device in devices]

a0 = qs.promote(resonator.ops["a"], r_indx, Ns)
a0_dag = qs.promote(resonator.ops["a_dag"], r_indx, Ns)

q0 = qs.promote(qubit.ops["a"], q_indx, Ns)
q0_dag = qs.promote(qubit.ops["a_dag"], q_indx, Ns)

couplings = []
couplings.append(-g_rq * (a0 - a0_dag) @ (q0 - q0_dag))

system = qs.System.create(devices, couplings=couplings)
system.params["g_rq"] = g_rq

Es, kets = system.calculate_eig()

# chi ----
χ_e = Es[1:, 1] - Es[:-1, 1]
χ_g = Es[1:, 0] - Es[:-1, 0]
χ = χ_e - χ_g

# kerr ----
# kerr[0,n] = (E(n+2, g) - E(n+1, g)) - (E(n+1, g) - E(n, g))
# kerr[1,n] = (E(n+2, e) - E(n+1, e)) - (E(n+1, e) - E(n, e))
K_g = (Es[2:, 0] - Es[1:-1, 0]) - (Es[1:-1, 0] - Es[0:-2, 0])
K_e = (Es[2:, 1] - Es[1:-1, 1]) - (Es[1:-1, 1] - Es[0:-2, 1])

χ, K_g, K_e
```



## Acknowledgements & History

**Core Devs:** [Shantanu A. Jha](https://github.com/Phionx), [Shoumik Chowdhury](https://github.com/shoumikdc)


This package was initially developed in early 2023 to aid in the design of a superconducting circuit device made for bosonic quantum error correction. This package was also briefly announced to the world at APS March Meeting 2023. Since then, this package has been open sourced and developed while conducting research in the Engineering Quantum Systems Group at MIT with invaluable advice from [Prof. William D. Oliver](https://equs.mit.edu/william-d-oliver/). 

## Citation

Thank you for taking the time to try our package out. If you found it useful in your research, please cite us as follows:

```bibtex
@software{jha2024jaxquantum,
  author = {Shantanu R. Jha and Shoumik Chowdhury and Max Hays and Jeff A. Grover and William D. Oliver},
  title  = {An auto differentiable and hardware accelerated software toolkit for quantum circuit design, simulation and control},
  url    = {https://github.com/EQuS/jaxquantum, https://github.com/EQuS/bosonic-jax, https://github.com/EQuS/qcsys},
  version = {0.1.0},
  year   = {2024},
}
```
> S. R. Jha, S. Chowdhury, M. Hays, J. A. Grover, W. D. Oliver. An auto differentiable and hardware accelerated software toolkit for quantum circuit design, simulation and control (2024), in preparation.


## Contributions & Contact

This package is open source and, as such, very open to contributions. Please don't hesitate to open an issue, report a bug, request a feature, or create a pull request. We are also open to deeper collaborations to create a tool that is more useful for everyone. If a discussion would be helpful, please email [shanjha@mit.edu](mailto:shanjha@mit.edu) to set up a meeting. 
