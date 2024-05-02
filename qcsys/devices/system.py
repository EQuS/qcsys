""" System."""

from functools import partial
from typing import List, Optional, Dict, Any, Union
import math

from flax import struct
from jax import jit, vmap, Array
from jax import config
import jaxquantum as jqt
import jax.numpy as jnp

from qcsys.devices.base import Device
from qcsys.devices.drive import Drive

config.update("jax_enable_x64", True)


@partial(jit, static_argnums=(0,))
def calculate_eig(Ns, H: jqt.Qarray):
    N_tot = math.prod(Ns)
    edxs = jnp.arange(N_tot)

    vals, kets = jnp.linalg.eigh(H.data)
    kets = kets.T

    def calc_quantum_number(edx):
        argmax = jnp.argmax(jnp.abs(kets[edx]))
        val = vals[edx]  # - vals[0]
        return val, argmax, kets[edx]

    quantum_numbers = vmap(calc_quantum_number)(edxs)

    def calc_order(edx):
        indx = jnp.argmin(jnp.abs(edx - quantum_numbers[1]))
        return quantum_numbers[0][indx], quantum_numbers[2][indx]

    Es, kets = vmap(calc_order)(edxs)

    return (
        jnp.reshape(Es, Ns),
        jnp.reshape(kets, (*Ns, -1)),
    )


def promote(op: jqt.Qarray, device_num, Ns):
    I_ops = [jqt.identity(N) for N in Ns]
    return jqt.tensor(*I_ops[:device_num], op, *I_ops[device_num + 1 :])


@struct.dataclass
class System:
    Ns: List[int] = struct.field(pytree_node=False)
    devices: List[Union[Device, Drive]]
    couplings: List[Array]
    params: Dict[str, Any]

    @classmethod
    def create(
        cls,
        devices: List[Union[Device, Drive]],
        couplings: Optional[List[Array]] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        Ns = tuple([device.N for device in devices])
        couplings = couplings if couplings is not None else []
        params = params if params is not None else {}
        return cls(Ns, devices, couplings, params)

    def promote(self, op, device_num):
        return promote(op, device_num, self.Ns)

    def get_H_bare(self):
        I_ops = [jqt.identity(N) for N in self.Ns]
        H = 0
        for j, device in enumerate(self.devices):
            H += self.promote(device.get_H(), j)
        return H

    def get_H_couplings(self):
        H = 0
        for coupling in self.couplings:
            H += coupling
        return H
    
    def get_H(self):
        H_bare = self.get_H_bare()
        H_couplings = self.get_H_couplings()
        return H_bare + H_couplings

    def calculate_eig(self):
        H = self.get_H()
        return calculate_eig(self.Ns, H)
