# transmon_generic_CPR.py
#
# This file is part of scqubits: a Python package for superconducting qubits,
# Quantum 5, 583 (2021). https://quantum-journal.org/papers/q-2021-11-17-583/
#
#    Copyright (c) 2019 and later, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

import math

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy as sp

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from scipy.sparse import diags
from scipy.sparse import spdiags
from scipy.sparse.linalg import eigsh

import scqubits.core.constants as constants
import scqubits.core.descriptors as descriptors
import scqubits.core.discretization as discretization
import scqubits.core.qubit_base as base
import scqubits.core.storage as storage
import scqubits.io_utils.fileio_serializers as serializers
import scqubits.utils.plot_defaults as defaults
import scqubits.utils.plotting as plot

from scqubits.core.discretization import Grid1d
from scqubits.core.noise import NoisySystem
from scqubits.core.storage import WaveFunction

LevelsTuple = Tuple[int, ...]
Transition = Tuple[int, int]
TransitionsTuple = Tuple[Transition, ...]

# Cooper pair box / transmon


class Transmon_g_CPR(base.QubitBaseClass1d, serializers.Serializable, NoisySystem):
    r"""Class for the Cooper-pair-box and transmon qubit. The Hamiltonian is
    represented in dense form in the phase basis,
    :math:`H_\text{CPB}=4E_\text{C}(\frac{i\partial}{\partial \hat{ \phi}}-n_g)^2-E_\text{J}(f(\phi))`.
    Initialize with, for example::

        Transmon(EJ=1.0, EC=2.0, ng=0.2, ncut=30)

    Parameters
    ----------
    EJ:
       Josephson energy
    CPR:
       function of the phase variable
    EC:
        charging energy
    ng:
        offset charge
    ncut:
        charge basis cutoff, `n = -ncut, ..., ncut`
    truncated_dim:
        desired dimension of the truncated quantum system; expected: truncated_dim > 1
    id_str:
        optional string by which this instance can be referred to in `HilbertSpace`
        and `ParameterSweep`. If not provided, an id is auto-generated.
    """
    EJ = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    EC = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    ng = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    ncut = descriptors.WatchedProperty(int, "QUANTUMSYSTEM_UPDATE")

    def __init__(
        self,
        EJ: float,
        CPR,
        EC: float,
        ng: float,
        ncut: int,
        truncated_dim: int = 6,
        id_str: Optional[str] = None,
        evals_method: Optional[str] = None,
        evals_method_options: Optional[dict] = None,
        esys_method: Optional[str] = None,
        esys_method_options: Optional[dict] = None,
    ) -> None:
        base.QubitBaseClass.__init__(
            self,
            id_str=id_str,
            evals_method=evals_method,
            evals_method_options=evals_method_options,
            esys_method=esys_method,
            esys_method_options=esys_method_options,
        )
        self.EJ = EJ
        self.CPR = CPR
        self.EC = EC
        self.ng = ng
        self.ncut = ncut
        self.truncated_dim = truncated_dim
        self._default_grid = discretization.Grid1d(-np.pi, np.pi, self.hilbertdim())
        self._default_n_range = (-5, 6)

    def CPR_default(x):
            return 1 - np.cos(x)
    @staticmethod
    def default_params(self) -> Dict[str, Any]:
        return {"EJ": 15.0, "EC": 0.3, "CPR"=self.CPR_default, "ng": 0.0, "ncut": 30, "truncated_dim": 10}

    @classmethod
    def supported_noise_channels(cls) -> List[str]:
        """Return a list of supported noise channels"""
        return [
            "tphi_1_over_f_cc",
            "tphi_1_over_f_ng",
            "t1_capacitive",
            "t1_charge_impedance",
        ]

    @classmethod
    def effective_noise_channels(cls) -> List[str]:
        """Return a default list of channels used when calculating effective t1 and
        t2 noise."""
        noise_channels = cls.supported_noise_channels()
        noise_channels.remove("t1_charge_impedance")
        return noise_channels

    def _hamiltonian_EC(self):
        N=self.hilbertdim()
        ng=self.ng
        dx=2*np.pi/N
        b=spdiags([1+1j*ng*dx], -N+1, N, N)
        b=(b+b.T.conjugate())/dx**2
        d2dx2 = b + diags([-1-1j*ng*dx, +2 +(ng**2)*(dx**2), -1 +1j*ng*dx], offsets=[-1, 0, 1], shape=(N, N))/(dx**2)
        T = 4*d2dx2*self.EC
        return T

    def _hamiltonian_EJ(self):
        x=self._default_grid.make_linspace()
        V = diags(CPR(x))*self.EJ
        return V

    def _evals_calc(self, evals_count: int) -> ndarray:
        H = self._hamiltonian_EC() + self._hamiltonian_EJ()
        evals = eigsh(H, which="SM", k=evals_count)[0]
        return evals

    def _esys_calc(self, evals_count: int) -> Tuple[ndarray, ndarray]:
        H = self._hamiltonian_EC() + self._hamiltonian_EJ()
        evals, evecs = eigsh(H, which="SM", k=evals_count)
        return evals, evecs

    @staticmethod
    def find_EJ_EC(
        E01: float, anharmonicity: float, ng=0, ncut=30
    ) -> Tuple[float, float]:
        """
        Finds the EJ and EC values given a qubit splitting `E01` and `anharmonicity`.

        Parameters
        ----------
            E01:
                qubit transition energy
            anharmonicity:
                absolute qubit anharmonicity, (E2-E1) - (E1-E0)
            ng:
                offset charge (default: 0)
            ncut:
                charge number cutoff (default: 30)

        Returns
        -------
            A tuple of the EJ and EC values representing the best fit.
        """
        tmon = Transmon_g_CPR(EJ=10.0, EC=0.1, ng=ng, ncut=ncut)
        start_EJ_EC = np.array([tmon.EJ, tmon.EC])

        def cost_func(EJ_EC: Tuple[float, float]) -> float:
            EJ, EC = EJ_EC
            tmon.EJ = EJ
            tmon.EC = EC
            energies = tmon.eigenvals(evals_count=3)
            computed_E01 = energies[1] - energies[0]
            computed_anharmonicity = energies[2] - energies[1] - computed_E01
            cost = (E01 - computed_E01) ** 2
            cost += (anharmonicity - computed_anharmonicity) ** 2
            return cost

        return sp.optimize.minimize(cost_func, start_EJ_EC).x

    def n_operator(
        self, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> ndarray:
        """
        Returns charge operator n in the charge or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns charge operator n in the charge basis.
            If `True`, energy eigenspectrum is computed, returns charge operator n in the energy eigenbasis.
            If `energy_esys = esys`, where `esys` is a tuple containing two ndarrays (eigenvalues and energy
            eigenvectors), returns charge operator n in the energy eigenbasis, and does not have to recalculate the
            eigenspectrum.

        Returns
        -------
            Charge operator n in chosen basis as ndarray.
            For `energy_esys=True`, n has dimensions of `truncated_dim` x `truncated_dim`.
            If an actual eigensystem is handed to `energy_sys`, then `n` has dimensions of m x m,
            where m is the number of given eigenvectors.
        """
        N=self.hilbertdim()
        dx=2*np.pi/N
        b=spdiags([1], -N+1, N, N)
        b=(b+b.T.conjugate())/dx**2
        d2dx2 = b + diags([-1, +2, -1], offsets=[-1, 0, 1], shape=(N, N))/(dx**2)
        T = d2dx2
        native = T.toarray()
        return self.process_op(native_op=native, energy_esys=energy_esys)


    def CPR_phi_operator(
        self, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> ndarray:
        """
        Returns operator :math:`\\cos \\varphi` in the charge or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns operator :math:`\\cos \\varphi` in the charge basis.
            If `True`, the energy eigenspectrum is computed, returns operator :math:`\\cos \\varphi` in the energy eigenbasis.
            If `energy_esys = esys`, where esys is a tuple containing two ndarrays (eigenvalues and energy eigenvectors),
            returns operator :math:`\\cos \\varphi` in the energy eigenbasis, and does not have to recalculate eigenspectrum.

        Returns
        -------
            Operator :math:`\\cos \\varphi` in chosen basis as ndarray. If the eigenenergy basis is chosen,
            unless energy_esys is specified, :math:`\\cos \\varphi` has dimensions of truncated_dim
            x `truncated_dim`. Otherwise, if eigenenergy basis is chosen, :math:`\\cos \\varphi` has dimensions of m x m,
            for m given eigenvectors.
        """
        x=self._default_grid.make_linspace()
        V = diags(CPR(x))
        return self.process_op(native_op=V, energy_esys=energy_esys)

    def d_CPR_phi_operator(
        self, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> ndarray:
        """
        Returns operator :math:`\\cos \\varphi` in the charge or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns operator :math:`\\cos \\varphi` in the charge basis.
            If `True`, the energy eigenspectrum is computed, returns operator :math:`\\cos \\varphi` in the energy eigenbasis.
            If `energy_esys = esys`, where esys is a tuple containing two ndarrays (eigenvalues and energy eigenvectors),
            returns operator :math:`\\cos \\varphi` in the energy eigenbasis, and does not have to recalculate eigenspectrum.

        Returns
        -------
            Operator :math:`\\cos \\varphi` in chosen basis as ndarray. If the eigenenergy basis is chosen,
            unless energy_esys is specified, :math:`\\cos \\varphi` has dimensions of truncated_dim
            x `truncated_dim`. Otherwise, if eigenenergy basis is chosen, :math:`\\cos \\varphi` has dimensions of m x m,
            for m given eigenvectors.
        """
        x=self._default_grid.make_linspace()
        V = diags(CPR(x))
        return self.process_op(native_op=V, energy_esys=energy_esys)



    def hamiltonian(
        self, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> ndarray:
        """
        Returns Hamiltonian in the charge or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns Hamiltonian in the charge basis.
            If `True`, the energy eigenspectrum is computed; returns Hamiltonian in the energy eigenbasis.
            If `energy_esys = esys`, where `esys` is a tuple containing two ndarrays (eigenvalues and energy
            eigenvectors); then return the Hamiltonian in the energy eigenbasis, do not recalculate eigenspectrum.

        Returns
        -------
            Hamiltonian in chosen basis as ndarray. For `energy_esys=False`, the Hamiltonian has dimensions of
            `truncated_dim` x `truncated_dim`. For `energy_sys=esys`, the Hamiltonian has dimensions of m x m,
            for m given eigenvectors.
        """
        H = (self._hamiltonian_EC() + self._hamiltonian_EJ()).toarray()
        return self.process_hamiltonian(
            native_hamiltonian=H, energy_esys=energy_esys
        )

    def d_hamiltonian_d_ng(
        self, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> ndarray:
        """
        Returns operator representing a derivative of the Hamiltonian with respect to
        charge offset `ng` in the charge or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns operator in the charge basis.
            If `True`, the energy eigenspectrum is computed, returns operator in the energy eigenbasis.
            If `energy_esys = esys`, where `esys` is a tuple containing two ndarrays (eigenvalues and energy
            eigenvectors), returns operator in the energy eigenbasis, and does not have to recalculate eigenspectrum.

        Returns
        -------
            Operator in chosen basis as ndarray. If the eigenenergy basis is chosen,
            unless `energy_esys` is specified, operator has dimensions of `truncated_dim`
            x `truncated_dim`. Otherwise, if eigenenergy basis is chosen, operator has dimensions of m x m,
            for m given eigenvectors.
        """
        native = -8 * self.EC * self.n_operator(energy_esys=energy_esys)
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def d_hamiltonian_d_EJ(
        self, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> ndarray:
        """
        Returns operator representing a derivative of the Hamiltonian with respect to
        EJ in the charge or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns operator in the charge basis.
            If `True`, the energy eigenspectrum is computed, returns operator in the energy eigenbasis.
            If `energy_esys = esys`, where `esys` is a tuple containing two ndarrays (eigenvalues and energy eigenvectors),
            returns operator in the energy eigenbasis, and does not have to recalculate eigenspectrum.

        Returns
        -------
            Operator in chosen basis as ndarray. If the eigenenergy basis is chosen,
            unless `energy_esys` is specified, operator has dimensions of `truncated_dim`
            x `truncated_dim`. Otherwise, if eigenenergy basis is chosen, operator has dimensions of m x m,
            for m given eigenvectors.
        """
        native = -self.d_CPR_phi_operator()
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def hilbertdim(self) -> int:
        """Returns Hilbert space dimension"""
        return 2 * self.ncut + 1

    def potential(self, phi: Union[float, ndarray]) -> ndarray:
        """Transmon phase-basis potential evaluated at `phi`.

        Parameters
        ----------
        phi:
            phase variable value
        """
        return -self.EJ * self.CPR(phi)


class Fluxonium_g_CPR(base.QubitBaseClass1d, serializers.Serializable, NoisySystem):
    r"""Class for the fluxonium qubit. Hamiltonian :math:`H_\text{fl}=-4E_\text{
    C}\partial_\phi^2-E_\text{J}\cos(\phi+\varphi_\text{ext}) +\frac{1}{2}E_L\phi^2`
    is represented in dense form. The employed basis is the EC-EL harmonic oscillator
    basis. The cosine term in the potential is handled via matrix exponentiation.
    Initialize with, for example::


        Transmon(EJ=1.0, EC=2.0, ng=0.2, ncut=30)

    Parameters
    ----------
    EJ:
       Josephson energy
    EL:
       Inductance energy
    CPR:
       function of the phase variable
    EC:
        charging energy
    ng:
        offset charge
    ncut:
        charge basis cutoff, `n = -ncut, ..., ncut`
    truncated_dim:
        desired dimension of the truncated quantum system; expected: truncated_dim > 1
    id_str:
        optional string by which this instance can be referred to in `HilbertSpace`
        and `ParameterSweep`. If not provided, an id is auto-generated.
    """
    EJ = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    EL = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    EC = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    ng = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    ncut = descriptors.WatchedProperty(int, "QUANTUMSYSTEM_UPDATE")
    flux = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")

    def __init__(
        self,
        EJ: float,
        EL: float,
        CPR,
        EC: float,
        ng: float,
        ncut: int,
        flux: float,
        truncated_dim: int = 6,
        id_str: Optional[str] = None,
        evals_method: Optional[str] = None,
        evals_method_options: Optional[dict] = None,
        esys_method: Optional[str] = None,
        esys_method_options: Optional[dict] = None,
    ) -> None:
        base.QubitBaseClass.__init__(
            self,
            id_str=id_str,
            evals_method=evals_method,
            evals_method_options=evals_method_options,
            esys_method=esys_method,
            esys_method_options=esys_method_options,
        )
        self.EJ = EJ
        self.EL = EL
        self.CPR = CPR
        self.EC = EC
        self.ng = ng
        self.ncut = ncut
        self.flux = flux
        self.truncated_dim = truncated_dim
        self._default_grid = discretization.Grid1d(-4.5*np.pi, 4.5*np.pi, self.hilbertdim())
        self._default_n_range = (-5, 6)

    def CPR_default(x):
            return 1 - np.cos(x)
    @staticmethod
    def default_params(self) -> Dict[str, Any]:
        return {"EL": 0.5, "EJ": 15.0, "EC": 0.3, "CPR"=self.CPR_default, "ng": 0.0, "ncut": 30, "truncated_dim": 10, "flux": 0.0,}

    @classmethod
    def supported_noise_channels(cls) -> List[str]:
        """Return a list of supported noise channels"""
        return [
            "tphi_1_over_f_cc",
            "tphi_1_over_f_ng",
            "t1_capacitive",
            "t1_charge_impedance",
        ]

    @classmethod
    def effective_noise_channels(cls) -> List[str]:
        """Return a default list of channels used when calculating effective t1 and
        t2 noise."""
        noise_channels = cls.supported_noise_channels()
        noise_channels.remove("t1_charge_impedance")
        return noise_channels

    def _hamiltonian_EC(self):
        N=self.hilbertdim()
        ng=self.ng
        dx=9*np.pi/N
        b=spdiags([1+1j*ng*dx], -N+1, N, N)
        b=(b+b.T.conjugate())/dx**2
        d2dx2 = b + diags([-1-1j*ng*dx, +2 +(ng**2)*(dx**2), -1 +1j*ng*dx], offsets=[-1, 0, 1], shape=(N, N))/(dx**2)
        T = 4*d2dx2*self.EC
        return T

    def _hamiltonian_EJ(self):
        x=self._default_grid.make_linspace()
        V = diags(CPR(x+2*np.pi*self.flux))*self.EJ-self.EL*diags(x**2)/2
        return V

    def _evals_calc(self, evals_count: int) -> ndarray:
        H = self._hamiltonian_EC() + self._hamiltonian_EJ()
        evals = eigsh(H, which="SM", k=evals_count)[0]
        return evals

    def _esys_calc(self, evals_count: int) -> Tuple[ndarray, ndarray]:
        H = self._hamiltonian_EC() + self._hamiltonian_EJ()
        evals, evecs = eigsh(H, which="SM", k=evals_count)
        return evals, evecs

    @staticmethod
    def find_EJ_EC(
        E01: float, anharmonicity: float, ng=0, ncut=30
    ) -> Tuple[float, float]:
        """
        Finds the EJ and EC values given a qubit splitting `E01` and `anharmonicity`.

        Parameters
        ----------
            E01:
                qubit transition energy
            anharmonicity:
                absolute qubit anharmonicity, (E2-E1) - (E1-E0)
            ng:
                offset charge (default: 0)
            ncut:
                charge number cutoff (default: 30)

        Returns
        -------
            A tuple of the EJ and EC values representing the best fit.
        """
        tmon = Transmon_g_CPR(EJ=10.0, EC=0.1, ng=ng, ncut=ncut)
        start_EJ_EC = np.array([tmon.EJ, tmon.EC])

        def cost_func(EJ_EC: Tuple[float, float]) -> float:
            EJ, EC = EJ_EC
            tmon.EJ = EJ
            tmon.EC = EC
            energies = tmon.eigenvals(evals_count=3)
            computed_E01 = energies[1] - energies[0]
            computed_anharmonicity = energies[2] - energies[1] - computed_E01
            cost = (E01 - computed_E01) ** 2
            cost += (anharmonicity - computed_anharmonicity) ** 2
            return cost

        return sp.optimize.minimize(cost_func, start_EJ_EC).x

    def n_operator(
        self, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> ndarray:
        """
        Returns charge operator n in the charge or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns charge operator n in the charge basis.
            If `True`, energy eigenspectrum is computed, returns charge operator n in the energy eigenbasis.
            If `energy_esys = esys`, where `esys` is a tuple containing two ndarrays (eigenvalues and energy
            eigenvectors), returns charge operator n in the energy eigenbasis, and does not have to recalculate the
            eigenspectrum.

        Returns
        -------
            Charge operator n in chosen basis as ndarray.
            For `energy_esys=True`, n has dimensions of `truncated_dim` x `truncated_dim`.
            If an actual eigensystem is handed to `energy_sys`, then `n` has dimensions of m x m,
            where m is the number of given eigenvectors.
        """
        N=self.hilbertdim()
        dx=9*np.pi/N
        b=spdiags([1], -N+1, N, N)
        b=(b+b.T.conjugate())/dx**2
        d2dx2 = b + diags([-1, +2, -1], offsets=[-1, 0, 1], shape=(N, N))/(dx**2)
        T = d2dx2
        native = T.toarray()
        return self.process_op(native_op=native, energy_esys=energy_esys)


    def CPR_phi_operator(
        self, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> ndarray:
        """
        Returns operator :math:`\\cos \\varphi` in the charge or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns operator :math:`\\cos \\varphi` in the charge basis.
            If `True`, the energy eigenspectrum is computed, returns operator :math:`\\cos \\varphi` in the energy eigenbasis.
            If `energy_esys = esys`, where esys is a tuple containing two ndarrays (eigenvalues and energy eigenvectors),
            returns operator :math:`\\cos \\varphi` in the energy eigenbasis, and does not have to recalculate eigenspectrum.

        Returns
        -------
            Operator :math:`\\cos \\varphi` in chosen basis as ndarray. If the eigenenergy basis is chosen,
            unless energy_esys is specified, :math:`\\cos \\varphi` has dimensions of truncated_dim
            x `truncated_dim`. Otherwise, if eigenenergy basis is chosen, :math:`\\cos \\varphi` has dimensions of m x m,
            for m given eigenvectors.
        """
        x=self._default_grid.make_linspace()
        V = diags(CPR(x+2*np.pi*self.flux))
        return self.process_op(native_op=V, energy_esys=energy_esys)

    def d_CPR_phi_operator(
        self, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> ndarray:
        """
        Returns operator :math:`\\cos \\varphi` in the charge or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns operator :math:`\\cos \\varphi` in the charge basis.
            If `True`, the energy eigenspectrum is computed, returns operator :math:`\\cos \\varphi` in the energy eigenbasis.
            If `energy_esys = esys`, where esys is a tuple containing two ndarrays (eigenvalues and energy eigenvectors),
            returns operator :math:`\\cos \\varphi` in the energy eigenbasis, and does not have to recalculate eigenspectrum.

        Returns
        -------
            Operator :math:`\\cos \\varphi` in chosen basis as ndarray. If the eigenenergy basis is chosen,
            unless energy_esys is specified, :math:`\\cos \\varphi` has dimensions of truncated_dim
            x `truncated_dim`. Otherwise, if eigenenergy basis is chosen, :math:`\\cos \\varphi` has dimensions of m x m,
            for m given eigenvectors.
        """
        x=self._default_grid.make_linspace()
        V = diags(CPR(x))
        return self.process_op(native_op=V, energy_esys=energy_esys)



    def hamiltonian(
        self, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> ndarray:
        """
        Returns Hamiltonian in the charge or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns Hamiltonian in the charge basis.
            If `True`, the energy eigenspectrum is computed; returns Hamiltonian in the energy eigenbasis.
            If `energy_esys = esys`, where `esys` is a tuple containing two ndarrays (eigenvalues and energy
            eigenvectors); then return the Hamiltonian in the energy eigenbasis, do not recalculate eigenspectrum.

        Returns
        -------
            Hamiltonian in chosen basis as ndarray. For `energy_esys=False`, the Hamiltonian has dimensions of
            `truncated_dim` x `truncated_dim`. For `energy_sys=esys`, the Hamiltonian has dimensions of m x m,
            for m given eigenvectors.
        """
        H = (self._hamiltonian_EC() + self._hamiltonian_EJ()).toarray()
        return self.process_hamiltonian(
            native_hamiltonian=H, energy_esys=energy_esys
        )

    def d_hamiltonian_d_ng(
        self, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> ndarray:
        """
        Returns operator representing a derivative of the Hamiltonian with respect to
        charge offset `ng` in the charge or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns operator in the charge basis.
            If `True`, the energy eigenspectrum is computed, returns operator in the energy eigenbasis.
            If `energy_esys = esys`, where `esys` is a tuple containing two ndarrays (eigenvalues and energy
            eigenvectors), returns operator in the energy eigenbasis, and does not have to recalculate eigenspectrum.

        Returns
        -------
            Operator in chosen basis as ndarray. If the eigenenergy basis is chosen,
            unless `energy_esys` is specified, operator has dimensions of `truncated_dim`
            x `truncated_dim`. Otherwise, if eigenenergy basis is chosen, operator has dimensions of m x m,
            for m given eigenvectors.
        """
        native = -8 * self.EC * self.n_operator(energy_esys=energy_esys)
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def d_hamiltonian_d_EJ(
        self, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> ndarray:
        """
        Returns operator representing a derivative of the Hamiltonian with respect to
        EJ in the charge or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns operator in the charge basis.
            If `True`, the energy eigenspectrum is computed, returns operator in the energy eigenbasis.
            If `energy_esys = esys`, where `esys` is a tuple containing two ndarrays (eigenvalues and energy eigenvectors),
            returns operator in the energy eigenbasis, and does not have to recalculate eigenspectrum.

        Returns
        -------
            Operator in chosen basis as ndarray. If the eigenenergy basis is chosen,
            unless `energy_esys` is specified, operator has dimensions of `truncated_dim`
            x `truncated_dim`. Otherwise, if eigenenergy basis is chosen, operator has dimensions of m x m,
            for m given eigenvectors.
        """
        native = -self.d_CPR_phi_operator()
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def hilbertdim(self) -> int:
        """Returns Hilbert space dimension"""
        return 2 * self.ncut + 1

    def potential(self, phi: Union[float, ndarray]) -> ndarray:
        """Transmon phase-basis potential evaluated at `phi`.

        Parameters
        ----------
        phi:
            phase variable value
        """
        return -self.EJ * self.CPR(phi + 2*np.pi*self.flux) + self.EL * phi**2 / 2
