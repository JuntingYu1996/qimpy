from __future__ import annotations
from typing import Optional

import torch
import numpy as np

from qimpy import rc, TreeNode
from qimpy.io import CheckpointPath, CheckpointContext, InvalidInputException
from qimpy.profiler import stopwatch
from qimpy.transport import material


class Light(TreeNode):
    ab_initio: material.ab_initio.AbInitio
    coherent: bool  #: Whether term is Coherent or Lindbladian
    gauge: str  #: Gauge: one of velocity or length
    A0: torch.Tensor  #: Vector potential amplitude
    E0: torch.Tensor  #: Electric field amplitude
    smearing: float  #: Width of Gaussian
    symmetric: bool  #: Whether symmetric lindblad is used
    omega: dict[int, torch.Tensor]  #: light frequency
    t0: dict[int, torch.Tensor]  #: center of Gaussian pulse, if sigma is non-zero
    sigma: dict[int, torch.Tensor]  #: width of Gaussian pulse in time, if non-zero
    amp_mat: dict[int, torch.Tensor]  #: Amplitude matrix, precomputed A0 . P or E0 . R
    plus: dict[int, torch.Tensor]  #: TODO: document
    plus_deg: dict[int, torch.Tensor]  #: TODO: document
    minus: dict[int, torch.Tensor]  #: TODO: document
    minus_deg: dict[int, torch.Tensor]  #: TODO: document

    constant_params: dict[str, torch.Tensor]  #: constant values of parameters

    def __init__(
        self,
        *,
        ab_initio: material.ab_initio.AbInitio,
        coherent: bool = True,
        gauge: str = "velocity",
        A0: Optional[list[complex]] = None,
        E0: Optional[list[complex]] = None,
        omega: float = 0.0,
        t0: float = 0.0,
        sigma: float = 0.0,
        smearing: float = 0.001,
        symmetric: bool = False,
        matrix_form: bool = True,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ) -> None:
        """
        Initialize coherent light interaction.

        Parameters
        ----------
        coherent
            :yaml:`Switch between coherent and Lindbladian implementations.`
        gauge
            :yaml:`Switch between 'velocity' or 'length' gauge.`
        A0
            :yaml:`Vector potential amplitude.`
            TODO: specify details about differences in CW vs pulse mode.
            Exactly one of A0 or E0 must be specified.
        E0:
            :yaml:`Electric-field amplitude.`
            Exactly one of A0 or E0 must be specified.
        omega
            :yaml:`Angular frequency / photon energy of the light.`
        t0
            :yaml:`Center of Gaussian pulse, used only if sigma is non-zero.`
        sigma
            :yaml:`Time width of Gaussian pulse, if non-zero.`
        smearing
            :yaml:`Width of Gaussian function to represent delta function.`
        """
        super().__init__()
        self.coherent = coherent
        self.ab_initio = ab_initio
        self.gauge = gauge
        self.symmetric = symmetric
        self.matrix_form = matrix_form

        # Get amplitude from A0 or E0:
        if (A0 is None) == (E0 is None):
            raise InvalidInputException("Exactly one of A0 and E0 must be specified")
        if A0 is not None:
            A0_t = torch.tensor(A0, device=rc.device)
        else:  # E0 is not None
            A0_t = torch.tensor(E0, device=rc.device) / omega
        if A0_t.shape[-1] == 2:  # handle complex tensors
            A0_t = torch.view_as_complex(A0_t)
        else:
            A0_t = A0_t.to(torch.complex128)

        self.constant_params = dict(
            A0=A0_t,
            omega=torch.tensor(omega, device=rc.device),
            t0=torch.tensor(t0, device=rc.device),
            sigma=torch.tensor(sigma, device=rc.device),
            smearing=torch.tensor(smearing, device=rc.device),
        )
        self.t0 = {}
        self.sigma = {}
        if self.coherent:
            self.amp_mat = {}
            self.omega = {}
        elif self.symmetric:
            self.plus = {}
            self.plus_deg = {}
            self.minus = {}
            self.minus_deg = {}
        else:
            if matrix_form:
                self.gamma_plus = {}
                self.gamma_minus = {}
                self.Gamma_plus = {}
                self.Gamma_minus = {}
            else:
                self.P = {}
                self.Peye = {}

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        attrs = cp_path.attrs
        attrs["A0"] = self.constant_params["A0"].to(rc.cpu)
        attrs["omega"] = self.constant_params["omega"].item()
        attrs["t0"] = self.constant_params["t0"].item()
        attrs["sigma"] = self.constant_params["sigma"].item()
        attrs["smearing"] = self.constant_params["smearing"].item()
        attrs["coherent"] = self.coherent
        attrs["gauge"] = self.gauge
        attrs["symmetric"] = self.symmetric
        return list(attrs.keys())

    def initialize_fields(self, params: dict[str, torch.Tensor], patch_id: int) -> None:
        self._initialize_fields(patch_id, **params)

    def _initialize_fields(
        self,
        patch_id: int,
        *,
        A0: torch.Tensor,
        omega: torch.Tensor,
        t0: torch.Tensor,
        sigma: torch.Tensor,
        smearing: torch.Tensor,
    ) -> None:
        ab_initio = self.ab_initio
        if self.gauge == "velocity":
            amp_mat = torch.einsum("i, kiab -> kab", A0, ab_initio.P)
        elif self.gauge == "length":
            assert ab_initio.R is not None
            amp_mat = torch.einsum("i, kiab -> kab", A0 * omega, ab_initio.R)
        else:
            raise InvalidInputException(
                "Parameter gauge should only be velocity or length"
            )

        # reshape omega here for convenient broadcasting
        omega = (omega * torch.ones([1, 1]).to(rc.device))[..., None, None, None]
        self.t0[patch_id] = t0
        self.sigma[patch_id] = sigma
        if self.coherent:
            self.amp_mat[patch_id] = amp_mat[None, None]
            self.omega[patch_id] = omega
        elif self.symmetric:  #: lindblad version
            prefac = torch.sqrt(torch.sqrt(torch.pi / (8 * smearing**2)))
            exp_factor = -1.0 / (4 * smearing**2)
            Nk, Nb = ab_initio.E.shape
            dE = (ab_initio.E[..., None] - ab_initio.E[:, None, :])[None, None]
            plus = prefac * amp_mat * torch.exp(exp_factor * ((dE + omega) ** 2))
            minus = prefac * amp_mat * torch.exp(exp_factor * ((dE - omega) ** 2))
            plus_deg = plus.swapaxes(-2, -1).conj()
            minus_deg = minus.swapaxes(-2, -1).conj()
            self.plus[patch_id] = plus
            self.plus_deg[patch_id] = plus_deg
            self.minus[patch_id] = minus
            self.minus_deg[patch_id] = minus_deg
        else: # conventional Markov
            prefac = torch.sqrt(torch.pi / (32 * smearing**2))
            dE = (ab_initio.E[..., None] - ab_initio.E[:, None, :])[None, None]
            exp_factor = -1.0 / (2 * smearing**2)
            delta = torch.exp(exp_factor * ((dE - omega) ** 2))
            if self.matrix_form:
                gamma_plus = amp_mat
                gamma_minus = amp_mat.swapaxes(-1,-2).conj()
                Gamma_minus = prefac * amp_mat * delta
                Gamma_plus = Gamma_minus.swapaxes(-1,-2).conj()
                self.gamma_plus[patch_id] = gamma_plus
                self.gamma_minus[patch_id] = gamma_minus
                self.Gamma_plus[patch_id] = Gamma_plus
                self.Gamma_minus[patch_id] = Gamma_minus
            else:
                P = prefac * (
                    torch.einsum("...kac, kac, kbd -> ...kabcd", delta, amp_mat, amp_mat.conj())
                    + torch.einsum("...kca, kca, kdb -> ...kabcd", delta, amp_mat.conj(), amp_mat)
                )
                Peye = torch.einsum("...kabcc -> ...kab", P)
                self.P[patch_id] = P
                self.Peye[patch_id] = Peye

    @stopwatch
    def rho_dot(self, rho: torch.Tensor, t: float, patch_id: int) -> torch.Tensor:
        t0 = self.t0[patch_id]
        sigma = self.sigma[patch_id]
        # shape of prefac must be (Nx, Ny, 1, 1, 1)
        if sigma > 0:
            prefac = (
                torch.exp(-((t - t0) ** 2) / (2 * sigma**2))
                / torch.sqrt(torch.sqrt(np.pi * sigma**2))
                * torch.ones(rho.shape[:2] + (1, 1, 1)).to(rc.device)
            )
        else:
            prefac = torch.ones(rho.shape[:2] + (1, 1, 1)).to(rc.device)

        if self.coherent:
            omega = self.omega[patch_id]
            prefac = (
                -0.5j * torch.exp(-1j * omega * t) * prefac
            )  # Louiville, symmetrization
            interaction = prefac * self.amp_mat[patch_id]
            return (interaction - interaction.swapaxes(-2, -1).conj()) @ rho
        elif self.symmetric:
            prefac = 0.5 * prefac**2
            I_minus_rho = self.ab_initio.eye_bands - rho
            plus = self.plus[patch_id]
            minus = self.minus[patch_id]
            plus_deg = self.plus_deg[patch_id]
            minus_deg = self.minus_deg[patch_id]
            return prefac * (
                commutator(I_minus_rho @ plus @ rho, plus_deg)
                + commutator(I_minus_rho @ minus @ rho, minus_deg)
            )
        else:
            prefac = prefac**2
            I_minus_rho = self.ab_initio.eye_bands - rho
            if self.matrix_form: # Faster
                gamma_plus = self.gamma_plus[patch_id]
                gamma_minus = self.gamma_minus[patch_id]
                Gamma_plus = self.Gamma_plus[patch_id]
                Gamma_minus = self.Gamma_minus[patch_id]
                return prefac * (
                    commutator(I_minus_rho @ Gamma_plus @ rho, gamma_plus) + 
                    commutator(I_minus_rho @ Gamma_minus @ rho, gamma_minus)
                )
            else:
                P = self.P[patch_id]
                Peye = self.Peye[patch_id]
                Prho = torch.einsum("...abcd, ...cd -> ...ab", P, rho)
                # return prefac * (I_minus_rho @ Prho - rho @ (Peye - Prho))
                return prefac * (Prho - rho @ Peye)


def commutator(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Commutator of two tensors (along final two dimensions)."""
    return A @ B - B @ A
