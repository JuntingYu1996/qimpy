from __future__ import annotations
from typing import Optional

import torch
import numpy as np

from qimpy import rc, TreeNode
from qimpy.io import CheckpointPath, CheckpointContext, InvalidInputException
from qimpy.profiler import stopwatch
from qimpy.transport import material
from .. import bose


class Spontaneous(TreeNode):
    ab_initio: material.ab_initio.AbInitio
    scale: float  #: Scale factor of spontaneous recombination

    constant_params: dict[str, torch.Tensor]  #: constant values of parameters

    def __init__(
        self,
        *,
        ab_initio: material.ab_initio.AbInitio,
        scale: float = 1.0,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ) -> None:
        """
        Initialize coherent light interaction.

        Parameters
        ----------
        scale
            :yaml:`Scale factor of spontaneous recombination`
        """
        super().__init__()
        self.scale = scale
        self.ab_initio = ab_initio
        self.constant_params = dict(
            scale=torch.tensor(scale, device=rc.device),
        )
        self.p = {}
        self.Gamma = {}

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        attrs = cp_path.attrs
        attrs["scale"] = self.scale
        return list(attrs.keys())

    def initialize_fields(self, params: dict[str, torch.Tensor], patch_id: int) -> None:
        self._initialize_fields(patch_id, **params)

    def _initialize_fields(
        self,
        patch_id: int,
        *,
        scale: torch.Tensor,
    ) -> None:
        ab_initio = self.ab_initio
        ab_initio.P
        dE = torch.abs(ab_initio.E[..., None] - ab_initio.E[:, None, :])
        Nk, Nb = ab_initio.E.shape
        Nmat = bose(dE, ab_initio.T)
        Nmat[Nmat == torch.inf] = 0.0
        triu = torch.triu_indices(Nb, Nb, 1).to(rc.device) # indices of upper triangle
        Nmat[:, triu[0], triu[1]] += 1
        Gamma = (Nmat * dE)[None, :, :, :] * ab_initio.P.swapaxes(0, 1)
        self.p[patch_id] = ab_initio.P.swapaxes(0, 1)[:, None, None]
        self.Gamma[patch_id] = Gamma[:, None, None]
        self.eye = torch.tile(torch.eye(Nb), (1, 1, Nk, 1, 1)).to(rc.device)

    @stopwatch
    def rho_dot(self, rho: torch.Tensor, t: float, patch_id: int) -> torch.Tensor:
        c = 137.03599968439292
        prefac = 2/(3 * c**3) * self.scale
        I_minus_rho = self.eye - rho
        return prefac * torch.sum(
            commutator(I_minus_rho[None] @ self.Gamma[patch_id] @ rho[None], 
                       self.p[patch_id]),
            axis = 0
        )

def commutator(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Commutator of two tensors (along final two dimensions)."""
    return A @ B - B @ A
