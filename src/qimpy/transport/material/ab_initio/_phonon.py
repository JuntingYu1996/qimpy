from __future__ import annotations
from typing import Union

import numpy as np
import torch

from qimpy import rc, log, TreeNode
from qimpy.io import (
    Checkpoint,
    CheckpointPath,
    InvalidInputException,
    CheckpointContext,
)
from qimpy.mpi import BufferView
from qimpy.math import ceildiv
from qimpy.profiler import stopwatch
from qimpy.transport import material
from .. import bose


class Phonon(TreeNode):
    ab_initio: material.ab_initio.AbInitio
    P: torch.Tensor  #: P and Pbar operators stacked together
    rho_dot0: torch.Tensor  #: rho_dot(rho0) for detailed balance correction
    smearing: float  #: Width of delta function
    form: str  #: Conventional Markov or Lindblad
    matrix_form: bool  #: Whether to use matrix form or P form

    constant_params: dict[str, torch.Tensor]  #: constant values of parameters
    scale_factor: dict[int, torch.Tensor]  #: scale factors per patch

    @stopwatch
    def __init__(
        self,
        *,
        ab_initio: material.ab_initio.AbInitio,
        data_file: Checkpoint,
        smearing: float = 0.001,
        scale_factor: float = 1.0,
        form: str = "conventional",
        matrix_form : bool = False,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ) -> None:
        """
        Initialize ab initio phonon scattering.

        Parameters
        ----------
        scale_factor
            :yaml:`Overall scale factor for scattering rate.`
        """
        super().__init__()
        self.ab_initio = ab_initio
        self.data_file = data_file
        self.smearing = smearing
        self.form = form
        self.matrix_form = matrix_form
        if not form in ["conventional", "lindblad"]:
            raise InvalidInputException("form must be conventional or lindblad")
        if not bool(data_file.attrs["ePhEnabled"]):
            raise InvalidInputException("No e-ph scattering available in data file")

        log.info("Constructing P tensor")
        nk = ab_initio.k_division.n_tot
        ik_start = ab_initio.k_division.i_start
        ik_stop = ab_initio.k_division.i_stop
        nk_mine = ab_initio.nk_mine
        n_bands = ab_initio.n_bands
        n_bands_sq = n_bands**2
        block_shape_flat = (-1, n_bands_sq, n_bands_sq)
        exp_factor = -0.5 / smearing**2
        prefactor = np.pi * ab_initio.wk / np.sqrt(2 * np.pi * smearing**2)

        # Collect together evecs for all k if needed:
        if (ab_initio.comm.size == 1) or (ab_initio.evecs is None):
            evecs = ab_initio.evecs
        else:
            sendbuf = ab_initio.evecs.contiguous()
            recvbuf = torch.zeros(
                (nk,) + sendbuf.shape[1:], dtype=sendbuf.dtype, device=rc.device
            )
            mpi_type = rc.mpi_type[sendbuf.dtype]
            recv_prev = ab_initio.k_division.n_prev * n_bands_sq
            ab_initio.comm.Allgatherv(
                (BufferView(sendbuf), np.prod(sendbuf.shape), 0, mpi_type),
                (BufferView(recvbuf), np.diff(recv_prev), recv_prev[:-1], mpi_type),
            )
            evecs = recvbuf

        def get_mine(ik) -> Union[torch.Tensor, slice, None]:
            """Utility to fetch efficient slices of relevant k-points."""
            if ab_initio.k_division.n_procs == 1:
                return slice(None)  # no split, so bypass search
            sel = torch.where(torch.logical_and(ik >= ik_start, ik < ik_stop))[0]
            if not len(sel):
                return None
            sel_start = sel[0].item()
            sel_stop = sel[-1].item() + 1
            if sel_stop - sel_start == len(sel):
                return slice(sel_start, sel_stop)  # contiguous
            return sel  # general selection

        def conventionalP(einsum_path, delta, g1, g2):
            return torch.einsum(einsum_path, delta, g1, g2).reshape(block_shape_flat)

        def lindbladP(einsum_path, G1, G2):
            return torch.einsum(einsum_path, G1, G2).reshape(block_shape_flat)

        # Operate in blocks to reduce working memory:
        cp_ikpair = data_file["ikpair"]
        n_pairs = cp_ikpair.shape[0]
        n_blocks = 1000  # may want to set this from input later on
        block_size = ceildiv(n_pairs, n_blocks)
        block_lims = np.minimum(
            np.arange(0, n_pairs + block_size - 1, block_size), n_pairs
        )
        cp_omega_ph = data_file["omega_ph"]
        cp_g = data_file["G"] # original electron-phonon matrix
        cp_E = torch.from_numpy(np.array(data_file["E"])).to(rc.device)
        if not matrix_form:
            P_shape = (2, nk_mine * nk, n_bands_sq, n_bands_sq)
            nbytes = 16 * np.prod(P_shape)
            log.info(f"Memory for P matrix: {bytes2memory(nbytes)}.")
            P = torch.zeros(P_shape, dtype=torch.complex128, device=rc.device)
            for block_start, block_stop in zip(block_lims[:-1], block_lims[1:]):
                # Read current slice of data:
                cur = slice(block_start, block_stop)
                ik, jk = torch.from_numpy(cp_ikpair[cur]).to(rc.device).T
                omega_ph = torch.from_numpy(cp_omega_ph[cur]).to(rc.device)
                g = torch.from_numpy(cp_g[cur]).to(rc.device)
                if evecs is not None:
                    g = torch.einsum("kba, kbc, kcd -> kad", evecs[ik].conj(), g, evecs[jk])
                bose_occ = bose(omega_ph, ab_initio.T)[:, None, None]
                wm = prefactor * bose_occ
                wp = prefactor * (bose_occ + 1.0)

                # Contributions to P(ik,jk):
                if (sel := get_mine(ik)) is not None:
                    i_pair = (ik[sel] - ik_start) * nk + jk[sel]
                    dE = cp_E[ik[sel]][..., None] - cp_E[jk[sel]][..., None, :]
                    gcur = g[sel]
                    if form == "conventional":
                        delta = torch.exp(exp_factor * (dE - omega_ph[sel, None, None])**2)
                        Gsq = conventionalP("kac, kac, kbd -> kabcd", delta, gcur, gcur.conj())
                    else:
                        sqrt_delta = torch.exp(
                            0.5 * exp_factor * (dE - omega_ph[sel, None, None])**2
                        )
                        Gcur = gcur * sqrt_delta
                        Gsq = lindbladP("kac, kbd -> kabcd", Gcur, Gcur.conj())
                    P[0].index_add_(0, i_pair, wm[sel] * Gsq)  # P contribution
                    P[1].index_add_(0, i_pair, wp[sel] * Gsq)  # Pbar contribution

                # Contributions to P(jk,ik):
                if (sel := get_mine(jk)) is not None:
                    i_pair = (jk[sel] - ik_start) * nk + ik[sel]
                    dE = cp_E[ik[sel]][..., None] - cp_E[jk[sel]][..., None, :]
                    gcur = g[sel]
                    if form == "conventional":
                        delta = torch.exp(exp_factor * (dE - omega_ph[sel, None, None])**2)
                        Gsq = conventionalP("kca, kca, kdb -> kabcd", delta, gcur.conj(), gcur)
                    else:
                        sqrt_delta = torch.exp(
                            0.5 * exp_factor * (dE - omega_ph[sel, None, None])**2
                        )
                        Gcur = gcur * sqrt_delta
                        Gsq = lindbladP("kca, kdb -> kabcd", Gcur.conj(), Gcur)
                    P[0].index_add_(0, i_pair, wp[sel] * Gsq)  # P contribution
                    P[1].index_add_(0, i_pair, wm[sel] * Gsq)  # Pbar contribution

            op_shape = (2, nk_mine * n_bands_sq, nk * n_bands_sq)
            self.P = P.unflatten(1, (nk_mine, nk)).swapaxes(2, 3).reshape(op_shape)
            self.P2_eye = apply_batched(
                self.P, 
                (1.0 + 0.0j) * torch.tile(ab_initio.eye_bands[None], (nk, 1, 1))[..., None]
            )[1]

        else: # matrix form is slower but may save memory, lindblad not implemented
            g_mine, Gamma1, Gamma2, kpair_mine = [], [], [], []
            for block_start, block_stop in zip(block_lims[:-1], block_lims[1:]):
                # Read current slice of data:
                cur = slice(block_start, block_stop)
                ik, jk = torch.from_numpy(cp_ikpair[cur]).to(rc.device).T
                omega_ph = torch.from_numpy(cp_omega_ph[cur]).to(rc.device)
                g = torch.from_numpy(cp_g[cur]).to(rc.device)
                if evecs is not None:
                    g = torch.einsum("kba, kbc, kcd -> kad", evecs[ik].conj(), g, evecs[jk])
                bose_occ = bose(omega_ph, ab_initio.T)[:, None, None]
                wm = prefactor * bose_occ
                wp = prefactor * (bose_occ + 1.0)

                # Contributions to P(ik,jk):
                if (sel := get_mine(ik)) is not None:
                    i_pair = (ik[sel] - ik_start) * nk + jk[sel]
                    dE = cp_E[ik[sel]][..., None] - cp_E[jk[sel]][..., None, :]
                    gcur = g[sel]
                    delta_minus = torch.exp(exp_factor * (dE - omega_ph[sel, None, None])**2)
                    delta_plus = torch.exp(exp_factor * (dE + omega_ph[sel, None, None])**2)
                    g_mine.append(gcur)
                    kpair_mine.append(i_pair)
                    Gamma1.append((wm[sel] * delta_minus + wp[sel] * delta_plus) * gcur)
                    Gamma2.append((wp[sel] * delta_minus + wm[sel] * delta_plus) * gcur)

            # get cumulative count of each kpair, which is used as mode id
            def get_cumcount(a):
                count = torch.unique(a, return_counts=True)[1]
                maxcount = count.max().item()
                idx = torch.cumsum(count, dim=0)
                id_arr = torch.ones(idx[-1],dtype=torch.long, device=rc.device)
                id_arr[0] = 0
                id_arr[idx[:-1]] = -count[:-1] + 1
                cumcount = torch.cumsum(id_arr, dim=0)[torch.argsort(a, stable=True)]
                return cumcount, maxcount

            kpair_mine = torch.cat(kpair_mine)
            modes_mine, nmode = get_cumcount(kpair_mine)
            Matrices = torch.zeros(
                (3, nmode, nk_mine * nk, n_bands, n_bands), 
                dtype=torch.complex128, 
                device=rc.device
            )
            Matrices[0, modes_mine, kpair_mine] = torch.cat(g_mine).swapaxes(-1,-2).conj()
            Matrices[1, modes_mine, kpair_mine] = torch.cat(Gamma1)
            Matrices[2, modes_mine, kpair_mine] = torch.cat(Gamma2)
            Matrices = Matrices.unflatten(2, (nk_mine, nk))
            self.Matrices = Matrices
            self.Gamma2gdeg = torch.einsum(
                "mkKab, mkKbc -> kac", self.Matrices[2], self.Matrices[0]
            )

        self.rho_dot0 = self._calculate(ab_initio.rho0 * (1.0+0.0j))
        self.constant_params = dict(
            scale_factor=torch.tensor(scale_factor, device=rc.device),
        )
        self.scale_factor = dict()

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        attrs = cp_path.attrs
        attrs["scale_factor"] = self.constant_params["scale_factor"].item()
        attrs["smearing"] = self.smearing
        attrs["form"] = self.form
        return list(attrs.keys())

    def initialize_fields(self, params: dict[str, torch.Tensor], patch_id: int) -> None:
        self._initialize_fields(patch_id, **params)

    def _initialize_fields(self, patch_id: int, *, scale_factor: torch.Tensor) -> None:
        self.scale_factor[patch_id] = scale_factor[..., None, None, None]

    @stopwatch
    def rho_dot(self, rho: torch.Tensor, t: float, patch_id: int) -> torch.Tensor:
        """drho/dt due to scattering in Schrodinger picture.
        Input and output rho are in unpacked (complex Hermitian) form."""
        return self.scale_factor[patch_id] * (self._calculate(rho) - self.rho_dot0)

    def _calculate(self, rho: torch.Tensor) -> torch.Tensor:
        """Internal drho/dt calculation without detailed balance / scaling."""
        eye = self.ab_initio.eye_bands
        rho_all = self._collectT(rho)  # all k
        if not self.matrix_form:
            Prho = apply_batched(self.P, rho_all)
            Prho[1] -= self.P2_eye  # convert [1] to Pbar @ (rho - eye)
            return (eye - rho) @ Prho[0] + rho @ Prho[1]  # my k only
        else:
            Trhog = torch.einsum(
                "imkKab, Kbc..., mkKcd -> i...kad", 
                self.Matrices[1:], 
                rho_all, 
                self.Matrices[0]
            )
            Trhog[1] -= self.Gamma2gdeg
            return (eye - rho) @ Trhog[0] + rho @ Trhog[1]

    def _collectT(self, rho: torch.Tensor) -> torch.Tensor:
        """Collect rho from all MPI processes and transpose batch dimension.
        Batch dimension is put at end for efficient matrix multiplication."""
        ab_initio = self.ab_initio
        if ab_initio.comm.size == 1:
            return torch.einsum("...kab -> kab...", rho)
        nk = ab_initio.k_division.n_tot
        n_bands = ab_initio.n_bands
        batch_shape = rho.shape[:-3]
        n_batch = int(np.prod(batch_shape))
        sendbuf = rho.reshape(n_batch, -1).T.contiguous()
        recvbuf = torch.zeros(
            (n_batch, nk * n_bands * n_bands), dtype=rho.dtype, device=rc.device
        )
        mpi_type = rc.mpi_type[rho.dtype]
        recv_prev = ab_initio.k_division.n_prev * n_bands * n_bands * n_batch
        ab_initio.comm.Allgatherv(
            (BufferView(sendbuf), np.prod(rho.shape), 0, mpi_type),
            (BufferView(recvbuf), np.diff(recv_prev), recv_prev[:-1], mpi_type),
        )
        return recvbuf.reshape((nk, n_bands, n_bands) + batch_shape)


def apply_batched(P: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    """Apply batched flattened-rho operator P on batched rho.
    Batch dimension is at end of input, and at beginning of output."""
    result = torch.einsum("ikK, K... -> i...k", P, rho.flatten(0, 2))
    return result.unflatten(-1, (-1,) + rho.shape[1:3])

def bytes2memory(nbytes):
    if nbytes >= 1024**3:
        return f"{nbytes / 1024**3 :.2f} GB"
    elif nbytes >= 1024**2:
        return f"{nbytes / 1024**2:.2f} MB"
    elif nbytes >= 1024:
        return f"{nbytes / 1024:.2f} KB"
    else:
        return f"{nbytes} B"
