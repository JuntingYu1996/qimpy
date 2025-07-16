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
    form: str  #: fermi_golden, conventional Markov or Lindblad

    constant_params: dict[str, torch.Tensor]  #: constant values of parameters
    scale_factor: dict[int, torch.Tensor]  #: scale factors per patch

    @stopwatch
    def __init__(
        self,
        *,
        ab_initio: material.ab_initio.AbInitio,
        data_file: Checkpoint,
        read_P: bool = False,
        smearing: float = 0.001,
        scale_factor: float = 1.0,
        form: str = "conventional",
        uniform_smearing: bool = True,
        omega_threshold: float = 1e-6,
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
        form = form.replace("-", "_").lower()
        self.form = form
        self.uniform_smearing = uniform_smearing
        if not form in ["fermi_golden", "conventional", "lindblad", "matrix", "conventional_fast"]:
            raise InvalidInputException("form must be fermi_golden, conventional, lindblad or matrix")
        if not bool(data_file.attrs["ePhEnabled"]):
            raise InvalidInputException("No e-ph scattering available in data file")

        nk = ab_initio.k_division.n_tot
        ik_start = ab_initio.k_division.i_start
        ik_stop = ab_initio.k_division.i_stop
        nk_mine = ab_initio.nk_mine
        n_bands = ab_initio.n_bands
        n_bands_sq = n_bands**2
        prefactor = np.sqrt(np.pi / 2)

        if read_P:
            if form == "fermi_golden":
                self.T = torch.zeros(
                    (2, nk_mine, nk, n_bands, n_bands),
                    dtype=torch.double,
                    device=rc.device
                )
                batch = 100
                nbatch = (nk_mine - 1) // batch + 1
                for i in range(nbatch):
                    mysel = slice(i*batch, min((i+1)*batch, nk_mine))
                    ksel = slice(mysel.start + ik_start, mysel.stop + ik_start)
                    self.T[:, mysel] = torch.from_numpy(
                        np.array(data_file["Tmat"][:, ksel])
                    ).to(rc.device)
                self.T2eye = self.T[1].sum((1, 3)) # "kKab -> ka"
            else:
                raise InvalidInputException("Not implemented.")
            self.rho_dot0 = self._calculate(ab_initio.rho0 * (1.0 + 0.0j))
            self.constant_params = dict(
                scale_factor=torch.tensor(scale_factor, device=rc.device),
            )
            self.scale_factor = dict()
            return

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
            return torch.einsum(
                einsum_path, delta, g1, g2
            ).reshape((-1, n_bands_sq, n_bands_sq))

        def lindbladP(einsum_path, G1, G2):
            return torch.einsum(
                einsum_path, G1, G2
            ).reshape((-1, n_bands_sq, n_bands_sq))

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
        if ab_initio.uniform:
            wk = ab_initio.wk * torch.ones([nk, 1, 1], device=rc.device)
        else:
            wk = torch.from_numpy(np.array(data_file["wk"])).to(rc.device)[:, None, None]
        if uniform_smearing:
            exp_factor = -0.5 / smearing**2
        else:
            cp_smearing = data_file["smearing_ph"]

        if form == "fermi_golden":
            log.info("Using Fermi's Golden rule for phonon scattering.")
            Tshape = (2, nk_mine * nk, n_bands, n_bands)
            T = torch.zeros(Tshape, dtype=torch.double, device=rc.device)
        elif form in ["conventional", "lindblad"]:
            log.info("Constructing P tensor")
            P_shape = (2, nk_mine * nk, n_bands_sq, n_bands_sq)
            nbytes = 16 * np.prod(P_shape)
            log.info(f"Memory for P matrix: {bytes2memory(nbytes)}.")
            P = torch.zeros(P_shape, dtype=torch.complex128, device=rc.device)
        elif form == "matrix":
            log.info("Using matrix form for electron-phonon")
            g_mine, Gamma1, Gamma2, kpair_mine = [], [], [], []
        elif form == "conventional_fast":
            log.info("Conventional Markov in lower complexity")
            P1_shape = (2, nk_mine * nk, n_bands, n_bands_sq)
            P2_shape = (2, nk_mine * nk, n_bands_sq, n_bands)
            nbytes = 32 * np.prod(P1_shape)
            log.info(f"Memory for P1 and P2 matrix: {bytes2memory(nbytes)}.")
            P1 = torch.zeros(P1_shape, dtype=torch.complex128, device=rc.device)
            P2 = torch.zeros(P2_shape, dtype=torch.complex128, device=rc.device)

        for block_start, block_stop in zip(block_lims[:-1], block_lims[1:]):
            # Read current slice of data:
            cur = slice(block_start, block_stop)
            ik, jk = torch.from_numpy(cp_ikpair[cur]).to(rc.device).T
            omega_ph = torch.from_numpy(cp_omega_ph[cur]).to(rc.device)
            g = torch.from_numpy(cp_g[cur]).to(rc.device)
            if not uniform_smearing:
                smearing_cur = torch.from_numpy(cp_smearing[cur]).to(rc.device)
            if evecs is not None:
                g = torch.einsum("kba, kbc, kcd -> kad", evecs[ik].conj(), g, evecs[jk])
            bose_occ = bose(omega_ph, ab_initio.T)[:, None, None]
            wm = prefactor * bose_occ * (omega_ph > omega_threshold)[:, None, None]
            wp = prefactor * (bose_occ + 1.0) * (omega_ph > omega_threshold)[:, None, None]
            if (sel := get_mine(ik)) is not None:
                i_pair = (ik[sel] - ik_start) * nk + jk[sel]
                dE = cp_E[ik[sel]][..., None] - cp_E[jk[sel]][..., None, :]
                gsel = g[sel]
                if not uniform_smearing:
                    smearing = smearing_cur[sel, None, None]
                    exp_factor = -0.5 / smearing ** 2
                if form == "fermi_golden":
                    delta = torch.exp(exp_factor * (dE - omega_ph[sel, None, None])**2) / smearing
                    Gsq = (gsel * gsel.conj()).real * delta
                    T[0].index_add_(0, i_pair, wk[jk[sel]] * wm[sel] * Gsq)
                    T[1].index_add_(0, i_pair, wk[jk[sel]] * wp[sel] * Gsq)
                elif form == "conventional":
                    delta = torch.exp(exp_factor * (dE - omega_ph[sel, None, None])**2) / smearing
                    Gsq = conventionalP("kac, kac, kbd -> kabcd", delta, gsel, gsel.conj())
                    P[0].index_add_(0, i_pair, wk[jk[sel]] * wm[sel] * Gsq)  # P contribution
                    P[1].index_add_(0, i_pair, wk[jk[sel]] * wp[sel] * Gsq)  # Pbar contribution
                elif form == "lindblad":
                    sqrt_delta = torch.exp(
                        0.5 * exp_factor * (dE - omega_ph[sel, None, None])**2
                    ) / smearing ** 0.5
                    Gsel = gsel * sqrt_delta
                    Gsq = lindbladP("kac, kbd -> kabcd", Gsel, Gsel.conj())
                    P[0].index_add_(0, i_pair, wk[jk[sel]] * wm[sel] * Gsq)  # P contribution
                    P[1].index_add_(0, i_pair, wk[jk[sel]] * wp[sel] * Gsq)  # Pbar contribution
                elif form == "matrix":
                    delta_minus = torch.exp(exp_factor * (dE - omega_ph[sel, None, None])**2) / smearing
                    delta_plus = torch.exp(exp_factor * (dE + omega_ph[sel, None, None])**2) / smearing
                    g_mine.append(gsel)
                    kpair_mine.append(i_pair)
                    Gamma1.append(wk[jk[sel]] * (wm[sel] * delta_minus + wp[sel] * delta_plus) * gsel)
                    Gamma2.append(wk[jk[sel]] * (wp[sel] * delta_minus + wm[sel] * delta_plus) * gsel)
                elif form == "conventional_fast":
                    delta = torch.exp(exp_factor * (dE - omega_ph[sel, None, None])**2) / smearing
                    Gsel = delta * gsel
                    gconj = gsel.conj()
                    Gsq1 = torch.einsum("kac, kad -> kacd", Gsel, gconj).reshape((-1, n_bands, n_bands_sq))
                    Gsq2 = torch.einsum("kac, kbc -> kabc", Gsel, gconj).reshape((-1, n_bands_sq, n_bands))
                    P1[0].index_add_(0, i_pair, wk[jk[sel]] * wm[sel] * Gsq1)
                    P1[1].index_add_(0, i_pair, wk[jk[sel]] * wp[sel] * Gsq1)
                    P2[0].index_add_(0, i_pair, wk[jk[sel]] * wm[sel] * Gsq2)
                    P2[1].index_add_(0, i_pair, wk[jk[sel]] * wp[sel] * Gsq2)

            if form != "matrix" and (sel := get_mine(jk)) is not None:
                i_pair = (jk[sel] - ik_start) * nk + ik[sel]
                dE = cp_E[ik[sel]][..., None] - cp_E[jk[sel]][..., None, :]
                gsel = g[sel]
                if not uniform_smearing:
                    smearing = smearing_cur[sel, None, None]
                    exp_factor = -0.5 / smearing ** 2
                if form == "fermi_golden":
                    delta = torch.exp(exp_factor * (dE - omega_ph[sel, None, None])**2) / smearing
                    Gsq = ((gsel * gsel.conj()).real * delta).swapaxes(-1, -2)
                    T[0].index_add_(0, i_pair, wk[ik[sel]] * wp[sel] * Gsq)
                    T[1].index_add_(0, i_pair, wk[ik[sel]] * wm[sel] * Gsq)
                elif form == "conventional":
                    delta = torch.exp(exp_factor * (dE - omega_ph[sel, None, None])**2) / smearing
                    Gsq = conventionalP("kca, kca, kdb -> kabcd", delta, gsel.conj(), gsel)
                    P[0].index_add_(0, i_pair, wk[ik[sel]] * wp[sel] * Gsq)  # P contribution
                    P[1].index_add_(0, i_pair, wk[ik[sel]] * wm[sel] * Gsq)  # Pbar contribution
                elif form == "lindblad":
                    sqrt_delta = torch.exp(
                        0.5 * exp_factor * (dE - omega_ph[sel, None, None])**2
                    ) / smearing ** 0.5
                    Gsel = gsel * sqrt_delta
                    Gsq = lindbladP("kca, kdb -> kabcd", Gsel.conj(), Gsel)
                    P[0].index_add_(0, i_pair, wk[ik[sel]] * wp[sel] * Gsq)  # P contribution
                    P[1].index_add_(0, i_pair, wk[ik[sel]] * wm[sel] * Gsq)  # Pbar contribution
                elif form == "conventional_fast":
                    delta = torch.exp(exp_factor * (dE - omega_ph[sel, None, None])**2) / smearing
                    Gsel = delta * gsel.conj()
                    Gsq1 = torch.einsum("kca, kda -> kacd", Gsel, gsel).reshape((-1, n_bands, n_bands_sq))
                    Gsq2 = torch.einsum("kca, kcb -> kabc", Gsel, gsel).reshape((-1, n_bands_sq, n_bands))
                    P1[0].index_add_(0, i_pair, wk[ik[sel]] * wp[sel] * Gsq1)
                    P1[1].index_add_(0, i_pair, wk[ik[sel]] * wm[sel] * Gsq1)
                    P2[0].index_add_(0, i_pair, wk[ik[sel]] * wp[sel] * Gsq2)
                    P2[1].index_add_(0, i_pair, wk[ik[sel]] * wm[sel] * Gsq2)

        if form == "fermi_golden":
            self.T = T.unflatten(1, (nk_mine, nk))
            self.T2eye = self.T[1].sum((1, 3)) # "kKab -> ka"
        elif form in ["conventional", "lindblad"]:
            op_shape = (2, nk_mine * n_bands_sq, nk * n_bands_sq)
            self.P = P.unflatten(1, (nk_mine, nk)).swapaxes(2, 3).reshape(op_shape)
            self.P2_eye = apply_batched(
                self.P, 
                (1.0 + 0.0j) * torch.tile(ab_initio.eye_bands[None], (nk, 1, 1))[..., None]
            )[1]
        elif form == "conventional_fast":
            op_shape1 = (2, nk_mine * n_bands, nk * n_bands_sq)
            self.P1 = P1.unflatten(1, (nk_mine, nk)).swapaxes(2, 3).reshape(op_shape1)
            op_shape2 = (2, nk_mine * n_bands_sq, nk * n_bands)
            self.P2 = ( # off-diagonal
                P2 * (1 - ab_initio.eye_bands).flatten()[None, None, :, None]
            ).unflatten(1, (nk_mine, nk)).swapaxes(2, 3).reshape(op_shape2)
            self.P_eye = apply_batched_fast(
                self.P1, self.P2,
                (1.0 + 0.0j) * torch.tile(ab_initio.eye_bands[None], (nk, 1, 1))[..., None]
            )[1]
        else:
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

        self.rho_dot0 = self._calculate(ab_initio.rho0 * (1.0 + 0.0j))
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
        if self.form == "fermi_golden":
            f = rho[..., range(self.ab_initio.n_bands), range(self.ab_initio.n_bands)]
            f_all = rho_all[:, range(self.ab_initio.n_bands), range(self.ab_initio.n_bands), ...].real
            Tf = torch.einsum("ikKab, Kb... -> i...ka", self.T, f_all)
            Tf[1] -= self.T2eye
            dfdt = (1 - f) * Tf[0] + f * Tf[1]
            return torch.diag_embed(dfdt)
        elif self.form in ["conventional", "lindblad"]:
            Prho = apply_batched(self.P, rho_all)
            Prho[1] -= self.P2_eye  # convert [1] to Pbar @ (rho - eye)
            return (eye - rho) @ Prho[0] + rho @ Prho[1]  # my k only
        elif self.form == "conventional_fast":
            Prho = apply_batched_fast(self.P1, self.P2, rho_all)
            Prho[1] -= self.P_eye
            return (eye - rho) @ Prho[0] + rho @ Prho[1]
        else:
            Trhog = torch.einsum(
                "imkKab, ...mkKbd -> i...kad",
                self.Matrices[1:],
                torch.einsum("Kbc..., mkKcd -> ...mkKbd", rho_all, self.Matrices[0])
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

def apply_batched1(P1: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    """Apply batched flattened-rho operator P on batched rho.
    Batch dimension is at end of input, and at beginning of output."""
    result = torch.einsum("ikK, K... -> i...k", P1, rho.flatten(0, 2))
    return result.unflatten(-1, (-1,) + rho.shape[1:2]) # (2, x, y, k, a)

def apply_batched2(P2: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    """Apply batched flattened-rho operator P on batched rho.
    Batch dimension is at end of input, and at beginning of output."""
    nbands = rho.shape[1]
    f = rho[:, range(nbands), range(nbands), ...].flatten(0,1)
    result = torch.einsum("ikK, K... -> i...k", P2, f)
    return result.unflatten(-1, (-1, nbands, nbands)) # (2, x, y, k, a, b)

def apply_batched_fast(P1: torch.Tensor, P2: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    """Apply batched flattened-rho operator P on batched rho.
    Batch dimension is at end of input, and at beginning of output."""
    nbands = rho.shape[1]
    f = rho[:, range(nbands), range(nbands), ...].flatten(0,1)
    result = torch.einsum(
        "ikK, K... -> i...k", P2, f
    ).unflatten(-1, (-1, nbands, nbands)) # off-diagonal
    result[..., range(nbands), range(nbands)] = torch.einsum(
        "ikK, K... -> i...k", P1, rho.flatten(0, 2)
    ).unflatten(-1, (-1, nbands)) # diagonal
    return result

def bytes2memory(nbytes):
    if nbytes >= 1024**3:
        return f"{nbytes / 1024**3 :.2f} GB"
    elif nbytes >= 1024**2:
        return f"{nbytes / 1024**2:.2f} MB"
    elif nbytes >= 1024:
        return f"{nbytes / 1024:.2f} KB"
    else:
        return f"{nbytes} B"
