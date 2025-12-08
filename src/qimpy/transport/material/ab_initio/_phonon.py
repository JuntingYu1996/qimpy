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
    uniform_smearing: bool  #: whether to use the same smearing for all pairs
    omega_threshold: float  #: leave out omega < omega_threshold
    enforcing_Econs: str  #: whether to enforce energy conservation to phonon occupation
    N_sigma: float  #: the phonon occupation of those energy less than N_sigma * smearing will be changed for Econs

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
        enforcing_Econs: str = "no",
        N_sigma: float = 1.0, 
        detail_balance: bool = False,
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
        if not form in [
            "fermi_golden", "conventional", "lindblad", "conventional_fast", "conventional_sparse"
        ]:
            raise InvalidInputException("form must be fermi_golden, conventional, lindblad")
        if not bool(data_file.attrs["ePhEnabled"]):
            raise InvalidInputException("No e-ph scattering available in data file")

        nk = ab_initio.k_division.n_tot
        ik_start = ab_initio.k_division.i_start
        ik_stop = ab_initio.k_division.i_stop
        nk_mine = ab_initio.nk_mine
        if "bands_sel" in data_file.attrs:
            self.use_partial = True
            bstart, bstop = data_file.attrs["bands_sel"]
            self.bands_sel = slice(bstart, bstop)
            n_bands = bstop - bstart
        else:
            self.use_partial = False
            n_bands = ab_initio.n_bands
            self.bands_sel = slice(0, n_bands)
        self.n_bands = n_bands
        self.eye_bands = torch.eye(n_bands, device=rc.device)
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
                Tmat_data = data_file["Tmat"]
                for i in range(nbatch):
                    mysel = slice(i*batch, min((i+1)*batch, nk_mine))
                    ksel = slice(mysel.start + ik_start, mysel.stop + ik_start)
                    if len(Tmat_data.shape) == 5:
                        read_data = np.array(Tmat_data[:, ksel])
                    else:
                        kpair = (
                            np.arange(ksel.start, ksel.stop)[:, None] * nk + np.arange(nk)[None]
                        ).reshape(-1)
                        read_data = np.array(
                            Tmat_data[:, kpair]
                        ).reshape((2, ksel.stop-ksel.start, nk, n_bands, n_bands))
                    self.T[:, mysel] = torch.from_numpy(read_data).to(rc.device)
                self.T2eye = self.T[1].sum((1, 3)) # "kKab -> ka"
            elif form == "conventional_sparse":
                h5P = data_file["Psparse"]
                sel_dict = {}
                nkpair_tot = 0
                for s1 in range(n_bands):
                    for s3 in range(n_bands):
                        k1_all = torch.from_numpy(
                            h5P[f"k_{s1}_{s3}" if s1 <= s3 else f"k_{s3}_{s1}"][:, 0 if s1 <= s3 else 1]
                        ).to(rc.device)
                        sel = torch.where(torch.logical_and(k1_all >= ik_start, k1_all < ik_stop))[0].cpu()
                        sel_dict[(s1, s3)] = sel
                        nkpair_tot += sel.shape[0]
                self.k1s1, self.k2s3 = torch.zeros((2, nkpair_tot), dtype=int, device=rc.device)
                self.P = torch.zeros((2, nkpair_tot, n_bands, n_bands), dtype=torch.complex128, device=rc.device)
                kpair_cur = 0
                for s1 in range(n_bands):
                    for s3 in range(n_bands):
                        log.info(f"k_{s1}_{s3}")
                        kpair_data = h5P[f"k_{s1}_{s3}" if s1 <= s3 else f"k_{s3}_{s1}"]
                        sel = sel_dict[(s1, s3)]
                        nkpair = sel.shape[0]
                        if nkpair == 0:
                            continue
                        k1, k2 = torch.from_numpy(kpair_data[sel]).to(rc.device).T[[0,1] if s1 <= s3 else [1,0]]
                        cur_slice = slice(kpair_cur, kpair_cur+nkpair)
                        self.k1s1[cur_slice] = (k1 - ik_start) * n_bands + s1
                        self.k2s3[cur_slice] = k2 * n_bands + s3
                        P_data = torch.from_numpy(h5P[f"P_{s1}_{s3}" if s1 <= s3 else f"P_{s3}_{s1}"][:, sel])
                        self.P[:, cur_slice] = (
                            P_data if s1 <= s3 else P_data[[1,0]].swapaxes(-1, -2).conj()
                        ).to(rc.device)
                        kpair_cur += nkpair
                self.P2_eye = self.apply_batched_sparse(
                    (1.0 + 0.0j) * torch.tile(ab_initio.eye_bands[None], (nk, 1, 1))[..., None]
                )[1]
                
                # k1s1list, k2s3list, Plist = [], [], []
                # for s1 in range(n_bands):
                #     for s3 in range(n_bands):
                #         log.info(f"k_{s1}_{s3}")
                #         kpair_data = h5P[f"k_{s1}_{s3}" if s1 <= s3 else f"k_{s3}_{s1}"]
                #         P_data = h5P[f"P_{s1}_{s3}" if s1 <= s3 else f"P_{s3}_{s1}"]
                #         k1_all = torch.from_numpy(
                #             kpair_data[:, 0 if s1 <= s3 else 1]
                #         ).to(rc.device)
                #         sel = torch.where(torch.logical_and(k1_all >= ik_start, k1_all < ik_stop))[0].cpu()
                #         nkpair = sel.shape[0]
                #         k1, k2 = torch.from_numpy(kpair_data[sel]).to(rc.device).T[[0,1] if s1 <= s3 else [1,0]]
                #         k1s1list.append((k1 - ik_start) * n_bands + s1)
                #         k2s3list.append(k2 * n_bands + s3)
                #         Plist.append(
                #             torch.from_numpy(P_data[:, sel]).to(rc.device) if s1 <= s3 else
                #             torch.from_numpy(P_data[:, sel]).to(rc.device)[[1,0]].swapaxes(-1, -2).conj()
                #         )
                # k1s1list, k2s3list, Plist = (
                #     torch.concatenate(k1s1list, axis=0),
                #     torch.concatenate(k2s3list, axis=0),
                #     torch.concatenate(Plist, axis=1)
                # )
                # self.k1s1, self.k2s3, self.P = k1s1list, k2s3list, Plist
                # self.P2_eye = self.apply_batched_sparse(
                #     (1.0 + 0.0j) * torch.tile(ab_initio.eye_bands[None], (nk, 1, 1))[..., None]
                # )[1]
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
            # bose_occ = bose(omega_ph, ab_initio.T)[:, None, None]
            # wm = prefactor * bose_occ * (omega_ph > omega_threshold)[:, None, None]
            # wp = prefactor * (bose_occ + 1.0) * (omega_ph > omega_threshold)[:, None, None]
            if (sel := get_mine(ik)) is not None:
                i_pair = (ik[sel] - ik_start) * nk + jk[sel]
                dE = cp_E[ik[sel]][..., self.bands_sel, None] - cp_E[jk[sel]][..., None, self.bands_sel]
                gsel = g[sel]
                omega_sel = omega_ph[sel, None, None]
                omega_nonzero = (omega_sel > omega_threshold)
                if enforcing_Econs == "all":
                    bose_occ = bose(torch.maximum(omega_sel, dE), ab_initio.T)
                    omega_scale = 1.0
                elif enforcing_Econs == "low_omega":
                    # wm_ = wm[sel]
                    # wp_ = wm_ * torch.exp(dE / ab_initio.T)
                    omega_eff = torch.tile(omega_sel, (1, n_bands, n_bands))
                    rep_flags = torch.minimum(dE, torch.tensor([N_sigma * smearing], device=rc.device)) >= omega_eff
                    omega_eff[rep_flags] = dE[rep_flags]
                    bose_occ = bose(omega_eff, ab_initio.T)
                    omega_scale = omega_sel / omega_eff
                else:
                    bose_occ = bose(omega_sel, ab_initio.T)
                    omega_scale = 1.0
                wm = prefactor * bose_occ * omega_scale * omega_nonzero
                if detail_balance:
                    wp = wm * torch.exp(dE / ab_initio.T)
                else:
                    wp = prefactor * (bose_occ + 1.0) * omega_scale * omega_nonzero
                if not uniform_smearing:
                    smearing = smearing_cur[sel, None, None]
                    exp_factor = -0.5 / smearing ** 2
                if form == "fermi_golden":
                    delta = torch.exp(exp_factor * (dE - omega_sel)**2) / smearing
                    Gsq = (gsel * gsel.conj()).real * delta
                    T[0].index_add_(0, i_pair, wk[jk[sel]] * wm * Gsq)
                    T[1].index_add_(0, i_pair, wk[jk[sel]] * wp * Gsq)
                elif form == "conventional":
                    delta = torch.exp(exp_factor * (dE - omega_sel)**2) / smearing
                    Gsq = conventionalP("kac, kac, kbd -> kabcd", delta, gsel, gsel.conj())
                    P[0].index_add_(0, i_pair, wk[jk[sel]] * wm * Gsq)  # P contribution
                    P[1].index_add_(0, i_pair, wk[jk[sel]] * wp * Gsq)  # Pbar contribution
                elif form == "lindblad":
                    sqrt_delta = torch.exp(
                        0.5 * exp_factor * (dE - omega_sel)**2
                    ) / smearing ** 0.5
                    Gsel = gsel * sqrt_delta
                    Gsq = lindbladP("kac, kbd -> kabcd", Gsel, Gsel.conj())
                    P[0].index_add_(0, i_pair, wk[jk[sel]] * wm * Gsq)  # P contribution
                    P[1].index_add_(0, i_pair, wk[jk[sel]] * wp * Gsq)  # Pbar contribution
                elif form == "conventional_fast":
                    delta = torch.exp(exp_factor * (dE - omega_sel)**2) / smearing
                    Gsel = delta * gsel
                    gconj = gsel.conj()
                    Gsq1 = torch.einsum("kac, kad -> kacd", Gsel, gconj)
                    Gsq2 = torch.einsum("kac, kbc -> kabc", Gsel, gconj)
                    P1[0].index_add_(0, i_pair, ((wk[jk[sel]] * wm)[:, :, :, None] * Gsq1).reshape((-1, n_bands, n_bands_sq)))
                    P1[1].index_add_(0, i_pair, ((wk[jk[sel]] * wp)[:, :, :, None] * Gsq1).reshape((-1, n_bands, n_bands_sq)))
                    P2[0].index_add_(0, i_pair, ((wk[jk[sel]] * wm)[:, :, None, :] * Gsq2).reshape((-1, n_bands_sq, n_bands)))
                    P2[1].index_add_(0, i_pair, ((wk[jk[sel]] * wp)[:, :, None, :] * Gsq2).reshape((-1, n_bands_sq, n_bands)))
                    # P1[0].index_add_(0, i_pair, (wk[jk[sel]] * wm)[:, :, :, None] * Gsq1)
                    # P1[1].index_add_(0, i_pair, (wk[jk[sel]] * wp)[:, :, :, None] * Gsq1)
                    # P2[0].index_add_(0, i_pair, (wk[jk[sel]] * wm)[:, :, None, :] * Gsq2)
                    # P2[1].index_add_(0, i_pair, (wk[jk[sel]] * wp)[:, :, None, :] * Gsq2)
                elif form == "conventional_sparse":
                    dE_plus, dE_minus = dE + omega_sel, dE - omega_sel
                    kKxsel, s1sel, s3sel = torch.where(torch.logical_or(
                        torch.abs(dE_plus) < 10 * smearing, 
                        torch.abs(dE_minus) < 10 * smearing
                    ))
                    k1sel, k2sel = ik[sel][kKxsel], jk[sel][kKxsel]
                    delta_plus = torch.exp(exp_factor * dE_plus[kKxsel, s1sel, s3sel]**2) / smearing
                    delta_minus = torch.exp(exp_factor * dE_minus[kKxsel, s1sel, s3sel]**2) / smearing
                    Gsq = gsel[kKxsel, s1sel, s2sel][:, None, None] * gsel[kKxsel].conj() # Ncd
                    P1sel = wk[k2sel] * (
                        wp[kKxsel] * delta_plus[:, None, None] + wm[kKxsel] * delta_minus[:, None, None]
                    ) * Gsq
                    P2sel = wk[k2sel] * (
                        wm[kKxsel] * delta_plus[:, None, None] + wp[kKxsel] * delta_minus[:, None, None]
                    ) * Gsq
                    k1s1, k2s3 = (k1sel - ik_start) * n_bands + s1sel, k2sel * n_bands + s3sel
                    

            if (sel := get_mine(jk)) is not None:
                i_pair = (jk[sel] - ik_start) * nk + ik[sel]
                dE = cp_E[ik[sel]][..., self.bands_sel, None] - cp_E[jk[sel]][..., None, self.bands_sel]
                gsel = g[sel]
                omega_sel = omega_ph[sel, None, None]
                omega_nonzero = (omega_sel > omega_threshold)
                if enforcing_Econs == "all":
                    bose_occ = bose(torch.maximum(omega_sel, dE.swapaxes(-1, -2)), ab_initio.T)
                    omega_scale = 1.0
                elif enforcing_Econs == "low_omega":
                    # wm_ = wm[sel]
                    # wp_ = wm_ * torch.exp(dE.swapaxes(-1,-2) / ab_initio.T)
                    omega_eff = torch.tile(omega_sel, (1, n_bands, n_bands))
                    rep_flags = torch.minimum(dE.swapaxes(-1, -2), torch.tensor([N_sigma * smearing], device=rc.device)) >= omega_eff
                    omega_eff[rep_flags] = dE.swapaxes(-1, -2)[rep_flags]
                    bose_occ = bose(omega_eff, ab_initio.T)
                    omega_scale = omega_sel / omega_eff
                else:
                    bose_occ = bose(omega_sel, ab_initio.T)
                    omega_scale = 1.0
                wm = prefactor * bose_occ * omega_scale * omega_nonzero
                if detail_balance:
                    wp = wm * torch.exp(dE.swapaxes(-1,-2) / ab_initio.T)
                else:
                    wp = prefactor * (bose_occ + 1.0) * omega_scale * omega_nonzero
                if not uniform_smearing:
                    smearing = smearing_cur[sel, None, None]
                    exp_factor = -0.5 / smearing ** 2
                if form == "fermi_golden":
                    delta = torch.exp(exp_factor * (dE - omega_sel)**2) / smearing
                    Gsq = ((gsel * gsel.conj()).real * delta).swapaxes(-1, -2)
                    T[0].index_add_(0, i_pair, wk[ik[sel]] * wp * Gsq)
                    T[1].index_add_(0, i_pair, wk[ik[sel]] * wm * Gsq)
                elif form == "conventional":
                    delta = torch.exp(exp_factor * (dE - omega_sel)**2) / smearing
                    Gsq = conventionalP("kca, kca, kdb -> kabcd", delta, gsel.conj(), gsel)
                    P[0].index_add_(0, i_pair, wk[ik[sel]] * wp * Gsq)  # P contribution
                    P[1].index_add_(0, i_pair, wk[ik[sel]] * wm * Gsq)  # Pbar contribution
                elif form == "lindblad":
                    sqrt_delta = torch.exp(
                        0.5 * exp_factor * (dE - omega_sel)**2
                    ) / smearing ** 0.5
                    Gsel = gsel * sqrt_delta
                    Gsq = lindbladP("kca, kdb -> kabcd", Gsel.conj(), Gsel)
                    P[0].index_add_(0, i_pair, wk[ik[sel]] * wp * Gsq)  # P contribution
                    P[1].index_add_(0, i_pair, wk[ik[sel]] * wm * Gsq)  # Pbar contribution
                elif form == "conventional_fast":
                    delta = torch.exp(exp_factor * (dE - omega_sel)**2) / smearing
                    Gsel = delta * gsel.conj()
                    Gsq1 = torch.einsum("kca, kda -> kacd", Gsel, gsel)
                    Gsq2 = torch.einsum("kca, kcb -> kabc", Gsel, gsel)
                    P1[0].index_add_(0, i_pair, ((wk[ik[sel]] * wp)[:, :, :, None] * Gsq1).reshape((-1, n_bands, n_bands_sq)))
                    P1[1].index_add_(0, i_pair, ((wk[ik[sel]] * wm)[:, :, :, None] * Gsq1).reshape((-1, n_bands, n_bands_sq)))
                    P2[0].index_add_(0, i_pair, ((wk[ik[sel]] * wp)[:, :, None, :] * Gsq2).reshape((-1, n_bands_sq, n_bands)))
                    P2[1].index_add_(0, i_pair, ((wk[ik[sel]] * wm)[:, :, None, :] * Gsq2).reshape((-1, n_bands_sq, n_bands)))
                    # P1[0].index_add_(0, i_pair, (wk[ik[sel]] * wp)[:, :, :, None] * Gsq1)
                    # P1[1].index_add_(0, i_pair, (wk[ik[sel]] * wm)[:, :, :, None] * Gsq1)
                    # P2[0].index_add_(0, i_pair, (wk[ik[sel]] * wp)[:, :, None, :] * Gsq2)
                    # P2[1].index_add_(0, i_pair, (wk[ik[sel]] * wm)[:, :, None, :] * Gsq2)

        if form == "fermi_golden":
            self.T = T.unflatten(1, (nk_mine, nk))
            self.T2eye = self.T[1].sum((1, 3)) # "kKab -> ka"
        elif form in ["conventional", "lindblad"]:
            op_shape = (2, nk_mine * n_bands_sq, nk * n_bands_sq)
            self.P = P.unflatten(1, (nk_mine, nk)).swapaxes(2, 3).reshape(op_shape)
            self.P2_eye = apply_batched(
                self.P, 
                (1.0 + 0.0j) * torch.tile(self.eye_bands[None], (nk, 1, 1))[..., None]
            )[1]
        elif form == "conventional_fast":
            op_shape1 = (2, nk_mine * n_bands, nk * n_bands_sq)
            self.P1 = P1.unflatten(1, (nk_mine, nk)).swapaxes(2, 3).reshape(op_shape1)
            op_shape2 = (2, nk_mine * n_bands_sq, nk * n_bands)
            self.P2 = ( # off-diagonal
                P2 * (1 - self.eye_bands).flatten()[None, None, :, None]
            ).unflatten(1, (nk_mine, nk)).swapaxes(2, 3).reshape(op_shape2)
            self.P2_eye = apply_batched_fast(
                self.P1, self.P2,
                (1.0 + 0.0j) * torch.tile(self.eye_bands[None], (nk, 1, 1))[..., None]
            )[1]

        self.rho_dot0 = self._calculate(
            ab_initio.rho0[..., self.bands_sel, self.bands_sel] * (1.0 + 0.0j)
        )
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
        if self.use_partial:
            bs = self.bands_sel
            rho_partial = rho[..., bs, bs]
            res = torch.zeros_like(rho)
            res[..., bs, bs] = self.scale_factor[patch_id] * (self._calculate(rho_partial) - self.rho_dot0)
        else:
            res = self.scale_factor[patch_id] * (self._calculate(rho) - self.rho_dot0)
        return res

    def _calculate(self, rho: torch.Tensor) -> torch.Tensor:
        """Internal drho/dt calculation without detailed balance / scaling."""
        eye = self.eye_bands
        rho_all = self._collectT(rho)  # all k
        if self.form == "fermi_golden":
            n_bands = self.n_bands
            f = rho[..., range(n_bands), range(n_bands)]
            f_all = rho_all[:, range(n_bands), range(n_bands), ...].real
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
            Prho[1] -= self.P2_eye
            return (eye - rho) @ Prho[0] + rho @ Prho[1]
        elif self.form == "conventional_sparse":
            Prho = self.apply_batched_sparse(rho_all)
            Prho[1] -= self.P2_eye
            return (eye - rho) @ Prho[0] + rho @ Prho[1]

    def _collectT(self, rho: torch.Tensor) -> torch.Tensor:
        """Collect rho from all MPI processes and transpose batch dimension.
        Batch dimension is put at end for efficient matrix multiplication."""
        ab_initio = self.ab_initio
        if ab_initio.comm.size == 1:
            return torch.einsum("...kab -> kab...", rho)
        nk = ab_initio.k_division.n_tot
        n_bands = self.n_bands
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

    def apply_batched_sparse(self, rho: torch.Tensor) -> torch.Tensor:
        # P: (2, Npair, s2, s4), rho: (k,s3,s4,x,y)
        nk_mine, n_bands = self.ab_initio.nk_mine, self.ab_initio.n_bands
        Prho = torch.zeros(
            (2, ) + rho.shape[3:] + (nk_mine * n_bands, n_bands), 
            dtype=torch.complex128, device=rc.device
        )
        Prho.index_add_(
            -2, self.k1s1, torch.einsum("iNbd, Nd... -> i...Nb", self.P, rho.flatten(0, 1)[self.k2s3])
        ) # i...Kb
        return Prho.unflatten(-2, (-1, rho.shape[1])) # i...kab


def apply_batched(P: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    """Apply batched flattened-rho operator P on batched rho.
    Batch dimension is at end of input, and at beginning of output."""
    result = torch.einsum("ikK, K... -> i...k", P, rho.flatten(0, 2))
    return result.unflatten(-1, (-1,) + rho.shape[1:3])

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
