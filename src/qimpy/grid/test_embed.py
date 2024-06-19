import matplotlib.pyplot as plt
import numpy as np
import torch
import qimpy

from typing import Sequence

from qimpy import rc
from qimpy.lattice import Lattice
from qimpy.grid import Grid
from qimpy.symmetries import Symmetries
from qimpy.io import log_config
from qimpy.lattice._wigner_seitz import WignerSeitz


def check_embed(grid: Grid, latticeCenter: Sequence[float], periodic: np.ndarray) -> None:
    """Check Coulomb embedding procedure on test system"""
    # Create fake data
    r = get_r(grid, torch.eye(3), space='R')  # In mesh coords
    sigma_r = 0.005
    blob = torch.exp(
        -torch.sum((r - torch.tensor(latticeCenter)) ** 2, dim=-1) / (2 * sigma_r))
    blob /= np.sqrt(2 * np.pi * sigma_r ** 2)

    data1, data2, data3 = extend_grid(blob, grid, periodic, torch.tensor(latticeCenter))

    fig, axs = plt.subplots(1, 3)
    im = []
    im.append(show(axs[0], data1.sum(axis=0), "Original data"))
    im.append(show(axs[1], data2.sum(axis=0), "Embedded data"))
    im.append(show(axs[2], data3.sum(axis=0), "Embedded->Original data"))
    for _im in im:
        fig.colorbar(_im, orientation='horizontal')
    plt.show()


def extend_grid(dataOrig: torch.Tensor, gridOrig: Grid, periodic: np.ndarray, latticeCenter: torch.Tensor):
    # Initialize embedding grid
    meshOrig = gridOrig.get_mesh('R', mine=True)
    Sorig = torch.tensor(gridOrig.shape)
    RbasisOrig = gridOrig.lattice.Rbasis
    dimScale = (1, 1, 1) + (periodic == False) # extend cell in non-periodic directions
    v1,v2,v3 = (RbasisOrig @ np.diag(dimScale)).T
    latticeEmbed = Lattice(vector1=(v1[0],v1[1],v1[2]),
                       vector2=(v2[0],v2[1],v2[2]),
                       vector3=(v3[0],v3[1],v3[2]))
    gridEmbed = Grid(lattice=latticeEmbed, symmetries=Symmetries(lattice=latticeEmbed),
                     shape=tuple(dimScale * gridOrig.shape), comm=rc.comm)
    Sembed = torch.tensor(gridEmbed.shape)
    # Shift center to origin and report embedding center in various coordinate systems:
    latticeCenter = torch.tensor(latticeCenter)
    shifts = torch.round(latticeCenter * torch.tensor(meshOrig.shape[0:3])).to(int)
    rCenter = RbasisOrig @ latticeCenter
    ivCenter = torch.round(latticeCenter @ (1.*torch.diag(torch.tensor(gridOrig.shape)))).to(int)
    print("Integer grid location selected as the embedding center:")
    print("\tGrid: {:6} {:6} {:6}".format(*tuple(ivCenter)))
    print("\tLattice: {:6.3f} {:6.3f} {:6.3f}".format(*tuple(latticeCenter)))
    print("\tCartesian: {:6.3f} {:6.3f} {:6.3f}".format(*tuple(rCenter)))
    # Setup Wigner-Seitz cells of original and embed meshes
    wsOrig = WignerSeitz(gridOrig.lattice.Rbasis)
    wsEmbed = WignerSeitz(gridEmbed.lattice.Rbasis)
    # Setup mapping between original and embedding meshes
    ivEmbed = gridEmbed.get_mesh('R', mine=True).reshape(-1, 3)
    diagSembedInv = torch.diag(1 / torch.tensor(gridEmbed.shape))
    ivEmbed_wsOrig = wsEmbed.reduceIndex(ivEmbed, Sembed)
    ivEquivOrig = (ivEmbed_wsOrig + shifts) % Sorig[None, :]
    iEmbed = ivEmbed[:, 2] + Sembed[2] * (ivEmbed[:, 1] + Sembed[1] * ivEmbed[:, 0])
    iEquivOrig = ivEquivOrig[:, 2] + Sorig[2] * (ivEquivOrig[:, 1] + Sorig[1] * ivEquivOrig[:, 0])
    # Symmetrize points on boundary using weight function "smoothTheta"
    xWS = (1. * ivEmbed_wsOrig @ diagSembedInv) @ gridEmbed.lattice.Rbasis.T
    weights = smoothTheta(wsOrig.ws_boundary_distance(xWS))
    bMap = torch.sparse_coo_tensor(np.array([iEquivOrig, iEmbed]), weights,
                                   device=rc.device)
    colSums = torch.sparse.sum(bMap, dim=1).to_dense()
    colNorms = torch.sparse.spdiags(1./colSums,offsets=torch.tensor([0]),
                                    shape=(Sorig.prod(), Sorig.prod()))
    bMap = torch.sparse.mm(colNorms, bMap)
    dataEmbed = (dataOrig.reshape(-1) @ bMap).reshape(gridEmbed.shape)
    dataOrig2 = (dataEmbed.reshape(-1) @ bMap.T).reshape(gridOrig.shape)
    return dataOrig, dataEmbed, dataOrig2


def smoothTheta(x) -> torch.Tensor:
    return torch.where(x <= -1, 1., torch.where(x >= 1, 0., 0.25*(2.-x*(3.-x**2))))


def get_r(grid, R, space='R'):
    mesh = grid.get_mesh(space, mine=True)
    diagSinv = torch.diag(1 / torch.tensor(mesh.shape[0:3]))
    M = torch.tensor(mesh, dtype=torch.double)
    return M @ diagSinv @ R.T


def show(ax, data, title=None):
    #ax.plot(center[1] * data.shape[0], center[2] * data.shape[1], 'rx')
    if title:
        ax.set_title(title)
    return ax.imshow(data, origin='lower')



def main():
    log_config()
    rc.init()
    if rc.is_head:
        shape = (48, 48, 48)
        lattice = Lattice(system="cubic", modification="body-centered", a=3.3)
        # lattice = Lattice(system="cubic", a=3.3)
        grid = Grid(lattice=lattice, symmetries=Symmetries(lattice=lattice),
                    shape=shape, comm=rc.comm)
        periodic = np.array([True, True, False])
        latticeCenter = (0.75, 0.75, 0.75)
        check_embed(grid, latticeCenter, periodic)


if __name__ == "__main__":
    main()
