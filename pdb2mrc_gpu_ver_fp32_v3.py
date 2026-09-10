"""PDB density generation using PyTorch, usable inside the persistent search worker.

The original density kernel is retained, including its radial exponential,
float32 output, C-style rounding and exclusive/clipped upper voxel bounds.
PyTorch is imported lazily so the optimizer process does not initialize it.
"""
import argparse
from functools import lru_cache

import numpy as np


element_data = {
    'H': (1., 1.00794), 'C': (6., 12.0107), 'N': (7., 14.00674),
    'O': (8., 15.9994), 'P': (15., 30.973761), 'S': (16., 32.066),
}


def atoms_to_atomic_numbers_and_masses(atoms):
    values = np.asarray([element_data.get(str(a).upper(), (0., 0.)) for a in atoms],
                        dtype=np.float64).reshape(-1, 2)
    return values[:, 0], values[:, 1]


@lru_cache(maxsize=8)
def _voxel_offsets(width, device_string):
    import torch
    # At a half-integer crossing zero, round-away-from-zero can give 2*w+1
    # positions rather than 2*w. The per-atom upper bound below selects them.
    axis = torch.arange(2 * width + 1, device=device_string, dtype=torch.int64)
    zz, yy, xx = torch.meshgrid(axis, axis, axis, indexing='ij')
    return torch.stack((xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)), dim=1)


def _round_away_from_zero(value):
    import torch
    value = value.to(torch.float64)
    return (torch.sign(value) * torch.floor(torch.abs(value) + 0.5)).to(torch.int64)


def density_from_coords(coordinates, atomic_numbers, boxsize, apix, res,
                        device='cpu', max_chunk_voxels=524288):
    """Return an unflipped float32 torch volume; no files or subprocesses.

    ``atomic_numbers`` must follow ``element_data`` (unsupported elements have
    zero weight). Chunks bound the temporary atom/voxel tensor size. The output
    is indexed [z, y, x], as in the original GPU implementation.
    """
    import torch
    boxsize = int(boxsize)
    apix, res = float(apix), float(res)
    if boxsize < 1 or not np.isfinite(apix) or not np.isfinite(res) or apix <= 0 or res <= 0:
        raise ValueError('boxsize, apix and resolution must be positive and finite')
    width = int(round(res * 3.0 / apix))
    if width < 3:
        raise ValueError('You do not have sufficient sampling for this resolution. Decrease apix.')
    rp = float(np.power(np.pi / (res / apix), 2))
    kn = float(np.power(rp / np.pi, 1.5))
    device = torch.device(device)
    xyz = torch.as_tensor(coordinates, dtype=torch.float32, device=device).reshape(-1, 3)
    numbers = torch.as_tensor(atomic_numbers, dtype=torch.float32, device=device).reshape(-1)
    if len(xyz) != len(numbers):
        raise ValueError('Coordinate and atom count differ')
    volume = torch.zeros((boxsize, boxsize, boxsize), dtype=torch.float32, device=device)
    if len(xyz) == 0:
        return volume
    if not bool(torch.isfinite(xyz).all()):
        raise ValueError('PDB coordinates must be finite')

    # CUDA originally divides by a double apix, adds integer box/2, then casts
    # into a float position. Do not replace this with float32 division.
    positions = (xyz.to(torch.float64) / apix + boxsize // 2).to(torch.float32)
    lower = _round_away_from_zero(positions - width)
    upper = _round_away_from_zero(positions + width)
    valid_atoms = ((lower < boxsize).all(dim=1) & (upper >= 0).all(dim=1) & (numbers != 0))
    positions = positions[valid_atoms]
    numbers = numbers[valid_atoms]
    lower = lower[valid_atoms].clamp(min=0)
    upper = upper[valid_atoms].clamp(max=boxsize - 1)
    offsets = _voxel_offsets(width, str(device))
    chunk_size = max(1, int(max_chunk_voxels) // len(offsets))
    flat_volume = volume.reshape(-1)
    with torch.no_grad():
        for start in range(0, len(positions), chunk_size):
            end = start + chunk_size
            voxels = lower[start:end, None, :] + offsets[None, :, :]
            valid = (voxels < upper[start:end, None, :]).all(dim=2)
            displacement = voxels.to(torch.float32) - positions[start:end, None, :]
            squared = displacement * displacement
            radius = torch.sqrt((squared[..., 0] + squared[..., 1]) + squared[..., 2])
            # expf receives a float argument even though rp is double; the
            # multiplication by kn is double before atomicAdd casts to float.
            exponent = (-radius.to(torch.float64) * rp).to(torch.float32)
            values = ((numbers[start:end, None].to(torch.float64) * kn)
                      * torch.exp(exponent).to(torch.float64)).to(torch.float32)
            indices = voxels[..., 0] + boxsize * (voxels[..., 1] + boxsize * voxels[..., 2])
            flat_volume.index_add_(0, indices[valid], values[valid])
    return volume


def yflip_vectorized(arr):
    """Apply the original EMAN y-flip (even boxes retain row zero)."""
    ny = arr.shape[1]
    indices = np.arange(ny)
    j1 = np.arange(ny // 2, ny)
    j2 = ny - j1 + (-1 if ny % 2 else 0)
    valid = j1 > j2
    indices[j1[valid]], indices[j2[valid]] = j2[valid], j1[valid]
    if isinstance(arr, np.ndarray):
        return arr[:, indices, :].copy()
    import torch
    return arr.index_select(1, torch.as_tensor(indices, device=arr.device))


def generate_mrc(pdb_path, output, boxsize, apix, res, device='cpu',
                 do_center=False, do_yflip=False, include_hetatm=False):
    """Read text PDB and write its MRC, or return a numpy array if output=None."""
    from pdb_text import read_pdb
    pdb = read_pdb(pdb_path, include_hetatm=include_hetatm, first_model_only=False)
    coordinates = pdb.coords
    atomic_numbers, masses = atoms_to_atomic_numbers_and_masses(pdb.elements)
    if do_center and len(coordinates):
        if masses.sum() <= 0:
            raise ValueError('Cannot center PDB: no supported atoms with nonzero mass')
        coordinates = coordinates - np.sum(coordinates * masses[:, None], axis=0) / masses.sum()
    volume = density_from_coords(coordinates, atomic_numbers, boxsize, apix, res, device=device)
    if do_yflip:
        volume = yflip_vectorized(volume)
    data = volume.detach().cpu().numpy()
    if output is None:
        return data
    import mrcfile
    with mrcfile.new(output, overwrite=True) as result:
        result.set_data(data)
    return 0


def create_parser():
    parser = argparse.ArgumentParser(description='Read text PDB and generate a 3-D MRC using PyTorch.')
    parser.add_argument('--i', required=True, help='Input PDB file')
    parser.add_argument('--o', required=True, help='Output MRC file')
    parser.add_argument('--box', type=int, default=256)
    parser.add_argument('--apix', type=float, default=1.0)
    parser.add_argument('--res', type=float, default=2.0)
    parser.add_argument('--center', action='store_true')
    parser.add_argument('--yflip', action='store_true')
    parser.add_argument('--includeHETATM', action='store_true')
    parser.add_argument('--gpuid', type=int, default=0)
    parser.add_argument('--device', help='PyTorch device, e.g. cuda:0 or cpu; overrides --gpuid')
    parser.add_argument('--dontWriteMRC', action='store_true')
    return parser


def run_pdb2mrc(args):
    import torch
    device = getattr(args, 'device', None)
    if device is None:
        device = 'cuda:' + str(args.gpuid) if torch.cuda.is_available() else 'cpu'
    return generate_mrc(args.i, None if args.dontWriteMRC else args.o,
                        args.box, args.apix, args.res, device=device,
                        do_center=args.center, do_yflip=args.yflip,
                        include_hetatm=args.includeHETATM)


def main():
    run_pdb2mrc(create_parser().parse_args())


if __name__ == '__main__':
    main()
