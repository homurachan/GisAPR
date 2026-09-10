"""In-memory geometric restraint with a reusable fixed-mainbody cache.

Refinement imports this module inside the permanent search worker. The CLI is
retained for existing standalone calls, and never writes intermediate PDB/MRC
files. No PyTorch import occurs until geometry is actually requested.
"""
import argparse

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation as R

from pdb2mrc_gpu_ver_fp32_v3 import (
    atoms_to_atomic_numbers_and_masses,
    density_from_coords,
    element_data,
    yflip_vectorized,
)

OUT_OF_BOUNDARY = False
IN_BOUNDARY = True


def read_pdb_split_chain_and_rest(pdb_filename, target_chain_id, do_include_HETATM=False):
    from pdb_text import read_pdb
    pdb = read_pdb(pdb_filename, include_hetatm=do_include_HETATM, first_model_only=False)
    selected = pdb.chains == target_chain_id
    if not np.any(selected):
        raise ValueError(f'Cannot find chain {target_chain_id}')
    return pdb.coords[selected], pdb.elements[selected], pdb.coords[~selected], pdb.elements[~selected]


def rotate_coords_rotvec(coordinates, rot, tilt, psi, xshift, yshift, zshift, pivot=None):
    """Retain the original geometry's float32 row-vector rotation convention."""
    if coordinates.shape[0] == 0:
        return coordinates.copy()
    geometric_center = (np.mean(coordinates, axis=0) if pivot is None
                        else np.asarray(pivot, dtype=np.float32))
    if geometric_center.shape != (3,) or not np.isfinite(geometric_center).all():
        raise ValueError('pivot must contain three finite coordinates in Angstrom')
    translated_coords = coordinates - geometric_center
    omega_deg = np.asarray([rot, tilt, psi], dtype=np.float32)
    rotation_matrix = R.from_rotvec(np.deg2rad(omega_deg)).as_matrix().astype(np.float32, copy=False)
    shift = np.asarray([xshift, yshift, zshift], dtype=np.float32)
    return ((translated_coords @ rotation_matrix) + geometric_center + shift).astype(np.float32, copy=False)


def run_pdb2mrc_from_coords(coordinates, atoms, boxsize, apix, res,
                          do_center=False, do_yflip=False, device='cpu'):
    numbers, masses = atoms_to_atomic_numbers_and_masses(atoms)
    if do_center and len(coordinates):
        # This compatibility helper historically used float32 atomic masses.
        masses = masses.astype(np.float32)
        if masses.sum() > 0:
            coordinates = coordinates - np.sum(coordinates * masses[:, None], axis=0) / masses.sum()
    volume = density_from_coords(coordinates, numbers, boxsize, apix, res, device=device)
    if do_yflip:
        volume = yflip_vectorized(volume)
    return volume.detach().cpu().numpy()


class GeometryRestraint:
    """Cache atom data, fixed mask and its distance transform for one chain.

    For two nonempty voxel masks A and B, min(EDT(~A)[B]) equals
    min(EDT(~B)[A]). Thus the original minimum-distance calculation can use
    the fixed mainbody distance transform for every subsequent sampled pose.
    Threshold defaults and the density kernel are unchanged. Decisions about
    rejecting a pose and the historical unused bias remain in the caller.
    """
    def __init__(self, pdb_path, chain_id, device='cpu', boxsize=256, apix=1.5,
                 threshold=1., include_hetatm=False):
        import torch
        self.device = torch.device(device)
        self.boxsize = int(boxsize)
        self.apix = float(apix)
        self.threshold = float(threshold)
        self.chain_coords, chain_atoms, rest_coords, rest_atoms = read_pdb_split_chain_and_rest(
            pdb_path, chain_id, do_include_HETATM=include_hetatm)
        chain_numbers, _ = atoms_to_atomic_numbers_and_masses(chain_atoms)
        self.chain_numbers = torch.as_tensor(chain_numbers, dtype=torch.float32, device=self.device)
        rest_numbers, _ = atoms_to_atomic_numbers_and_masses(rest_atoms)
        rest_volume = density_from_coords(rest_coords, rest_numbers, self.boxsize,
                                         self.apix, 2. * self.apix, device=self.device)
        self.mainbody_mask = rest_volume > self.threshold
        del rest_volume
        self.has_mainbody = bool(self.mainbody_mask.any())
        self.mainbody_distance = None
        self._empty_chain_distance = None
        if self.has_mainbody:
            mask_cpu = self.mainbody_mask.cpu().numpy()
            distance = distance_transform_edt(~mask_cpu)
            # Keep double precision to match SciPy's original distance output.
            self.mainbody_distance = torch.as_tensor(distance, dtype=torch.float64, device=self.device)

    def evaluate(self, pose, pivot=None):
        """Return (overlapped voxel count, minimum separation in Angstrom).

        ``pose`` is [rot, tilt, psi, xshift, yshift, zshift]. To skip geometry
        entirely the worker caller omits this request and cache construction.
        """
        import torch
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (6,) or not np.isfinite(pose).all():
            raise ValueError('pose must contain six finite rotation/translation values')
        if not self.has_mainbody:
            return 0, 0.0
        coordinates = rotate_coords_rotvec(self.chain_coords, *pose, pivot=pivot)
        volume = density_from_coords(coordinates, self.chain_numbers, self.boxsize,
                                     self.apix, 2. * self.apix, device=self.device)
        chain_mask = volume > self.threshold
        del volume
        overlap = int(torch.count_nonzero(chain_mask & self.mainbody_mask).item())
        if overlap:
            return overlap, 0.0
        if bool(chain_mask.any()):
            distance = float(self.mainbody_distance[chain_mask].min().item()) * self.apix
            return overlap, distance
        # Preserve the old SciPy all-ones EDT behavior when every selected-chain
        # density voxel is outside the box/below threshold. SciPy measures to
        # the implicit point (-1, 0, 0) in this special case.
        if self._empty_chain_distance is None:
            zz, yy, xx = np.nonzero(self.mainbody_mask.cpu().numpy())
            squared = (zz + 1) ** 2 + yy ** 2 + xx ** 2
            self._empty_chain_distance = float(np.sqrt(np.min(squared))) * self.apix
        return overlap, self._empty_chain_distance


def create_BOUNDARY_CHECK_parser():
    parser = argparse.ArgumentParser(description='Check whether a subunit hits a physical boundary.')
    parser.add_argument('--i', required=True)
    parser.add_argument('--chainID', required=True)
    parser.add_argument('--rot', type=float, required=True)
    parser.add_argument('--tilt', type=float, required=True)
    parser.add_argument('--psi', type=float, required=True)
    parser.add_argument('--centerX', type=float, default=0.)
    parser.add_argument('--centerY', type=float, default=0.)
    parser.add_argument('--centerZ', type=float, default=0.)
    parser.add_argument('--pivot', type=float, nargs=3, metavar=('X', 'Y', 'Z'))
    parser.add_argument('--maskBoxsize', type=int, default=256)
    parser.add_argument('--apix', type=float, default=1.5)
    parser.add_argument('--outputRoot', required=True)
    parser.add_argument('--thresholdForMask', type=float, default=1.)
    parser.add_argument('--thresholdChainIsTooFar', type=float, default=50.)
    parser.add_argument('--thresholdPixelHits', type=int, default=5)
    parser.add_argument('--outputFile', required=True)
    parser.add_argument('--noUse', action='store_true', help='Legacy placeholder')
    parser.add_argument('--gpuid', type=int, default=0)
    parser.add_argument('--device', help='PyTorch device, e.g. cuda:0 or cpu; overrides --gpuid')
    parser.add_argument('--skip-geometric-restraint', action='store_true')
    return parser


def convert_float_into_my_format(number):
    integer_part = int(np.fabs(number))
    decimal_part = abs(number - integer_part)
    dp = f'{decimal_part:.1f}'.split('.')[1]
    return ('N' if number < 0 else '') + str(integer_part) + 'p' + dp


def main():
    args = create_BOUNDARY_CHECK_parser().parse_args()
    pose = [args.rot, args.tilt, args.psi, args.centerX, args.centerY, args.centerZ]
    if args.skip_geometric_restraint:
        overlap, min_distance = 0, 0.0
    else:
        import torch
        device = args.device or ('cuda:' + str(args.gpuid) if torch.cuda.is_available() else 'cpu')
        restraint = GeometryRestraint(args.i, args.chainID, device=device,
                                     boxsize=args.maskBoxsize, apix=args.apix,
                                     threshold=args.thresholdForMask)
        overlap, min_distance = restraint.evaluate(pose, pivot=args.pivot)
    print('Minimum distance between the rotated chains and mainbody is (in Angstrom):', min_distance)
    values = [args.chainID] + [convert_float_into_my_format(x) for x in pose] + [str(overlap), str(min_distance)]
    with open(args.outputFile, 'w') as handle:
        handle.write('\t'.join(values) + '\n')


if __name__ == '__main__':
    main()
