import numpy as np
import cupy as cp
import os, sys, math
import argparse
import mrcfile
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation as R
from Bio.PDB import PDBParser
import xpdb
# changelog ver4
# Now this script returns overlapped pixels.
# add time.sleep(3) before reading the output_chain_file_mrc_filename, in order to prevent from reading null file.
# changelog ver41
# replace pdb2mrc to pdb2mrc_path. The pdb2mrc_remove_verbose.exe is a standalone binary file.
# changelog ver5
# add search for psi.
# Note, the handedness does not affect the Geometric Restrain computation.
# changelog ver6
# Now output the PDB command to a text file.
# changelog ver7
# Change pdb2mrc_remove_verbose.exe to pdb2mrc_gpu_ver_fp32_v3.py
# changelog ver8
# copy pdb2mrc_gpu_ver to here. Now the program will not generate intermediate mrc files.
# fix when only one chain in the pdb, the min_distance could return an error.
# changelog ver8 New
# Completely rewrite the codes by chatGPT. Now only takes 3 seconds comparing to old version 8~11 seconds 
OUT_OF_BOUNDARY = False
IN_BOUNDARY = True

BLOCKSIZE = 1024
BLOCKDIM = lambda x: (x - 1) // BLOCKSIZE + 1

element_data = {
	'H': (1., 1.00794),
	'C': (6., 12.0107),
	'N': (7., 14.00674),
	'O': (8., 15.9994),
	'P': (15., 30.973761),
	'S': (16., 32.066),
}

def read_pdb_split_chain_and_rest(pdb_filename, target_chain_id, do_include_HETATM=False):
	parser = PDBParser(PERMISSIVE=True, structure_builder=xpdb.SloppyStructureBuilder())
	structure = parser.get_structure('FULL_PDB', pdb_filename)

	chain_coords = []
	chain_atoms = []
	rest_coords = []
	rest_atoms = []

	found_chain = False

	for model in structure:
		for chain in model:
			is_target = (chain.id == target_chain_id)
			if is_target:
				found_chain = True

			for residue in chain:
				if not do_include_HETATM:
					tags = residue.get_full_id()
					if tags[3][0] != " ":
						continue

				for atom in residue:
					coord = atom.get_coord()
					elem = atom.element
					if is_target:
						chain_coords.append(coord)
						chain_atoms.append(elem)
					else:
						rest_coords.append(coord)
						rest_atoms.append(elem)

	if not found_chain:
		raise ValueError(f"Cannot find chain {target_chain_id}")

	chain_coords = np.asarray(chain_coords, dtype=np.float32)
	rest_coords = np.asarray(rest_coords, dtype=np.float32)

	chain_atoms = np.asarray(chain_atoms)
	rest_atoms = np.asarray(rest_atoms)

	return chain_coords, chain_atoms, rest_coords, rest_atoms

def rotate_coords_rotvec(coordinates, rot, tilt, psi, xshift, yshift, zshift):
	if coordinates.shape[0] == 0:
		return coordinates.copy()

	geometric_center = np.mean(coordinates, axis=0)
	translated_coords = coordinates - geometric_center

	omega_deg = np.array([rot, tilt, psi], dtype=np.float32)
	omega_rad = np.deg2rad(omega_deg)
	rotation_matrix = R.from_rotvec(omega_rad).as_matrix().astype(np.float32, copy=False)

	rotated_coords = translated_coords @ rotation_matrix
	shift_coordinates = np.array([xshift, yshift, zshift], dtype=np.float32)

	new_coords = rotated_coords + geometric_center + shift_coordinates
	return new_coords.astype(np.float32, copy=False)

def atoms_to_atomic_numbers_and_masses(atoms):
	atomic_numbers = np.zeros(len(atoms), dtype=np.float32)
	atomic_masses = np.zeros(len(atoms), dtype=np.float32)

	for i, symbol in enumerate(atoms):
		if symbol in element_data:
			atomic_numbers[i] = element_data[symbol][0]
			atomic_masses[i] = element_data[symbol][1]

	return atomic_numbers, atomic_masses

def run_pdb2mrc_from_coords(coordinates, atoms, boxsize, apix, res, do_center=False, do_yflip=False):
	device_id = 0
	cp.cuda.Device(device_id).use()

	if coordinates.shape[0] == 0:
		return np.zeros((boxsize, boxsize, boxsize), dtype=np.float32)

	atomic_numbers, atomic_masses = atoms_to_atomic_numbers_and_masses(atoms)
	natoms = len(atoms)

	if do_center:
		mass_sum = np.sum(atomic_masses)
		if mass_sum > 0:
			mass_center = np.sum(coordinates * atomic_masses[:, np.newaxis], axis=0) / mass_sum
			coordinates = coordinates - mass_center

	rp = res / apix
	rp = np.power(np.pi / rp, 2)
	kn = np.power(rp / np.pi, 1.5)
	w = int(round(res * 3.0 / apix))
	if w < 3:
		raise ValueError("You do not have sufficient sampling for this resolution. Decrease apix.")

	coordinates_X_gpu = cp.asarray(coordinates[:, 0], dtype=cp.float32)
	coordinates_Y_gpu = cp.asarray(coordinates[:, 1], dtype=cp.float32)
	coordinates_Z_gpu = cp.asarray(coordinates[:, 2], dtype=cp.float32)
	atomic_numbers_gpu = cp.asarray(atomic_numbers, dtype=cp.float32)

	volume = incert_atoms_to_cupy_array(
		coordinates_X_gpu, coordinates_Y_gpu, coordinates_Z_gpu,
		atomic_numbers_gpu, natoms, boxsize, w, rp, kn, apix
	)
	if do_yflip:
		volume = yflip_vectorized(volume)
	cpu_volume = volume.get().astype(np.float32)
	del coordinates_X_gpu, coordinates_Y_gpu, coordinates_Z_gpu, atomic_numbers_gpu, volume
	cp.get_default_memory_pool().free_all_blocks()
	return cpu_volume

def incert_atoms_to_cupy_array(X, Y, Z, atomic_numbers, natoms, boxsize, w, rp, kn, apix):
	ker_project = cp.RawKernel(r'''
extern "C" __global__ void incert_atoms_to_cupy_array(float* d, const float* atoms_X,const float* atoms_Y,const float* atoms_Z,const float* atomic_numbers, const int natoms, const double apix, const int boxsize, const int w, const double rp, const double kn)
{
	long long l = blockIdx.x * blockDim.x + threadIdx.x;
	int box=boxsize;
	if (l >= natoms) return;

	float xx = (atoms_X[l] / apix) + float(box / 2);
	float yy = (atoms_Y[l] / apix) + float(box / 2);
	float zz = (atoms_Z[l] / apix) + float(box / 2);

	int x[2], y[2], z[2];
	x[0] = round(xx - w);
	x[1] = round(xx + w);
	y[0] = round(yy - w);
	y[1] = round(yy + w);
	z[0] = round(zz - w);
	z[1] = round(zz + w);

	if (x[0] >= box || y[0] >= box || z[0] >= box || x[1] < 0 || y[1] < 0 || z[1] < 0) return;
	if (x[0] < 0) x[0] = 0;
	if (x[1] >= box) x[1] = box - 1;
	if (y[0] < 0) y[0] = 0;
	if (y[1] >= box) y[1] = box - 1;
	if (z[0] < 0) z[0] = 0;
	if (z[1] >= box) z[1] = box - 1;

	float XX=0.;
	float YY=0.;
	float ZZ=0.;
	for (int k = z[0]; k < z[1]; k++) {
		for (int j = y[0]; j < y[1]; j++) {
			for (int i = x[0]; i < x[1]; i++) {
				XX=(float)i - xx;
				YY=(float)j - yy;
				ZZ=(float)k - zz;
				float r = sqrtf(XX * XX + YY * YY + ZZ * ZZ);
				atomicAdd(&d[i + j * box + k * box * box], kn * atomic_numbers[l] * expf(-r * rp));
			}
		}
	}
}''', 'incert_atoms_to_cupy_array')

	volume = cp.zeros((boxsize, boxsize, boxsize), dtype=cp.float32)
	ker_project((BLOCKDIM(natoms),), (BLOCKSIZE,), (volume, X, Y, Z, atomic_numbers, natoms, apix, boxsize, w, rp, kn))
	cp.cuda.Stream.null.synchronize()
	return volume

def yflip_vectorized(arr):
	# The handedness does not affect the Geometric Restrain values. nouse.
	xp = cp.get_array_module(arr)
	arr = arr.copy()
	nz, ny, nx = arr.shape
	dj = -1 if ny % 2 else 0

	j_start = ny // 2
	j1 = xp.arange(j_start, ny)
	j2 = ny - j1 + dj

	mask = j1 > j2
	j1 = j1[mask]
	j2 = j2[mask]

	for k in range(nz):
		temp = arr[k, j1, :].copy()
		arr[k, j1, :] = arr[k, j2, :]
		arr[k, j2, :] = temp

	return arr

def create_BOUNDARY_CHECK_parser():
	parser = argparse.ArgumentParser(description="Check_if_subunit_hit_physical_boundary.")
	parser.add_argument("--i", type=str, required=True, help="Input original PDB file")
	parser.add_argument("--chainID", type=str, required=True, help="Chain ID to be checked")
	parser.add_argument("--rot", type=float, required=True, help="Angle rot in degree")
	parser.add_argument("--tilt", type=float, required=True, help="Angle tilt in degree")
	parser.add_argument("--psi", type=float, required=True, help="Angle psi in degree")
	parser.add_argument("--centerX", type=float, default=0., help="Additional center X")
	parser.add_argument("--centerY", type=float, default=0., help="Additional center Y")
	parser.add_argument("--centerZ", type=float, default=0., help="Additional center Z")
	parser.add_argument("--maskBoxsize", type=int, default=256, help="The boxsize of mask")
	parser.add_argument("--apix", type=float, default=1.5, help="The pixel size for checking")
	parser.add_argument("--outputRoot", type=str, required=True, help="Root name")
	parser.add_argument("--thresholdForMask", type=float, default=1., help="Threshold for mask")
	parser.add_argument("--thresholdChainIsTooFar", type=float, default=50., help="Maximum distance allowed")
	parser.add_argument("--thresholdPixelHits", type=int, default=5, help="Maximum allowed overlapped pixels")
	parser.add_argument("--outputFile", type=str, required=True, help="Output file")
	parser.add_argument("--noUse", action='store_true', help="Place holder")
	return parser

def main():
	parser = create_BOUNDARY_CHECK_parser()
	args = parser.parse_args()

	subunit_serial = args.chainID
	apix = args.apix
	thresholdForMask = args.thresholdForMask
	maskBoxsize = args.maskBoxsize
	xshift = args.centerX
	yshift = args.centerY
	zshift = args.centerZ
#	time0=time.time()
	chain_coords, chain_atoms, rest_coords, rest_atoms = read_pdb_split_chain_and_rest(
		args.i, subunit_serial, do_include_HETATM=False
	)

	data_MRC_chain_original = run_pdb2mrc_from_coords(chain_coords, chain_atoms, maskBoxsize, apix, 2. * apix)
	data_MRC_MainBody = run_pdb2mrc_from_coords(rest_coords, rest_atoms, maskBoxsize, apix, 2. * apix)

	data_MRC_chain_original_Mask = (data_MRC_chain_original > thresholdForMask).astype(np.int8)
	data_MRC_MainBody_Mask = (data_MRC_MainBody > thresholdForMask).astype(np.int8)

	count_Original_Chain_MainBody = 0
	threshold_for_mask_checking_boundary = 1.
	count_threshold_for_mask_checking_boundary = count_Original_Chain_MainBody + args.thresholdPixelHits

	rotated_chain_coords = rotate_coords_rotvec(
		chain_coords, args.rot, args.tilt, args.psi, xshift, yshift, zshift
	)

	data_Rotated_MRC = run_pdb2mrc_from_coords(rotated_chain_coords, chain_atoms, maskBoxsize, apix, 2. * apix)
	data_MRC_rotated_chain_Mask = (data_Rotated_MRC > thresholdForMask).astype(np.int8)

	Add_rotated_Chain_MainBody = data_MRC_rotated_chain_Mask + data_MRC_MainBody_Mask
	count_rotated_Chain_MainBody = np.sum(Add_rotated_Chain_MainBody > threshold_for_mask_checking_boundary)

	dist_transform = distance_transform_edt(1 - data_MRC_rotated_chain_Mask)
	coords2 = np.argwhere(data_MRC_MainBody_Mask == 1)
	try:
		min_distance = float(np.min(dist_transform[tuple(coords2.T)]) * apix)
	except:
		min_distance = 0.0

	rot = convert_float_into_my_format(args.rot)
	tilt = convert_float_into_my_format(args.tilt)
	psi = convert_float_into_my_format(args.psi)
	XSHIFT = convert_float_into_my_format(xshift)
	YSHIFT = convert_float_into_my_format(yshift)
	ZSHIFT = convert_float_into_my_format(zshift)

	print("Minimum distance between the rotated chains and mainbody is (in Angstrom):", min_distance)

	with open(args.outputFile, "w") as ff:
		ff.write(
			str(args.chainID) + "\t" + str(rot) + "\t" + str(tilt) + "\t" + str(psi) + "\t" +
			str(XSHIFT) + "\t" + str(YSHIFT) + "\t" + str(ZSHIFT) + "\t" +
			str(count_rotated_Chain_MainBody) + "\t" + str(min_distance) + "\n"
		)
#	time1=time.time()
#	print(f"runtime = {time1-time0}")
	
def convert_float_into_my_format(number):
	integer_part = int(np.fabs(number))
	sign = (number < 0)
	decimal_part = abs(number - integer_part)
	dp = f"{decimal_part:.1f}".split('.')[1]
	tmp = ""
	if sign:
		tmp = "N"
	output = tmp + str(integer_part) + "p" + str(dp)
	return output

if __name__ == "__main__":
	main()
