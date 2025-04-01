import numpy as np
import cupy as cp
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.vectors import Vector
import xpdb
import argparse
import mrcfile
import time
# changelog v3
# add yflip feature
# remove unused imports and functions
# split the main into functions
# add an option to return 3d numpy array, and no write to disk
# pdb2mrc.exe exclude HETATMs by default. so I did here.
BLOCKSIZE = 1024
BLOCKDIM = lambda x : (x - 1) // BLOCKSIZE + 1
element_data = {
	'H': (1., 1.00794),
	'C': (6., 12.0107),
	'N': (7., 14.00674),
	'O': (8., 15.9994),
	'P': (15., 30.973761),
	'S': (16., 32.066),
}
def create_parser():
	parser = argparse.ArgumentParser(description="Read PDB file and generate 3D mrc.")
	parser.add_argument("--i", type=str, required=True, help="Input PDB file")
	parser.add_argument("--o", type=str, required=True, help="Output 3D MRC")
	parser.add_argument("--box", type=int, default=256, help="The boxsize of 3D MRC, default = 256 (pixel)")
	parser.add_argument("--apix", type=float, default=1.0, help="The pixel size of 3D MRC, default = 1.0 (Angstrom/pixel)")
	parser.add_argument("--res", type=float, default=2.0, help="The resolution limit of 3D MRC, should not be exceed with 2*apix, default = 2.0 (Angstrom^-1)")
	parser.add_argument("--center", action='store_true', help="Whether move the center-of-mass of PDB to the center of 3D MRC. default = False")
	parser.add_argument("--yflip", action='store_true', help="Do yflip to the resulting volume. default = False")
	parser.add_argument("--includeHETATM", action='store_true', help="Whether to include all HETATMs. default = False")
	parser.add_argument("--gpuid", type=int, default=0, help="The specified GPU ID, default = 0")
	parser.add_argument("--dontWriteMRC", action='store_true', help="When set, run_pdb2mrc will return a 3D numpy array rather than writing a mrc file. default = False")
	return parser
def main():
	parser = create_parser()
	args = parser.parse_args()
	run_pdb2mrc(args)
	
def run_pdb2mrc(args):
	time0=time.time()
	pdb_filename = args.i
	output=args.o
	boxsize=args.box
	apix=args.apix
	res=args.res
	do_center=args.center
	do_include_HETATM=args.includeHETATM
	do_yflip=args.yflip
	do_dontWriteMRC = args.dontWriteMRC
	device_id=args.gpuid
	cp.cuda.Device(device_id).use()
	parser = PDBParser(PERMISSIVE=True, structure_builder=xpdb.SloppyStructureBuilder())
	# xpdb must be used to read large (>100,000 atoms in a subunit) PDBs
	structure = parser.get_structure('FULL_PDB', pdb_filename)
	coordinates = []
	atoms = []
	chain_A = None
	for model in structure:
		for chain in model:
			for residue in chain:
				if(not do_include_HETATM):
					tags = residue.get_full_id()
					if tags[3][0] != " ":
						# The residue is a heteroatom
						continue				
				for atom in residue:
					coordinates.append(atom.get_coord())
					atoms.append(atom.element)

	coordinates = np.array(coordinates)
	# e.g. coordinates[i]=[100.0 115.0 123.0]
	atoms=np.array(atoms)
	natoms=len(atoms)
	print("Total of atoms: ",natoms)
	
	atomic_numbers = np.zeros_like(atoms, dtype=float)
	atomic_masses = np.zeros_like(atoms, dtype=float)
	for i, symbol in enumerate(atoms):
		if symbol in element_data:
			atomic_numbers[i] = element_data[symbol][0]
			atomic_masses[i] = element_data[symbol][1]
	# if the atoms are not in the table, regard as zero.		
	# e.g. atomic_numbers[i]=6, atomic_masses[i]=12.0107
	MASS_ARRAY=coordinates*atomic_masses[:, np.newaxis]
#	print(MASS_ARRAY[1500],coordinates[1500],atomic_masses[1500])
	Mass_center = np.sum(MASS_ARRAY, axis=0)/np.sum(atomic_masses[:, np.newaxis])
#	print("Center-of-mass = ",Mass_center)
	if(do_center):
		coordinates=coordinates-Mass_center
		print("Shift center-of-mass to center of 3D MRC.")
#	amin = np.min(coordinates, axis=0)
#	amax = np.max(coordinates, axis=0)
#	print(amin,amax)
#	xt = float(boxsize)/2. - (amax[0]-amin[0])/(2.*apix)
#	yt = float(boxsize)/2. - (amax[1]-amin[1])/(2.*apix)
#	zt = float(boxsize)/2. - (amax[2]-amin[2])/(2.*apix)
	# The gaussian fall-off
	rp=res/apix
	rp=np.power(np.pi/rp,2)
	kn=np.power(rp/np.pi,1.5)
	w=int(round(res*3.0/apix))
	if (w<3):
		print("You do not have sufficient sampling for this resolution. Decrease apix.")
		quit()
	time1=time.time()
#	print(w,rp,kn)
#	print("Run time before GPU = ",round(time1-time0,4)," seconds")
	coordinates_X_gpu=cp.array(coordinates[:,0],dtype=cp.float32)
	coordinates_Y_gpu=cp.array(coordinates[:,1],dtype=cp.float32)
	coordinates_Z_gpu=cp.array(coordinates[:,2],dtype=cp.float32)
#	print(coordinates_X_gpu.dtype)
	atomic_numbers_gpu=cp.array(atomic_numbers,dtype=cp.float32)
	volume = incert_atoms_to_cupy_array(coordinates_X_gpu,coordinates_Y_gpu,coordinates_Z_gpu,atomic_numbers_gpu,natoms,boxsize,w,rp,kn,apix)
	if(do_yflip):
		volume = yflip_vectorized(volume)
	time2=time.time()
#	print("Run time of GPU = ",round(time2-time1,4)," seconds")
	if(do_dontWriteMRC):
		cpu_volume = (volume.get()).astype(np.float32)
		cp.get_default_memory_pool().free_all_blocks()
		return cpu_volume
	with mrcfile.new(output,overwrite=True) as output_volume:
		output_volume.set_data((volume.get()).astype(np.float32))
	output_volume.close()
	cp.get_default_memory_pool().free_all_blocks()
	return 0
def incert_atoms_to_cupy_array(X,Y,Z,atomic_numbers,natoms,boxsize,w,rp,kn,apix):
	# const float is very different from const double!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	# It's very confusing since the typeof is very important.
	ker_project = cp.RawKernel(r'''
extern "C" __global__ void incert_atoms_to_cupy_array(float* d, const float* atoms_X,const float* atoms_Y,const float* atoms_Z,const float* atomic_numbers, const int natoms, const double apix, const int boxsize, const int w, const double rp, const double kn)
{
	long long l = blockIdx.x * blockDim.x + threadIdx.x;
	int box=boxsize;
	if (l >= natoms) return;
//	printf("atoms_X compute l, %f\t%f\t%d\t%d\n", atoms_X[l],apix,box/2,l);
	float xx = (atoms_X[l] / apix) + float(box / 2);
	float yy = (atoms_Y[l] / apix) + float(box / 2);
	float zz = (atoms_Z[l] / apix) + float(box / 2);
//	printf("xx yy zz l, %f\t%f\t%f\t%d\n", xx,yy,zz,l);
	int x[2], y[2], z[2];
	x[0] = round(xx - w);
	x[1] = round(xx + w);
	y[0] = round(yy - w);
	y[1] = round(yy + w);
	z[0] = round(zz - w);
	z[1] = round(zz + w);
//	printf("xx w x0 x1, %f\t%d\t%d\t%d\n",xx,w,x[0],x[1]);
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

	assert isinstance(X, cp.ndarray)
	assert isinstance(Y, cp.ndarray)
	assert isinstance(Z, cp.ndarray)
	assert isinstance(atomic_numbers, cp.ndarray)
	volume = cp.zeros((boxsize, boxsize, boxsize), dtype = cp.float32)
	ker_project((BLOCKDIM(volume.size), ), (BLOCKSIZE, ), (volume, X,Y,Z, atomic_numbers, natoms, apix, boxsize, w, rp, kn))
	cp.cuda.Stream.null.synchronize()
	return volume
def yflip_vectorized(arr):
	"""
	Flip a 3D array along the y-axis, matching EMAN's logic with dj correction.
	Works for both NumPy and CuPy arrays.
	"""
	xp = cp.get_array_module(arr)  # Automatically use numpy or cupy

	arr = arr.copy()
	nz, ny, nx = arr.shape
	dj = -1 if ny % 2 else 0

	# Indices to swap
	j_start = ny // 2
	j1 = xp.arange(j_start, ny)
	j2 = ny - j1 + dj

	# Only keep pairs where j1 > j2 to avoid double-swapping or same index
	mask = j1 > j2
	j1 = j1[mask]
	j2 = j2[mask]

	# Use broadcasting to index and swap
	# Create (len(j),) and broadcast to (len(j), nz, nx)
	for k in range(nz):
		temp = arr[k, j1, :].copy()
		arr[k, j1, :] = arr[k, j2, :]
		arr[k, j2, :] = temp

	return arr
if __name__== "__main__":
	main()
