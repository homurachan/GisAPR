import numpy as np
import cupy as cp
import os,sys,math
import argparse
import mrcfile,time
from scipy.ndimage import distance_transform_edt
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.vectors import Vector
import xpdb
OUT_OF_BOUNDARY = False
IN_BOUNDARY= True
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
def run_pdb2mrc_return_numpy_volume(pdb_filename,boxsize,apix,res):
	time0=time.time()
	do_center=False
	do_include_HETATM=False
	do_dontWriteMRC = True
	do_yflip = False
	device_id=0
	# The default gpuid. I don't think this will affect much. The program only consumes 1GB memory.
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
#	print("Total of atoms: ",natoms)
	
	atomic_numbers = np.zeros_like(atoms, dtype=float)
	atomic_masses = np.zeros_like(atoms, dtype=float)
	for i, symbol in enumerate(atoms):
		if symbol in element_data:
			atomic_numbers[i] = element_data[symbol][0]
			atomic_masses[i] = element_data[symbol][1]
	# if the atoms are not in the table, regard as zero.		
	# e.g. atomic_numbers[i]=6, atomic_masses[i]=12.0107
	MASS_ARRAY=coordinates*atomic_masses[:, np.newaxis]
	Mass_center = np.sum(MASS_ARRAY, axis=0)/np.sum(atomic_masses[:, np.newaxis])
	if(do_center):
		coordinates=coordinates-Mass_center
		print("Shift center-of-mass to center of 3D MRC.")

	# The gaussian fall-off
	rp=res/apix
	rp=np.power(np.pi/rp,2)
	kn=np.power(rp/np.pi,1.5)
	w=int(round(res*3.0/apix))
	if (w<3):
		print("You do not have sufficient sampling for this resolution. Decrease apix.")
		quit()
	time1=time.time()
	coordinates_X_gpu=cp.array(coordinates[:,0],dtype=cp.float32)
	coordinates_Y_gpu=cp.array(coordinates[:,1],dtype=cp.float32)
	coordinates_Z_gpu=cp.array(coordinates[:,2],dtype=cp.float32)

	atomic_numbers_gpu=cp.array(atomic_numbers,dtype=cp.float32)
	volume = incert_atoms_to_cupy_array(coordinates_X_gpu,coordinates_Y_gpu,coordinates_Z_gpu,atomic_numbers_gpu,natoms,boxsize,w,rp,kn,apix)
	if(do_yflip):
		volume = yflip_vectorized(volume)
	time2=time.time()

	cpu_volume = (volume.get()).astype(np.float32)
	cp.get_default_memory_pool().free_all_blocks()
	return cpu_volume
	
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
def create_BOUNDARY_CHECK_parser():
	parser = argparse.ArgumentParser(description="Check_if_subunit_hit_physical_boundary.")
	parser.add_argument("--i", type=str, required=True, help="Input original PDB file")
	parser.add_argument("--chainID", type=str, required=True, help="Chain ID to be checked")
	parser.add_argument("--rot", type=float, required=True, help="Angle rot in degree")
	parser.add_argument("--tilt", type=float, required=True, help="Angle tilt in degree")
	parser.add_argument("--psi", type=float, required=True, help="Angle psi in degree")
	parser.add_argument("--centerX", type=float, default=0., help="Additional center X. To be added to geometric center of chain_ID. default = 0.0 (Angstrom)")
	parser.add_argument("--centerY", type=float, default=0., help="Additional center Y. To be added to geometric center of chain_ID. default = 0.0 (Angstrom)")
	parser.add_argument("--centerZ", type=float, default=0., help="Additional center Z. To be added to geometric center of chain_ID. default = 0.0 (Angstrom)")
	parser.add_argument("--maskBoxsize", type=int, default=256, help="The boxsize of mask for checking, default = 256 (pixel)")
	parser.add_argument("--apix", type=float, default=1.5, help="The pixel size for checking, don't be confused by actual pixel size of the models or templates. default = 1.5")
	parser.add_argument("--outputRoot", type=str, required=True, help="The root name for mainbody and chain PDBs")
	parser.add_argument("--thresholdForMask", type=float, default=1., help="In creating mask, use this threshold.")
	parser.add_argument("--thresholdChainIsTooFar", type=float, default=50., help="The maximum distance allowed between the closest atom in selected chain and main body. default =  50 (Angstrom)")
	parser.add_argument("--thresholdPixelHits", type=int, default=5, help="The maximum allowed overlapped pixels, default = 5")
	parser.add_argument("--outputFile", type=str, required=True, help="The output file that stores ")
	parser.add_argument("--noUse", action='store_true', help="Place holder here. default = False")
	
	return parser
def main():
	## reading parameters
	
	parser = create_BOUNDARY_CHECK_parser()
	args = parser.parse_args()
	subunit_serial=args.chainID
	apix=args.apix
	thresholdForMask=args.thresholdForMask
	maskBoxsize=args.maskBoxsize
	xshift=args.centerX
	yshift=args.centerY
	zshift=args.centerZ
	pdb2mrc_path = "python pdb2mrc_gpu_ver_fp32_v3.py"
	# 0. split the selected chain and main body.
	output_chain_file = args.outputRoot+"_selected_chain_"+str(subunit_serial)+".pdb"
	output_rest_file = args.outputRoot+"_the_Main_Body.pdb"
	Name_py_split_chains = "read_pdb_split_to_transform_and_nochange.py"
	
	if (os.path.exists(output_chain_file) and os.path.exists(output_rest_file)):
		sksksk=1
	else:
		Command_step_0 = "python "+Name_py_split_chains+" "+args.i+" "+args.chainID+" "+args.outputRoot
		os.system(Command_step_0)

	# 1. pdb2mrc without the selected chain
	# 2. pdb2mrc only the selected chain
	'''
	output_chain_file_mrc_filename=args.outputRoot+"_selected_chain_"+str(subunit_serial)+"_for_Boundary_check.mrc"
	output_rest_file_mrc_filename=args.outputRoot+"_the_Main_Body.mrc"
	if (os.path.exists(output_chain_file_mrc_filename) and os.path.exists(output_rest_file_mrc_filename)):
		sksks=1
	else:
		Command_step_1 = pdb2mrc_path+" --i "+output_chain_file+" --o "+output_chain_file_mrc_filename+" --box "+str(maskBoxsize)+" --apix "+str(apix)+" --res "+str(2.*apix)
		Command_step_2 = pdb2mrc_path+" --i "+output_rest_file+" --o "+output_rest_file_mrc_filename+" --box "+str(maskBoxsize)+" --apix "+str(apix)+" --res "+str(2.*apix)
		os.system(Command_step_1)
		os.system(Command_step_2)
	'''
	data_MRC_chain_original = run_pdb2mrc_return_numpy_volume(output_chain_file,maskBoxsize,apix,2.*apix)
	data_MRC_MainBody = run_pdb2mrc_return_numpy_volume(output_rest_file,maskBoxsize,apix,2.*apix)
	# 3. generate masks for both mrc-s.
	# new version, don't use relion_mask_create anymore.
	# Need to sleep for 3seconds, or some error may pop up.
	'''
	time.sleep(3)
	with mrcfile.mmap(output_chain_file_mrc_filename, mode='r') as MRC_chain_original:
		data_MRC_chain_original=MRC_chain_original.data
	with mrcfile.mmap(output_rest_file_mrc_filename, mode='r') as MRC_MainBody:
		data_MRC_MainBody=MRC_MainBody.data
	'''
	data_MRC_chain_original_Mask = (data_MRC_chain_original > thresholdForMask).astype(int)
	data_MRC_MainBody_Mask= (data_MRC_MainBody > thresholdForMask).astype(int)

	# 4. add them up, if number of pixels whose values are greater than 1, larger than a given threshold, return hit boundary condition.
	# first, we need to check how many hit points are in the original PDB files.
	# then we add the thresholdPixelHits to the original results.

	Add_Original_Chain_MainBody = data_MRC_chain_original_Mask+data_MRC_MainBody_Mask
	# The values of mask are either zero or one.
	threshold_for_mask_checking_boundary = 1.
#	count_Original_Chain_MainBody = np.sum(Add_Original_Chain_MainBody > threshold_for_mask_checking_boundary)
	count_Original_Chain_MainBody = 0
	# Just in case the original model has overlaps. A very small overlap will not affect our results.
	count_threshold_for_mask_checking_boundary = count_Original_Chain_MainBody + args.thresholdPixelHits
#	print("DEBUG count_Original_Chain_MainBody = ",count_Original_Chain_MainBody)
	# Now compute the rotated chain.
	
	######## To be added: apply centerX/Y/Z into rotate_subunit_of_PDB.py
	rot=convert_float_into_my_format(args.rot)
	tilt=convert_float_into_my_format(args.tilt)
	psi=convert_float_into_my_format(args.psi)
	XSHIFT=convert_float_into_my_format(xshift)
	YSHIFT=convert_float_into_my_format(yshift)
	ZSHIFT=convert_float_into_my_format(zshift)
	Rotate_Chain_Python_Name="python rotate_translate_subunit_of_PDB_v2.py "
	Rotated_PDB_Name=args.outputRoot+"_chain_"+subunit_serial+"_rot"+str(rot)+"_tilt"+str(tilt)+"_psi"+str(psi)+"deg_trans"+str(XSHIFT)+"_"+str(YSHIFT)+"_"+str(ZSHIFT)+"ANG.pdb"
	Command_Rotate_Chain=Rotate_Chain_Python_Name+args.i+" "+subunit_serial+" "+str(args.rot)+" "+str(args.tilt)+" "+str(args.psi)+" "+str(xshift)+" "+str(yshift)+" "+str(zshift)+" "+Rotated_PDB_Name
#	print("Command_Rotate_Chain = ",Command_Rotate_Chain)
	os.system(Command_Rotate_Chain)
	xxx=open("func_check.log","a")
	xxx.write(Command_Rotate_Chain)
	xxx.write("\n")
	xxx.close()
	outputRoot_intermediate = args.outputRoot+"_tmp_chain_"+subunit_serial+"_rot"+str(rot)+"_tilt"+str(tilt)+"_psi"+str(psi)+"deg_trans"+str(XSHIFT)+"_"+str(YSHIFT)+"_"+str(ZSHIFT)
	output_rotated_chain_file = outputRoot_intermediate+"_selected_chain_"+str(subunit_serial)+".pdb"
	output_rotated_nouse_mainbody_file=outputRoot_intermediate+"_the_Main_Body.pdb"
	
	Command_split_rotated_chain_and_mainbody = "python "+Name_py_split_chains+" "+Rotated_PDB_Name+" "+args.chainID+" "+outputRoot_intermediate
#	print("Command_split_rotated_chain_and_mainbody = ",Command_split_rotated_chain_and_mainbody)
	os.system(Command_split_rotated_chain_and_mainbody)
	
	data_Rotated_MRC = run_pdb2mrc_return_numpy_volume(output_rotated_chain_file,maskBoxsize,apix,2.*apix)
	
#	Rotated_MRC_Name = outputRoot_intermediate+"ANG.mrc"
#	Command_pdb2mrc_rotated_chain = pdb2mrc_path+" --i "+output_rotated_chain_file+" --o "+Rotated_MRC_Name+" --box "+str(maskBoxsize)+" --apix "+str(apix)+" --res "+str(2.*apix)
#	print("Command_pdb2mrc_rotated_chain = ",Command_pdb2mrc_rotated_chain)
#	os.system(Command_pdb2mrc_rotated_chain)
#	print("debug, rot,tilt,Rotated_MRC_Name = ",args.rot,args.tilt,Rotated_MRC_Name)
#	Rotated_MRC_Mask_Name= outputRoot_intermediate+"_"+subunit_serial+"_rot"+str(rot)+"_tilt"+str(tilt)+"deg_mask.mrc"
#	with mrcfile.mmap(Rotated_MRC_Name, mode='r') as Rotated_MRC:
#		data_Rotated_MRC=Rotated_MRC.data
	data_MRC_rotated_chain_Mask= (data_Rotated_MRC > thresholdForMask).astype(int)
#	Command_create_mask_rotated_chain = "relion_mask_create --i "+Rotated_MRC_Name+" --o "+Rotated_MRC_Mask_Name+" --ini_threshold "+str(thresholdForMask)+" --j 4"
#	os.system(Command_create_mask_rotated_chain)
	
#	with mrcfile.mmap(Rotated_MRC_Mask_Name, mode='r') as MRC_rotated_chain_Mask:
#		data_MRC_rotated_chain_Mask=MRC_rotated_chain_Mask.data
	Add_rotated_Chain_MainBody = data_MRC_rotated_chain_Mask+data_MRC_MainBody_Mask
	count_rotated_Chain_MainBody = np.sum(Add_rotated_Chain_MainBody > threshold_for_mask_checking_boundary)
#	print("DEBUG count_rotated_Chain_MainBody = ",count_rotated_Chain_MainBody)
#	if(count_rotated_Chain_MainBody>count_threshold_for_mask_checking_boundary):
#		return OUT_OF_BOUNDARY
	
	# 5. also need to check if the selected chain is too far away from the remaining proteins.
	# 6. compute the minimum distance of selected chain to the rests. return hit boundary condition if the distance is larger than a given threshold.
	
	dist_transform = distance_transform_edt(1 - data_MRC_rotated_chain_Mask)
	coords2 = np.argwhere(data_MRC_MainBody_Mask == 1)  # Get coordinates of ones in volume2
	min_distance = float(np.min(dist_transform[tuple(coords2.T)])*apix)  # Minimum distance to volume1
#	Rotated_MRC.close()
#	MRC_MainBody.close()
#	MRC_chain_original.close()
	print("Minimum distance between the rotated chains and mainbody is (in Angstrom):", min_distance)
	ff=open(args.outputFile,"w")
	ff.write(str(args.chainID)+"\t"+str(rot)+"\t"+str(tilt)+"\t"+str(psi)+"\t"+str(XSHIFT)+"\t"+str(YSHIFT)+"\t"+str(ZSHIFT)+"\t"+str(count_rotated_Chain_MainBody)+"\t"+str(min_distance)+"\n")
	ff.close()
	
	if(os.name =='nt'):
		Command_to_remove = "del "+Rotated_PDB_Name+" "+output_rotated_chain_file+" "+output_rotated_nouse_mainbody_file
	else:
		Command_to_remove = "rm "+Rotated_PDB_Name+" "+output_rotated_chain_file+" "+output_rotated_nouse_mainbody_file
	os.system(Command_to_remove)
	
def convert_float_into_my_format(number):
	integer_part = int(np.fabs(number))
	sign = (number <0)
	decimal_part = abs(number - integer_part)
	dp=f"{decimal_part:.1f}".split('.')[1]
	tmp=""
	if(sign):
		tmp="N"
	output=tmp+str(integer_part)+"p"+str(dp)
	return output
if __name__== "__main__":
	main()
