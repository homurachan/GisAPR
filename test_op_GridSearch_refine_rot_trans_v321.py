import numpy as np
from scipy.optimize import minimize
import os, sys
import itertools
import concurrent.futures
import pickle,argparse
from functools import partial
from func import func_gpuid
# changelog ver2
# do the Geometric search first, then read the results and do the GPU search.
# changelog ver3
# modification for running under GUI.
# changelog ver31
# Add handedness to read_pdb_index_generate_sh_3DEG_local_v32.py. It should also be used together with the modified relion_project.
# changelog ver32
# The name of the search script will be fixed. currently the local value is changing, leading to some error.
# The func is now moving to a seperate file.
# Update func_check_boundary_for_testing to v8. (20250330)
# Add apix_PDB (20250415)
# changelog ver321
# Fix gpu only runs on the first device when --gpuid contains more than one device_id.
Geo_optimization_history = []
optimization_history = []
def create_gridsearch_parser():
	parser = argparse.ArgumentParser(description="Grid Refine Execution")
	parser.add_argument("--PDB_NAME", type=str, required=True)
	parser.add_argument("--STAR_NAME", type=str, required=True)
	parser.add_argument("--rotate_chain", type=str, required=True)
	parser.add_argument("--output_name_root", type=str, default="output")
	parser.add_argument("--gpuid", type=str, default="0")
	parser.add_argument("--ang", type=str, default="/groups/kyouko/mydata/c1_3deg_remove_rotLzero_200kV.star")
	parser.add_argument("--boxsize", type=int, default=256)
	parser.add_argument("--apix", type=float, default=1.58)
	parser.add_argument("--apix_PDB", type=float, default=1.58)
	parser.add_argument("--newboxsize", type=int, default=160)
	parser.add_argument("--search_script", type=str, default="/groups/kyouko/mydata/test1_with_isspa_weight_varingKK_search_translation_also_v6032.py")
	parser.add_argument("--fsc_file", type=str, default="ribo_recons_masked_vs_7k00_masked.fsc")
	parser.add_argument("--transRange", type=int, default=0)
	parser.add_argument("--voltage", type=float, default=300.0)
	parser.add_argument("--cs", type=float, default=2.7)
	parser.add_argument("--psiStep", type=int, default=3)
	parser.add_argument("--kk", type=int, default=3)
	parser.add_argument("--do_local_search", action="store_true")
	parser.add_argument("--local_stepsize", type=float, default=30)
	parser.add_argument("--do_ignoreFSC", action="store_true")
	parser.add_argument("--yflip", action="store_true")
	parser.add_argument("--doEnableGpuProj", action='store_true')
	parser.add_argument("--doSplitDiffGpu", action='store_true', help="If enabled, wrap_to_search will use different gpuid. The inital gpuid is provided by --gpuid. default = False")
	parser.add_argument("--SplitParticles", type=int, default=1, help="Split the starfile into these sections. Default = 1")
	parser.add_argument("--do_run_CC", action="store_true")
	parser.add_argument("--do_simple_sum", action="store_true")
	parser.add_argument("--maskRadius", type=int, default=110)
	parser.add_argument("--Geometric_restrain_Scaling_Factor", type=float, default=1.0)
	parser.add_argument("--chain_MASS_in_residues", type=int, default=275)
	parser.add_argument("--MAXIUM_ALLOWED_overlapped_pixels", type=int, default=130)
	parser.add_argument("--MAX_MinDistance_Allowed", type=float, default=20.)
	# PSO parameters
	parser.add_argument("--Grid_Bounds", type=str, default="(-15,15),(-15,15),(-15,15),(-10,10),(-10,10),(-10,10)")
	parser.add_argument("--Grid_Rotation_Stepsize", type=int, default=5)
	parser.add_argument("--Grid_Translation_Stepsize", type=int, default=10)
	parser.add_argument("--Grid_dimensions", type=int, default=6)
	parser.add_argument("--max_workers_CPU", type=int, default=8)
	parser.add_argument("--max_workers_GPU", type=int, default=3)
	return parser
def convert_number_to_filename(number):
	integer_part = int(np.fabs(number))
	sign = (number <0)
	decimal_part = abs(number - integer_part)
	dp=f"{decimal_part:.1f}".split('.')[1]
	tmp=""
	if(sign):
		tmp="N"
	output=tmp+str(integer_part)+"p"+str(dp)
	return output
####
def func_geo(x, args):
	PDB_NAME = args.PDB_NAME
	STAR_NAME = args.STAR_NAME
	rotate_chain = args.rotate_chain
	output_name_root = args.output_name_root
	gpuid = args.gpuid
	ang = args.ang
	boxsize = args.boxsize
	apix = args.apix
	newboxsize = args.newboxsize
	search_script = args.search_script
	fsc_file = args.fsc_file
	transRange = args.transRange
	voltage = args.voltage
	cs = args.cs
	psiStep = args.psiStep
	kk = args.kk
	do_local_search = args.do_local_search
	local_stepsize = args.local_stepsize
	do_ignoreFSC = args.do_ignoreFSC
	maskRadius = args.maskRadius
	Geometric_restrain_Scaling_Factor=args.Geometric_restrain_Scaling_Factor
	chain_MASS_in_residues = args.chain_MASS_in_residues
	MAXIUM_ALLOWED_overlapped_pixels = args.MAXIUM_ALLOWED_overlapped_pixels
	MAX_MinDistance_Allowed = args.MAX_MinDistance_Allowed
	yflip=args.yflip
	rot=x[0]
	tilt=x[1]
	psi=x[2]
	xshift=x[3]
	yshift=x[4]
	zshift=x[5]
	score_in_this_conformation = 9999.
	str_Rot=convert_number_to_filename(rot)
	str_Tilt=convert_number_to_filename(tilt)
	str_Psi=convert_number_to_filename(psi)
	str_XSHIFT=convert_number_to_filename(xshift)
	str_YSHIFT=convert_number_to_filename(yshift)
	str_ZSHIFT=convert_number_to_filename(zshift)

	# Step: 
	# 0. Compute geometric restrain. If overlapped pixels are too many, skip all the rest computation.
	Step0_Python_Name="python func_check_boundary_for_testing_v8.py "
	Geometric_restrain_Result_filename = output_name_root+rotate_chain+"_rot"+str_Rot+"_tilt"+str_Tilt+"_psi"+str_Psi+"deg_trans"+str_XSHIFT+"_"+str_YSHIFT+"_"+str_ZSHIFT+"_GeometricRestrain_Result.txt"
	To_Run_Command_Step0 = Step0_Python_Name+"--i "+PDB_NAME+" --chainID "+rotate_chain+" --rot "+str(rot)+" --tilt "+str(tilt)+" --psi "+str(psi)\
	+" --centerX "+str(xshift)+" --centerY "+str(yshift)+" --centerZ "+str(zshift)+" --outputRoot "+output_name_root+" --outputFile "+Geometric_restrain_Result_filename+"\n"
	os.system(To_Run_Command_Step0)
	Geometric_restrain_Result_FILE = open(Geometric_restrain_Result_filename,"r")
	Geometric_restrain_Result_FILE_lines=Geometric_restrain_Result_FILE.readlines()
	Geometric_restrain_Overlapped_Pixels=float(Geometric_restrain_Result_FILE_lines[0].split()[7])
	Min_Distance = float(Geometric_restrain_Result_FILE_lines[0].split()[8])
	print("debug, Overlapped pixels, Min_Distance = ",Geometric_restrain_Overlapped_Pixels,Min_Distance)
	return Geometric_restrain_Overlapped_Pixels,Min_Distance
def generate_grid(bound, stepsize,dimensions):
	""" Generate all grid points based on the given boundary and step sizes. """
	grid_axes = [np.arange(bound[i, 0], bound[i, 1] + stepsize[i], stepsize[i]) for i in range(dimensions)]
	return np.array(list(itertools.product(*grid_axes)))  # Generate all 6D points
def remake_grid(positions,dimensions):
	"""Recreate the grid based on filtered positions."""
	# Get unique values for each dimension
	unique_axes = [np.unique(positions[:, i]) for i in range(dimensions)]
	# Generate the new grid
	return np.array(list(itertools.product(*unique_axes)))
def convert_Bounds_to_bounds_for_grid(args):
	string_bounds = args.Grid_Bounds
	dimensions = args.Grid_dimensions
	bounds=np.zeros((dimensions,2))
	for i in range(dimensions):
		part_N=string_bounds.split(')')[i]
		remove_left = part_N.split('(')[1]
		LOW=float(remove_left.split(',')[0])
		HIGH=float(remove_left.split(',')[1])
		bounds[i,0]=LOW
		bounds[i,1]=HIGH
	return bounds

if __name__ == "__main__":
	
	parser = create_gridsearch_parser()
	args = parser.parse_args()
	BOUNDS = convert_Bounds_to_bounds_for_grid(args)
	rot_step_size=args.Grid_Rotation_Stepsize
	shift_step_size=args.Grid_Translation_Stepsize
	dimensions=args.Grid_dimensions
	STEPSIZE = np.array([rot_step_size, rot_step_size, rot_step_size, shift_step_size, shift_step_size, shift_step_size])  # Step sizes for each dimension
	positions = generate_grid(BOUNDS, STEPSIZE,dimensions)
	FILE_LOG=open("GridSearch_GEO.log","a")
	func_geo_partial = partial(func_geo, args=args)
	with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers_CPU) as executor:
		results = list(executor.map(func_geo_partial, positions))
		Geo_optimization_history.append([results, positions])
		FILE_LOG.write(str(Geo_optimization_history))
	FILE_LOG.close()

	combined_data = [(pos, A, B) for pos, (A, B) in zip(positions, results)]
	with open('positions_and_results.pkl', 'wb') as f:
		pickle.dump(combined_data, f)
	print(f"Saved {len(combined_data)} positions and results to 'positions_and_results.pkl'.")

	filtered_positions = [pos for pos, (Geo_score, Geo_Mindist) in zip(positions, results) if Geo_score < args.MAXIUM_ALLOWED_overlapped_pixels and Geo_Mindist < args.MAX_MinDistance_Allowed]
	with open('filtered_positions.pkl', 'wb') as f:
		pickle.dump(filtered_positions, f)
	print(f"Filtered data saved to 'filtered_positions.pkl'.")
	# Ensure there are positions to process
	if filtered_positions:
		filtered_positions = np.array(filtered_positions)
		new_grid = remake_grid(filtered_positions,dimensions)
		print(f"Original grid size: {len(positions)}")
		print(f"Filtered positions: {len(filtered_positions)}")
		print(f"New grid size: {len(new_grid)}")

		# Display the first 5 new grid positions
		for pos in new_grid:
			print(f"New Grid Position: {pos}")
	else:
		print("No positions met the threshold criteria.")

	FILE_LOG2=open("GridSearch_GPU.log","a")
	for pos in range(len(new_grid)):
		FILE_LOG2.write(str(new_grid[pos]))
	gpuid = args.gpuid
	gpuid_list = [str(x) for x in gpuid.split(":")]
	if(len(gpuid_list)<len(new_grid)):
		new_gpuid_list = []
		for J in range(len(new_grid)):
			new_gpuid_list.append(gpuid_list[J%len(gpuid_list)])
		gpuid_list = new_gpuid_list
	args_list = zip(new_grid, gpuid_list)
	func_partial = partial(func_gpuid, args=args)
	with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers_GPU) as executor2:
		results_search = list(executor2.map(func_partial, args_list))
		optimization_history.append([results_search, args_list])
		FILE_LOG2.write(str(optimization_history))
	FILE_LOG2.close()
