import numpy as np
from scipy.optimize import minimize
import os, sys
import itertools
import concurrent.futures
import pickle,argparse
from functools import partial
from func import func_gpuid, prepare_workers, shutdown_workers, finalize_run
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
	parser.add_argument("--ang", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "c1_3deg_remove_rotLzero_200kV.star"))
	parser.add_argument("--boxsize", type=int, default=256)
	parser.add_argument("--apix", type=float, default=1.58)
	parser.add_argument("--apix_PDB", type=float, default=1.58)
	parser.add_argument("--newboxsize", type=int, default=160)
	parser.add_argument("--search_script", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "test1_with_isspa_weight_varingKK_search_translation_also_v606_torch_optimized_standalone.py"))
	parser.add_argument("--fsc_file", type=str, default=None, help="Optional FSC curve; omit to use uniform Fourier weights.")
	parser.add_argument("--skip_geometric_restraint", action="store_true", help="Skip geometric restraint calculation and its score penalty.")
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

def main():
	parser = create_gridsearch_parser()
	args = parser.parse_args()
	bounds = convert_Bounds_to_bounds_for_grid(args)
	steps = np.array([args.Grid_Rotation_Stepsize] * 3 + [args.Grid_Translation_Stepsize] * 3)
	positions = generate_grid(bounds, steps, args.Grid_dimensions)
	gpuids = args.gpuid.split(":")
	tasks = [(pose, gpuids[i % len(gpuids)]) for i, pose in enumerate(positions)]
	# The permanent worker performs the geometric gate once per candidate,
	# before density generation and image scoring, including grid searches.
	try:
		prepare_workers(args)
		with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers_GPU) as executor:
			results = list(executor.map(partial(func_gpuid, args=args), tasks))
		with open("GridSearch_GPU.log", "a") as handle:
			handle.write(str([[results, positions]]) + "\n")
		with open("positions_and_results.pkl", "wb") as handle:
			pickle.dump(list(zip(positions, results)), handle)
		finalize_run(args)
	finally:
		shutdown_workers(args)


if __name__ == "__main__":
	main()
