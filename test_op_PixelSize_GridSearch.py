import numpy as np
from scipy.optimize import minimize
import os, sys
import itertools
import concurrent.futures
import pickle,argparse
from functools import partial
from func_for_PixelSize_search import func_for_PixelSize_search_gpuid
from func import prepare_workers, shutdown_workers, finalize_run
# Initial version
# This is made from gridsearch. It runs grid search on the pixel sizes.
# (20250505) Add doEnableGpuProj
# (20250527) Add doSplitDiffGpu and SplitParticles
Geo_optimization_history = []
optimization_history = []
def create_gridsearch_parser():
	parser = argparse.ArgumentParser(description="Grid Refine Execution")
	parser.add_argument("--PDB_NAME", type=str, required=True)
	parser.add_argument("--STAR_NAME", type=str, required=True)
	parser.add_argument("--rotate_chain", type=str, required=True, help="In pixel size searches, you can input any existing chain.")
	parser.add_argument("--output_name_root", type=str, default="output")
	parser.add_argument("--gpuid", type=str, default="0", help="The gpuid to be run on. It's like 0:0:0, default = 0")
	parser.add_argument("--ang", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "c1_3deg_remove_rotLzero_200kV.star"))
	parser.add_argument("--boxsize", type=int, default=256)
	parser.add_argument("--apix", type=float, default=1.58)
	parser.add_argument("--apix_PDB", type=float, default=1.58, help="The PDB pixel sizes to be searched, default = 1.58")
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
	parser.add_argument("--maskRadius", type=int, default=110)
	parser.add_argument("--Geometric_restrain_Scaling_Factor", type=float, default=1.0)
	parser.add_argument("--chain_MASS_in_residues", type=int, default=275)
	parser.add_argument("--MAXIUM_ALLOWED_overlapped_pixels", type=int, default=9999999999999999)
	parser.add_argument("--MAX_MinDistance_Allowed", type=float, default=99999999999.)
	parser.add_argument("--doEnableGpuProj", action='store_true', help="Enable GPU projection. Will consume large amount of device memory. default = False")
	parser.add_argument("--doSplitDiffGpu", action='store_true', help="If enabled, wrap_to_search will use different gpuid. The inital gpuid is provided by --gpuid. default = False")
	parser.add_argument("--SplitParticles", type=int, default=1, help="Split the starfile into these sections. Default = 1")
	# Pixel-size scoring historically did not apply geometric penalties.
	parser.set_defaults(skip_geometric_restraint=True)
	parser.add_argument("--enable_geometric_restraint", dest="skip_geometric_restraint", action="store_false", help="Opt in to geometric restraint during pixel-size search.")
	# PSO parameters
	parser.add_argument("--Pixelsize_Bounds", type=str, default="(-0.1,0.1)", help="The search range of the pixel sizes, default = (-0.1,0.1). The actual range adds to apix_PDB")
	parser.add_argument("--Pixelsize_stepsize", type=float, default=0.01, help="The search stepsize of the pixel sizes, default = 0.01")
	parser.add_argument("--Grid_dimensions", type=int, default=1)
	parser.add_argument("--max_workers_CPU", type=int, default=8)
	parser.add_argument("--max_workers_GPU", type=int, default=3, help="The number of GPUs to be utilized, default = 3")
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
def generate_grid(bound, stepsize,dimensions):
	""" Generate all grid points based on the given boundary and step sizes. """
	grid_axes = [np.arange(bound[i, 0], bound[i, 1] + stepsize[i], stepsize[i]) for i in range(dimensions)]
	return np.array(list(itertools.product(*grid_axes)))  # Generate all points
def remake_grid(positions,dimensions):
	"""Recreate the grid based on filtered positions."""
	# Get unique values for each dimension
	unique_axes = [np.unique(positions[:, i]) for i in range(dimensions)]
	# Generate the new grid
	return np.array(list(itertools.product(*unique_axes)))
def convert_Bounds_to_bounds_for_grid(args):
	string_bounds = args.Pixelsize_Bounds
	dimensions = args.Grid_dimensions
	bounds=np.zeros((dimensions,2))
	apix_PDB=args.apix_PDB
	for i in range(dimensions):
		part_N=string_bounds.split(')')[i]
		remove_left = part_N.split('(')[1]
		LOW=float(remove_left.split(',')[0])
		HIGH=float(remove_left.split(',')[1])
		bounds[i,0]=LOW+apix_PDB
		bounds[i,1]=HIGH+apix_PDB
	return bounds

def main():
	args = create_gridsearch_parser().parse_args()
	bounds = convert_Bounds_to_bounds_for_grid(args)
	positions = generate_grid(bounds, np.array([args.Pixelsize_stepsize]), args.Grid_dimensions)
	gpuids = args.gpuid.split(":")
	tasks = [(pose, gpuids[i % len(gpuids)]) for i, pose in enumerate(positions)]
	try:
		prepare_workers(args)
		with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers_GPU) as executor:
			results = list(executor.map(partial(func_for_PixelSize_search_gpuid, args=args), tasks))
		with open("PixelSize_Search.log", "a") as handle:
			handle.write(str([[results, positions]]) + "\n")
		finalize_run(args)
	finally:
		shutdown_workers(args)


if __name__ == "__main__":
	main()
