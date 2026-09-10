import numpy as np
from scipy.optimize import minimize,Bounds
import os,sys,argparse,math
from functools import partial
from scipy.spatial.transform import Rotation as R
from func import func_gpuid, prepare_workers, shutdown_workers, finalize_run
from optimizer_checkpoint import checkpoint_nelder_mead
# changelog ver4
# search for rot-tilt-psi and X-Y-Z. Add bounds to the minimize.
# changelog ver5
# change the convert_number_to_filename function.
# changelog ver6
# modification for running under GUI.
# changelog ver7
# (SKIP this)add nelder_mead_callback(simplex), force convergence if the distance in rotation translation is too small.
# Remove duplicated computing to func
# changelog ver71
# Add handedness to read_pdb_index_generate_sh_3DEG_local_v32.py. It should also be used together with the modified relion_project.
# changelog ver72
# The name of the search script will be fixed. currently the local value is changing, leading to some error.
# The func is now moving to a seperate file.
# Add apix_PDB (20250415)
# Now supports multi-threading.
def create_simplex_parser():
	parser = argparse.ArgumentParser(description="Simplex Refine Execution")
	parser.add_argument("--max_workers", type=int, default=3)
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
	parser.add_argument("--MAXIUM_ALLOWED_overlapped_pixels", type=int, default=300)
	parser.add_argument("--MAX_MinDistance_Allowed", type=float, default=30.)
	# PSO parameters
	parser.add_argument("--Simplex_Bounds", type=str, default="(-15,15),(-15,15),(-15,15),(-20,20),(-20,20),(-20,20)")
	parser.add_argument("--Simplex_dimensions", type=int, default=6)
	parser.add_argument("--Simplex_xatol", type=float, default=0.2)
	parser.add_argument("--Simplex_fatol", type=float, default=1.0)
	parser.add_argument("--Simplex_maxiter", type=int, default=100)
#	parser.add_argument("--Initial_Simplex", type=str, default = "")
	parser.add_argument("--Simplex_continue", "--continue_run", action="store_true")
	parser.add_argument("--Simplex_continue_file", "--continue_file", type=str, default=None)
	parser.add_argument("--Simplex_continue_more_rounds", "--continue_more_rounds", type=int, default=20)
	return parser

def convert_Bounds_to_bounds(args):
	string_bounds = args.Simplex_Bounds
	dimensions = args.Simplex_dimensions
	bounds=np.zeros((dimensions,2))
	lower_bounds = [-5] * dimensions
	upper_bounds = [5] * dimensions
	for i in range(dimensions):
		part_N=string_bounds.split(')')[i]
		remove_left = part_N.split('(')[1]
		LOW=float(remove_left.split(',')[0])
		HIGH=float(remove_left.split(',')[1])
		bounds[i,0]=LOW
		bounds[i,1]=HIGH
		lower_bounds[i]=LOW
		upper_bounds[i]=HIGH
	BOUNDS=Bounds(lower_bounds, upper_bounds)
	return bounds,BOUNDS
def run_downhill_simplex(x0, objective, BOUNDS, options, args=None):
	resume_file = None
	if args is not None and args.Simplex_continue:
		if not args.Simplex_continue_file:
			raise ValueError("--Simplex_continue requires --Simplex_continue_file")
		resume_file = args.Simplex_continue_file
	result = checkpoint_nelder_mead(objective, x0, BOUNDS, options,
		resume_file=resume_file,
		additional_iterations=getattr(args, "Simplex_continue_more_rounds", 20))
	print("Optimal point:", result.x)
	print("Function value at the optimal point:", result.fun)
	return result


def main():
	# Instantiate PSO
	parser = create_simplex_parser()
	args = parser.parse_args()
	bounds,BOUNDS = convert_Bounds_to_bounds(args)
	dimensions = args.Simplex_dimensions
#	Initial_Simplex=args.Initial_Simplex
	Initial_Simplex = np.random.uniform(low=[b[0] for b in bounds],high=[b[1] for b in bounds],size=(dimensions+1, dimensions))
	print("Bounds:", BOUNDS)
	print("Initial_Simplex: ",Initial_Simplex)
	options = {
		'xatol': args.Simplex_xatol,  # Minimum step-size in terms of change in parameters
		'fatol': args.Simplex_fatol,  # Minimum step-size in terms of change in function value
		'maxiter': args.Simplex_maxiter,  # Maximum number of iterations (optional)
		'initial_simplex': Initial_Simplex,	# initial_simplex should be N+1 dimension. For testing, D(x)=2, so we start with 3 points.
		'disp': True, # Set to True to print convergence messages.
		'return_all': True, # Set to True to return a list of the best solution at each of the iterations.
	}

	# set an inititive. x0 will not be used since we have set Initial_Simplex.
	TWO_TIMES_RANGE = 30.
	x01=np.round(TWO_TIMES_RANGE*(np.random.random()-0.5))
	x02=np.round(TWO_TIMES_RANGE*(np.random.random()-0.5))
	x03=np.round(TWO_TIMES_RANGE*(np.random.random()-0.5))
	x04=np.round(TWO_TIMES_RANGE*(np.random.random()-0.5))
	x05=np.round(TWO_TIMES_RANGE*(np.random.random()-0.5))
	x06=np.round(TWO_TIMES_RANGE*(np.random.random()-0.5))
	if(dimensions==6):
		x0 = np.array([x01, x02,x03, x04,x05,x06])
	else:
		x0 = np.array([x01, x02,x03])
	x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
	# If not, the minimize sometimes reports "Initial guess is not within the specified bounds".
	# Note: x0 would not be used if Initial_Simplex had been set.
	# Note 2: BOUNDS is for the Bounds class. It's 6 by 2. While bounds is for array iteration. it's 2 by 6.
	
	# Nelder--Mead evaluates sequentially; keep one permanent GPU worker
	# available per requested device and dispatch successive evaluations round robin.
	gpuids = args.gpuid.split(":")
	evaluation_index = 0
	def objective(pose):
		nonlocal evaluation_index
		gpu = gpuids[evaluation_index % len(gpuids)]
		evaluation_index += 1
		return func_gpuid((pose, gpu), args=args)
	try:
		prepare_workers(args)
		run_downhill_simplex(x0, objective, BOUNDS, options, args=args)
		finalize_run(args)
	finally:
		shutdown_workers(args)

####
# Note: Is it possible to interpolate the values inside our search points?
# So that we only need to compute the values outside. Therefore reduce computational cost.
# Maybe only for large domains.


if __name__ == "__main__":
	main()
