import numpy as np
import re
from scipy.optimize import minimize,Bounds
import os,sys,argparse,math
from functools import partial
from scipy.spatial.transform import Rotation as R
from func import func_gpuid
from dataclasses import dataclass
import concurrent.futures
import time
# chatgpt generated pattern search
# changelog ver2
# Add multithreading.
# changelog 20251015
# Fix a bug in multithreading, where the scores and inputs are mismatched.
# changelog ver3
# Add output. Add Continue run.

def create_simplex_parser():
	parser = argparse.ArgumentParser(description="Pattern Refine Execution")
	parser.add_argument("--max_workers", type=int, default=3)
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
	parser.add_argument("--MAXIUM_ALLOWED_overlapped_pixels", type=int, default=300)
	parser.add_argument("--MAX_MinDistance_Allowed", type=float, default=30.)
	parser.add_argument("--Pattern_Bounds", type=str, default="(-15,15),(-15,15),(-15,15),(-20,20),(-20,20),(-20,20)")
	parser.add_argument("--Pattern_dimensions", type=int, default=6)
	parser.add_argument("--Pattern_given_initial",action="store_true")
	parser.add_argument("--Pattern_initial",type=str, default="[10,10,10,10,10,10]")
	parser.add_argument("--Pattern_stepsize", type=float, default=1.0)
	parser.add_argument("--Pattern_tol", type=float, default=0.01)
	parser.add_argument("--Pattern_max_iter", type=int, default=500)
	parser.add_argument("--Pattern_shrink", type=float, default=0.5)
	parser.add_argument("--Pattern_expand", type=float, default=1.2)
	parser.add_argument("--Pattern_continue", action="store_true")
	parser.add_argument("--Pattern_continue_file", type=str, default=None)
	return parser

def convert_Bounds_to_bounds(args):
	string_bounds = args.Pattern_Bounds
	dimensions = args.Pattern_dimensions
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
_BOUNDS_PAIR_RE = re.compile(
	r"\(\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*,\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*\)"
)

def parse_bounds_str(bounds_str: str, dimensions: int):
	"""
	Parse a string like '(-15,15),(-15,15),...,(-20,20)' into (lb, ub) arrays.

	Rules:
	  - If exactly 1 pair is provided, it will be broadcast to all dimensions.
	  - If N pairs are provided, N must equal `dimensions`.
	  - Ensures lb[i] <= ub[i].
	"""
	pairs = [(float(a), float(b)) for a, b in _BOUNDS_PAIR_RE.findall(bounds_str)]
	if not pairs:
		raise ValueError(f"No '(low,high)' pairs found in: {bounds_str!r}")

	if len(pairs) == 1 and dimensions > 1:
		pairs = pairs * dimensions
	elif len(pairs) != dimensions:
		raise ValueError(
			f"Found {len(pairs)} bound pairs but dimensions={dimensions}. "
			"Provide one pair to broadcast or exactly `dimensions` pairs."
		)

	lb = np.empty(dimensions, dtype=float)
	ub = np.empty(dimensions, dtype=float)
	for i, (lo, hi) in enumerate(pairs):
		if lo > hi:
			lo, hi = hi, lo  # auto-fix swapped inputs
		if lo == hi:
			raise ValueError(f"Degenerate bound at dim {i}: ({lo},{hi})")
		lb[i], ub[i] = lo, hi

	return lb, ub

def convert_Bounds_to_arrays(args):
	lb, ub = parse_bounds_str(args.Pattern_Bounds, args.Pattern_dimensions)
	# If you still need SciPy Bounds elsewhere:
	BOUNDS = Bounds(lb, ub)
	return (lb, ub), BOUNDS
def scalar_step_from_bounds(lb, ub, frac=0.1):
	span = np.asarray(ub) - np.asarray(lb)
	return float(np.mean(span) * frac)
@dataclass
class PSResult:
	x: np.ndarray
	fun: float
	nit: int
	nfev: int
	step_size: float
	success: bool
	message: str
	history: list  # [(iter, f, step_size, x.copy())]


def _project_to_bounds(x, lb, ub):
	if lb is None and ub is None:
		return x
	if lb is not None:
		x = np.maximum(x, lb)
	if ub is not None:
		x = np.minimum(x, ub)
	return x


def _evaluate_one(func, args,candidate,gpuid=None):
	return func(candidate, args=args)
def evaluate_wrapper(inp):
	return _evaluate_one(*inp)
def save_pattern_state_round(x,fval,poll_dirs,lb,ub,nit,nfev,step_size,tol,max_evals,max_iter,history,expand,shrink):
	prefix = "my_pattern_state_round_"
	iteration = nit
	filename = f"{prefix}{iteration}.npz"
	np.savez(
		filename,
		x=x,
		hyperparams={
			'x': x,
			'fval': fval,
			'poll_dirs': poll_dirs,
			'lb': lb,
			'ub': ub,
			'nit': nit,
			'nfev': nfev,
			'step_size': step_size,
			'tol': tol,
			'max_evals': max_evals,
			'max_iter': max_iter,
			'history': history,
			'expand': expand,
			'shrink': shrink
		}
	)
	print(f"Pattern state saved for iteration {iteration}: {filename}")
def from_file(filename):
	data = np.load(filename, allow_pickle=True)
	hyperparams=data['hyperparams'].item()
	x          =hyperparams['x']
	fval       =hyperparams['fval']
	poll_dirs  =hyperparams['poll_dirs']
	lb         =hyperparams['lb']
	ub         =hyperparams['ub']
	nit        =hyperparams['nit']
	nfev       =hyperparams['nfev']
	step_size  =hyperparams['step_size']
	tol        =hyperparams['tol']
	max_evals  =hyperparams['max_evals']
	max_iter   =hyperparams['max_iter']
	history    =hyperparams['history']
	expand     =hyperparams['expand']
	shrink     =hyperparams['shrink']

	print(f"Pattern state loaded from: {filename}")
	print(f" -> Best score so far: {fval} , best position: {x}")
	return x,fval,poll_dirs,lb,ub,nit,nfev,step_size,tol,max_evals,max_iter, history,expand,shrink	
def pattern_search(
    func,
    x0,
    step_size=1.0,
    tol=1e-2,
    max_iter=500,
    max_evals=np.inf,
    shrink=0.5,
    expand=1.2,
    bounds=None,
    directions=None,
    opportunistic=True,
    args=None,
    gpuid_list=None,
    max_workers=3,
):
	x = np.copy(x0)
	n = len(x)
	if directions is None:
		directions = np.eye(n)
	poll_dirs = np.vstack((directions, -directions))

	lb, ub = (None, None)
	if bounds is not None:
		lb, ub = bounds
	
	xx=[x,gpuid_list[0]]
	print(f"debug0, {x},{xx},{gpuid_list}")
	AA=open("Pattern_optimization_history.log","a")
	AA.write(f"debug0, {x},{xx},{gpuid_list}")
	AA.close()
	fval= 99999999999
	nfev = 1
	nit = 0
	history = [(nit, fval, step_size, x.copy())]
	if not args.Pattern_continue:
		fval = func(xx,args=args)
		history = [(nit, fval, step_size, x.copy())]
	if(args.Pattern_continue):
		continue_file = args.Pattern_continue_file
		x,fval,poll_dirs,lb,ub,nit,nfev,step_size,tol,max_evals,max_iter,history,expand,shrink = from_file(continue_file)
	while step_size > tol and nfev < max_evals and nit < max_iter:
		
		candidates = x + step_size * poll_dirs
		candidates = _project_to_bounds(candidates, lb, ub)

		if gpuid_list is not None:
			task_inputs = [(func,args, [candidates[i],gpuid_list[i % len(gpuid_list)]]) for i in range(len(candidates))]
			print(f"candidates = {candidates}")
			AA=open("Pattern_optimization_history.log","a")
			AA.write(f"nit = {nit}, candidates = {candidates}")
			AA.close()
		else:
			print("No gpuid. EXIT")
			quit()

		vals = np.empty(len(candidates))
		
		with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
			vals = list(executor.map(evaluate_wrapper, task_inputs))
		nfev += 12

		best_idx = np.argmin(vals)
		if vals[best_idx] < fval:
			x = candidates[best_idx]
			fval = vals[best_idx]
			step_size *= expand
			print (f"Finish one iteraltion. Current best x = {x}, value = {fval}, stepsize = {step_size}")
			AA=open("Pattern_optimization_history.log","a")
			AA.write(f"Finish one iteraltion. Current best x = {x}, value = {fval}, stepsize = {step_size}")
			AA.close()
			message = "Accepted better point"
		else:
			step_size *= shrink
			message = "No improvement"
			print (f"Finish one iteraltion. Current best x = {x}, value = {fval}, stepsize = {step_size}")
			AA=open("Pattern_optimization_history.log","a")
			AA.write(f"Finish one iteraltion. Current best x = {x}, value = {fval}, stepsize = {step_size}")
			AA.close()
		nit += 1
		history.append((nit, fval, step_size, x.copy()))
		save_pattern_state_round(x,fval,poll_dirs,lb,ub,nit,nfev,step_size,tol,max_evals,max_iter,history,expand,shrink)
	success = step_size <= tol
	return PSResult(
		x=x,
		fun=fval,
		nit=nit,
		nfev=nfev,
		step_size=step_size,
		success=success,
		message=message,
		history=history,
	)

if __name__ == "__main__":
	# Instantiate PSO
	parser = create_simplex_parser()
	args = parser.parse_args()
	bounds,BOUNDS = convert_Bounds_to_bounds(args)
	dimensions = args.Pattern_dimensions
#	Initial_Simplex=args.Initial_Simplex
	Initial_Simplex = np.random.uniform(low=[b[0] for b in bounds],high=[b[1] for b in bounds],size=(dimensions+1, dimensions))
	print("Bounds:", BOUNDS)
	
	TWO_TIMES_RANGE = 40.
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
	if(args.Pattern_given_initial):
		x01=float(args.Pattern_initial.split(",")[0].split("[")[1])
		x02=float(args.Pattern_initial.split(",")[1])
		x03=float(args.Pattern_initial.split(",")[2])
		x04=float(args.Pattern_initial.split(",")[3])
		x05=float(args.Pattern_initial.split(",")[4])
		x06=float(args.Pattern_initial.split(",")[5].split("]")[0])
		x0 = np.array([x01, x02,x03, x04,x05,x06])
	x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
	# If not, the minimize sometimes reports "Initial guess is not within the specified bounds".
	# Note: x0 would not be used if Initial_Simplex had been set.
	# Note 2: BOUNDS is for the Bounds class. It's 6 by 2. While bounds is for array iteration. it's 2 by 6.

	(lb, ub), _ = convert_Bounds_to_arrays(args)

#	x0 = np.clip(np.asarray(x0, float), lb, ub)  # ensure feasible start
	step_size = scalar_step_from_bounds(lb, ub, frac=0.1)

	gpuid = args.gpuid
	gpuid_list = [str(x) for x in gpuid.split(":")]
	if(len(gpuid_list)<args.max_workers):
		new_gpuid_list = []
		for J in range(args.max_workers):
			new_gpuid_list.append(gpuid_list[J%len(gpuid_list)])
		gpuid_list = new_gpuid_list
	result = pattern_search(
		func=func_gpuid,
		x0=x0,
		step_size=args.Pattern_stepsize,
		tol=args.Pattern_tol, # noise -> looser tol
		max_iter=args.Pattern_max_iter,
		shrink=args.Pattern_shrink, # slightly gentler shrinking
		expand=args.Pattern_expand, # mild expansion
		bounds=(lb, ub),
		args=args,
		gpuid_list=gpuid_list,
		max_workers=args.max_workers,
	)
