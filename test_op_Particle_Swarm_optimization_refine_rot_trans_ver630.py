import numpy as np
from scipy.optimize import minimize
import os, sys
import argparse
import concurrent.futures
from functools import partial
from func import func_gpuid
# changelog ver32
# add geometric restrain as bias to final values.
# changelog ver33
# remove the search boundary.
# changelog ver40
# implement adaptive inertia and coefficient.
# changelog ver41
# remove the hard boundary, use additional forces to help convergence
# changelog ver423
# re-add hard bound, set initial velocity to have the directions to the center, remove other restrains
# return score = Geometric_restrain_Overlapped_Pixels + MAX_MinDistance_Allowed if over the GR.
# changelog ver424
# use func_check_boundary_for_testing_v41.py and read_pdb_index_generate_sh_3DEG_local_v31.py. I also recompiled relion_image_handler and relion_project, so no progress bar anymore.
# changelog ver425
# when hit boundary, the particle's velocity would re-initialize and point to center
# changelog ver5
# major change. We search for rot-tilt-psi and X-Y-Z shifts. The psi must be searched by our coordinate system.
# changelog ver51
# Add additional velocities to center (pivot point) if the particles hit the GR bound.
# changelog ver52
# try except for continue run. Also change the convert_number_to_filename function
# changelog ver6
# modification for running under GUI.

# changelog ver61
# Add handedness to read_pdb_index_generate_sh_3DEG_local_v32.py. It should also be used together with the modified relion_project.
# changelog ver62
# The name of the search script will be fixed. currently the local value is changing, leading to some error.
# The func is now moving to a seperate file.
# Fix the continue_run_filename initialization error.
# Add apix_PDB (20250415)

# changelog ver621
# Fix gpu only runs on the first device when --gpuid contains more than one device_id.
# changelog ver622
# Can run continously by --PSO_continue . Filename --PSO_continue_file --PSO_continue_more_rounds should be entered.
# (20250527) Add doSplitDiffGpu and SplitParticles
# changelog ver6221
# Enable reseting velocities to random numbers after continue run
# Enable add random Gaussian noise to velocities, so hopefully it won't stuck to local minima
def create_PSO_parser():
	parser = argparse.ArgumentParser(description="PSO Parallel Execution")
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
	# PSO parameters
	parser.add_argument("--PSO_Bounds", type=str, default="(-15,15),(-15,15),(-15,15),(-20,20),(-20,20),(-20,20)")
	parser.add_argument("--PSO_num_particles", type=int, default=3)
	parser.add_argument("--PSO_dimensions", type=int, default=6)
	parser.add_argument("--PSO_iterations", type=int, default=30)
	parser.add_argument("--PSO_wmax", type=float, default=0.9)
	parser.add_argument("--PSO_wmin", type=float, default=0.4)
	parser.add_argument("--PSO_add_noise_velocities", action="store_true")
	parser.add_argument("--PSO_noise_strength", type=float, default=1.0)
	parser.add_argument("--PSO_noise_decay_per_round", type=float, default=0.99)
	parser.add_argument("--PSO_continue", action="store_true")
	parser.add_argument("--PSO_continue_file", type=str, default=None)
	parser.add_argument("--PSO_continue_more_rounds", type=int, default=20)
	parser.add_argument("--PSO_continue_reset_velocities", action="store_true")
	return parser

class PSO:
	def __init__(self, objective_func, args, bounds, callback=None):
		self.obj_func = objective_func
		self.num_particles = args.PSO_num_particles
		self.dimensions = args.PSO_dimensions
		self.bounds = bounds
		self.max_workers=args.max_workers
		self.MAXIUM_ALLOWED_overlapped_pixels=args.MAXIUM_ALLOWED_overlapped_pixels
		
		#############
		self.do_add_noise_velocities = args.PSO_add_noise_velocities
		self.noise_strength = args.PSO_noise_strength
		self.noise_decay_per_round = args.PSO_noise_decay_per_round
		self.sigma = self.noise_strength
		#############
		#
		self.gpuid = args.gpuid
		self.gpuid_list = [str(x) for x in self.gpuid.split(":")]
		if(len(self.gpuid_list)<self.num_particles):
			new_gpuid_list = []
			for J in range(self.num_particles):
				new_gpuid_list.append(self.gpuid_list[J%len(self.gpuid_list)])
			self.gpuid_list = new_gpuid_list
		#
		#### These two below are no longer used.
		self.min_distance_rotation = 0.2
		self.min_distance_translation = 1.0
		self.args=args
		####
		self.callback = callback
		
		# Store the initial planned number of iterations
		self.max_iterations = args.PSO_iterations
		
		# Keep track of how many iterations we've already done.
		self.current_iteration = 0
		
		# Initialize particles
		self.positions = self.initialize_particles()
		directions = -self.positions
		self.V_LOW_SCALE=0.1
		self.V_HIGH_SCALE=0.2
		random_magnitudes = np.random.uniform(-1.0*self.V_HIGH_SCALE,self.V_HIGH_SCALE,size=(self.num_particles,1))
		self.velocities = directions*random_magnitudes
		print("self.positions,self.velocities: ",self.positions,self.velocities)
		self.pbest_positions = np.copy(self.positions)
		self.pbest_scores = np.full(self.num_particles, np.inf)
		self.gbest_position = None
		self.gbest_score = np.inf

		# PSO hyperparameters
		self.w_max = args.PSO_wmax  # Maximum inertia weight
		self.w_min = args.PSO_wmin # Minimum inertia weight
		self.c1 = 2.0       # Cognitive coefficient
		self.c2 = 2.0       # Social coefficient
	def initialize_particles(self):
	#	init_bounds = [(-10., 10.), (-10., 10.),(-10., 10.),(-20., 20.), (-20., 20.), (-20., 20.)]
		init_bounds= self.bounds
		positions = np.random.uniform(
			low=[b[0] for b in init_bounds],
			high=[b[1] for b in init_bounds],
			size=(self.num_particles, self.dimensions)
		)
		return positions
	def run_iterations(self, n):
		optimization_history = []
		# Run the PSO loop for `n` more iterations (no re-initialization!).
		for iteration in range(n):
			# actual_iter = self.current_iteration + iteration  # total iteration count so far
			actual_iter = self.current_iteration
			# Example linear decay for inertia weight across the total planned range
			# (You can choose to freeze w if you prefer.)
			w = self.compute_inertia_weight(actual_iter, self.max_iterations)
			
			print(f"Iteration {actual_iter + 1}/{self.max_iterations}, w={w}")
			
			args_list = zip(self.positions, self.gpuid_list)
			with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:	
				func_partial = partial(self.obj_func, args=self.args)
				scores = list(executor.map(func_partial, args_list))
			scores = np.array(scores)
			print("PSO debug, scores, positions = ", scores, self.positions)
			optimization_history.append([scores, self.positions])
			
			# Update pbest
			better_mask = scores < self.pbest_scores
			self.pbest_scores[better_mask] = scores[better_mask]
			self.pbest_positions[better_mask] = self.positions[better_mask]

			# Update gbest
			min_score_idx = np.argmin(scores)
			if scores[min_score_idx] < self.gbest_score:
				self.gbest_score = scores[min_score_idx]
				self.gbest_position = self.positions[min_score_idx]

			# Update velocities
			r1 = np.random.uniform(size=(self.num_particles, self.dimensions))
			r2 = np.random.uniform(size=(self.num_particles, self.dimensions))
			c1_rest_tomultipy = r1 * (self.pbest_positions - self.positions)
			c2_rest_tomultipy = r2 * (self.gbest_position - self.positions)
			cognitive,social = self.generate_cognitive(c1_rest_tomultipy,c2_rest_tomultipy)
			
			self.velocities = w * self.velocities + cognitive + social
			# Apply boundary force instead of hard clipping

			positions_try = self.positions+self.velocities
			
			for d in range(self.dimensions):
				# Identify particles that hit the boundary
				hit_lower = positions_try[:, d] <= self.bounds[d][0]
				hit_upper = positions_try[:, d] >= self.bounds[d][1]

				# Clip positions to stay within bounds
				positions_try[:, d] = np.clip(positions_try[:, d], self.bounds[d][0], self.bounds[d][1])

				# Reset velocity if the boundary is hit
				# Option 1: Reverse velocity (bounce effect)
			#	self.velocities[hit_lower | hit_upper, d] *= -1
				# Tested and didn't work.
				# Option 2: Reinitialize velocity (randomized reset)
				random_magnitudes = np.random.uniform(self.V_LOW_SCALE, self.V_HIGH_SCALE, size=(self.num_particles,))
				self.velocities[hit_lower | hit_upper, d] = -self.positions[hit_lower | hit_upper, d] * random_magnitudes[hit_lower | hit_upper]
			for ii in range(self.num_particles):
				if scores[ii] >= self.MAXIUM_ALLOWED_overlapped_pixels:
					central_velocity_magnitudes = np.random.uniform(low=self.V_LOW_SCALE, high=self.V_HIGH_SCALE, size=(self.dimensions,))
					self.velocities[ii] += central_velocity_magnitudes * -self.positions[ii]
					
			if(self.do_add_noise_velocities):
				noise = np.random.normal(0.0, self.sigma, size=self.velocities.shape)
				self.velocities + = noise
			# Yes, you can replace the -self.positions into a pivot point
			self.positions += self.velocities
			# (Optional) callback
			if self.callback:
				self.callback(self.positions, scores,self.velocities,self.gbest_position,self.gbest_score)
			save_pso_state_round(self, self.current_iteration, prefix="my_pso_state_round_")
			self.current_iteration += 1
			self.sigma *= self.noise_decay_per_round
			# After finishing these n iterations, update the "current_iteration"
		
		print(f"Finished {n} more iterations (total so far: {self.current_iteration}).")
		AA=open("PSO_optimization_history.log","a")
		AA.write(str(optimization_history))
		AA.close()
	def generate_cognitive(self, c1_rest_tomultipy, c2_rest_tomultipy):
	#	c1_orient = 1.5  # Lower cognitive influence
	#	c2_orient = 2.5  # Higher social influence (faster convergence)
	#	c1_trans = 2.5  # Higher cognitive influence (more independent exploration)
	#	c2_trans = 1.5  # Lower social influence (reduces premature convergence)
		c1_orient = 2.
		c2_orient = 2.
		c1_trans = 2.
		c2_trans = 2.
		if(self.dimensions < 4):
			# refine only the orientations.
			cognitive = c1_orient*c1_rest_tomultipy
			social = c2_orient*c2_rest_tomultipy
		else:
			cognitive = c1_rest_tomultipy
			social = c2_rest_tomultipy
			for i in range(self.num_particles):
				for j in range(0,3):
					cognitive[i][j]*=c1_orient
					social[i][j]*=c2_orient
				for j in range(3,6):
					cognitive[i][j]*=c1_trans
					social[i][j]*=c2_trans
		return cognitive,social
	def compute_inertia_weight(self, iteration, max_iterations):
		"""
		Computes an adaptive inertia weight for each parameter.
		- P1, P2 (orientation) -> faster decay
		- P3, P4, P5 (translation) -> slower decay
		"""
		if max_iterations <= 1:
			return self.w_min
		if iteration >= max_iterations:
			return self.w_min
		decay_factor = 1.0
		if(self.dimensions<4):
			# refine only the orientations.
			decay_factor = 1.0
			fraction = (iteration / (max_iterations - 1)) ** decay_factor
			w = self.w_max - (self.w_max - self.w_min) * fraction
		else:
			w = np.zeros((self.num_particles, self.dimensions))
			for i in range(self.num_particles):
				for j in range(0,3):
					decay_factor = 1.0
					fraction = (iteration / (max_iterations - 1)) ** decay_factor
					w[i][j] = self.w_max - (self.w_max - self.w_min) * fraction
				for j in range(3,6):
					decay_factor = 1.0
					fraction = (iteration / (max_iterations - 1)) ** decay_factor
					w[i][j] = self.w_max - (self.w_max - self.w_min) * fraction
		# If you want to clamp at w_min beyond the last iteration, do:
		# w = max(w, self.w_min)
		return w
	def optimize(self):
		"""
		Convenience method: runs the initially planned number of iterations 
		(i.e., self.max_iterations) from the current state. 
		If current_iteration is 0, that means from scratch; 
		otherwise it's a continuation as well.
		"""
		print(f"Starting optimization from iteration {self.current_iteration} "
			  f"to iteration {self.current_iteration + self.max_iterations}")
		
		self.run_iterations(self.max_iterations - self.current_iteration)
		
		print("\nOptimization complete.")
		print(f"Best position found: {self.gbest_position}")
		print(f"Best score: {self.gbest_score}")
	def continue_optimization(self, additional_iterations):
		"""
		Continues optimization for `additional_iterations` more steps, 
		preserving the current swarm state.
		"""
		# Optionally, update self.max_iterations
		self.max_iterations += additional_iterations  # If you want w-decay to keep scaling
		self.run_iterations(additional_iterations)
		
		print("\nContinuing optimization complete.")
		print(f"New best position found: {self.gbest_position}")
		print(f"New best score: {self.gbest_score}")
	def from_file(self, filename, objective_func, callback=None,reset_velocities = False):
		"""
		Class method: create a new PSO instance from a saved state file (NPZ).
		You must supply 'objective_func'. 'callback' is optional.
		"""
		data = np.load(filename, allow_pickle=True)
		
		# Extract hyperparameters
		hyperparams = data['hyperparams'].item()
		
		# Construct a new PSO instance with matching hyperparams
		# Note, the args and bounds have to be the identical to original entry.
		pso = PSO(
			objective_func=func_gpuid,
			args=args,
			bounds=bounds,
			callback=my_callback
		)
		
		# Now overwrite its internal arrays with the saved state
		pso.positions        = data['positions']
		pso.velocities       = data['velocities']
		pso.pbest_positions  = data['pbest_positions']
		pso.pbest_scores     = data['pbest_scores']
		pso.gbest_position   = data['gbest_position']
		pso.gbest_score      = data['gbest_score']
		if(reset_velocities):
			pso.velocities = np.random.uniform(-1.0,1.0,size=(self.num_particles,6))
			pso.positions += pso.velocities
			print(f"Reset velocities. pso.velocities = {pso.velocities }")
			print(f"New positions = {pso.positions }")
		# current_iteration needs to be cast back from the array
		pso.current_iteration = int(data['current_iteration'][0])
		
		# Done! Now 'pso' should have the exact same state as before.
		print(f"PSO state loaded from: {filename}")
		print(f"  -> Current iteration was: {pso.current_iteration}")
		print(f"  -> Best score so far: {pso.gbest_score}")
		return pso
def save_pso_state_round(pso, iteration, prefix="my_pso_state_round_"):
	"""
	Saves the PSO state to NPZ file. The filename will be:
		my_pso_state_round_{iteration}.npz
	"""
	filename = f"{prefix}{iteration}.npz"

	np.savez(
		filename,
		positions=pso.positions,
		velocities=pso.velocities,
		pbest_positions=pso.pbest_positions,
		pbest_scores=pso.pbest_scores,
		gbest_position=pso.gbest_position,
		gbest_score=pso.gbest_score,
		current_iteration=np.array([pso.current_iteration]),
		
		# Save hyperparams in a dictionary if you want to reconstruct
		hyperparams={
			'num_particles': pso.num_particles,
			'dimensions': pso.dimensions,
			'bounds': pso.bounds,
			'min_distance_rotation': pso.min_distance_rotation,
			'min_distance_translation': pso.min_distance_translation,
			'w_max': pso.w_max,
			'w_min': pso.w_min,
			'c1': pso.c1,
			'c2': pso.c2,
			'max_iterations': pso.max_iterations
		}
	)
	print(f"PSO state saved for iteration {iteration}: {filename}")

def my_callback(positions, scores,velocities,best_positions,best_scores):
	print("Callback invoked.")
	print("Current positions:\n", positions)
	print("Current velocities:\n", velocities)
	print("Current scores:\n", scores)
	print("Global best positions:\n", best_positions)
	print("Global best scores:\n", best_scores)

def convert_PSO_Bounds_to_bounds(args):
	string_bounds = args.PSO_Bounds
	dimensions = args.PSO_dimensions
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
	# Instantiate PSO
	parser = create_PSO_parser()
	args = parser.parse_args()
	bounds = convert_PSO_Bounds_to_bounds(args)
	# First 2 are rot/tilt in degree, the last 3 are transX/Y/Z in Angstrom
	pso = PSO(
		objective_func=func_gpuid,
		args=args,
		bounds=bounds,
		callback=my_callback
	)
	run_continue=args.PSO_continue
	continue_rounds=args.PSO_continue_more_rounds
	continue_run_filename=args.PSO_continue_file
	reset_velocities = args.PSO_continue_reset_velocities
	# Run optimization
	if(not run_continue):
		pso.optimize()
		print("Finish PSO.")
	else:
		try:
			pso_restored = pso.from_file(filename=continue_run_filename, objective_func=func_gpuid,reset_velocities=reset_velocities)

			# The new 'pso_restored' now has the same positions, velocities, etc.
			print("Restored iteration:", pso_restored.current_iteration)
			print("Restored best score:", pso_restored.gbest_score)

			# Continue optimization for continue_rounds more iterations, for example
			pso_restored.run_iterations(continue_rounds)

		except FileNotFoundError:
			print(f"Your continue file '{continue_run_filename}' does not exist. EXIT.")
