import numpy as np
from scipy.optimize import minimize
import os, sys
import argparse
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.compose import TransformedTargetRegressor
import concurrent.futures
from functools import partial
from func import func_gpuid, prepare_workers, shutdown_workers, finalize_run
import joblib
import glob
import json
import ast
from optimizer_checkpoint import atomic_npz
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

# changelog ver640
# test, add experimental mlp model for predicting the positions of swarms.
def create_PSO_parser():
	parser = argparse.ArgumentParser(description="PSO Parallel Execution")
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
	parser.add_argument("--PSO_Bounds", type=str, default="(-15,15),(-15,15),(-15,15),(-20,20),(-20,20),(-20,20)")
	parser.add_argument("--PSO_num_particles", type=int, default=3)
	parser.add_argument("--PSO_dimensions", type=int, default=6)
	parser.add_argument("--PSO_iterations", type=int, default=30)
	parser.add_argument("--PSO_wmax", type=float, default=0.9)
	parser.add_argument("--PSO_wmin", type=float, default=0.4)
	parser.add_argument("--PSO_add_noise_velocities", action="store_true")
	parser.add_argument("--PSO_noise_strength", type=float, default=1.0)
	parser.add_argument("--PSO_noise_decay_per_round", type=float, default=0.99)
	parser.add_argument("--PSO_continue", "--continue_run", action="store_true")
	parser.add_argument("--PSO_continue_file", "--continue_file", type=str, default=None)
	parser.add_argument("--PSO_continue_more_rounds", "--continue_more_rounds", type=int, default=20)
	parser.add_argument("--PSO_continue_reset_velocities", action="store_true")
	parser.add_argument("--PSO_pivot_point", type=str, default=None, help="Attraction point in pose parameter space: rot,tilt,psi (degrees),tx,ty,tz (angstrom). Default all zeros; this is not a physical rotation center.")
	# mlp model parameters
	parser.add_argument("--PSO_use_surrogate", action="store_true")
	parser.add_argument("--PSO_surrogate_start_round", type=int, default=5, help="Start training surrogate after this many completed PSO rounds.")
	parser.add_argument("--PSO_surrogate_update_every", type=int, default=5, help="Retrain surrogate every N rounds.")
	parser.add_argument("--PSO_surrogate_min_points", type=int, default=20, help="Minimum number of real evaluated points before training surrogate.")
	parser.add_argument("--PSO_surrogate_weight", type=float, default=0.5, help="Strength of surrogate guidance term.")
	parser.add_argument("--PSO_surrogate_hidden", type=str, default="128,128,64", help="Hidden layer sizes for surrogate MLP, e.g. 128,128,64")
	parser.add_argument("--PSO_surrogate_random_state", type=int, default=0)
	parser.add_argument("--PSO_SKIP_surrogate_THRESHOLD", type=float, default=1.0)
	return parser

class PSO:
	def __init__(self, objective_func, args, bounds, callback=None):
		self.obj_func = objective_func
		self.num_particles = args.PSO_num_particles
		self.dimensions = args.PSO_dimensions
		self.bounds = bounds
		pivot_text = getattr(args, "PSO_pivot_point", None)
		self.pivot_point = np.zeros(self.dimensions, dtype=float) if not pivot_text else np.asarray(ast.literal_eval(pivot_text), dtype=float)
		if self.pivot_point.shape != (self.dimensions,) or not np.all(np.isfinite(self.pivot_point)):
			raise ValueError("--PSO_pivot_point needs one finite value per PSO dimension")
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
		####
		self.SKIP_surrogate_THRESHOLD = args.PSO_SKIP_surrogate_THRESHOLD
		# -------- surrogate settings --------
		self.use_surrogate = args.PSO_use_surrogate
		self.surrogate_start_round = args.PSO_surrogate_start_round
		self.surrogate_update_every = args.PSO_surrogate_update_every
		self.surrogate_min_points = args.PSO_surrogate_min_points
		self.surrogate_weight = args.PSO_surrogate_weight
		self.surrogate_random_state = args.PSO_surrogate_random_state

		self.surrogate_model = None
		self.surrogate_is_ready = False
		self.history_X = []
		self.history_y = []

		self.surrogate_hidden = tuple(int(x.strip()) for x in args.PSO_surrogate_hidden.split(",") if x.strip())
		# Store the initial planned number of iterations
		self.max_iterations = args.PSO_iterations
		
		# Keep track of how many iterations we've already done.
		self.current_iteration = 0
		
		# Initialize particles
		self.positions = self.initialize_particles()
		directions = self.pivot_point - self.positions
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
	
	def decode_relion_style_number_from_result_name(self, s):
		"""
		Convert strings like:
		  N3p6 -> -3.6
		  0p5  -> 0.5
		  1p6  -> 1.6
		"""
		neg = s.startswith("N")
		if neg:
			s = s[1:]
		s = s.replace("p", ".")
		val = float(s)
		return -val if neg else val

	def parse_result_filename_to_x(self, filename):
		"""
		Parse filename like:
		ReSuLt_..._rot0p1_tilt6p3_psiN16p7deg_trans7p1_N5p7_N5p1ANG.pdb.txt

		Return:
			np.array([rot, tilt, psi, tx, ty, tz], dtype=np.float64)
		"""
		import re

		base = os.path.basename(filename)
		pattern = re.compile(
			r"rot(?P<rot>N?\d+p\d+)_"
			r"tilt(?P<tilt>N?\d+p\d+)_"
			r"psi(?P<psi>N?\d+p\d+)deg_"
			r"trans(?P<tx>N?\d+p\d+)_(?P<ty>N?\d+p\d+)_(?P<tz>N?\d+p\d+)ANG"
		)

		m = pattern.search(base)
		if m is None:
			raise ValueError(f"Cannot parse x from result filename: {filename}")

		rot = self.decode_relion_style_number_from_result_name(m.group("rot"))
		tilt = self.decode_relion_style_number_from_result_name(m.group("tilt"))
		psi = self.decode_relion_style_number_from_result_name(m.group("psi"))
		tx = self.decode_relion_style_number_from_result_name(m.group("tx"))
		ty = self.decode_relion_style_number_from_result_name(m.group("ty"))
		tz = self.decode_relion_style_number_from_result_name(m.group("tz"))

		return np.array([rot, tilt, psi, tx, ty, tz], dtype=np.float64)

	def parse_result_file_score(self, filename):
		"""
		ReSuLt file content:
			line 1: search filename (unused)
			line 2: first column is real score

		Return:
			float score
		"""
		with open(filename, "r", encoding="utf-8") as f:
			lines = [line.strip() for line in f if line.strip()]

		if len(lines) < 2:
			raise ValueError(f"Result file has too few lines: {filename}")

		parts = lines[1].split()
		if len(parts) < 1:
			raise ValueError(f"Second line has no score: {filename}")

		return float(parts[0])

	def load_history_from_result_files(self, result_glob="ReSuLt_*.pdb.txt"):
		"""
		Load historical evaluated points from result files in current directory.

		Fills:
			self.history_X
			self.history_y

		Returns:
			n_loaded
		"""
		files = sorted(glob.glob(result_glob))
		if len(files) == 0:
			print(f"No result files found matching: {result_glob}")
			return 0

		X_list = []
		y_list = []
		n_bad = 0

		for fn in files:
			try:
				x = self.parse_result_filename_to_x(fn)
				y = self.parse_result_file_score(fn)

				if np.all(np.isfinite(x)) and np.isfinite(y):
					X_list.append(x)
					y_list.append(float(y))
				else:
					n_bad += 1
			except Exception as e:
				print(f"Skip bad result file {fn}: {e}")
				n_bad += 1

		if len(X_list) == 0:
			print("No valid historical result files could be loaded.")
			return 0

		self.history_X = [np.array(x, dtype=np.float64) for x in X_list]
		self.history_y = [float(y) for y in y_list]

		print(f"Loaded {len(self.history_X)} historical points from result files. Skipped {n_bad}.")
		return len(self.history_X)
	def save_surrogate_model(self, filename_prefix="pso_surrogate"):
		if self.surrogate_model is None:
			print("No surrogate model to save.")
			return

		model_file = f"{filename_prefix}_round_{self.current_iteration}.joblib"
		meta_file = f"{filename_prefix}_round_{self.current_iteration}.json"

		joblib.dump(self.surrogate_model, model_file)

		meta = {
			"round": int(self.current_iteration),
			"n_points": int(len(self.history_X)),
			"dimensions": int(self.dimensions),
			"feature_order": ["rot_x_deg", "tilt_y_deg", "psi_z_deg", "tx_A", "ty_A", "tz_A"],
			"surrogate_hidden": list(self.surrogate_hidden),
			"surrogate_weight": float(self.surrogate_weight),
			"surrogate_start_round": int(self.surrogate_start_round),
			"surrogate_update_every": int(self.surrogate_update_every),
		}

		with open(meta_file, "w", encoding="utf-8") as f:
			json.dump(meta, f, indent=2)

		print(f"Saved surrogate model to: {model_file}")
		print(f"Saved surrogate metadata to: {meta_file}")
	def add_history(self, positions, scores):
		"""
		Store real evaluated (x, score) pairs for surrogate training.
		"""
		for pos, sc in zip(positions, scores):
			if np.all(np.isfinite(pos)) and np.isfinite(sc):
				self.history_X.append(np.array(pos, dtype=np.float64))
				self.history_y.append(float(sc))

	def build_surrogate_model(self):
		"""
		Build MLP surrogate with X-scaling and y-scaling.
		"""
		mlp = Pipeline([
			("x_scaler", StandardScaler()),
			("mlp", MLPRegressor(
				hidden_layer_sizes=self.surrogate_hidden,
				activation="relu",
				solver="adam",
				alpha=1e-4,
				learning_rate_init=3e-4,
				max_iter=3000,
				early_stopping=False,
				random_state=self.surrogate_random_state,
			))
		])

		model = TransformedTargetRegressor(
			regressor=mlp,
			transformer=StandardScaler()
		)
		return model


	def maybe_train_surrogate(self, force=False):
		"""
		Train / retrain surrogate.

		If force=True, skip round-based gating and train immediately as long as
		enough historical points are available.
		"""
		if not self.use_surrogate:
			print("Surrogate disabled, skip training.")
			return False

		if (not force) and (self.current_iteration < self.surrogate_start_round):
			print(f"Surrogate not started yet: current_iteration={self.current_iteration}, "
				  f"start_round={self.surrogate_start_round}")
			return False

		if len(self.history_X) < self.surrogate_min_points:
			print(f"Not enough history points for surrogate: {len(self.history_X)} < {self.surrogate_min_points}")
			return False

		if (not force) and ((self.current_iteration - self.surrogate_start_round) % self.surrogate_update_every != 0):
			print(f"Not surrogate update round: current_iteration={self.current_iteration}, "
				  f"update_every={self.surrogate_update_every}")
			return False

		X = np.asarray(self.history_X, dtype=np.float64)
		y = np.asarray(self.history_y, dtype=np.float64)

		mask = np.isfinite(y)
		if np.sum(mask) < self.surrogate_min_points:
			print(f"Not enough valid finite points after filtering: {np.sum(mask)} < {self.surrogate_min_points}")
			return False

		X = X[mask]
		y = y[mask]

		model = self.build_surrogate_model()
		model.fit(X, y)

		self.surrogate_model = model
		self.surrogate_is_ready = True
		print(f"Surrogate trained at round {self.current_iteration}, using {len(X)} points.")
		self.save_surrogate_model(filename_prefix="pso_surrogate")
		return True

	def predict_with_surrogate(self, X):
		"""
		Predict objective values for candidate positions.
		"""
		if (not self.use_surrogate) or (self.surrogate_model is None):
			return None
		X = np.asarray(X, dtype=np.float64)
		return self.surrogate_model.predict(X)
	def run_iterations(self, n):
		optimization_history = []
		GO_SKIP_surrogate = False
		
		# --------------------------------------------------
		# Continue-mode bootstrap:
		# load historical points from existing ReSuLt_*.pdb.txt
		# and train surrogate before starting new iterations.
		# --------------------------------------------------
		if self.use_surrogate and getattr(self.args, "PSO_continue", False):
			if len(self.history_X) == 0:
				print("PSO_continue detected. Loading all historical result files in your working folder for surrogate bootstrap ...")
				n_loaded = self.load_history_from_result_files(result_glob="ReSuLt_*.pdb.txt")
				print(f"Bootstrap loaded history points: {n_loaded}")
				if n_loaded > 0:
					ok = self.maybe_train_surrogate(force=True)
					print(f"Bootstrap surrogate training success = {ok}")

		with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
			for iteration in range(n):
				GO_SKIP_surrogate = False
				actual_iter = self.current_iteration
				w = self.compute_inertia_weight(actual_iter, self.max_iterations)

				print(f"Iteration {actual_iter + 1}/{self.max_iterations}, w={w}")

				# --------------------------------------------------
				# 1) real evaluation at current positions
				# --------------------------------------------------
				args_list = zip(self.positions, self.gpuid_list)
				func_partial = partial(self.obj_func, args=self.args)
				scores = list(executor.map(func_partial, args_list))

				scores = np.array(scores, dtype=np.float64)
				print("PSO debug, scores, positions = ", scores, self.positions)
				optimization_history.append([scores.copy(), self.positions.copy()])

				# store new real evaluated data for surrogate
				self.add_history(self.positions, scores)

				# maybe train / retrain surrogate
				self.maybe_train_surrogate()

				# --------------------------------------------------
				# 2) update pbest / gbest
				# --------------------------------------------------
				better_mask = scores < self.pbest_scores
				self.pbest_scores[better_mask] = scores[better_mask]
				self.pbest_positions[better_mask] = self.positions[better_mask]

				min_score_idx = np.argmin(scores)
				if scores[min_score_idx] < self.gbest_score:
					self.gbest_score = scores[min_score_idx]
					self.gbest_position = self.positions[min_score_idx].copy()

				# --------------------------------------------------
				# 3) standard PSO velocity terms
				# --------------------------------------------------
				r1 = np.random.uniform(size=(self.num_particles, self.dimensions))
				r2 = np.random.uniform(size=(self.num_particles, self.dimensions))

				c1_rest_tomultipy = r1 * (self.pbest_positions - self.positions)
				c2_rest_tomultipy = r2 * (self.gbest_position - self.positions)

				cognitive, social = self.generate_cognitive(c1_rest_tomultipy, c2_rest_tomultipy)
				base_velocities = w * self.velocities + cognitive + social
			
				# no surrogate_term when the base_velocities are smaller than self.SKIP_surrogate_THRESHOLD.
				max_velocity = np.max(np.abs(base_velocities))

				if (max_velocity< self.SKIP_surrogate_THRESHOLD):
					GO_SKIP_surrogate = True
					print(f"max_velocity = {max_velocity}, will skip surrogate if surrogate_is_ready is true.")
				# --------------------------------------------------
				# 4) optional surrogate guidance
				# --------------------------------------------------
				if self.surrogate_is_ready and (not GO_SKIP_surrogate):
					positions_try_for_surrogate = self.positions + base_velocities
					sur_scores_try = self.predict_with_surrogate(positions_try_for_surrogate)

					if sur_scores_try is not None and np.all(np.isfinite(sur_scores_try)):
						best_sur_idx = np.argmin(sur_scores_try)
						surrogate_target = positions_try_for_surrogate[best_sur_idx].copy()
						surrogate_best_score = sur_scores_try[best_sur_idx]

						# conservative gating:
						# only apply surrogate if it predicts something clearly better
						margin = 0.0
						if surrogate_best_score < self.gbest_score - margin:
							r3 = np.random.uniform(size=(self.num_particles, self.dimensions))
							surrogate_term = self.surrogate_weight * r3 * (surrogate_target - self.positions)
							self.velocities = base_velocities + surrogate_term

							print("Surrogate guidance active.")
							print("Surrogate best predicted score:", surrogate_best_score)
							print("Surrogate target:", surrogate_target)
						else:
							self.velocities = base_velocities
					else:
						self.velocities = base_velocities
				else:
					self.velocities = base_velocities
			
				# --------------------------------------------------
				# 5) boundary handling / overlap handling
				# --------------------------------------------------
				positions_try = self.positions + self.velocities

				for d in range(self.dimensions):
					hit_lower = positions_try[:, d] <= self.bounds[d][0]
					hit_upper = positions_try[:, d] >= self.bounds[d][1]

					positions_try[:, d] = np.clip(
						positions_try[:, d],
						self.bounds[d][0],
						self.bounds[d][1]
					)

					random_magnitudes = np.random.uniform(
						self.V_LOW_SCALE,
						self.V_HIGH_SCALE,
						size=(self.num_particles,)
					)
					self.velocities[hit_lower | hit_upper, d] = \
						(self.pivot_point[d] - self.positions[hit_lower | hit_upper, d]) * random_magnitudes[hit_lower | hit_upper]

				for ii in range(self.num_particles):
					if not getattr(self.args, "skip_geometric_restraint", False) and scores[ii] >= self.MAXIUM_ALLOWED_overlapped_pixels:
						central_velocity_magnitudes = np.random.uniform(
							low=self.V_LOW_SCALE,
							high=self.V_HIGH_SCALE,
							size=(self.dimensions,)
						)
						self.velocities[ii] += central_velocity_magnitudes * (self.pivot_point - self.positions[ii])

				if self.do_add_noise_velocities:
					noise = np.random.normal(0.0, self.sigma, size=self.velocities.shape)
					self.velocities += noise

				# --------------------------------------------------
				# 6) move particles
				# --------------------------------------------------
				self.positions += self.velocities

				if self.callback:
					self.callback(self.positions, scores, self.velocities, self.gbest_position, self.gbest_score)

				self.current_iteration += 1
				self.sigma *= self.noise_decay_per_round
				save_pso_state_round(self, self.current_iteration, prefix="my_pso_state_round_")

		print(f"Finished {n} more iterations (total so far: {self.current_iteration}).")
		AA = open("PSO_optimization_history.log", "a")
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
			objective_func=objective_func,
			args=self.args,
			bounds=np.asarray(hyperparams.get('bounds', self.bounds)),
			callback=self.callback if callback is None else callback
		)
		if data['positions'].shape != (pso.num_particles, pso.dimensions):
			raise ValueError("Checkpoint dimensions / particle count do not match the supplied PSO settings")
		if not getattr(self.args, 'PSO_pivot_point', None):
			pso.pivot_point = np.asarray(hyperparams.get('pivot_point', np.zeros(pso.dimensions)))
		
		# Now overwrite its internal arrays with the saved state
		pso.positions        = data['positions']
		pso.velocities       = data['velocities']
		pso.pbest_positions  = data['pbest_positions']
		pso.pbest_scores     = data['pbest_scores']
		pso.gbest_position   = data['gbest_position']
		pso.gbest_score      = data['gbest_score']
		if(reset_velocities):
			pso.velocities = np.random.uniform(-1.0,1.0,size=(self.num_particles,self.dimensions))
			pso.positions += pso.velocities
			print(f"Reset velocities. pso.velocities = {pso.velocities }")
			print(f"New positions = {pso.positions }")
		# current_iteration needs to be cast back from the array
		pso.current_iteration = int(data['current_iteration'][0])
		version = int(data['checkpoint_version']) if 'checkpoint_version' in data else 1
		if version < 2:
			# Legacy saves were written after movement, but before increasing nit.
			pso.current_iteration += 1
		if 'sigma' in data:
			pso.sigma = float(data['sigma'])
		else:
			pso.sigma = pso.noise_strength * pso.noise_decay_per_round ** pso.current_iteration
		if 'rng_state' in data and not reset_velocities:
			np.random.set_state(tuple(data['rng_state']))
		if 'history_X' in data:
			pso.history_X = list(data['history_X'])
			pso.history_y = list(data['history_y'])
		pso.max_iterations = int(hyperparams.get('max_iterations', pso.max_iterations))
		data.close()
		
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

	atomic_npz(
		filename,
		checkpoint_version=np.array(2),
		sigma=np.array(pso.sigma),
		rng_state=np.array(np.random.get_state(), dtype=object),
		history_X=np.asarray(pso.history_X),
		history_y=np.asarray(pso.history_y),
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
			'pivot_point': pso.pivot_point,
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

def my_callback(positions, scores, velocities, best_positions, best_scores):
	with np.printoptions(precision=4, suppress=True):
		print("Callback invoked.")
		print("Current positions:\n", positions)
		print("Current velocities:\n", velocities)

	with np.printoptions(precision=6, suppress=True):
		print("Current scores:\n", scores)

	with np.printoptions(precision=4, suppress=True):
		print("Global best positions:\n", best_positions if best_positions is not None else best_positions)
		print("Global best scores:\n", best_scores if best_scores is not None else best_scores)
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
def main():
	parser = create_PSO_parser()
	args = parser.parse_args()
	bounds = convert_PSO_Bounds_to_bounds(args)
	pso = PSO(objective_func=func_gpuid, args=args, bounds=bounds, callback=my_callback)
	if args.PSO_continue:
		if not args.PSO_continue_file:
			parser.error("--PSO_continue requires --PSO_continue_file")
		pso = pso.from_file(args.PSO_continue_file, objective_func=func_gpuid,
			reset_velocities=args.PSO_continue_reset_velocities)
	try:
		prepare_workers(args)
		if args.PSO_continue:
			pso.run_iterations(args.PSO_continue_more_rounds)
		else:
			pso.optimize()
		print("Finish PSO.")
		finalize_run(args)
	finally:
		shutdown_workers(args)


if __name__ == "__main__":
	main()
