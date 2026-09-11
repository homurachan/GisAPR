import tkinter as tk
from tkinter import ttk, filedialog
import json
from pathlib import Path
import shlex
import sys


PROGRAM_ROOT = Path(__file__).resolve().parent
SEARCH_SCRIPT_NAME = "test1_with_isspa_weight_varingKK_search_translation_also_v606_torch_optimized_standalone.py"
ALGORITHM_SCRIPTS = {
	"Grid Search": "test_op_GridSearch_refine_rot_trans_v321.py",
	"Simplex": "test_op_Downhill_simplex_optimization_refine_rot_trans_rnd_init_v72.py",
	"Particle Swarm Optimization (PSO)": "test_op_Particle_Swarm_optimization_refine_rot_trans_ver622.py",
	"Pattern search": "test_op_pattern_search_try_multithreading_v3.py",
}


def resolve_search_script(value="", program_root=None):
	"""Resolve packaged scripts relative to the GUI, preserving custom scripts."""
	root = Path(program_root or PROGRAM_ROOT).resolve()
	value = str(value or "").strip()
	if not value:
		return str(root / SEARCH_SCRIPT_NAME)
	path = Path(value).expanduser()
	# Older GUI versions saved an author-specific default that is not distributed.
	if not path.exists() and value.startswith("/groups/kyouko/mydata/test1_with_isspa_"):
		return str(root / SEARCH_SCRIPT_NAME)
	return str((path if path.is_absolute() else root / path).resolve())


def _append_options(argv, options):
	for key, value in options.items():
		if value is not None and str(value).strip():
			argv.extend(("--" + key, str(value)))


def build_refinement_argv(input_params, search_params, refine_params,
		continue_params=None, program_root=None, python_executable=None):
	"""Build the exact CLI without creating a Tk window or changing the working directory.

	A blank FSC field is omitted. Input paths remain relative to the caller's working
	directory; only program paths are resolved against this GUI's directory.
	"""
	root = Path(program_root or PROGRAM_ROOT).resolve()
	algorithm = refine_params.get("algorithm", "Simplex")
	if algorithm not in ALGORITHM_SCRIPTS:
		raise ValueError("Unknown optimization algorithm: " + str(algorithm))
	argv = [str(python_executable or sys.executable), str(root / ALGORITHM_SCRIPTS[algorithm])]
	general = {
		"PDB_NAME": input_params.get("PDB file", ""),
		"STAR_NAME": input_params.get("Particle Star file", ""),
		"rotate_chain": input_params.get("Subunit", ""),
		"output_name_root": refine_params.get("Output root name", "default_name_"),
		"gpuid": refine_params.get("gpuid", "0"),
		"ang": input_params.get("Angle list", ""),
		"boxsize": input_params.get("Boxsize", 256),
		"apix_PDB": input_params.get("PDB Apix", 1.0),
		"apix": input_params.get("Particle Apix", 1.0),
		"newboxsize": search_params.get("Search boxsize in pixel", 160),
		"search_script": resolve_search_script(search_params.get("Search script"), root),
		"voltage": input_params.get("Voltage", 300),
		"cs": input_params.get("Cs", 2.7),
		"psiStep": search_params.get("Search in-plane rotation step in deg", 3),
		"kk": search_params.get("isSPA n", 3),
		"maskRadius": search_params.get("Mask Radius in pixel", 110),
		"Geometric_restrain_Scaling_Factor": refine_params.get("Geometric_restrain_Scaling_Factor", 1.0),
		"chain_MASS_in_residues": input_params.get("Subunit mass in residues", 275),
		"MAXIUM_ALLOWED_overlapped_pixels": refine_params.get("MAXIUM_ALLOWED_overlapped_pixels", 150),
		"MAX_MinDistance_Allowed": refine_params.get("MAX_MinDistance_Allowed", 30.0),
		"SplitParticles": search_params.get("Split particle starfile in these parts", 1),
	}
	for key in ("PDB_NAME", "STAR_NAME", "rotate_chain", "ang"):
		if not str(general[key]).strip():
			raise ValueError("Missing required input: " + key)
	fsc_file = str(input_params.get("FSC file") or "").strip()
	if fsc_file:
		general["fsc_file"] = fsc_file
	_append_options(argv, general)
	if search_params.get("Do local search", True):
		argv.append("--do_local_search")
		_append_options(argv, {"local_stepsize": search_params.get("Local Search Range in deg", 30)})
	if not fsc_file or search_params.get("Do ignore FSC", True):
		argv.append("--do_ignoreFSC")
	for field, flag, default in (
		("Inverse handedness", "yflip", False),
		("Do run CC", "do_run_CC", True),
		("Do simple sum", "do_simple_sum", False),
		("Do GPU projection", "doEnableGpuProj", False),
		("Do run split particle starfile in diff GPUs", "doSplitDiffGpu", False),
	):
		if search_params.get(field, default):
			argv.append("--" + flag)
	if refine_params.get("Skip geometric restraint", False):
		argv.append("--skip_geometric_restraint")
	bounds = refine_params.get("Bounds", "(-15,15),(-15,15),(-15,15),(-20,20),(-20,20),(-20,20)")
	workers = refine_params.get("max_GPU_workers", 3)
	if algorithm == "Grid Search":
		options = {"Grid_Bounds": bounds,
			"Grid_Rotation_Stepsize": refine_params.get("Orientation Step Size", 5),
			"Grid_Translation_Stepsize": refine_params.get("Translation Step Size", 10),
			"max_workers_CPU": refine_params.get("max_CPU_workers", 8),
			"max_workers_GPU": workers}
	elif algorithm == "Simplex":
		options = {"Simplex_Bounds": bounds,
			"Simplex_xatol": refine_params.get("xatol", 0.2),
			"Simplex_fatol": refine_params.get("fatol", 1.0),
			"Simplex_maxiter": refine_params.get("Maximum iterations", 100)}
	elif algorithm == "Pattern search":
		options = {"Pattern_Bounds": bounds, "max_workers": workers,
			"Pattern_stepsize": refine_params.get("Pattern stepsize", 2.0),
			"Pattern_tol": refine_params.get("Pattern tolerance", 0.01),
			"Pattern_shrink": refine_params.get("Pattern shrink eff", 0.6),
			"Pattern_expand": refine_params.get("Pattern expand eff", 1.05),
			"Pattern_max_iter": refine_params.get("Pattern Maximum iterations", 500)}
		if refine_params.get("Pattern give initial position", False):
			argv.append("--Pattern_given_initial")
			options["Pattern_initial"] = refine_params.get("Pattern initial point", "[10,10,10,5,5,5]")
	else:
		options = {"PSO_Bounds": bounds, "max_workers": workers,
			"PSO_num_particles": refine_params.get("Number of particles", 6),
			"PSO_iterations": refine_params.get("Maximum iterations", 30),
			"PSO_pivot_point": refine_params.get("Pivot Points", "(0,0,0,0,0,0)"),
			"PSO_wmax": refine_params.get("w_max", 0.9),
			"PSO_wmin": refine_params.get("w_min", 0.4)}
	_append_options(argv, options)
	continued = continue_params or {}
	enabled = continued.get("Do continue run", continued.get("Do Continue PSO", False)
		and algorithm == "Particle Swarm Optimization (PSO)")
	if enabled:
		prefix = {"Simplex": "Simplex", "Pattern search": "Pattern",
			"Particle Swarm Optimization (PSO)": "PSO"}.get(algorithm)
		if prefix is None:
			raise ValueError("Grid Search does not support checkpoint continuation.")
		checkpoint = str(continued.get("Checkpoint file", continued.get("PSO Continue Run File", ""))).strip()
		if not checkpoint:
			raise ValueError("Select a checkpoint file in Continue Run Parameters.")
		rounds = int(continued.get("Continue Run this more rounds", 20))
		if rounds < 1:
			raise ValueError("Additional continuation rounds must be greater than zero.")
		argv.append("--" + prefix + "_continue")
		_append_options(argv, {prefix + "_continue_file": checkpoint,
			prefix + "_continue_more_rounds": rounds})
	return argv


def build_refinement_command(*args, **kwargs):
	"""Return a POSIX-shell command with each argument safely quoted."""
	return shlex.join(build_refinement_argv(*args, **kwargs))


def build_once_argv(once_params, program_root=None, python_executable=None):
	"""Build a one-shot search command; its optional FSC behaves as in refinement."""
	root = Path(program_root or PROGRAM_ROOT).resolve()
	argv = [str(python_executable or sys.executable), str(root / "wrap_to_search_v2.py")]
	angle_list = once_params.get("Angle list") or once_params.get("Template Star file", "")
	options = {
		"p": once_params.get("Particle Star file", ""),
		"o": once_params.get("Output root name", "default_name_"),
		"gpuid": once_params.get("gpuid", "0"),
		# The current search retains --i for compatibility; --ang supplies orientations.
		"i": angle_list,
		"ang": angle_list,
		"mrc": once_params.get("Model MRC file", ""),
		"oriboxsize": once_params.get("Boxsize", 256),
		"apix": once_params.get("Particle Apix", 1.0),
		"newboxsize": once_params.get("Search boxsize in pixel", 160),
		"script": resolve_search_script(once_params.get("Search script"), root),
		"voltage": once_params.get("Voltage", 300),
		"cs": once_params.get("Cs", 2.7),
		"psiStep": once_params.get("Search in-plane rotation step in deg", 1),
		"kk": once_params.get("isSPA n", 3),
		"maskRadius": once_params.get("Mask Radius in pixel", 110),
		"localRange": once_params.get("Local Search Range in deg", 12),
		"maskEdge": once_params.get("Mask Soft Edge in pixel", 6),
		"SplitParticles": once_params.get("Split particle starfile in these parts", 1),
	}
	for key, field in (("p", "Particle Star file"), ("ang", "Angle list"), ("mrc", "Model MRC file")):
		if not str(options[key]).strip():
			raise ValueError("Supply " + field + " in Refine Once Parameters.")
	psi_step = float(options["psiStep"])
	if not 0 < psi_step < float("inf"):
		raise ValueError("Search in-plane rotation step must be a positive finite number in degrees.")
	options["psiStep"] = psi_step
	fsc_file = str(once_params.get("FSC file") or "").strip()
	if fsc_file:
		options["FSC"] = fsc_file
	_append_options(argv, options)
	argv.append("--doLocalSearch")
	if not fsc_file or once_params.get("Do ignore FSC", True):
		argv.append("--ignoreFSC")
	if once_params.get("Do run split particle starfile in diff GPUs", False):
		argv.append("--doSplitDiffGpu")
	return argv


def build_once_command(*args, **kwargs):
	return shlex.join(build_once_argv(*args, **kwargs))

# changelog ver2
# After clicking submit bottom, next time when you run the GUI, GUI will display your inputs.
# changelog ver201
# Program_RootName changes to Path(__file__).resolve().parent
# changelog ver202
# Add yflip to pdb2mrc, in case the reconstruction has the inversed handedness. Also run with read_pdb_index_generate_sh_3DEG_local_v32.py, no the modified relion_project should be used.
# changelog ver203
# The name of the search script will be fixed. currently the local value is changing, leading to some error.
# The func is now moving to a seperate file.
# Change the default search version from 6032 to 6033.
# changelog ver204
# Now apply the pdb2mrc_gpu_ver_fp32_v3.py and project3d_and_whiten.py in our programs.
# Only changed the func.py, read_pdb_index_generate_sh_3DEG_local_v34.py and func_check_boundary_for_testing_v7.py
# changelog ver205
# Now browse only the specific file extensions.
# changelog ver206
# Add pdb pixel size and particle pixel size. Upgrade read_pdb_index_generate_sh_3DEG_local_v34.py to read_pdb_index_generate_sh_3DEG_local_v35.py
# changelog ver207
# Update new_method_to_fit_the_2nd_Gaussian_PEAK_v31.py to new_method_to_fit_the_2nd_Gaussian_PEAK_v4.py
# Add --do_run_CC and --do_simple_sum. If do_run_CC = False, we fit the Zscore histogram. if do_simple_sum = True, we skip integration and do simple sum to the histogram.
# changelog ver208
# Add continue run for PSO. Add option to enable GPU in project3d_and_whiten. Update to read_pdb_index_generate_sh_3DEG_local_v36.py
# changelog ver209
# Update to read_pdb_index_generate_sh_3DEG_local_v37.py. Add doSplitDiffGpu and SplitParticles.
# changelog ver2091
# Add refine_one_time botton to the left panel. Only refine once using fine sets of templates.
# Provide a model MRC, an angle STAR, and a particle STAR for the current search.
# changelog ver2092
# Add Pattern Search in refine panel.
class MyApp(tk.Tk):
	def __init__(self):
		super().__init__()
		self.title("GisAPR        Author: Dongjie 'Homurachan' Zhu, ChatGPT 4o to latest")
		self.geometry("800x720")

		# Title region with image
		self.title_frame = tk.Frame(self, height=150, bg="white")
		self.title_frame.pack(side="top", fill="x")
		self.load_title_image()

		# Main layout: Left column and major region
		self.main_frame = tk.Frame(self)
		self.main_frame.pack(expand=True, fill="both")

		# Left column
		self.left_frame = tk.Frame(self.main_frame, width=300, bg="lightgray")
		self.left_frame.pack(side="left", fill="y")
		self.create_left_menu()

		# Scroll the parameter panel so every option remains accessible on small screens.
		self.panel_container = tk.Frame(self.main_frame, bg="white")
		self.panel_container.pack(expand=True, fill="both")
		self.panel_canvas = tk.Canvas(self.panel_container, bg="white", highlightthickness=0)
		panel_horizontal_scrollbar = ttk.Scrollbar(self.panel_container, orient="horizontal", command=self.panel_canvas.xview)
		panel_horizontal_scrollbar.pack(side="bottom", fill="x")
		panel_scrollbar = ttk.Scrollbar(self.panel_container, orient="vertical", command=self.panel_canvas.yview)
		panel_scrollbar.pack(side="right", fill="y")
		self.panel_canvas.pack(side="left", expand=True, fill="both")
		self.panel_canvas.configure(yscrollcommand=panel_scrollbar.set, xscrollcommand=panel_horizontal_scrollbar.set)
		self.major_region = tk.Frame(self.panel_canvas, bg="white")
		self.panel_canvas.create_window((0, 0), window=self.major_region, anchor="nw")
		self.major_region.bind("<Configure>", lambda event: self.panel_canvas.configure(scrollregion=self.panel_canvas.bbox("all")))
		
		# Store input fields
		self.INPUT_file_entries = {}
		self.INPUT_numeric_vars = {}
		self.SEARCH_file_entries = {}
		self.SEARCH_numeric_vars = {}
		self.SEARCH_boolean_vars = {}
		self.refine_vars = {}
		self.exclusive_frames = {}
		self.exclusive_vars = {}
		self.CONTINUE_file_entries={}
		self.CONTINUE_numeric_vars={}
		self.CONTINUE_boolean_vars = {}
		self.ONCE_file_entries={}
		self.ONCE_numeric_vars={}
		self.ONCE_boolean_vars = {}
		# Load existing parameters if available
		self.input_params = self.load_parameters("input_params.json")
		self.search_params = self.load_parameters("search_params.json")
		self.refine_params = self.load_parameters("refine_params.json")
		self.continue_params = self.load_parameters("continue_params.json")
		self.once_params = self.load_parameters("once_params.json")
		# Left panels
		self.panels = {
			"Input Files": self.create_InputFiles_panel(),
			"Search Parameters": self.create_SearchParameters_panel(),
			"Refine Parameters": self.create_RefineParameters_panel(),
			"Continue Run Parameters": self.create_ContinueRunParameters_panel(),
			"Refine Once Parameters": self.create_OnceParameters_panel()
		}
		self.default_panel = tk.Label(self.major_region, text="Welcome! Please select an option from the left.", font=("Arial", 14))
		self.default_panel.pack(expand=True)
		self.current_panel = self.default_panel
	def load_parameters(self, filename):
		"""Load parameters from a JSON file, or return an empty dictionary if the file does not exist."""
		try:
			with open(filename, "r") as f:
				return json.load(f)
		except FileNotFoundError:
			return {}
	def load_title_image(self):
		try:
			from PIL import Image, ImageTk
			image = Image.open(PROGRAM_ROOT / "title.png")
			image = image.resize((600, 150), Image.LANCZOS)
			self.photo = ImageTk.PhotoImage(image)
			label = tk.Label(self.title_frame, image=self.photo, bg="white")
			label.pack()
		except Exception as e:
			label = tk.Label(self.title_frame, text="Title Image Not Found", font=("Arial", 18), bg="gray", fg="white")
			label.pack()
	def create_left_menu(self):
		buttons = [("Input Files", "lightblue"),
		("Search Parameters", "lightblue"), 
		("Refine Parameters", "lightblue"), 
		("Continue Run Parameters", "lightcoral"),
		("Refine Once Parameters", "lightblue")]
		for text, color in buttons:
		#	btn = tk.Button(self.left_frame, text=text, bg=color, fg="blue", font=("Arial", 10))
			btn = tk.Button(self.left_frame, text=text, bg=color, fg="blue", font=("Arial", 10), 
				command=lambda t=text: self.show_panel(t))
			btn.pack(pady=10, padx=10, fill="x")
			
	def show_panel(self, panel_name):
		if self.current_panel:
			self.current_panel.pack_forget()
		self.current_panel = self.panels.get(panel_name, self.default_panel)
		self.current_panel.pack(expand=True)
		self.panel_canvas.yview_moveto(0)
		

	def create_InputFiles_panel(self):
		panel = tk.Frame(self.major_region, bg="white")

		def open_file_dialog(entry_widget, field):
			# Filetype filters for each field
			extension_map = {
				"Particle Star file": [("STAR files", "*.star")],
				"Angle list": [("STAR files", "*.star")],
				"PDB file": [("PDB files", "*.pdb")],
				"FSC file": [("STAR files", "*.star *.fsc")],
			}
			
			# Append the "All files" option to every field
			filetypes = extension_map.get(field, []) + [("All files", "*.*")]
			
			filename = filedialog.askopenfilename(
				title=f"Select {field}",
				filetypes=filetypes
			)
			
			if filename:
				entry_widget.delete(0, tk.END)
				entry_widget.insert(0, filename)
		
		fields = ["Particle Star file", "PDB file", "Angle list", "FSC file"]
		for idx, field in enumerate(fields):
			tk.Label(panel, text="FSC file (optional)" if field == "FSC file" else field).grid(row=idx, column=0, padx=5, pady=5, sticky="w")
			entry = tk.Entry(panel, width=40)
			entry.insert(0, self.input_params.get(field, ""))  # Load saved value if exists
			entry.grid(row=idx, column=1, padx=5, pady=5)
		#	tk.Button(panel, text="Browse", command=lambda e=entry: open_file_dialog(e)).grid(row=idx, column=2, padx=5, pady=5)
			tk.Button(panel, text="Browse", command=lambda e=entry, f=field: open_file_dialog(e, f)).grid(row=idx, column=2, padx=5, pady=5)

			self.INPUT_file_entries[field] = entry
		
		additional_fields = {
			"Subunit": tk.StringVar(value=self.input_params.get("Subunit", "")),
			"Subunit mass in residues": tk.IntVar(value=self.input_params.get("Subunit mass in residues", 275)),
			"Boxsize": tk.IntVar(value=self.input_params.get("Boxsize", 256)),
			"Particle Apix": tk.DoubleVar(value=self.input_params.get("Particle Apix", 1.00)),
			"PDB Apix": tk.DoubleVar(value=self.input_params.get("PDB Apix", 1.00)),
			"Voltage": tk.DoubleVar(value=self.input_params.get("Voltage", 300)),
			"Cs": tk.DoubleVar(value=self.input_params.get("Cs", 2.7))
		}
		
		for idx, (field, var_type) in enumerate(additional_fields.items(), start=len(fields)):
			tk.Label(panel, text=field).grid(row=idx, column=0, padx=5, pady=5, sticky="w")
			entry = tk.Entry(panel, textvariable=var_type, width=40)
			entry.grid(row=idx, column=1, padx=5, pady=5)
			self.INPUT_file_entries[field] = var_type
		
		# Submit button
		submit_btn = tk.Button(panel, text="Submit your input parameters", command=self.get_inputs)
		submit_btn.grid(row=len(fields) + len(additional_fields), column=0, columnspan=3, pady=10)
		
		return panel
	def create_SearchParameters_panel(self):
		panel = tk.Frame(self.major_region, bg="white")

		def open_file_dialog(entry_widget, field):
			filetypes = [
				("Python scripts", "*.py"),
				("All files", "*.*")
			]
			filename = filedialog.askopenfilename(title="Select a file", filetypes=filetypes)
			if filename:
				entry_widget.delete(0, tk.END)
				entry_widget.insert(0, filename)
		
		fields1 = ["Search script"]
		for idx, field in enumerate(fields1):
			tk.Label(panel, text=field).grid(row=idx, column=0, padx=5, pady=4, sticky="w")
			entry = tk.Entry(panel, width=40)
			if field == "Search script":
				entry.insert(0, resolve_search_script(self.search_params.get(field)))
			entry.grid(row=idx, column=1, padx=5, pady=4)
		#	tk.Button(panel, text="Browse", command=lambda e=entry: open_file_dialog(e)).grid(row=idx, column=2, padx=5, pady=5)
			tk.Button(panel, text="Browse", command=lambda e=entry, f=field: open_file_dialog(e, f)).grid(row=idx, column=2, padx=5, pady=5)
			self.SEARCH_file_entries[field] = entry
		
		additional_fields1 = {
			"Search in-plane rotation step in deg": tk.IntVar(value=self.search_params.get("Search in-plane rotation step in deg", 3)),
			"Local Search Range in deg": tk.IntVar(value=self.search_params.get("Local Search Range in deg", 30)),
			"Search boxsize in pixel": tk.IntVar(value=self.search_params.get("Search boxsize in pixel", 160)),
			"Mask Radius in pixel": tk.IntVar(value=self.search_params.get("Mask Radius in pixel", 110)),
			"isSPA n": tk.IntVar(value=self.search_params.get("isSPA n", 3)),
			"Split particle starfile in these parts": tk.IntVar(value=self.search_params.get("Split particle starfile in these parts", 1))
		}
		
		row_index = len(fields1)
		for field, var_type in additional_fields1.items():
			tk.Label(panel, text=field).grid(row=row_index, column=0, padx=5, pady=4, sticky="w")
			entry = tk.Entry(panel, textvariable=var_type, width=40)
			entry.grid(row=row_index, column=1, padx=5, pady=4)
			self.SEARCH_numeric_vars[field] = var_type
			row_index += 1
		boolean_fields1 = {
			"Do local search": tk.BooleanVar(value=self.search_params.get("Do local search", True)),
			"Do ignore FSC": tk.BooleanVar(value=self.search_params.get("Do ignore FSC", True)),
			"Inverse handedness": tk.BooleanVar(value=self.search_params.get("Inverse handedness", False)),
			"Do run CC": tk.BooleanVar(value=self.search_params.get("Do run CC", True)),
			"Do simple sum": tk.BooleanVar(value=self.search_params.get("Do simple sum", False)),
			"Do GPU projection": tk.BooleanVar(value=self.search_params.get("Do GPU projection", False)),
			"Do run split particle starfile in diff GPUs": tk.BooleanVar(value=self.search_params.get("Do run split particle starfile in diff GPUs", False))
		}
		for field, var in boolean_fields1.items():
			tk.Label(panel, text=field).grid(row=row_index, column=0, padx=5, pady=4, sticky="w")
			self.SEARCH_boolean_vars[field] = var
			tk.Checkbutton(panel, variable=var).grid(row=row_index, column=1, padx=5, pady=4)
			row_index += 1
		# Submit button
		submit_btn = tk.Button(panel, text="Submit your search parameters", command=self.get_search_parameters)
		submit_btn.grid(row=row_index, column=0, columnspan=3, pady=10)
		
		return panel
	def create_ContinueRunParameters_panel(self):
		panel = tk.Frame(self.major_region, bg="white")

		def open_file_dialog(entry_widget, field):
			filetypes = [
				("Stored Numpy files", "*.npz"),
				("All files", "*.*")
			]
			filename = filedialog.askopenfilename(title="Select a file", filetypes=filetypes)
			if filename:
				entry_widget.delete(0, tk.END)
				entry_widget.insert(0, filename)
		
		fields1 = ["Checkpoint file"]
		for idx, field in enumerate(fields1):
			tk.Label(panel, text=field).grid(row=idx, column=0, padx=5, pady=5, sticky="w")
			entry = tk.Entry(panel, width=40)
			entry.insert(0, self.continue_params.get(field, self.continue_params.get("PSO Continue Run File", "")))
			entry.grid(row=idx, column=1, padx=5, pady=5)
			tk.Button(panel, text="Browse", command=lambda e=entry, f=field: open_file_dialog(e, f)).grid(row=idx, column=2, padx=5, pady=5)
			self.CONTINUE_file_entries[field] = entry
		additional_fields1 = {
			"Continue Run this more rounds": tk.IntVar(value=self.continue_params.get("Continue Run this more rounds", 20))
		}
		row_index = len(fields1)
		for field, var_type in additional_fields1.items():
			tk.Label(panel, text=field).grid(row=row_index, column=0, padx=5, pady=5, sticky="w")
			entry = tk.Entry(panel, textvariable=var_type, width=40)
			entry.grid(row=row_index, column=1, padx=5, pady=5)
			self.CONTINUE_numeric_vars[field] = var_type
			row_index += 1
		boolean_fields1 = {
			"Do continue run": tk.BooleanVar(value=self.continue_params.get("Do continue run", self.continue_params.get("Do Continue PSO", False))),
		}
		for field, var in boolean_fields1.items():
			tk.Label(panel, text=field).grid(row=row_index, column=0, padx=5, pady=5, sticky="w")
			self.CONTINUE_boolean_vars[field] = var
			tk.Checkbutton(panel, variable=var).grid(row=row_index, column=1, padx=5, pady=5)
			row_index += 1
		tk.Label(panel, text="Uses the algorithm selected in Refine Parameters.\nSupports PSO, Pattern search, and Simplex checkpoints.", justify="left").grid(row=row_index, column=0, columnspan=3, padx=5, pady=5, sticky="w")
		row_index += 1
		# Submit button
		submit_btn = tk.Button(panel, text="Submit your continue parameters", command=self.get_CONTINUE_parameters)
		submit_btn.grid(row=row_index, column=0, columnspan=3, pady=10)
		
		return panel
	def create_OnceParameters_panel(self):
		panel = tk.Frame(self.major_region, bg="white")

		def open_file_dialog(entry_widget, field):
			# Filetype filters for each field
			extension_map = {
				"Search script": [("Python files","*.py")],
				"Particle Star file": [("STAR files", "*.star")],
				"Model MRC file": [("MRC files", "*.mrc *.map")],
				"Angle list": [("STAR files", "*.star")],
				"FSC file": [("FSC files", "*.fsc")],
				
			}
			
			# Append the "All files" option to every field
			filetypes = extension_map.get(field, []) + [("All files", "*.*")]
			
			filename = filedialog.askopenfilename(
				title=f"Select {field}",
				filetypes=filetypes
			)
			
			if filename:
				entry_widget.delete(0, tk.END)
				entry_widget.insert(0, filename)
		
		fields = ["Search script", "Particle Star file", "Model MRC file", "Angle list", "FSC file"]
		for idx, field in enumerate(fields):
			tk.Label(panel, text="FSC file (optional)" if field == "FSC file" else field).grid(row=idx, column=0, padx=5, pady=5, sticky="w")
			entry = tk.Entry(panel, width=40)
			value = self.once_params.get(field, self.input_params.get(field, ""))
			if field == "Angle list":
				value = self.once_params.get(field) or self.once_params.get("Template Star file") or value
			entry.insert(0, resolve_search_script(value) if field == "Search script" else value)
			entry.grid(row=idx, column=1, padx=5, pady=5)
		#	tk.Button(panel, text="Browse", command=lambda e=entry: open_file_dialog(e)).grid(row=idx, column=2, padx=5, pady=5)
			tk.Button(panel, text="Browse", command=lambda e=entry, f=field: open_file_dialog(e, f)).grid(row=idx, column=2, padx=5, pady=5)

			self.ONCE_file_entries[field] = entry
		row_index = len(fields)
		additional_fields = {
			"Output root name": tk.StringVar(value=self.once_params.get("Output root name", "default_name_")),
			"Boxsize": tk.IntVar(value=self.once_params.get("Boxsize", 256)),
			"Particle Apix": tk.DoubleVar(value=self.once_params.get("Particle Apix", 1.00)),
			"Voltage": tk.DoubleVar(value=self.once_params.get("Voltage", 300)),
			"Cs": tk.DoubleVar(value=self.once_params.get("Cs", 2.7)),
			"Search in-plane rotation step in deg": tk.DoubleVar(value=self.once_params.get("Search in-plane rotation step in deg", 1.0)),
			"Local Search Range in deg": tk.IntVar(value=self.once_params.get("Local Search Range in deg", 12)),
			"Search boxsize in pixel": tk.IntVar(value=self.once_params.get("Search boxsize in pixel", 160)),
			"Mask Radius in pixel": tk.IntVar(value=self.once_params.get("Mask Radius in pixel", 110)),
			"Mask Soft Edge in pixel": tk.IntVar(value=self.once_params.get("Mask Soft Edge in pixel", 6)),
			"isSPA n": tk.IntVar(value=self.once_params.get("isSPA n", 3)),
			"Split particle starfile in these parts": tk.IntVar(value=self.once_params.get("Split particle starfile in these parts", 1)),
			"gpuid": tk.StringVar(value=self.once_params.get("gpuid", "0")),
		}
		
		for idx, (field, var_type) in enumerate(additional_fields.items(), start=len(fields)):
			tk.Label(panel, text=field).grid(row=idx, column=0, padx=5, pady=5, sticky="w")
			entry = tk.Entry(panel, textvariable=var_type, width=40)
			entry.grid(row=idx, column=1, padx=5, pady=5)
			self.ONCE_file_entries[field] = var_type
			row_index+=1
		boolean_fields1 = {
			"Do ignore FSC": tk.BooleanVar(value=self.once_params.get("Do ignore FSC", True)),
			"Do run split particle starfile in diff GPUs": tk.BooleanVar(value=self.once_params.get("Do run split particle starfile in diff GPUs", False))
		}
		for field, var in boolean_fields1.items():
			tk.Label(panel, text=field).grid(row=row_index, column=0, padx=5, pady=4, sticky="w")
			self.ONCE_boolean_vars[field] = var
			tk.Checkbutton(panel, variable=var).grid(row=row_index, column=1, padx=5, pady=4)
			row_index += 1
		# Submit button
		submit_btn = tk.Button(panel, text="Submit your input parameters", command=self.get_ONCE_parameters)
		submit_btn.grid(row=row_index, column=0, columnspan=3, pady=10)
		
		return panel
	def create_RefineParameters_panel(self):
		panel = tk.Frame(self.major_region, bg="white")
		
		# Shared parameters
		shared_params = {
			"Bounds": tk.StringVar(value=self.refine_params.get("Bounds", "(-15,15),(-15,15),(-15,15),(-20,20),(-20,20),(-20,20)")),
			"MAXIUM_ALLOWED_overlapped_pixels": tk.IntVar(value=self.refine_params.get("MAXIUM_ALLOWED_overlapped_pixels", 150)),
			"MAX_MinDistance_Allowed": tk.DoubleVar(value=self.refine_params.get("MAX_MinDistance_Allowed", 30.0)),
			"Geometric_restrain_Scaling_Factor": tk.DoubleVar(value=self.refine_params.get("Geometric_restrain_Scaling_Factor", 1.0)),
			"max_CPU_workers": tk.IntVar(value=self.refine_params.get("max_CPU_workers", 8)),
			"max_GPU_workers": tk.IntVar(value=self.refine_params.get("max_GPU_workers", 3)),
			"gpuid": tk.StringVar(value=self.refine_params.get("gpuid", "0:0:0")),
			"Output root name": tk.StringVar(value=self.refine_params.get("Output root name", "default_name_"))
		}
		row_index = 0
		for param, var in shared_params.items():
			tk.Label(panel, text=param).grid(row=row_index, column=0, padx=5, pady=5, sticky="w")
			tk.Entry(panel, textvariable=var, width=40).grid(row=row_index, column=1, padx=5, pady=5)
			self.refine_vars[param] = var
			row_index += 1
		self.refine_vars["Skip geometric restraint"] = tk.BooleanVar(value=self.refine_params.get("Skip geometric restraint", False))
		tk.Label(panel, text="Skip geometric restraint").grid(row=row_index, column=0, padx=5, pady=5, sticky="w")
		tk.Checkbutton(panel, variable=self.refine_vars["Skip geometric restraint"]).grid(row=row_index, column=1, padx=5, pady=5)
		row_index += 1
		
		# Algorithm selection
		tk.Label(panel, text="Optimization Algorithm").grid(row=row_index, column=0, padx=5, pady=5, sticky="w")
		self.refine_vars["algorithm"] = tk.StringVar(value=self.refine_params.get("algorithm", "Simplex"))
		algo_combobox = ttk.Combobox(panel, textvariable=self.refine_vars["algorithm"], 
					 values=list(ALGORITHM_SCRIPTS), state="readonly")
		algo_combobox.grid(row=row_index, column=1, padx=5, pady=5)
		algo_combobox.bind("<<ComboboxSelected>>", lambda event: self.update_exclusive_params(event))
		row_index += 1
		
		# Exclusive parameter frames
		self.exclusive_frame = tk.Frame(panel, bg="white")
		self.exclusive_frame.grid(row=row_index, column=0, columnspan=2, sticky="w")

		self.exclusive_frames["Grid Search"] = self.create_grid_search_params(self.exclusive_frame)
		self.exclusive_frames["Simplex"] = self.create_simplex_params(self.exclusive_frame)
		self.exclusive_frames["Particle Swarm Optimization (PSO)"] = self.create_pso_params(self.exclusive_frame)
		self.exclusive_frames["Pattern search"] = self.create_pattern_params(self.exclusive_frame)
		# Show default selection
		self.update_exclusive_params(None)
		
		# Submit button
		row_index += 1
		submit_btn = tk.Button(panel, text="Submit Refine Parameters", command=self.submit_refine_parameters)
		submit_btn.grid(row=row_index, column=0, columnspan=2, pady=10)
		
		return panel
	
	def update_exclusive_params(self, event):
		selected_algo = self.refine_vars["algorithm"].get()
		for algo, frame in self.exclusive_frames.items():
			if algo == selected_algo:
				frame.grid()
			else:
				frame.grid_remove()

	def create_grid_search_params(self, parent):
		frame = tk.Frame(parent, bg="white")
		params = {
			"Orientation Step Size": tk.IntVar(value=self.refine_params.get("Orientation Step Size", 5)),
			"Translation Step Size": tk.IntVar(value=self.refine_params.get("Translation Step Size", 10))
		}
		self.exclusive_vars["Grid Search"] = params
		for idx, (param, var) in enumerate(params.items()):
			tk.Label(frame, text=param).grid(row=idx, column=0, padx=5, pady=5, sticky="w")
			tk.Entry(frame, textvariable=var, width=40).grid(row=idx, column=1, padx=5, pady=5)
		return frame
	
	def create_simplex_params(self, parent):
		frame = tk.Frame(parent, bg="white")
		params = {
			"Maximum iterations": tk.IntVar(value=self.refine_params.get("Maximum iterations", 100)),
			"xatol": tk.DoubleVar(value=self.refine_params.get("xatol", 0.2)),
			"fatol": tk.DoubleVar(value=self.refine_params.get("fatol", 1.0))
		}
		self.exclusive_vars["Simplex"] = params
		for idx, (param, var) in enumerate(params.items()):
			tk.Label(frame, text=param).grid(row=idx, column=0, padx=5, pady=5, sticky="w")

			tk.Entry(frame, textvariable=var, width=40).grid(row=idx, column=1, padx=5, pady=5)
		return frame

	def create_pso_params(self, parent):
		frame = tk.Frame(parent, bg="white")
		params = {
			"Number of particles": tk.IntVar(value=self.refine_params.get("Number of particles", 6)),
			"Maximum iterations": tk.IntVar(value=self.refine_params.get("Maximum iterations", 30)),
			"Pivot Points": tk.StringVar(value=self.refine_params.get("Pivot Points", "(0,0,0,0,0,0)")),
			"w_max": tk.DoubleVar(value=self.refine_params.get("w_max", 0.9)),
			"w_min": tk.DoubleVar(value=self.refine_params.get("w_min", 0.4))
		}
		self.exclusive_vars["Particle Swarm Optimization (PSO)"] = params
		for idx, (param, var) in enumerate(params.items()):
			label = "Pivot (rot, tilt, psi in deg; x, y, z in Å)" if param == "Pivot Points" else param
			tk.Label(frame, text=label).grid(row=idx, column=0, padx=5, pady=5, sticky="w")
			tk.Entry(frame, textvariable=var, width=40).grid(row=idx, column=1, padx=5, pady=5)
		tk.Label(frame, text="Pivot is the PSO attraction point in parameter space.", justify="left").grid(row=len(params), column=0, columnspan=2, padx=5, sticky="w")
		return frame
	def create_pattern_params(self, parent):
		frame = tk.Frame(parent, bg="white")
		params = {
			"Pattern give initial position": tk.BooleanVar(value=self.refine_params.get("Pattern give initial position", False)),
			"Pattern initial point": tk.StringVar(value=self.refine_params.get("Pattern initial point", "[10,10,10,5,5,5]")),
			"Pattern stepsize": tk.DoubleVar(value=self.refine_params.get("Pattern stepsize", 2.0)),
			"Pattern tolerance": tk.DoubleVar(value=self.refine_params.get("Pattern tolerance", 0.01)),
			"Pattern shrink eff": tk.DoubleVar(value=self.refine_params.get("Pattern shrink eff", 0.6)),
			"Pattern expand eff": tk.DoubleVar(value=self.refine_params.get("Pattern expand eff", 1.05)),
			"Pattern Maximum iterations": tk.IntVar(value=self.refine_params.get("Pattern Maximum iterations", 500))
		}
		self.exclusive_vars["Pattern search"] = params
		for idx, (param, var) in enumerate(params.items()):
			tk.Label(frame, text=param).grid(row=idx, column=0, padx=5, pady=5, sticky="w")
			if isinstance(var, tk.BooleanVar):
				tk.Checkbutton(frame, variable=var).grid(row=idx, column=1, padx=5, pady=5)
			else:
				tk.Entry(frame, textvariable=var, width=40).grid(row=idx, column=1, padx=5, pady=5)
		return frame
	def save_parameters(self, filename, data):
		"""Save dictionary data to a JSON file."""
		with open(filename, "w") as f:
			json.dump(data, f, indent=4)
		print(f"Saved parameters to {filename}")
	def submit_refine_parameters(self):
		selected_algo = self.refine_vars["algorithm"].get()
		inputs = {key: var.get() for key, var in self.refine_vars.items()}  # Shared parameters
		if selected_algo in self.exclusive_vars:
			inputs.update({key: var.get() for key, var in self.exclusive_vars[selected_algo].items()})
		print("Refine Parameters Submitted:", inputs)
		self.save_parameters("refine_params.json", inputs)
	def get_search_parameters(self):
		inputs = {key: entry.get() for key, entry in self.SEARCH_file_entries.items() if key == "Search script"}
		inputs.update({key: var.get() for key, var in self.SEARCH_numeric_vars.items()})
		inputs.update({key: var.get() for key, var in self.SEARCH_boolean_vars.items()})
		print("Search Parameters:", inputs)
		self.save_parameters("search_params.json", inputs)
	def get_CONTINUE_parameters(self):
		inputs = {key: entry.get() for key, entry in self.CONTINUE_file_entries.items()}
		inputs.update({key: var.get() for key, var in self.CONTINUE_numeric_vars.items()})
		inputs.update({key: var.get() for key, var in self.CONTINUE_boolean_vars.items()})
		print("Continue Parameters:", inputs)
		self.save_parameters("continue_params.json", inputs)
	def get_ONCE_parameters(self):
		inputs = {key: entry.get() for key, entry in self.ONCE_file_entries.items()}
		inputs.update({key: var.get() for key, var in self.ONCE_numeric_vars.items()})
		inputs.update({key: var.get() for key, var in self.ONCE_boolean_vars.items()})
		print("Refine Once Parameters:", inputs)
		self.save_parameters("once_params.json", inputs)
	def get_inputs(self):
		inputs = {key: entry.get() for key, entry in self.INPUT_file_entries.items()}
		inputs.update({key: var.get() for key, var in self.INPUT_numeric_vars.items()})
		print("Collected Inputs:", inputs)
		self.save_parameters("input_params.json", inputs)

def load_parameters(filename):
	# Load dictionary data from a JSON file.
	try:
		with open(filename, "r") as f:
			data = json.load(f)
		print(f"Loaded parameters from {filename}")
		return data
	except FileNotFoundError:
		print(f"File {filename} not found.")
		return {}
def load_continue_parameters(filename):
	# Load dictionary data from a JSON file.
	try:
		with open(filename, "r") as f:
			data = json.load(f)
		print(f"Loaded parameters from {filename}")
		return data
	except FileNotFoundError:
		print("No continue file found. Go without continue run.")
		return {}
def load_Once_parameters(filename):
	# Load dictionary data from a JSON file.
	try:
		with open(filename, "r") as f:
			data = json.load(f)
		print(f"Loaded parameters from {filename}")
		return data
	except FileNotFoundError:
		print("No refine once file found. Go without refine once.")
		return {}
def main():
	app = MyApp()
	app.mainloop()
	once_params = load_Once_parameters("once_params.json")
	# Preserve the existing convention: a submitted one-shot configuration takes priority.
	if once_params:
		try:
			command = build_once_command(once_params)
		except (ValueError, TypeError) as exc:
			print("Cannot generate the refine once command:", exc)
			return 1
		print("The refine once command is:")
		print(command)
		print("\nDelete once_params.json to generate a conventional refinement command.")
		return 0
	refine_params = load_parameters("refine_params.json")
	search_params = load_parameters("search_params.json")
	input_params = load_parameters("input_params.json")
	continue_params = load_continue_parameters("continue_params.json")
	missing = [label for label, params in (("Input", input_params),
		("Search", search_params), ("Refine", refine_params)) if not params]
	if missing:
		print("Submit parameters on these pages before closing: " + ", ".join(missing))
		return 1
	try:
		command = build_refinement_command(input_params, search_params, refine_params, continue_params)
	except (ValueError, TypeError) as exc:
		print("Cannot generate the refinement command:", exc)
		return 1
	print("\nThe refinement command is:")
	print(command)
	print("\nRun this command from your data working directory.")
	return 0


if __name__ == "__main__":
	sys.exit(main())
