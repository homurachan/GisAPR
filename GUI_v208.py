import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
from PIL import Image, ImageTk  # For displaying PNG images
import json
from pathlib import Path

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
class MyApp(tk.Tk):
	def __init__(self):
		super().__init__()
		self.title("GisAPR        Author: Dongjie 'Homurachan' Zhu")
		self.geometry("800x600")

		# Title region with image
		self.title_frame = tk.Frame(self, height=100, bg="gray")
		self.title_frame.pack(side="top", fill="x")
		self.load_title_image()

		# Main layout: Left column and major region
		self.main_frame = tk.Frame(self)
		self.main_frame.pack(expand=True, fill="both")

		# Left column
		self.left_frame = tk.Frame(self.main_frame, width=300, bg="lightgray")
		self.left_frame.pack(side="left", fill="y")
		self.create_left_menu()

		# Major region (currently blank)
		self.major_region = tk.Frame(self.main_frame, bg="white")
		self.major_region.pack(expand=True, fill="both")
		
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
		# Load existing parameters if available
		self.input_params = self.load_parameters("input_params.json")
		self.search_params = self.load_parameters("search_params.json")
		self.refine_params = self.load_parameters("refine_params.json")
		self.continue_params = self.load_parameters("continue_params.json")
		# Left panels
		self.panels = {
			"Input Files": self.create_InputFiles_panel(),
			"Search Parameters": self.create_SearchParameters_panel(),
			"Refine Parameters": self.create_RefineParameters_panel(),
			"Continue Run Parameters": self.create_ContinueRunParameters_panel()
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
			image = Image.open("title.png")  # Ensure title.png exists
			image = image.resize((600, 100), Image.LANCZOS)
			self.photo = ImageTk.PhotoImage(image)
			label = tk.Label(self.title_frame, image=self.photo, bg="gray")
			label.pack()
		except Exception as e:
			label = tk.Label(self.title_frame, text="Title Image Not Found", font=("Arial", 18), bg="gray", fg="white")
			label.pack()
	def create_left_menu(self):
		buttons = [("Input Files", "lightblue"),
		("Search Parameters", "lightblue"), 
		("Refine Parameters", "lightblue"), 
		("Continue Run Parameters", "lightcoral")]
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
			tk.Label(panel, text=field).grid(row=idx, column=0, padx=5, pady=5, sticky="w")
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
				entry.insert(0, self.search_params.get(field, "/groups/kyouko/mydata/test1_with_isspa_weight_varingKK_search_translation_also_v6034.py"))
			entry.grid(row=idx, column=1, padx=5, pady=4)
		#	tk.Button(panel, text="Browse", command=lambda e=entry: open_file_dialog(e)).grid(row=idx, column=2, padx=5, pady=5)
			tk.Button(panel, text="Browse", command=lambda e=entry, f=field: open_file_dialog(e, f)).grid(row=idx, column=2, padx=5, pady=5)
			self.SEARCH_file_entries[field] = entry
		
		additional_fields1 = {
			"Search in-plane rotation step in deg": tk.IntVar(value=self.search_params.get("Search in-plane rotation step in deg", 3)),
			"Local Search Range in deg": tk.IntVar(value=self.search_params.get("Local Search Range in deg", 30)),
			"Search boxsize in pixel": tk.IntVar(value=self.search_params.get("Search boxsize in pixel", 160)),
			"Mask Radius in pixel": tk.IntVar(value=self.search_params.get("Mask Radius in pixel", 110)),
			"Mask Soft Edge in pixel": tk.IntVar(value=self.search_params.get("Mask Soft Edge in pixel", 6)),
			"isSPA n": tk.IntVar(value=self.search_params.get("isSPA n", 3))
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
			"Do GPU projection": tk.BooleanVar(value=self.search_params.get("Do GPU projection", False))
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
		
		fields1 = ["PSO Continue Run File"]
		for idx, field in enumerate(fields1):
			tk.Label(panel, text=field).grid(row=idx, column=0, padx=5, pady=5, sticky="w")
			entry = tk.Entry(panel, width=40)
			entry.insert(0, self.continue_params.get(field, ""))  # Load saved value if exists
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
			"Do Continue PSO": tk.BooleanVar(value=self.continue_params.get("Do Continue PSO", False)),
		}
		for field, var in boolean_fields1.items():
			tk.Label(panel, text=field).grid(row=row_index, column=0, padx=5, pady=5, sticky="w")
			self.CONTINUE_boolean_vars[field] = var
			tk.Checkbutton(panel, variable=var).grid(row=row_index, column=1, padx=5, pady=5)
			row_index += 1
		# Submit button
		submit_btn = tk.Button(panel, text="Submit your continue parameters", command=self.get_CONTINUE_parameters)
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
		
		# Algorithm selection
		tk.Label(panel, text="Optimization Algorithm").grid(row=row_index, column=0, padx=5, pady=5, sticky="w")
		self.refine_vars["algorithm"] = tk.StringVar(value="Simplex")
		algo_combobox = ttk.Combobox(panel, textvariable=self.refine_vars["algorithm"], 
					 values=["Grid Search", "Simplex", "Particle Swarm Optimization (PSO)"])
		algo_combobox.grid(row=row_index, column=1, padx=5, pady=5)
		algo_combobox.bind("<<ComboboxSelected>>", lambda event: self.update_exclusive_params(event))
		row_index += 1
		
		# Exclusive parameter frames
		self.exclusive_frame = tk.Frame(panel, bg="white")
		self.exclusive_frame.grid(row=row_index, column=0, columnspan=2, sticky="w")

		self.exclusive_frames["Grid Search"] = self.create_grid_search_params(self.exclusive_frame)
		self.exclusive_frames["Simplex"] = self.create_simplex_params(self.exclusive_frame)
		self.exclusive_frames["Particle Swarm Optimization (PSO)"] = self.create_pso_params(self.exclusive_frame)
		
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
			"Randomize initial simplex": tk.BooleanVar(value=self.refine_params.get("Randomize initial simplex", True)),
			"Initial simplex": tk.StringVar(value=self.refine_params.get("Randomize initial simplex", "[15,15,15,15,15,15],[-15,15,15,15,15,15],[-15,-15,15,15,15,15],[-15,-15,-15,15,15,15],[-15,-15,-15,-15,15,15],[-15,-15,-15,-15,-15,15],[-15,-15,-15,-15,-15,-15]")),
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
			"Pivot Points": tk.StringVar(value=self.refine_params.get("Maximum iterations","(0,0,0,0,0,0)")),
			"w_max": tk.DoubleVar(value=self.refine_params.get("w_max", 0.9)),
			"w_min": tk.DoubleVar(value=self.refine_params.get("w_min", 0.4))
		}
		self.exclusive_vars["Particle Swarm Optimization (PSO)"] = params
		for idx, (param, var) in enumerate(params.items()):
			tk.Label(frame, text=param).grid(row=idx, column=0, padx=5, pady=5, sticky="w")
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
if __name__ == "__main__":
	app = MyApp()
	app.mainloop()
	refine_params = load_parameters("refine_params.json")
	search_params = load_parameters("search_params.json")
	input_params = load_parameters("input_params.json")
	continue_params = load_continue_parameters("continue_params.json")
	NEED_TO_QUIT=False
	if(input_params == {}):
		print("You didn't click the submit botton on Input page.")
		NEED_TO_QUIT=True
	if(search_params == {}):
		print("You didn't click the submit botton on Search page.")
		NEED_TO_QUIT=True
	if(refine_params == {}):
		print("You didn't click the submit botton on Refine page.")
		NEED_TO_QUIT=True
	if(NEED_TO_QUIT):
		print("GUI EXIT.")
		quit()
	do_continue_run = False
	if(continue_params == {}):
		do_continue_run = False
	# Reading the input_params
	General_params = {
		"PDB_NAME": input_params["PDB file"],
		"STAR_NAME": input_params["Particle Star file"],
		"rotate_chain": input_params["Subunit"],
		"output_name_root": refine_params["Output root name"],
		"gpuid": refine_params["gpuid"],
		"ang": input_params["Angle list"],
		"boxsize": input_params["Boxsize"],
		"apix_PDB": input_params["PDB Apix"],
		"apix": input_params["Particle Apix"],
		"newboxsize": search_params["Search boxsize in pixel"],
		"search_script": search_params["Search script"],
		"fsc_file": input_params["FSC file"],
		"voltage": input_params["Voltage"],
		"cs": input_params["Cs"],
		"psiStep": search_params["Search in-plane rotation step in deg"],
		"kk": search_params["isSPA n"],
		"maskRadius": search_params["Mask Radius in pixel"],
		"Geometric_restrain_Scaling_Factor": refine_params["Geometric_restrain_Scaling_Factor"],
		"chain_MASS_in_residues": input_params["Subunit mass in residues"],
		"MAXIUM_ALLOWED_overlapped_pixels": refine_params["MAXIUM_ALLOWED_overlapped_pixels"],
		"MAX_MinDistance_Allowed": refine_params["MAX_MinDistance_Allowed"],
	}
	max_CPU_workers = refine_params["max_CPU_workers"]
	max_GPU_workers = refine_params["max_GPU_workers"]
	do_local_search = search_params["Do local search"]
	do_ignoreFSC = search_params["Do ignore FSC"]
	yflip = search_params["Inverse handedness"]
	do_run_CC = search_params["Do run CC"]
	do_simple_sum = search_params["Do simple sum"]
	do_GPU_projection = search_params["Do GPU projection"]
	Additional_params = ""
	if (do_local_search == True):
		Additional_params+=" --do_local_search --local_stepsize "+str(search_params["Local Search Range in deg"])
	if(do_ignoreFSC == True):
		Additional_params += " --do_ignoreFSC"
	if(yflip == True):
		Additional_params += " --yflip"
	if(do_run_CC == True):
		Additional_params += " --do_run_CC"
	if(do_simple_sum == True):
		Additional_params += " --do_simple_sum"
	if(do_GPU_projection == True):
		Additional_params += " --doEnableGpuProj"
		print("Warning! Enable GPU projection is not recommanded for < 32GB graphics memory.")
		print("It will easily lead to out-of-memory even on RTX 4090")
		print("")
	Bounds = "\""+refine_params["Bounds"]+"\""
	Program_RootName = Path(__file__).resolve().parent
	General_params_for_CMD = " ".join(f"--{key} {str(value).lower() if isinstance(value, bool) else value}"for key, value in General_params.items()) + Additional_params
	if(refine_params["algorithm"]=="Grid Search"):
		Grid_Bounds = Bounds
		Grid_Rotation_Stepsize = refine_params["Orientation Step Size"]
		Grid_Translation_Stepsize = refine_params["Translation Step Size"]
		print("Specify doing Grid Refinement")
		
		Grid_refine_warp = Program_RootName / "test_op_GridSearch_refine_rot_trans_v321.py"
		Command = "python "+str(Grid_refine_warp)+" "+General_params_for_CMD\
		+" --Grid_Bounds "+Grid_Bounds+" --Grid_Rotation_Stepsize "+str(Grid_Rotation_Stepsize)+" --Grid_Translation_Stepsize "+str(Grid_Translation_Stepsize)\
		+" --max_workers_CPU "+str(max_CPU_workers)+" --max_workers_GPU "+str(max_GPU_workers)
		
		# do grid refine
	if(refine_params["algorithm"]=="Simplex"):
		Simplex_Bounds = Bounds
	#	refine_params["Randomize initial simplex"]
		Simplex_xatol = refine_params["xatol"]
		Simplex_fatol = refine_params["fatol"]
		Simplex_maxiter = refine_params["Maximum iterations"]
		# 'Randomize initial simplex' is True in this version. So no provided Initial_Simplex.
		# Will add this in the future.
		# do simplex refine
		print("Specify doing Simplex Refinement")
		
		Simplex_refine_warp = Program_RootName / "test_op_Downhill_simplex_optimization_refine_rot_trans_rnd_init_v72.py"
		Command = "python "+str(Simplex_refine_warp)+" "+General_params_for_CMD\
		+" --Simplex_Bounds "+str(Simplex_Bounds)+" --Simplex_xatol "+str(Simplex_xatol)+" --Simplex_fatol "+str(Simplex_fatol)+" --Simplex_maxiter "+str(Simplex_maxiter)
		
		# Simplex can only run single thread.

	if(refine_params["algorithm"]=="Particle Swarm Optimization (PSO)"):
		PSO_Bounds = Bounds
		PSO_num_particles = refine_params["Number of particles"]
		PSO_iterations = refine_params["Maximum iterations"]
		PSO_wmax = refine_params["w_max"]
		PSO_wmin = refine_params["w_min"]
		# The pivot point is skipped in this version.
		# do PSO refine
		print("Specify doing PSO Refinement")
		do_continue = False
		if(continue_params != {}):
			if(continue_params["Do Continue PSO"]):
				print("Continue running PSO")
				do_continue = True
				PSO_continue_file = continue_params["PSO Continue Run File"]
				PSO_continue_more_rounds = continue_params["Continue Run this more rounds"]
		####### TODO complete continue
		#######
		PSO_refine_warp = Program_RootName / "test_op_Particle_Swarm_optimization_refine_rot_trans_ver622.py"
		Command = "python "+str(PSO_refine_warp)+" "+General_params_for_CMD\
		+" --PSO_Bounds "+str(PSO_Bounds)+" --PSO_num_particles "+str(PSO_num_particles)+" --PSO_iterations "+str(PSO_iterations)\
		+" --PSO_wmax "+str(PSO_wmax)+" --PSO_wmin "+str(PSO_wmin)+ " --max_workers "+str(max_GPU_workers)
		if(do_continue):
			Command +=" --PSO_continue --PSO_continue_file "+str(PSO_continue_file) +" --PSO_continue_more_rounds "+str(PSO_continue_more_rounds)
		# PSO can run in parallel.
	print()
	print("The refinement command is:")
	print (Command)
	print()
	print("Please re-check the parameters before running.")
	if(refine_params["algorithm"]=="Grid Search"):
		print("When running Grid refine, using too fine grids will cost tremendous amount of time. Good luck.")
	
