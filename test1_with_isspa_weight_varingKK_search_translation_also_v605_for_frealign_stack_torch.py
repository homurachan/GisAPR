import numpy as np
import mrcfile
import os,sys,math
import time
import pickle
import random, argparse
from CTF_cupy import CTF
try:
	from optparse import OptionParser
except:
	from optik import OptionParser

from functools import partial
import concurrent.futures
from project3d_and_whiten_cuda_nowrite import *
import gc
import torch
import torch
import torch.nn.functional as F

class _TorchMemoryPool:
    def free_all_blocks(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    def used_bytes(self):
        if torch.cuda.is_available():
            return int(torch.cuda.memory_allocated())
        return 0
    def total_bytes(self):
        if torch.cuda.is_available():
            return int(torch.cuda.memory_reserved())
        return 0

class _TorchDeviceCtx:
    def __init__(self, parent, idx=None):
        self.parent = parent
        self.idx = idx
    def use(self):
        if self.idx is None or (not torch.cuda.is_available()):
            self.parent._device = torch.device('cpu')
        else:
            self.parent._device = torch.device(f'cuda:{self.idx}')
            torch.cuda.set_device(self.idx)
    @property
    def mem_info(self):
        if torch.cuda.is_available() and str(self.parent._device).startswith('cuda'):
            free_b, total_b = torch.cuda.mem_get_info(self.parent._device)
            return (free_b, total_b)
        return (0, 0)

class _TorchCudaStreamNull:
    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

class _TorchCudaStream:
    null = _TorchCudaStreamNull()

class _TorchCuda:
    def __init__(self, parent):
        self.parent = parent
        self.Stream = _TorchCudaStream
    def Device(self, idx=None):
        return _TorchDeviceCtx(self.parent, idx)

class _TorchFFT:
    @staticmethod
    def fft2(x):
        return torch.fft.fft2(x)
    @staticmethod
    def fftshift(x, axes=None):
        if axes is None:
            return torch.fft.fftshift(x)
        return torch.fft.fftshift(x, dim=axes)
    @staticmethod
    def irfft2(x, axes=None):
        if axes is None:
            return torch.fft.irfft2(x)
        return torch.fft.irfft2(x, dim=axes)

class _TorchLinalg:
    @staticmethod
    def norm(x):
        return torch.linalg.norm(x)

class _CPCompat:
    float32 = torch.float32
    complex64 = torch.complex64
    int32 = torch.int32
    newaxis = None
    def __init__(self):
        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.cuda = _TorchCuda(self)
        self.fft = _TorchFFT()
        self.linalg = _TorchLinalg()
    def _to_device(self, x, dtype=None):
        if isinstance(x, torch.Tensor):
            return x.to(device=self._device, dtype=dtype if dtype is not None else x.dtype)

        if isinstance(x, np.ndarray):
            if not x.flags.writeable:
                x = np.array(x, copy=True)
            return torch.as_tensor(x, dtype=dtype, device=self._device)
        return torch.as_tensor(x, dtype=dtype, device=self._device)
    def array(self, x, dtype=None):
        return self._to_device(x, dtype)
    def asarray(self, x, dtype=None):
        return self._to_device(x, dtype)
    def zeros(self, shape, dtype=torch.float32):
        return torch.zeros(shape, dtype=dtype, device=self._device)
    def empty(self, shape, dtype=torch.float32):
        return torch.empty(shape, dtype=dtype, device=self._device)
    def zeros_like(self, x):
        return torch.zeros_like(x)
    def copy(self, x):
        return x.clone()
    def argwhere(self, x):
        return torch.argwhere(x)
    def arange(self, *args, **kwargs):
        kwargs.setdefault('device', self._device)
        return torch.arange(*args, **kwargs)
    def pad(self, array, pad_width=1, mode='reflect'):
        if array.ndim != 2:
            raise NotImplementedError('pad currently supports 2D tensors only')
        if isinstance(pad_width, int):
            pad = (pad_width, pad_width, pad_width, pad_width)
        else:
            raise NotImplementedError('tuple pad_width not implemented')
        return F.pad(array.unsqueeze(0).unsqueeze(0), pad, mode=mode).squeeze(0).squeeze(0)
    def max(self, x, axis=None):
        return torch.max(x) if axis is None else torch.amax(x, dim=axis)
    def mean(self, x):
        return torch.mean(x)
    def std(self, x):
        return torch.std(x, correction=0)
    def where(self, cond):
        return torch.where(cond)
    def conj(self, x):
        return torch.conj(x)
    def real(self, x):
        return torch.real(x)
    def concatenate(self, xs, axis=0):
        return torch.cat(xs, dim=axis)
    def unravel_index(self, indices, shape):
        return torch.unravel_index(indices, shape)
    def argmax(self, x, axis=None):
        return torch.argmax(x) if axis is None else torch.argmax(x, dim=axis)
    def get_default_memory_pool(self):
        return _TorchMemoryPool()
    def get_default_pinned_memory_pool(self):
        return _TorchMemoryPool()

cp = _CPCompat()

def affine_transform(image, mat, offset, order=1):
    if image.ndim != 2:
        raise ValueError('Only 2D affine_transform is supported')
    H, W = image.shape
    yy, xx = torch.meshgrid(
        torch.arange(H, device=image.device, dtype=torch.float32),
        torch.arange(W, device=image.device, dtype=torch.float32),
        indexing='ij'
    )
    coords = torch.stack([yy, xx], dim=-1)
    mat_t = mat.to(device=image.device, dtype=torch.float32)
    off_t = torch.as_tensor(offset, device=image.device, dtype=torch.float32)
    src = coords @ mat_t.T + off_t
    y = src[..., 0]
    x = src[..., 1]
    if H > 1:
        y_norm = 2.0 * y / (H - 1) - 1.0
    else:
        y_norm = torch.zeros_like(y)
    if W > 1:
        x_norm = 2.0 * x / (W - 1) - 1.0
    else:
        x_norm = torch.zeros_like(x)
    grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)
    out = F.grid_sample(
        image.unsqueeze(0).unsqueeze(0).to(torch.float32),
        grid,
        mode='bilinear' if order == 1 else 'nearest',
        padding_mode='zeros',
        align_corners=True,
    )
    return out.squeeze(0).squeeze(0).to(image.dtype)

BLOCKSIZE = 64
BLOCKDIM = lambda x : (x - 1) // BLOCKSIZE + 1
####
# changelog since v53
# search in-plane rotation, using CCG, using argparse;
# using rfft (cropped from cp.fft.fft2) that reduce device memory usage;
# chunk size and pre-read image number adjusted for the search step 5deg;
# remove unused functions;
# final_result is float16, multiply by 1024 for better precision; cp.unravel_index is replaced by cp.where, so that 3x faster.
# speed now: cropped to 160, rotation 5deg, psi 5deg, 27400 images in 6150 seconds
# fix in line: "if (indices[0].size == 0):". cp.where medhods can result in not found any element in final_result. So make a backup.

# changelog 541
# add DTYPE_for_finalresult here
# in final_result, change MULT position (before the 1024*result will not work)
# change from: max_values = cp.empty(final_result.shape[0]*NUM_PSI_ROTATION); to: max_values = cp.empty(final_result.shape[0])
# dropping the NUM_PSI_ROTATION. Need tests to check whether the Z-score should consider the psi (inplane rotation) term.

# changelog 542
# IN this version Z-score consider the psi. max_values is converted to cp.float32 before computing cp.mean and cp.std to prevent from running into inf

# changelog 544
# Output both the Z-score and CC-score

# changelog 544_1
# wrap the main as a seperate function.

# changelog 546
# sizeof final_result  is now num_proj*num_psi*3.

# changelog 550
# output all the result that Z-score > 4 (p>0.999968)

# new version 600
# In this version, the search area will be controlled only near the given rot/tilt/psi-s .
# Similar to RELION local (finer) refinement, a range should be given.
# TODO: pre-sorting images by their defoci. 
# In CTF_cupy, each ctf costs 0.006~0.01s to be generated. That's OK.
# Idea: first search for the near orientation samples. Then stack the particles and do the search at one time. 
# Warning!! This v600 only localized the psi search. rot/tilt localization will be imported in v601

# changelog 601
# Import search rot/tilt localized.

# changelog 6011
# Test, change rotate_fourier_image and translation_twoD_image to CPU, if it's faster.
# No, it's 20% slower in the RTX 4090. Revert to v601

# TODO: change CTF. In the local search, each particle only takes 0.035s to run. So the CTF 0.01s is a big problem.
# Current power usage ~ 140w. So we have to re-optimize all the codes. Start tomorrow.

# changelog 602
# Read in the star and looking for the local orientations is very time consuming (33s for 27600 particles). Now only search for the first time, then store it into a pickle file.

# changelog 603
# Try to make this code faster by removing the chunk and block in local search. FAILED!!!!!! 2x slower.
# So chunks must always be used. Adjusted the Chunk size A from 32 to 16, ~15% faster in RTX4090
# Now let's change the functions.
# Change the chop_local_search_SN_Function function. It saves 0.02seconds/particle.
# Change the frequency of doing cp.get_default_memory_pool().free_all_blocks().
# In local search, calling cp.get_default_memory_pool().free_all_blocks() costs around 0.003seconds.

# changelog 6031
# Don't display the time during search.

# changelog 6032
# in line 850
# only if we run a continue, then out=open(output,"a") 
# preventing from writing the same file over and over again.

# changelog 6033
# in function rotate_fourier_image
# only call for cupy at the very end, leading to 13% faster.
# discard function remove_spots_with_neighbors
# No output Lx anymore (Not working actually)

# changelog 6034
# Remove some arrays that won't be used.

# changelog 6035
# Remove cp.copy on process_chunk
# --FSC is not mandatory for now.

# changelog 604
# Add split the input particle starfile into multiple section, so it will run in parallel. (It's even slower than single threading. Abandoned)
# Add --end to set the end of the search range. Default = -1, search all the starfile.
# The idea now is to run two this program in the same GPU, initializing from func.py. 10% faster on RTX 5070 Ti (Abandoned, too complicated and only 10% faster)

# changelog 6041
# Write the output when 96 particles are written.
# Always generate Full localsearch .pkl to prevent bug.

# changelog 6042
# set --psistep to float

# changelog 60421
# some changes from chatGPT.

# changelog 60422
# change rotate_fourier_image, now it's 10% faster in small local range search.

# changelog 605
# Now change the project3D into this. So we don't need to write the projection stack.
DTYPE_for_finalresult=cp.float32

def report_gpu_mem(tag=""):
	torch.cuda.synchronize()

	dev = torch.cuda.current_device()

	t_alloc = torch.cuda.memory_allocated(dev)
	t_resv  = torch.cuda.memory_reserved(dev)

	mp = cp.get_default_memory_pool()
	c_used = mp.used_bytes()
	c_total = mp.total_bytes()

	print(f"\n===== {tag} =====")
	print(f"PyTorch allocated : {t_alloc / 1024**2:.1f} MB")
	print(f"PyTorch reserved  : {t_resv  / 1024**2:.1f} MB")
	print(f"CuPy used         : {c_used  / 1024**2:.1f} MB")
	print(f"CuPy total(pool)  : {c_total / 1024**2:.1f} MB")
def get_current_memory_usage():
	device = cp.cuda.Device()
	mem_info = device.mem_info
	used_mem = mem_info[1] - mem_info[0]
	return used_mem / (1024 ** 2)
	# return in MB
def remove_spots_with_neighbors(array):
	# this function replace the hot/cold spots into the average of the 4 nearest values.
#	array1=cp.copy(array)
	mean = cp.mean(array)
	std_dev = cp.std(array)
	# the threshold is 6 s.d.
	threshold_high = mean + 6 * std_dev
	threshold_low = mean - 6 * std_dev

	hot_spots = (array > threshold_high)
	cold_spots = (array < threshold_low)

	mask = hot_spots | cold_spots
	padded_array = cp.pad(array, pad_width=1, mode='reflect')
	for i, j in cp.argwhere(mask):
		array[i, j] = (padded_array[i, j+1]+padded_array[i+2, j+1]+padded_array[i+1, j]+padded_array[i+1, j+2])/4.
	return array

def gen_soft_maskout_radius(array, radius,edge_width):
	# Calculate the coordinates of the center
	center = np.array(array.shape) // 2

	# Create a meshgrid of coordinates
	y, x = np.indices(array.shape)

	# Calculate the Euclidean distance from the center
	distances = np.sqrt((x - center[1])**2 + (y - center[0])**2)

	mask = np.zeros_like(distances)
	mask[distances <= radius] = 1
	edge_region = (distances > radius) & (distances <= radius + edge_width)
	mask[edge_region] = 0.5 * (1 + np.cos(np.pi * (distances[edge_region] - radius) / edge_width))

	# Select elements within the radius
#	array = array * mask
	return mask
def expand_to_2d(arr_1d, size):
	center = size // 2
	arr_2d = np.zeros((size, size), dtype=arr_1d.dtype)
	for i in range(size):
		for j in range(size):
			radius = int(np.sqrt((i - center) ** 2 + (j - center) ** 2))
			if radius < len(arr_1d):
				arr_2d[i, j] = arr_1d[radius]
			else:
				arr_2d[i, j] = 0.0
	return arr_2d

def translation_twoD_image(image,trans_X,trans_Y):
	# transX/Y are all in pixels. Ingore the decimal part.
	# image should be cp array and in real space.
	# crop the density outside after trans.
	trans_x=int(np.floor(trans_X+0.5))
	trans_y=int(np.floor(trans_Y+0.5))
	ysize, xsize = image.shape

	# Determine the start and end indices for slicing
	start_x = max(0, trans_x)
	end_x = xsize if trans_x >= 0 else xsize + trans_x
	start_y = max(0, trans_y)
	end_y = ysize if trans_y >= 0 else ysize + trans_y

	# Initialize the translated image with zeros
	translated_image = cp.zeros_like(image)

	# Fill the translated image with the appropriate slice of the original image
	translated_image[start_y:end_y, start_x:end_x] = image[max(0, -trans_y):ysize - max(0, trans_y), max(0, -trans_x):xsize - max(0, trans_x)]

	return translated_image

def rotate_fourier_image(image, angle):
	# cache: key = (shape0, shape1, rounded_angle)
	if not hasattr(rotate_fourier_image, "_cache"):
		rotate_fourier_image._cache = {}

	# prevent cache miss
	angle_key = float(np.round(angle, 6))
	key = (int(image.shape[0]), int(image.shape[1]), angle_key)

	cached = rotate_fourier_image._cache.get(key, None)
	if cached is None:
		theta = np.deg2rad(angle_key)
		c = np.float32(np.cos(theta))
		s = np.float32(np.sin(theta))

		mat = cp.asarray([[c, -s],
						  [s,  c]], dtype=cp.float32)

		center = np.asarray(image.shape, dtype=np.float32) / 2.0
		rot_np = np.array([[c, -s],
						   [s,  c]], dtype=np.float32)
		offset = center - rot_np @ center

		cached = (mat, offset)
		rotate_fourier_image._cache[key] = cached

	mat, offset = cached
	return affine_transform(image, mat, offset=offset, order=1)
# TODO 05/14. Setup and check CTF. (Done)
# TODO: test correlation. (Done) Skip MSE, since normalization is very hard to do.
# TODO: test weighting function. (Done)
def input_FT_write_Real_debug_map(filename,FT_array):
	FT_array=cp.fft.fftshift(FT_array)
	FT_array=cp.fft.ifft2(FT_array).real
#	Name=str(image_serial_int)+"_at_"+real_filename
	with mrcfile.new(filename,overwrite=True) as debug_mrc:
		debug_mrc.set_data((FT_array.detach().cpu().numpy()).astype(np.float32))
	debug_mrc.close()

# Full_Particle_Star_Info_To_Search_3D_Stack_SN = preread_particle_star_and_sorting(pstar_line,Real_Start_Line,Real_End_Line,p_ROT_index,p_TILT_index,Model_Rot,Model_Tilt,local_search_angular_distance)
def preread_particle_star_and_sorting(pstar_line, p_mline, START_LINE, p_ROT_index, p_TILT_index, Model_Rot, Model_Tilt, local_search_angular_distance):
	# Time spent: 27000 particles, 3deg, thres30deg, in 33seconds.
	# Initialize START_LINE if needed
	if START_LINE < 0:
		START_LINE = p_mline

	# Initialize lists
	Full_Particle_Star_Info_SN = []
	Full_Particle_Star_Info_To_Search_3D_Stack_SN = []

	# Collect rotation and tilt data
	Full_Particle_Star_Info_ROT = []
	Full_Particle_Star_Info_TILT = []

	for l in range(START_LINE, len(pstar_line)):
		tokens = pstar_line[l].split()
		if tokens:
			p_ROT = float(tokens[p_ROT_index])
			p_TILT = float(tokens[p_TILT_index])
			Full_Particle_Star_Info_SN.append(l)
			Full_Particle_Star_Info_ROT.append(p_ROT)
			Full_Particle_Star_Info_TILT.append(p_TILT)
			Full_Particle_Star_Info_To_Search_3D_Stack_SN.append([l])

	# Convert collected rotations and tilts to NumPy arrays
	Full_Particle_Star_Info_ROT = np.array(Full_Particle_Star_Info_ROT)
	Full_Particle_Star_Info_TILT = np.array(Full_Particle_Star_Info_TILT)

	# Convert Model_Rot and Model_Tilt to NumPy arrays
	Model_Rot_V = np.array(Model_Rot)
	Model_Tilt_V = np.array(Model_Tilt)

	# Loop through each particle entry
	expanded_lsad = 1.*local_search_angular_distance
	for n in range(len(Full_Particle_Star_Info_ROT)):
		# Get current particle rotation and tilt
		p_rot = Full_Particle_Star_Info_ROT[n]
		p_tilt = Full_Particle_Star_Info_TILT[n]
		
		# Vectorized criteria for filtering within the local search angular distance
		rot_diff = np.fabs((Model_Rot_V - p_rot + 180) % 360 - 180)
		tilt_diff = np.fabs((Model_Tilt_V - p_tilt + 180) % 360 - 180)
		
		# Create a boolean mask where both criteria are met
		valid_indices = np.where((rot_diff <= expanded_lsad) & (tilt_diff <= expanded_lsad))[0]
		
		# Filter valid Model_Rot and Model_Tilt based on the mask
		valid_rot = Model_Rot_V[valid_indices]
		valid_tilt = Model_Tilt_V[valid_indices]
		
		# Calculate distances only for filtered indices
		for i, m in enumerate(valid_indices):
			distance = calculateAngularDistance(valid_rot[i], valid_tilt[i], 0.0, p_rot, p_tilt, 0.0)
			if np.fabs(distance) <= local_search_angular_distance:
				Full_Particle_Star_Info_To_Search_3D_Stack_SN[n].append(m)

	return Full_Particle_Star_Info_To_Search_3D_Stack_SN
	# So Full_Particle_Star_Info_To_Search_3D_Stack_SN[n][m] contains the Orientations to be searched.
	# Now we only need to read this list, and slicing the 3D_array. However, the performance must be re-checked.
# 06/06/2024: try to fix the translation search problem. now projection does not store the .conj
def read_Series_Particle_NUM(pstar_line,Real_Start_Line,Real_End_Line,p_IMG_index,p_ROT_index,p_TILT_index,p_PSI_index,p_DFU_index,p_DFV_index, \
	p_DFA_index,TRANS_MAX,ORIGINAL_BOXSIZE,NEW_BOXSIZE,XSIZE,YSIZE,ccss,VOLTAGE,do_discard_mask,gpu_mask,NEW_APIX,kk,time0,p_mline,PSI_STEP,do_local_search,local_search_angular_distance,\
	last_filename,last_data,last_mrc):

	NUM_PSI_ROTATION = int(360 // PSI_STEP)
	if NUM_PSI_ROTATION < 1:
		NUM_PSI_ROTATION = 1

	TMPP = 999999
	if do_local_search:
		TMPP = local_search_angular_distance // PSI_STEP
		if TMPP * 2 + 1 < NUM_PSI_ROTATION:
			NUM_PSI_ROTATION = int(TMPP * 2 + 1)

	rfft_NEW_BOXSIZE = NEW_BOXSIZE // 2 + 1

	if Real_Start_Line < p_mline:
		Real_Start_Line = p_mline
	if Real_End_Line > len(pstar_line):
		Real_End_Line = len(pstar_line)

	alloc_size = int((Real_End_Line - Real_Start_Line) * NUM_PSI_ROTATION) + 1
	particle_array = cp.zeros((alloc_size, NEW_BOXSIZE, rfft_NEW_BOXSIZE), dtype=cp.complex64)

	particle_ROT = []
	particle_TILT = []
	particle_PSI = []
	particle_LINE = []
	particle_T_X = []
	particle_T_Y = []
	particle_fake_PSI = []
	particle_norm_array = cp.array([0.], dtype=cp.float32)

	FULL_SIZE = 1

	# local bindings
	cp_asarray = cp.asarray
	cp_fft2 = cp.fft.fft2
	cp_fftshift = cp.fft.fftshift
	cp_conj = cp.conj
	cp_linalg_norm = cp.linalg.norm
	rotate_func = rotate_fourier_image
	get_orig = get_original_filenames
	CTF_cls = CTF

	# crop indices
	if NEW_BOXSIZE != ORIGINAL_BOXSIZE:
		crop_l = ORIGINAL_BOXSIZE // 2 - NEW_BOXSIZE // 2
		crop_r = ORIGINAL_BOXSIZE // 2 + NEW_BOXSIZE // 2
	else:
		crop_l = None
		crop_r = None

	# small cache for repeated CTF params
	# key: (dfu, dfv, dfa) or ("noctf",)
	ctf_cache = {}

	for l in range(Real_Start_Line, Real_End_Line):
		tokens = pstar_line[l].split()
		if not tokens:
			continue

		p_PSI = 0.0
		p_image_name1 = tokens[p_IMG_index]
		image_serial_str, p_filename1 = p_image_name1.split('@', 1)
		p_image_serial_int = int(image_serial_str) - 1
		p_real_filename = get_orig(file_path=p_filename1)

		p_ROT = float(tokens[p_ROT_index])
		p_TILT = float(tokens[p_TILT_index])

		if p_PSI_index != -1:
			p_PSI = float(tokens[p_PSI_index])

		if p_DFU_index != -1:
			p_DFU = float(tokens[p_DFU_index])
			p_DFV = float(tokens[p_DFV_index])
			p_DFA = float(tokens[p_DFA_index])

		# read mrc/mrcs
		if p_real_filename != last_filename:
			if last_mrc is not None:
				last_mrc.close()
			last_mrc = mrcfile.mmap(p_real_filename, mode='r')
			last_data = last_mrc.data
			last_filename = p_real_filename

		image2 = last_data

		if len(image2.shape) < 3:
			p_XSIZE = image2.shape[1]
			p_YSIZE = image2.shape[0]
			p_image1_slice_i = image2
		else:
			p_XSIZE = image2.shape[2]
			p_YSIZE = image2.shape[1]
			p_image1_slice_i = image2[p_image_serial_int, :, :]

		if p_XSIZE != XSIZE or p_YSIZE != YSIZE:
			print("X/YSIZE of models and particles mismatch!")
			break

		trans_X = 0
		trans_Y = 0

		# CTF
		if p_DFU_index == -1:
			ctf_key = ("noctf",)
		else:
			# round a bit to make cache more useful while still safe
			ctf_key = (round(p_DFU, 6), round(p_DFV, 6), round(p_DFA, 6))

		gpu_ctf_image = ctf_cache.get(ctf_key, None)
		if gpu_ctf_image is None:
			if p_DFU_index == -1:
				ctf1 = CTF_cls(defocus_U=10000.0, defocus_V=5000.0, defocus_A=30.0,
							   CS=2.7, voltage=300.0, pixel_size=NEW_APIX,
							   XSIZE=NEW_BOXSIZE, YSIZE=NEW_BOXSIZE)
				ctf_image = np.ones((NEW_BOXSIZE, NEW_BOXSIZE), dtype=np.complex64)
			else:
				ctf1 = CTF_cls(defocus_U=p_DFU, defocus_V=p_DFV, defocus_A=p_DFA,
							   CS=ccss, voltage=VOLTAGE, pixel_size=NEW_APIX,
							   XSIZE=NEW_BOXSIZE, YSIZE=NEW_BOXSIZE)
				ctf_image = ctf1.getFftwImage_with_isSPA_weight(False, False, False, False, kk=kk)

			gpu_ctf_image = cp_asarray(ctf_image, dtype=cp.complex64)
			ctf_cache[ctf_key] = gpu_ctf_image

		gpu_image2_ori_bak = cp.asarray(p_image1_slice_i, dtype=cp.float32)
		gpu_image2_ori_bak *= -1.0

		REM_NEAREST_PSI = round(((p_PSI + 360.0) % 360.0) / PSI_STEP) * PSI_STEP
		REM_NEAREST_PSI %= 360.0

		for psi in range(int(NUM_PSI_ROTATION)):
			particle_ROT.append(p_ROT)
			particle_TILT.append(p_TILT)
			particle_PSI.append((p_PSI + 360.0) % 360.0)
			particle_LINE.append(l)
			particle_T_X.append(trans_X)
			particle_T_Y.append(trans_Y)

			psi_rot_degree = PSI_STEP * psi
			if do_local_search:
				psi_rot_degree = REM_NEAREST_PSI - TMPP * PSI_STEP + psi * PSI_STEP

			particle_fake_PSI.append(psi_rot_degree)

			if abs(psi_rot_degree) < 0.5:
				psi_rot_degree = 0.5

			rotated = rotate_func(gpu_image2_ori_bak, psi_rot_degree)

			if not do_discard_mask:
				rotated = rotated * gpu_mask

			fft2 = cp_fft2(rotated)
			fft2_shift = cp_fftshift(fft2)

			if crop_l is not None:
				fft2_shift = fft2_shift[crop_l:crop_r, crop_l:crop_r]

			Corr_fft2_shift = fft2_shift * gpu_ctf_image

			# fft2 -> rfft2 packed form
			Corr_fft2_shift = cp_fftshift(Corr_fft2_shift)
			Corr_fft2_shift = Corr_fft2_shift[:, :Corr_fft2_shift.shape[1] // 2 + 1]

			Corr_fft2_shift_conj = cp_conj(Corr_fft2_shift)
			norm_val = cp_linalg_norm(Corr_fft2_shift_conj)
			Corr_fft2_shift_conj = Corr_fft2_shift_conj / norm_val

			particle_array[FULL_SIZE, :, :] = Corr_fft2_shift_conj
			FULL_SIZE += 1

#	cp.cuda.Stream.null.synchronize()

	if not do_local_search:
		cp.get_default_memory_pool().free_all_blocks()

	return (particle_array, particle_norm_array, particle_ROT, particle_TILT,
			particle_PSI, particle_T_X, particle_T_Y, particle_LINE,
			particle_fake_PSI, NUM_PSI_ROTATION, last_filename, last_data,last_mrc)
	# Now len(particle_ROT)=particle_array.shape[0]-1, since the particle_array[0,:,:] is all zero. 
def process_chunk_old1(chunkA, chunkB):
	# Broadcasting and vectorized multiplication

	result = chunkA[:, cp.newaxis, :, :] * chunkB[cp.newaxis, :, :, :]

	# Apply fftshift to each slice in the 4D tensor
	shifted_result = cp.fft.fftshift(result, axes=(-2, -1))

	# Perform the inverse Fourier transform on each shifted slice in a batched manner
	ifft_result = cp.fft.irfft2(shifted_result, axes=(-2, -1))
	ifft_result_shifted = cp.fft.fftshift(ifft_result, axes=(-2, -1))
	## Should cp.fft.fftshift(ifft_result, axes=(-2, -1)) be applied??????
	# Convert the result back to real values if needed
	ifft_result_real = cp.real(ifft_result_shifted)
	del result
	del shifted_result
	del ifft_result
	del ifft_result_shifted
	return ifft_result_real
def process_chunk(chunkA, chunkB):
	# [na, 1, y, x] * [1, nb, y, x] -> [na, nb, y, x]
	result = chunkA[:, cp.newaxis, :, :] * chunkB[cp.newaxis, :, :, :]

	# 保持你原先流程不变
	result = cp.fft.fftshift(result, axes=(-2, -1))
	result = cp.fft.irfft2(result, axes=(-2, -1))
	result = cp.fft.fftshift(result, axes=(-2, -1))

	return result
def find_max_and_position_old0(Arr):
	# Get the shape
	ni, nj, nx, ny = Arr.shape

	# Reshape the array to [ni, nj, nx * ny]
	reshaped_arr = Arr.reshape(ni, nj, nx * ny)

	# Find the maximum values and their flattened positions in the reshaped array
	max_values = cp.max(reshaped_arr, axis=2)
	max_indices = cp.argmax(reshaped_arr, axis=2)

	# Convert the flattened indices back to the (nx, ny) grid
	max_pos_nx = max_indices // ny  # Integer division to get the nx positions
	max_pos_ny = max_indices % ny   # Modulus to get the ny positions

	# Create the RESULT array with shape [ni, nj, 3]
	RESULT = cp.zeros((ni, nj, 3), dtype=cp.float32)

	# Store the max values, nx positions, and ny positions
	RESULT[:, :, 0] = cp.array(max_values,dtype=cp.float32)
	RESULT[:, :, 1] = cp.array(max_pos_nx,dtype=cp.float32)
	RESULT[:, :, 2] = cp.array(max_pos_ny,dtype=cp.float32)
	return RESULT
def find_max_and_position(Arr):
	ni, nj, nx, ny = Arr.shape
	reshaped_arr = Arr.reshape(ni, nj, nx * ny)

	max_values = cp.max(reshaped_arr, axis=2)
	max_indices = cp.argmax(reshaped_arr, axis=2)

	RESULT = cp.empty((ni, nj, 3), dtype=cp.float32)
	RESULT[:, :, 0] = max_values.to(dtype=cp.float32)
	RESULT[:, :, 1] = (max_indices // ny).to(dtype=cp.float32)
	RESULT[:, :, 2] = (max_indices % ny).to(dtype=cp.float32)
	return RESULT
def chop_3D_array(three_d_array,Model_Rot,Model_Tilt,one_particle_local_search_SN):
#	Chopped_3D_Array = cp.array([three_d_array[j + 1] for j in one_particle_local_search_SN])
#	ChatGPT suggested.
	idx = cp.asarray(one_particle_local_search_SN, dtype=cp.int32) + 1
	Chopped_3D_Array = three_d_array[idx]
	Chopped_Model_Rot = [Model_Rot[j] for j in one_particle_local_search_SN]
	Chopped_Model_Tilt = [Model_Tilt[j] for j in one_particle_local_search_SN]
	return Chopped_3D_Array,Chopped_Model_Rot,Chopped_Model_Tilt
def chop_particle_array(particle_array,particle_ROT,particle_TILT,particle_PSI,particle_T_X,particle_T_Y,particle_LINE,particle_fake_PSI,N,In_Plane_Rotation_Sample_Size):
	START=N*In_Plane_Rotation_Sample_Size
	END=(N+1)*In_Plane_Rotation_Sample_Size
	Chopped_Particle_Array = particle_array[START+1:END+1]
	Chopped_Particle_Rot=particle_ROT[START:END]
	Chopped_Particle_Tilt=particle_TILT[START:END]
	Chopped_Particle_Psi=particle_PSI[START:END]
	Chopped_Particle_T_X=particle_T_X[START:END]
	Chopped_Particle_T_Y=particle_T_Y[START:END]
	Chopped_Particle_Fake_PSI=particle_fake_PSI[START:END]
	Chopped_Particle_LINE = particle_LINE[START:END]
	return Chopped_Particle_Array, Chopped_Particle_Rot, Chopped_Particle_Tilt, Chopped_Particle_Psi,Chopped_Particle_T_X, Chopped_Particle_T_Y,Chopped_Particle_Fake_PSI,Chopped_Particle_LINE
def search_locally_one_by_one(particle_array,three_d_array,Model_Rot,Model_Tilt,CC_THRESHOLD,ACCURACY,TRANS_SCALE_FACTOR,time0,\
	particle_ROT,particle_TILT,particle_PSI,particle_T_X,particle_T_Y,particle_LINE,particle_fake_PSI,chopped_local_search_SN,\
	NUM_PSI_ROTATION,output_Z_L5,chunk_size_A,chunk_size_B,cropsize,MULT,crop_low,crop_up,NUMBER_OF_TOPS):
	timexx=time.time()
	RESULTS=[]
	chunk_size_A = 16
	chunk_size_B = 32
	# 1. get the number of particles to be search.
	set_particle_LINE=sorted(list(set(particle_LINE)))
	# Warning!!!! set(particle_LINE) is in RANDOM ORDER!!!!!!!!!!!!
	NUMBER_OF_PARTICLES=len(set_particle_LINE)
	# In my codes, the in-plane rotation sample points are consistant. So we can simply divide particle_array.shape[0]-1 to NUMBER_OF_PARTICLES to get the psi sample size.
	In_Plane_Rotation_Sample_Size=NUM_PSI_ROTATION
	if(NUMBER_OF_PARTICLES>0):
		In_Plane_Rotation_Sample_Size= int((particle_array.shape[0]-1)//NUMBER_OF_PARTICLES)
	if(In_Plane_Rotation_Sample_Size!=NUM_PSI_ROTATION):
		print("BUG, In_Plane_Rotation_Sample_Size!=NUM_PSI_ROTATION. In_Plane_Rotation_Sample_Size NUM_PSI_ROTATION =",In_Plane_Rotation_Sample_Size,NUM_PSI_ROTATION)
	for N in range(NUMBER_OF_PARTICLES):

		if(set_particle_LINE[N]!=chopped_local_search_SN[N][0]):
			print("BUG, set_particle_LINE[N]!=chopped_local_search_SN[N][0]. Return an empty result.")
			print("debug,particle_LINE=",particle_LINE)
			print("debug,set_particle_LINE=",set_particle_LINE)
			print("debug,NUMBER_OF_PARTICLES=",NUMBER_OF_PARTICLES)
			print("debug,chopped_local_search_SN[N][0]=",chopped_local_search_SN[N][0])
			return RESULTS
		one_particle_local_search_SN=chopped_local_search_SN[N][1:]
		Chopped_3D_Array, Chopped_Model_Rot, Chopped_Model_Tilt = chop_3D_array(three_d_array,Model_Rot,Model_Tilt,one_particle_local_search_SN)

		Chopped_Particle_Array, Chopped_Particle_Rot, Chopped_Particle_Tilt, Chopped_Particle_Psi,Chopped_Particle_T_X, Chopped_Particle_T_Y,Chopped_Particle_Fake_PSI,Chopped_Particle_LINE\
		= chop_particle_array(particle_array,particle_ROT,particle_TILT,particle_PSI,particle_T_X,particle_T_Y,particle_LINE,particle_fake_PSI,N,In_Plane_Rotation_Sample_Size)
		result_shape = (Chopped_3D_Array.shape[0], In_Plane_Rotation_Sample_Size, 3)
		final_result = cp.zeros(result_shape, dtype=cp.float32)
		for i in range(0, Chopped_3D_Array.shape[0], chunk_size_A):
			endA = i + chunk_size_A
			if(endA>Chopped_3D_Array.shape[0]):
				endA=Chopped_3D_Array.shape[0]
			for j in range(0, Chopped_Particle_Array.shape[0], chunk_size_B):
				endB = j + chunk_size_B
				if(endB>Chopped_Particle_Array.shape[0]):
					endB=Chopped_Particle_Array.shape[0]
				chunkA = Chopped_3D_Array[i:endA, :, :]
				chunkB = Chopped_Particle_Array[j:endB, :, :]	
				result_chunk = process_chunk(chunkA, chunkB)
				Result_chunk_processed = MULT*result_chunk[:,:,crop_low:crop_up,crop_low:crop_up]
				final_result[i:endA, j:endB,:] = find_max_and_position(Result_chunk_processed)
				del result_chunk,Result_chunk_processed
	#	cp.cuda.Stream.null.synchronize()

		time1=time.time()
		SCORE_MAX=-9999.0
		REM_PSI=-9999999.
		REM_I=-1
		REM_XSHIFT=-9999.
		REM_YSHIFT=-9999.
		REM_FAKE_PSI=-1
		J_lower_limit=0
		J_upper_limit=NUM_PSI_ROTATION
		# Only one particle here.
		cut_FR=final_result

		reshaped_result = cut_FR.reshape(-1, final_result.shape[2])

		# Compute the maximum values along the last two axes
		max_values = cp.array(reshaped_result[:,0])
		try:
			peak_value = cp.max(max_values)
		except:
			return RESULTS
		indices_where = cp.where(cut_FR[:,:,0] == peak_value)
		indices = (indices_where[0], indices_where[1])
		peak_position = (cut_FR[int(indices[0][0]), int(indices[1][0]),1], cut_FR[int(indices[0][0]), int(indices[1][0]),2])
		REM_I=int(indices[0][0])

		REM_XSHIFT=float(cropsize//2-peak_position[1])/TRANS_SCALE_FACTOR
		REM_YSHIFT=float(cropsize//2-peak_position[0])/TRANS_SCALE_FACTOR
		REM_FAKE_PSI=int(indices[1][0])+J_lower_limit
		Angle_distance_CC2RELION=calculateAngularDistance(Chopped_Model_Rot[REM_I],Chopped_Model_Tilt[REM_I],0.0,Chopped_Particle_Rot[J_lower_limit], Chopped_Particle_Tilt[J_lower_limit],0.0)
		max_values=cp.asarray(max_values,dtype=cp.float32)
		CC_MEAN=cp.mean(max_values)
		CC_SIGMA=cp.std(max_values)
		CC_T1=(peak_value-CC_MEAN)/CC_SIGMA
		CC_L3=0
		if(CC_T1>CC_THRESHOLD):
			CC_L3=1
		CC_DIS_ACCU=0
		if(Angle_distance_CC2RELION<ACCURACY):
			CC_DIS_ACCU=1
		RESULTS.append(["cc: ",Chopped_Particle_LINE[J_lower_limit],float(CC_T1),CC_L3,CC_DIS_ACCU,Chopped_Model_Rot[REM_I],Chopped_Model_Tilt[REM_I],Angle_distance_CC2RELION,round(REM_XSHIFT),round(REM_YSHIFT), Chopped_Particle_T_X[J_lower_limit],Chopped_Particle_T_Y[J_lower_limit]])
		RESULTS.append(["psi: ",Chopped_Particle_LINE[J_lower_limit],float(Chopped_Particle_Fake_PSI[REM_FAKE_PSI]),Chopped_Particle_Psi[J_lower_limit]])
		RESULTS.append(["rawCC: ",Chopped_Particle_LINE[J_lower_limit],float(peak_value)])
		if(output_Z_L5):
			cut_FR[int(indices[0][0]), int(indices[1][0]),0]=-9999.0
			
			for ff in range(NUMBER_OF_TOPS):
				cp_cut_FR=cp.copy(cut_FR)
				cp_reshaped_result = cp_cut_FR.reshape(-1, final_result.shape[2])
				# Compute the maximum values along the last two axes
				cp_max_values = cp.array(cp_reshaped_result[:,0])
				cp_peak_value = cp.max(cp_max_values)
				cp_indices_where = cp.where(cp_cut_FR[:,:,0] == cp_peak_value)
				
				cp_indices = (cp_indices_where[0], cp_indices_where[1])
				cp_peak_position = (cut_FR[int(cp_indices[0][0]), int(cp_indices[1][0]),1], cut_FR[int(cp_indices[0][0]), int(cp_indices[1][0]),2])
				cp_REM_I=int(cp_indices[0][0])

				cp_REM_XSHIFT=float(cropsize//2-cp_peak_position[1])/TRANS_SCALE_FACTOR
				cp_REM_YSHIFT=float(cropsize//2-cp_peak_position[0])/TRANS_SCALE_FACTOR

				cp_REM_FAKE_PSI=int(cp_indices[1][0])+J_lower_limit
				cp_CC_T1=(cp_peak_value-CC_MEAN)/CC_SIGMA
				cp_Angle_distance_CC2RELION=calculateAngularDistance(Chopped_Model_Rot[cp_REM_I],Chopped_Model_Tilt[cp_REM_I],0.0,Chopped_Particle_Rot[J_lower_limit], Chopped_Particle_Tilt[J_lower_limit],0.0)
				PreFix="L"+str(ff+1)
				RESULTS.append([PreFix,"cc: ",Chopped_Particle_LINE[J_lower_limit],float(cp_CC_T1),CC_L3,CC_DIS_ACCU,Model_Rot[cp_REM_I],Model_Tilt[cp_REM_I],cp_Angle_distance_CC2RELION,round(cp_REM_XSHIFT),round(cp_REM_YSHIFT),particle_T_X[J_lower_limit],particle_T_Y[J_lower_limit]])
				RESULTS.append([PreFix,"psi: ",Chopped_Particle_LINE[J_lower_limit],float(Chopped_Particle_Fake_PSI[cp_REM_FAKE_PSI]),Chopped_Particle_Psi[J_lower_limit]])
				RESULTS.append([PreFix,"rawCC: ",Chopped_Particle_LINE[J_lower_limit],float(cp_peak_value)])
				cut_FR[int(cp_indices[0][0]), int(cp_indices[1][0]),0]=-9999.0
			del cp_cut_FR
		del final_result
	cp.cuda.Stream.null.synchronize()
	time2=time.time()
	
	return RESULTS
def search_stack_of_particles(particle_array,particle_norm_array,three_d_array,three_d_norm_array,Model_Rot,Model_Tilt,Model_Psi,out,CC_THRESHOLD,ACCURACY,TRANS_SCALE_FACTOR,output,time0, \
	particle_ROT,particle_TILT,particle_PSI,particle_T_X,particle_T_Y,particle_LINE,NEW_BOXSIZE,particle_fake_PSI,NUM_PSI_ROTATION, \
	do_local_search,chopped_local_search_SN):
	NUM_PSI_ROTATION=int(NUM_PSI_ROTATION)
	RESULTS=[]
	time0=time.time()
	output_Z_L5 = False
	NUMBER_OF_TOPS=4
	# when output_Z_L5 = True, output top NUMBER_OF_TOPS +1 orients and shifts.
	chunk_size_A = 32
	chunk_size_B = 32
#	cropsize=96	# for refine model
	cropsize=32	# for refine map
	# In this version, time consumption of cropsize 96 ~ that of 128
	MULT = cp.asarray(1024.*NEW_BOXSIZE*NEW_BOXSIZE, dtype=cp.float32)
	crop_low=NEW_BOXSIZE//2-cropsize//2
	crop_up=NEW_BOXSIZE//2+cropsize//2
	if(do_local_search):
	#	search one by one. Since the search samples can be different by each particle.
		RESULTS = search_locally_one_by_one(particle_array,three_d_array,Model_Rot,Model_Tilt,CC_THRESHOLD,ACCURACY,TRANS_SCALE_FACTOR,time0,\
		particle_ROT,particle_TILT,particle_PSI,particle_T_X,particle_T_Y,particle_LINE,particle_fake_PSI,chopped_local_search_SN,\
		NUM_PSI_ROTATION,output_Z_L5,chunk_size_A,chunk_size_B,cropsize,MULT,crop_low,crop_up,NUMBER_OF_TOPS)
		return RESULTS
	else:
		# do global search. Can do it all in once.
		result_shape = (three_d_array.shape[0]-1, particle_array.shape[0]-1, 3)
		final_result = cp.zeros(result_shape, dtype=cp.float32)
		for i in range(1, three_d_array.shape[0], chunk_size_A):
			endA = i + chunk_size_A
			if(endA>three_d_array.shape[0]):
				endA=three_d_array.shape[0]
			for j in range(1, particle_array.shape[0], chunk_size_B):
				endB = j + chunk_size_B
				if(endB>particle_array.shape[0]):
					endB=particle_array.shape[0]
				chunkA = three_d_array[i:endA, :, :]
				chunkB = particle_array[j:endB, :, :]	
				result_chunk = process_chunk(chunkA, chunkB)
				Result_chunk_processed = MULT*result_chunk[:,:,crop_low:crop_up,crop_low:crop_up]
				final_result[i-1:endA-1, j-1:endB-1,:] = find_max_and_position(Result_chunk_processed)
				del result_chunk,Result_chunk_processed

	#	cp.cuda.Stream.null.synchronize()
		cp.get_default_memory_pool().free_all_blocks()

		for k in range(final_result.shape[1]//NUM_PSI_ROTATION):
			time1=time.time()
			SCORE_MAX=-9999.0
			REM_PSI=-9999999.
			REM_I=-1
			REM_XSHIFT=-9999.
			REM_YSHIFT=-9999.
			REM_FAKE_PSI=-1
			J_lower_limit=k*NUM_PSI_ROTATION
			J_upper_limit=k*NUM_PSI_ROTATION+NUM_PSI_ROTATION
			cut_FR=final_result[:, J_lower_limit:J_upper_limit,:]
			reshaped_result = cut_FR.reshape(-1, final_result.shape[2])
			# Compute the maximum values along the last two axes
			max_values = cp.array(reshaped_result[:,0])
			peak_value = cp.max(max_values)
			indices_where = cp.where(cut_FR[:,:,0] == peak_value)
			
			indices = (indices_where[0], indices_where[1])
			peak_position = (cut_FR[int(indices[0][0]), int(indices[1][0]),1], cut_FR[int(indices[0][0]), int(indices[1][0]),2])
			REM_I=int(indices[0][0])

			REM_XSHIFT=float(cropsize//2-peak_position[1])/TRANS_SCALE_FACTOR
			REM_YSHIFT=float(cropsize//2-peak_position[0])/TRANS_SCALE_FACTOR
			#### Note: peak_position[1]=xshift, peak_position[0]=yshift
			####
			REM_FAKE_PSI=int(indices[1][0])+J_lower_limit
			Angle_distance_CC2RELION=calculateAngularDistance(Model_Rot[REM_I],Model_Tilt[REM_I],0.0,particle_ROT[J_lower_limit], particle_TILT[J_lower_limit],0.0)
			max_values=cp.asarray(max_values,dtype=cp.float32)
			CC_MEAN=cp.mean(max_values)
			CC_SIGMA=cp.std(max_values)
			
			CC_T1=(peak_value-CC_MEAN)/CC_SIGMA
			CC_L3=0
			if(CC_T1>CC_THRESHOLD):
				CC_L3=1
			CC_DIS_ACCU=0
			if(Angle_distance_CC2RELION<ACCURACY):
				CC_DIS_ACCU=1
			RESULTS.append(["cc: ",particle_LINE[J_lower_limit],float(CC_T1),CC_L3,CC_DIS_ACCU,Model_Rot[REM_I],Model_Tilt[REM_I],Angle_distance_CC2RELION,round(REM_XSHIFT),round(REM_YSHIFT),particle_T_X[J_lower_limit],particle_T_Y[J_lower_limit]])
			RESULTS.append(["psi: ",particle_LINE[J_lower_limit],float(particle_fake_PSI[REM_FAKE_PSI]),particle_PSI[J_lower_limit]])
			RESULTS.append(["rawCC: ",particle_LINE[J_lower_limit],float(peak_value)])
			if(output_Z_L5):
				cut_FR[int(indices[0][0]), int(indices[1][0]),0]=-9999.0
				# set the max to very negative.
				
				for ff in range(NUMBER_OF_TOPS):
					cp_cut_FR=cp.copy(cut_FR)
					cp_reshaped_result = cp_cut_FR.reshape(-1, final_result.shape[2])
					# Compute the maximum values along the last two axes
					cp_max_values = cp.array(cp_reshaped_result[:,0])
					cp_peak_value = cp.max(cp_max_values)
					cp_indices_where = cp.where(cp_cut_FR[:,:,0] == cp_peak_value)
					
					cp_indices = (cp_indices_where[0], cp_indices_where[1])
					cp_peak_position = (cut_FR[int(cp_indices[0][0]), int(cp_indices[1][0]),1], cut_FR[int(cp_indices[0][0]), int(cp_indices[1][0]),2])
					cp_REM_I=int(cp_indices[0][0])

					cp_REM_XSHIFT=float(cropsize//2-cp_peak_position[1])/TRANS_SCALE_FACTOR
					cp_REM_YSHIFT=float(cropsize//2-cp_peak_position[0])/TRANS_SCALE_FACTOR

					cp_REM_FAKE_PSI=int(cp_indices[1][0])+J_lower_limit
					cp_CC_T1=(cp_peak_value-CC_MEAN)/CC_SIGMA
					cp_Angle_distance_CC2RELION=calculateAngularDistance(Model_Rot[cp_REM_I],Model_Tilt[cp_REM_I],0.0,particle_ROT[J_lower_limit], particle_TILT[J_lower_limit],0.0)
					PreFix="L"+str(ff+1)
					RESULTS.append([PreFix,"cc: ",particle_LINE[J_lower_limit],float(cp_CC_T1),CC_L3,CC_DIS_ACCU,Model_Rot[cp_REM_I],Model_Tilt[cp_REM_I],cp_Angle_distance_CC2RELION,round(cp_REM_XSHIFT),round(cp_REM_YSHIFT),particle_T_X[J_lower_limit],particle_T_Y[J_lower_limit]])
					RESULTS.append([PreFix,"psi: ",particle_LINE[J_lower_limit],float(particle_fake_PSI[cp_REM_FAKE_PSI]),particle_PSI[J_lower_limit]])
					RESULTS.append([PreFix,"rawCC: ",particle_LINE[J_lower_limit],float(cp_peak_value)])
					cut_FR[int(cp_indices[0][0]), int(cp_indices[1][0]),0]=-9999.0
				del cp_cut_FR	
		cp.cuda.Stream.null.synchronize()
		time2=time.time()
		del final_result
	return RESULTS

#### Important Note below:
# If B=cp.fft.fft2(A), C=cp.fft.fftshift(B)
# you need to do C=cp.fft.fftshift(C) that shift back the zero freq to corner!
# then use D=C[:, :C.shape[1]//2 + 1] to convert fft2 array to rfft2 array.
# to translate back to real space, use arr=cp.fft.irfft2(D).real No need to use ifftshift here since zero-freq is in the corner.

## 2024/07/17: replace the final_result = cp.concatenate(results, axis=0). Because it consumes double the device memory.
## Now the maximum memory usage is sampling_point (825 for 5deg) * pre-read_rot_particles(8 * (360/8deg) in my program.) * cropsize*cropsize (96 in my program)* 4byte ~ 11Gb.
## Additional 2Gb should be added. So total of 13Gb will be consumed.
def create_SEARCH_parser_and_read():
	parser = argparse.ArgumentParser(description="GRID_SEARCH_CC_PR_WITH_TRANSLATION.")
	parser.add_argument("--i", type=str, required=True, help="Input model file")
	parser.add_argument("--p", type=str, required=True, help="Input particle file")
	parser.add_argument("--FSC", type=str, default = "", help="Input FSC file")
	parser.add_argument("--o", type=str, required=True, help="Output file")
	parser.add_argument("--kk", type=float, default=0., help="The kk value, default = 0")
	parser.add_argument("--gpuid", type=int, default=0, help="The specified GPU ID, default = 0")
	parser.add_argument("--oriboxsize", type=int, default=256, help="The original boxsize, default = 256 (pixel)")
	parser.add_argument("--newboxsize", type=int, default=256, help="The new boxsize, default = 256 (pixel)")
	parser.add_argument("--apix", type=float, default=1.42, help="The ORIGINAL pixel size, default = 1.42")
	parser.add_argument("--transRange", type=int, default=30, help="Translation search range in pixel, default = 30. When set to 0, no translation would be searched.")
	parser.add_argument("--voltage", type=float, default=300, help="The voltage in kV, default = 300")
	parser.add_argument("--cs", type=float, default=2.7, help="The cc in mm, default = 2.7")
	parser.add_argument("--maskRadius", type=int, default=110, help="The softmask radius in pixel, corresponding to original boxsize, default = 110")
	parser.add_argument("--maskEdge", type=int, default=6, help="The softmask edge width in pixel, default = 6")
	parser.add_argument("--ignoreFSC", action='store_true', help="For testing purpose, ignoring the FSC weight. default = False")
	parser.add_argument("--discardMask", action='store_true', help="For testing purpose, apply NO soft mask. default = False")
	parser.add_argument("--psiStep", type=float, default=15, help="The psi angle search step in deg, default = 15")
	parser.add_argument("--start", type=int, default=-1, help="The start line of the particle starfile, default = -1, meaning from top")
	parser.add_argument("--end", type=int, default=-1, help="The end of the starfile. -1 means the end of starfile Default = -1")
	parser.add_argument("--doLocalSearch", action='store_true', help="Only search for the local orientations. Should combine with --localRange. default = False")
	parser.add_argument("--localRange", type=float, default=20., help="The local search range in +- this degree. Also applies to psi search.")
	parser.add_argument("--doContinue", action='store_true', help="Do continue run, so the output file doesn't get overlapped. default = False")
	
	###
	parser.add_argument("--ang", type=str, required=True, help="Input angle star file")
	parser.add_argument("--mrc", type=str, required=True, help="Input 3D mrc file")
	
	return parser
def main():
#	(model_star, particle_star,FSCfile,output,kk,device_id) =  parse_command_line()
	## reading parameters
	parser = create_SEARCH_parser_and_read()
	args = parser.parse_args()
	run_search(args)
	
def run_search(args):
	IS_A_CONTINUE_RUN=False
	model_star = args.i
	particle_star = args.p
	FSCfile=args.FSC
	output=args.o
	kk=args.kk
	device_id=args.gpuid
	ORIGINAL_BOXSIZE=args.oriboxsize
	NEW_BOXSIZE=args.newboxsize
	ORI_APIX=args.apix
	TRANS_MAX=args.transRange
	VOLTAGE=args.voltage
	ccss=args.cs
	radius=args.maskRadius
	edge_width=args.maskEdge
	do_PR_calc=False
	do_ignore_FSC=args.ignoreFSC
	do_discard_mask=args.discardMask
	PSI_STEP=args.psiStep
	NUM_PSI_ROTATION=int(360//PSI_STEP)
	START_LINE=args.start
	END_LINE = args.end
	if(args.doContinue):
		IS_A_CONTINUE_RUN=True
	do_local_search=args.doLocalSearch
	local_search_angular_distance=args.localRange
	TMPP=999999
	if(do_local_search):
		TMPP=local_search_angular_distance//PSI_STEP
		if(TMPP*2+1<NUM_PSI_ROTATION):
			NUM_PSI_ROTATION=TMPP*2+1
	NUM_PSI_ROTATION=int(NUM_PSI_ROTATION)
	mrc_model = args.mrc
	angle_starfile = args.ang
#	print (args)
	
	## end reading parameters
	
	cp.cuda.Device(device_id).use()
	## start to project model
#	report_gpu_mem("before")
	projection_array = run_projection(angle_starfile,mrc_model,True,False,True,device_id,model_star)
	
	## end projection
	# ver 6032 change starts
	if(IS_A_CONTINUE_RUN):
		out=open(output,"a")
		out.close()
	else:
		out=open(output,"w")
		out.close()
		# overwrite a new file if not a continue run.
#	out=open(output,"a")
	# ver 6032 change ends
	
	NEW_APIX=ORIGINAL_BOXSIZE/NEW_BOXSIZE*ORI_APIX
	TRANS_SCALE_FACTOR=ORI_APIX/NEW_APIX
	# angle step
	ACCURACY=11.0
	#test trans
	peak_devmem_usage = 0.

	CC_THRESHOLD=3.
	# The CC_THRESHOLD is adjustable. When only search for orientation, it=3.0. However, it should be >5 (Not sure) when search for both orientation and translation.
	# Generate Mask
	FSC_data =[]
	if(FSCfile==""):
		for i in range(0,ORIGINAL_BOXSIZE):
			FSC_val=1.0
			FSC_data.append([FSC_val])
	else:
		FSC_file=open(FSCfile,'r')
		FSC_file_line=FSC_file.readlines()

		for i in range(0,len(FSC_file_line)):
			if (FSC_file_line[i].split()):
				FSC_val=float(FSC_file_line[i].split()[1])
				if(do_ignore_FSC):
					FSC_val=1.0
				FSC_data.append([FSC_val])
		FSC_file.close()
	FSC_data=np.array(FSC_data)
	FSC_value = expand_to_2d(FSC_data, NEW_BOXSIZE)
	gpu_FSC_value=cp.asarray(FSC_value,dtype=cp.float32)
	norm_gpu_FSC_value=cp.linalg.norm(gpu_FSC_value)
	Mask0=np.zeros(ORIGINAL_BOXSIZE*ORIGINAL_BOXSIZE)
	Mask0=np.reshape(Mask0,(ORIGINAL_BOXSIZE,ORIGINAL_BOXSIZE))
	mask=gen_soft_maskout_radius(Mask0, radius,edge_width)
	gpu_mask=cp.asarray(mask,dtype=cp.float32)
	# read the model star
	aa=open(model_star,"r")
	instar_line=aa.readlines()
	relion30=1
	relion30=judge_relion30_or_relion31(inline=instar_line)
	print ("Is target star relion3.0? = "+str(relion30))
	mline=-1
	if(relion30):
		mline=judge_mline0(inline=instar_line)
	else:
		#regard as relion3.1
		MLINE=judge_mline0(inline=instar_line)
		mline=judge_mline1(inline=instar_line,start=MLINE)
	if(mline<0):
		print ("Model starfile error or this script cannot handle it. EXIT.")
		quit()
	print ("star mline = "+str(mline))
	for i in range(0,mline):
		if (instar_line[i].split()):
			if (str(instar_line[i].split()[0])=="_rlnImageName"):
				IMG_index=int(instar_line[i].split('#')[1])-1
			if (str(instar_line[i].split()[0])=="_rlnOpticsGroup"):
				OGN_index=int(instar_line[i].split('#')[1])-1
				HAVE_OPTICSGROUP=True
			if (str(instar_line[i].split()[0])=="_rlnAngleRot"):
				ROT_index=int(instar_line[i].split('#')[1])-1
			if (str(instar_line[i].split()[0])=="_rlnAngleTilt"):
				TILT_index=int(instar_line[i].split('#')[1])-1
	# read the particle star		
	bb=open(particle_star,"r")
	pstar_line=bb.readlines()
	p_relion30=1
	p_relion30=judge_relion30_or_relion31(inline=pstar_line)
	print ("Is target star relion3.0? = "+str(p_relion30))
	p_mline=-1
	if(p_relion30):
		p_mline=judge_mline0(inline=pstar_line)
	else:
		#regard as relion3.1
		p_MLINE=judge_mline0(inline=pstar_line)
		p_mline=judge_mline1(inline=pstar_line,start=p_MLINE)
	if(p_mline<0):
		print ("Particle starfile error or this script cannot handle it. EXIT.")
		quit()
	print ("star p_mline = "+str(p_mline))
	p_DFU_index=-1
	p_DFV_index=-1
	p_DFA_index=-1
	p_PSI_index=-1
	for i in range(0,p_mline):
		if (pstar_line[i].split()):
			if (str(pstar_line[i].split()[0])=="_rlnImageName"):
				p_IMG_index=int(pstar_line[i].split('#')[1])-1
			if (str(pstar_line[i].split()[0])=="_rlnOpticsGroup"):
				p_OGN_index=int(pstar_line[i].split('#')[1])-1
				p_HAVE_OPTICSGROUP=True
			if (str(pstar_line[i].split()[0])=="_rlnAngleRot"):
				p_ROT_index=int(pstar_line[i].split('#')[1])-1
			if (str(pstar_line[i].split()[0])=="_rlnAngleTilt"):
				p_TILT_index=int(pstar_line[i].split('#')[1])-1
			if (str(pstar_line[i].split()[0])=="_rlnAnglePsi"):
				p_PSI_index=int(pstar_line[i].split('#')[1])-1
			if (str(pstar_line[i].split()[0])=="_rlnDefocusU"):
				p_DFU_index=int(pstar_line[i].split('#')[1])-1
			if (str(pstar_line[i].split()[0])=="_rlnDefocusV"):
				p_DFV_index=int(pstar_line[i].split('#')[1])-1
			if (str(pstar_line[i].split()[0])=="_rlnDefocusAngle"):
				p_DFA_index=int(pstar_line[i].split('#')[1])-1
			
	time0=time.time()
	rfft_NEW_BOXSIZE=NEW_BOXSIZE//2+1
	FULL_SIZE=0
	for l in range(mline,len(instar_line)):
		xx=len(instar_line[l].split())
		if (instar_line[l].split()):
			FULL_SIZE+=1
	three_d_array=cp.zeros((FULL_SIZE+1,NEW_BOXSIZE,rfft_NEW_BOXSIZE),dtype=cp.complex64)

	Model_Rot=[]
	Model_Tilt=[]
	Model_Psi=[]
	three_d_norm_array= cp.array([0.],dtype=cp.float32)
	FULL_SIZE = 1
	for l in range(mline,len(instar_line)):
		xx=len(instar_line[l].split())
		if (instar_line[l].split()):
			image_name1=str(instar_line[l].split()[IMG_index])
			image_serial=str(image_name1.split('@')[0])
			image_serial_int=int(image_serial)-1
			filename1=str(image_name1.split('@')[1])
			real_filename = get_original_filenames(file_path=filename1)
			ROT=float(instar_line[l].split()[ROT_index])
			TILT=float(instar_line[l].split()[TILT_index])
		#	start reading the mrcs file.
		
		#	with mrcfile.mmap(real_filename,mode='r') as mrc1:
		#		image1=mrc1.data
			image1=projection_array
			if(len(image1.shape)<3):
				XSIZE=image1.shape[1]
				YSIZE=image1.shape[0]
				ZSIZE=1
			else:
				XSIZE=image1.shape[2]
				YSIZE=image1.shape[1]
				ZSIZE=image1.shape[0]
		#	read image_serial_int@real_filename
			if(len(image1.shape)<3):
				image1_slice_i=image1
			else:
				image1_slice_i=image1[image_serial_int,:,:]	
			gpu_image1=cp.asarray(image1_slice_i,dtype=cp.float32)
		#	print("gpu_image1.dtype=",gpu_image1.dtype)
			if(not do_discard_mask):
				gpu_image1 = gpu_image1*gpu_mask

		#	gpu_image1_psi_slice=rotate_fourier_image(gpu_image1,psi_rot_degree)
			gpu_image1_psi_slice=gpu_image1
			fft1=cp.fft.fft2(gpu_image1_psi_slice)
		#	print("fft2.dtype",fft2.dtype)
			fft1_shift=cp.fft.fftshift(fft1)
			if(NEW_BOXSIZE!=ORIGINAL_BOXSIZE):
				fft1_shift=fft1_shift[ORIGINAL_BOXSIZE//2-NEW_BOXSIZE//2: ORIGINAL_BOXSIZE//2+NEW_BOXSIZE//2,ORIGINAL_BOXSIZE//2-NEW_BOXSIZE//2: ORIGINAL_BOXSIZE//2+NEW_BOXSIZE//2,]
			conjugate_fft1=fft1_shift*gpu_FSC_value
			#### fft2 to rfft2 START
			conjugate_fft1=cp.fft.fftshift(conjugate_fft1)
			conjugate_fft1=conjugate_fft1[:, :conjugate_fft1.shape[1]//2 + 1]
			#### fft2 to rfft2 END
			NORM_FFT=cp.linalg.norm(conjugate_fft1)
			conjugate_fft1=conjugate_fft1/NORM_FFT
			three_d_array[FULL_SIZE,:,:]=conjugate_fft1
			Model_Rot.append(ROT)
			Model_Tilt.append(TILT)
			Model_Psi.append(0.)
			FULL_SIZE+=1
			del gpu_image1
			del fft1
			del fft1_shift
			del conjugate_fft1
		#	mrc1.close()
		#	cp.get_default_memory_pool().free_all_blocks()
	del projection_array,image1
	del gpu_FSC_value
	gc.collect()
	torch.cuda.empty_cache()
	cp.get_default_memory_pool().free_all_blocks()
	cp.get_default_pinned_memory_pool().free_all_blocks()

#	report_gpu_mem("after cleanup")
	if(START_LINE<p_mline):
		START_LINE=p_mline
	if(END_LINE<0):
		END_LINE=len(pstar_line)
	if(END_LINE<p_mline):
		END_LINE=p_mline
	if(END_LINE>len(pstar_line)):
		END_LINE=len(pstar_line)
	# localsearch
	Full_Particle_Star_Info_To_Search_3D_Stack_SN=[]
	if(do_local_search):
		TIMEX=time.time()
		print("Start reading full star and find the near orientations.")
		Pickle_FileName = particle_star+'_LocalSearch_Angle_'+str(int(local_search_angular_distance))+'_data.pkl'
		if os.path.exists(Pickle_FileName):
			# Load the data from the file
			with open(Pickle_FileName, 'rb') as FFF:
				Full_Particle_Star_Info_To_Search_3D_Stack_SN = pickle.load(FFF)
			print("Data loaded from file: ",Pickle_FileName)
		else:
			# Run the function and save the data
		#	Full_Particle_Star_Info_To_Search_3D_Stack_SN = preread_particle_star_and_sorting(pstar_line,p_mline,START_LINE,p_ROT_index,p_TILT_index,Model_Rot,Model_Tilt,local_search_angular_distance)
			# always generate full starfile table.
			Full_Particle_Star_Info_To_Search_3D_Stack_SN = preread_particle_star_and_sorting(pstar_line,p_mline,p_mline,p_ROT_index,p_TILT_index,Model_Rot,Model_Tilt,local_search_angular_distance)
			with open(Pickle_FileName, 'wb') as FFF:
				pickle.dump(Full_Particle_Star_Info_To_Search_3D_Stack_SN, FFF)
			print("Data generated and saved to file: ",Pickle_FileName)
		TEMP_Full_Particle_Star_Info_To_Search_3D_Stack_SN=Full_Particle_Star_Info_To_Search_3D_Stack_SN[START_LINE-p_mline:END_LINE]
		Full_Particle_Star_Info_To_Search_3D_Stack_SN = TEMP_Full_Particle_Star_Info_To_Search_3D_Stack_SN
		del TEMP_Full_Particle_Star_Info_To_Search_3D_Stack_SN
		# remove the unused lines.
		print("End reading full star and find the near orientations. Time = ",time.time()-TIMEX)

	time1=time.time()

	print("Total model prepare time = ",round(time1-time0,4)," seconds")
	print("Shape of the three_d_array:" + str(three_d_array.shape))
	
	cp.get_default_memory_pool().free_all_blocks()
	INTER_COUNT=0
	FREE_BLOCK_ROUND_IN_LOCAL_SEARCH = 96
	out=open(output,"a")

	if(END_LINE<0):
		END_LINE = len(pstar_line)
	if(START_LINE<p_mline):
		START_LINE=p_mline
	Series_Particle_NUM=1
	SF=100//Series_Particle_NUM
	HOW_MANY_ITERATION=int(np.ceil((END_LINE-START_LINE)/Series_Particle_NUM))
	RESULTS_Towrite = []
	last_filename = None
	last_data = None
	last_mrc = None
	for N in range(0,HOW_MANY_ITERATION):
		timex=time.time()
		particle_norm_array = cp.array([0.],dtype=cp.float32)
		Real_Start_Line=START_LINE+Series_Particle_NUM*N
		Real_End_Line=START_LINE+Series_Particle_NUM*(N+1)
		if(Real_End_Line>END_LINE):
			Real_End_Line=END_LINE
		INTER_COUNT+=(Real_End_Line-Real_Start_Line)
		particle_array,particle_norm_array,particle_ROT,particle_TILT,particle_PSI,particle_T_X,particle_T_Y,particle_LINE,particle_fake_PSI,NUM_PSI_ROTATION,last_filename,last_data,last_mrc = \
		read_Series_Particle_NUM(pstar_line,Real_Start_Line,Real_End_Line,p_IMG_index,p_ROT_index,p_TILT_index,p_PSI_index,p_DFU_index,p_DFV_index, \
		p_DFA_index,TRANS_MAX,ORIGINAL_BOXSIZE,NEW_BOXSIZE,XSIZE,YSIZE,ccss,VOLTAGE,do_discard_mask,gpu_mask,NEW_APIX,kk,time0,p_mline,PSI_STEP,do_local_search,local_search_angular_distance,\
		last_filename,last_data,last_mrc)
	#	cp.cuda.Stream.null.synchronize()
		if(len(particle_ROT)==0):
			continue
		if(not do_local_search):
			cp.get_default_memory_pool().free_all_blocks()
		chopped_local_search_SN=[]
		if(do_local_search):
			chopped_local_search_SN,Full_Particle_Star_Info_To_Search_3D_Stack_SN=chop_local_search_SN_Function(Full_Particle_Star_Info_To_Search_3D_Stack_SN,particle_LINE)
		
		RESULTS=search_stack_of_particles(particle_array,particle_norm_array,three_d_array,three_d_norm_array,Model_Rot,Model_Tilt,Model_Psi,out,CC_THRESHOLD,ACCURACY,TRANS_SCALE_FACTOR,output,time0, \
		particle_ROT,particle_TILT,particle_PSI,particle_T_X,particle_T_Y,particle_LINE,NEW_BOXSIZE,particle_fake_PSI,NUM_PSI_ROTATION,do_local_search,chopped_local_search_SN)
	#	cp.cuda.Stream.null.synchronize()
		RESULTS_Towrite.append(RESULTS)
		del particle_array
		if(do_local_search):
			if(INTER_COUNT%FREE_BLOCK_ROUND_IN_LOCAL_SEARCH==0):
			#	cp.get_default_memory_pool().free_all_blocks()
				for CC in range(0,len(RESULTS_Towrite)):
					for FF in range(0,len(RESULTS_Towrite[CC])):
						out.write(str(RESULTS_Towrite[CC][FF])+"\n")
				out.close()
				out=open(output,"a")
				RESULTS_Towrite = []
		else:
			cp.get_default_memory_pool().free_all_blocks()
			for CC in range(0,len(RESULTS_Towrite)):
				for FF in range(0,len(RESULTS_Towrite[CC])):
					out.write(str(RESULTS_Towrite[CC][FF])+"\n")
			out.close()
			out=open(output,"a")
			RESULTS_Towrite = []
	for CC in range(0,len(RESULTS_Towrite)):
		for FF in range(0,len(RESULTS_Towrite[CC])):
			out.write(str(RESULTS_Towrite[CC][FF])+"\n")
	out.close()
	RESULTS_Towrite = []
	del three_d_array
	del gpu_mask
	cp.get_default_memory_pool().free_all_blocks()

	time_calc_one_particle=time.time()
	print("Total execution time = ",round(time_calc_one_particle-time0,4)," seconds")
	aa.close()
	out.close()

def chop_local_search_SN_Function(Full_Particle_Star_Info_To_Search_3D_Stack_SN, particle_LINE):
	# Step 1: Get unique serial numbers in descending order
	unique_serial_numbers = sorted(set(particle_LINE), reverse=True)
	chopped_local_search_SN = []
	index = 0  # Track position in unique_serial_numbers

	# Step 3: Process Full_Particle_Star_Info_To_Search_3D_Stack_SN with early stopping
	i = 0  # Use a manual index to allow in-place deletion
	while i < len(Full_Particle_Star_Info_To_Search_3D_Stack_SN):
		entry = Full_Particle_Star_Info_To_Search_3D_Stack_SN[i]
		serial_number = entry[0]
		# Check if the serial number matches the next one in unique_serial_numbers
		if serial_number == unique_serial_numbers[index]:
			chopped_local_search_SN.append(entry)
			del Full_Particle_Star_Info_To_Search_3D_Stack_SN[i]  # Remove in-place
			# Move to the next unique serial number if matched
			index += 1
			if index >= len(unique_serial_numbers):  # If all unique numbers are found, break early
				break
		else:
			i += 1  # Only increment i if we didn’t delete to avoid skipping entries

	return chopped_local_search_SN, Full_Particle_Star_Info_To_Search_3D_Stack_SN
def cryosparc_filename(imagename):
	image_filename_tmp=(imagename.split('@')[1]).split('_')[1:]
	return (image_filename_tmp)
def relion_filename(imagename):
	image_filename_tmp=os.path.basename(imagename)
	return (image_filename_tmp)
def get_original_filenames(file_path):
	# Check if it's a symbolic link
	original_path=file_path
	if os.path.islink(file_path):
		# Resolve the symbolic link to get the original file path
		original_path = os.path.realpath(file_path)
	return original_path

def SQR(x):
	y=float(x)
	return(y*y)
def judge_relion30_or_relion31(inline):
	trys=3
	RELION30=1
	for i in range (0,trys):
		
		if(inline[i].split()):
			if(len(inline[i].split())>=2):
				if(inline[i].split()[0]=="#"):
					if(len(inline[i].split())>=4):
						if(str(inline[i].split()[3])=="3.0.8"):
							break
				if(int(str(inline[i].split()[2]))>30000):
					RELION30=0
					break
	return RELION30
def judge_mline0(inline):
	trys=60
	intarget=-1
	for i in range (0,trys):
		if(inline[i].split()):
			if(inline[i].split()[0][0]!="_"):
				if(intarget==1):
					return i
					break
				else:
					continue
			if(inline[i].split()[0][0]=="_"):
				intarget=1
def judge_mline1(inline,start):
	trys=70
	intarget=-1
	for i in range (start,trys):
		if(inline[i].split()):
			if(inline[i].split()[0][0]!="_"):
				if(intarget==1):
					return i
					break
				else:
					continue
			if(inline[i].split()[0][0]=="_"):
				intarget=1				
def Euler_angles2direction(alpha, beta):

	alpha = DEG2RAD(alpha)
	beta = DEG2RAD(beta)
	v=[]
	for i in range(0,3):
		v.append([])
	ca = math.cos(alpha)
	cb = math.cos(beta)
	sa = math.sin(alpha)
	sb = math.sin(beta)
	sc = sb * ca
	ss = sb * sa

	v[0]= sc
	v[1] = ss
	v[2] = cb
	return v

def DEG2RAD(x):
	return(x/180.0*3.14159265359)
def Euler_angles2matrix(alpha, beta, gamma):
	alpha = DEG2RAD(alpha)
	beta  = DEG2RAD(beta)
	gamma = DEG2RAD(gamma)
	ca =  math.cos(alpha)
	cb =  math.cos(beta)
	cg =  math.cos(gamma)
	sa =  math.sin(alpha)
	sb =  math.sin(beta)
	sg =  math.sin(gamma)
	cc =  cb * ca
	cs =  cb * sa
	sc =  sb * ca
	ss =  sb * sa
	A=[]
	for i in range(0,3):
		A.append([])
		for j in range(0,3):
			A[i].append([])
	A[0][0] =  cg * cc - sg * sa
	A[0][1] =  cg * cs + sg * ca
	A[0][2] = -cg * sb
	A[1][0] = -sg * cc - cg * sa
	A[1][1] = -sg * cs + cg * ca
	A[1][2] = sg * sb
	A[2][0] =  sc
	A[2][1] =  ss
	A[2][2] = cb
	return A

def calculateAngularDistance(rot1, tilt1,psi1,rot2, tilt2,psi2):

#	direction1=Euler_angles2direction(alpha=rot1, beta=tilt1)
#	direction2=Euler_angles2direction(alpha=rot2, beta=tilt2)
	min_axes_dist = 3600.0

	E1=Euler_angles2matrix(alpha=rot1, beta=tilt1, gamma=psi1)
	E2=Euler_angles2matrix(alpha=rot2, beta=tilt2, gamma=psi2)
	v1=[]
	v2=[]
	axes_dist = 0;
	for i in range(0,3):
		v1=E1[i]
		v2=E2[i]
		axes_dist += math.acos(CLIP(a=dotProduct(v1, v2),b=-1., c=1.))*180.0/3.14159265359
	axes_dist=axes_dist/3.0
	if (axes_dist < min_axes_dist):
		min_axes_dist = axes_dist
	return min_axes_dist
def selfNormalize(v):
	tmp=0.0
	for i in range(0,len(v)):
		tmp+=SQR(v[i])
	tmp=math.sqrt(tmp)
	if (tmp>1E-6):
		for i in range(0,len(v)):
			v[i]/=tmp
	else:
		for i in range(0,len(v)):
			v[i]=0.0
	return v
def Euler_direction2angles(v0):

#	v[0]=sb * ca,v[1]=sb * sa,v[2]=math.cos(beta)
	v=selfNormalize(v0)
#	print "debug ",v
	alpha = math.degrees(math.atan2(v[1], v[0]))
	beta =  math.degrees(math.acos(v[2]))

	if ((math.fabs(beta) < 0.001) or (math.fabs(beta - 180.) < 0.001)):
		alpha = 0.;

	return (alpha,beta)

def CLIP(a,b,c):
	if(float(a)<float(b)):
		return float(b)
	else:
		if(float(a)>float(c)):
			return float(c)
		return float(a)
def dotProduct(v1,v2):
	if(len(v1)!=len(v2)):
		return -9999999.0
	sum=0.0
	for i in range(0,len(v1)):
		sum+=float(v1[i])*float(v2[i])
	return sum

if __name__== "__main__":
	main()
