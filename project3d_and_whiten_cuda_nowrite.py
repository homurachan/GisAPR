import numpy as np
import mrcfile
import os,math,argparse
from typing import Sequence, Tuple
import torch
import einops
import time
#import cupy as cp
torch.set_num_threads(4)  # Set to your desired number
# Initial version.
# The extract_central_slices_rfft comes from libtilt team: https://github.com/teamtomo/libtilt
# Ver 20260319
# Adapt normalization into pytorch, 50% faster when using GPU. Now I suggest always using GPU, as this version will run 1deg projection in RTX 4090.
# The maximum device memory usage is 10GB, a significant improvement.
def create_project3d_parser():
	parser = argparse.ArgumentParser(description="Project models from relion angle star then whitening them.")
	parser.add_argument("--i", type=str, required=True, help="Input model in mrc form.")
	parser.add_argument("--ang", type=str, required=True, help="The RELION angle starfile.")
	parser.add_argument("--o", type=str, required=True, help="The output starfile name.")
	parser.add_argument("--skip_padding", action='store_true', help="Skip the padding process. default = False")
	parser.add_argument("--normal_background_powerspectrum", action='store_true', help="Apply whitening filter to output projections. default = False")
	parser.add_argument("--gpuid", type=str,default = None, help="The gpuid, only one gpuid should be given. If not given, use CPU.")
	parser.add_argument("--return_array", action='store_true', help="Return numpy array instead of writing the stack. default = False")
	return parser

def build_radius_map_torch(ysize, xsize, device):
	y = torch.arange(ysize, device=device, dtype=torch.float32)
	x = torch.arange(xsize, device=device, dtype=torch.float32)
	yy, xx = torch.meshgrid(y, x, indexing='ij')
	cy = ysize // 2
	cx = xsize // 2
	dist = torch.round(torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)).long()
	return dist

def build_xy_grid_torch(ysize, xsize, device):
	y = torch.arange(ysize, device=device, dtype=torch.float32)
	x = torch.arange(xsize, device=device, dtype=torch.float32)
	yy, xx = torch.meshgrid(y, x, indexing='ij')
	return yy, xx
'''
def main():

	parser = create_project3d_parser()
	args = parser.parse_args()
	output_filename = extract_filename(args.o)
	particle_star=args.ang
	mrc_volume = args.i
	whitening = args.normal_background_powerspectrum
	do_skip_padding = args.skip_padding
	do_return_array = args.return_array
	gpuid=args.gpuid
	_=run_projection(args)
'''	
def run_projection(particle_star,mrc_volume,whitening,do_skip_padding,do_return_array,gpuid,output_filename):

#	particle_star=args.ang
#	mrc_volume = args.i
#	whitening = args.normal_background_powerspectrum
#	do_skip_padding = args.skip_padding
#	do_return_array = args.return_array
#	gpuid=args.gpuid
	device = 'cpu'
	device_for_cupy = None
	do_use_GPU = False
	if(gpuid != None):
		device = 'cuda:'+str(gpuid)
	#	cp.cuda.Device(gpuid).use()
		do_use_GPU = True
#	output_filename = extract_filename(args.o)
	output_mrcs_filename = output_filename+".mrcs"
	output_star_filename = output_filename
	aa=open(particle_star,"r")
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
		print ("Particle starfile error or this script cannot handle it. EXIT.")
		quit()
	# read the opticsgroup
	if(not relion30):
		for i in range(0,MLINE):
			if (instar_line[i].split()):
				if (str(instar_line[i].split()[0])=="_rlnImagePixelSize"):
					APIX_index=int(instar_line[i].split('#')[1])-1
		apix=float(instar_line[MLINE].split()[APIX_index])
		print ("star mline = "+str(mline)+" , input pixel size = "+str(apix))
	else:
		print ("star mline = "+str(mline))
	for i in range(0,mline):
		if (instar_line[i].split()):
			if (str(instar_line[i].split()[0])=="_rlnImageName"):
				IMG_index=int(instar_line[i].split('#')[1])-1
			if (str(instar_line[i].split()[0])=="_rlnAngleRot"):
				ROT_index=int(instar_line[i].split('#')[1])-1
			if (str(instar_line[i].split()[0])=="_rlnAngleTilt"):
				TILT_index=int(instar_line[i].split('#')[1])-1
			if (str(instar_line[i].split()[0])=="_rlnAnglePsi"):
				PSI_index=int(instar_line[i].split('#')[1])-1

	ROT_list=[]
	TILT_list=[]
	PSI_list=[]
	IS_WRITTEN=[]
	for l in range(mline,len(instar_line)):
		xx=len(instar_line[l].split())
		if (instar_line[l].split()):
			ROT=float(instar_line[l].split()[ROT_index])
			TILT=float(instar_line[l].split()[TILT_index])
			PSI=float(instar_line[l].split()[PSI_index])
			ROT_list.append(ROT)
			TILT_list.append(TILT)
			PSI_list.append(PSI)
			IS_WRITTEN.append(0)
	with mrcfile.mmap(mrc_volume,mode='r') as mrc1:
		image1=mrc1.data
	i1=image1.copy()
	# the tensor must be writable
	# Even the RTX 4090 is out of memory. CPU is quite fast since it runs multithreading automatically. 
	
	volume=torch.from_numpy(i1).to(device)

	N = image1.shape[0]

	pad = not do_skip_padding
	if pad is True:
		pad_length = volume.shape[-1] // 2
		volume = torch.nn.functional.pad(volume, pad=[pad_length] * 6, mode='constant', value=0)
	grid = fftfreq_grid(
		image_shape=volume.shape,
		rfft=False,
		fftshift=True,
		norm=True,
		device=volume.device
	)
	volume = volume * torch.sinc(grid) ** 2
	dft = torch.fft.fftshift(volume, dim=(-3, -2, -1))  # volume center to array origin
	dft = torch.fft.rfftn(dft, dim=(-3, -2, -1))
	dft = torch.fft.fftshift(dft, dim=(-3, -2,))  # actual fftshift of rfft

	stack = np.zeros((len(ROT_list), N, N), dtype = np.float32)
	radius_map_torch = build_radius_map_torch(N, N, volume.device)
	xx_torch, yy_torch = build_xy_grid_torch(N, N, volume.device)
	time0=time.time()
	
	for i in range(len(ROT_list)):
		rotation_matrix = Euler_angles2matrix(ROT_list[i], TILT_list[i], 0.0)
		rotation_matrix = np.transpose(np.array(rotation_matrix)).astype(np.float32)
		rotation_matrix = torch.from_numpy(rotation_matrix).reshape(1, 3, 3).to(device)

		projections = extract_central_slices_rfft(
			dft=dft,
			image_shape=volume.shape,
			rotation_matrices=rotation_matrix,
			rotation_matrix_zyx=False
		)

		projections = torch.fft.ifftshift(projections, dim=(-2,))
		projections = torch.fft.irfftn(projections, dim=(-2, -1))
		projections = torch.fft.ifftshift(projections, dim=(-2, -1))

		if pad is True:
			projections = projections[..., pad_length:-pad_length, pad_length:-pad_length]

		img = torch.real(projections[0]).to(torch.float32)

		if whitening:
			img = getSpectrum_divideBySpectrum_torch(img, radius_map=radius_map_torch)
			img = normalize_torch(img, xx_torch, yy_torch)
		stack[i, :, :] = img.detach().cpu().numpy()
		IS_WRITTEN[i] = 1
	if(stack.dtype!=np.float32):
		stack=np.array(stack,dtype=np.float32)
	return_array = None
	if(do_return_array):
		return_array=stack
	else:
		with mrcfile.new(output_mrcs_filename,overwrite=True) as output_mrcs:
			output_mrcs.set_data(stack)
		output_mrcs.close()
	output_star = open(output_star_filename,'w')
	for i in range(0,mline):
		output_star.write(instar_line[i])
	for l in range(mline,len(instar_line)):
		xx=len(instar_line[l].split())
		if (instar_line[l].split()):
			if(IS_WRITTEN[l-mline] > 0):
				towrite=""
				new_image_name=str(l-mline+1)+'@'+output_mrcs_filename
				for j in range(0,len(instar_line[l].split())):
					if(j!=IMG_index):
						towrite+=str(instar_line[l].split()[j])
					if(j==IMG_index):
						towrite+=new_image_name
					if(j<len(instar_line[l].split())-1):
						towrite+="\t"
					if(j>=len(instar_line[l].split())-1):
						towrite+="\n"
				output_star.write(towrite)
			else:
				print("In line ",l,", the projection is broken. Skipping this.")
	output_star.close()
	time1=time.time()
	try:
		del rotation_matrix, projections, img
	except:
		pass
	torch.cuda.empty_cache()
	print("Project3D, total execution time = ",round(time1-time0,4)," seconds")
	return return_array
############
############

def getSpectrum_divideBySpectrum_torch(img, radius_map=None):
	"""
	img: torch.Tensor, shape (H, W), real-valued
	return: torch.Tensor, shape (H, W), real-valued
	"""
	if img.ndim != 2:
		raise ValueError("img must be 2D")

	device = img.device
	H, W = img.shape

	if radius_map is None:
		radius_map = build_radius_map_torch(H, W, device)

	img_FFT = torch.fft.fftshift(torch.fft.fft2(img))

	dist_flat = radius_map.reshape(-1)
	fft_abs_flat = torch.abs(img_FFT).reshape(-1)

	max_radius = int(radius_map.max().item()) + 1

	spectrum = torch.zeros(max_radius, device=device, dtype=fft_abs_flat.dtype)
	count_freq = torch.zeros(max_radius, device=device, dtype=fft_abs_flat.dtype)

	spectrum.scatter_add_(0, dist_flat, fft_abs_flat)
	count_freq.scatter_add_(0, dist_flat, torch.ones_like(fft_abs_flat))

	div_spec = torch.ones_like(spectrum)
	nonzero = count_freq > 0
	div_spec[nonzero] = count_freq[nonzero] / spectrum[nonzero]
	div_spec[0] = 1.0

	img_FFT = img_FFT * div_spec[radius_map]
	img_IFT = torch.fft.ifft2(torch.fft.ifftshift(img_FFT)).real
	return img_IFT
def normalize_torch(img, xx=None, yy=None, eps=1e-12):
	"""
	img: (H, W) torch tensor, real
	xx, yy: meshgrid with indexing='ij'
			xx should be x-coordinates, yy should be y-coordinates
	"""
	if img.ndim != 2:
		raise ValueError("img must be 2D")

	H, W = img.shape
	device = img.device

	img64 = img.to(torch.float32)

	if xx is None or yy is None:
		y = torch.arange(H, device=device, dtype=torch.float32)
		x = torch.arange(W, device=device, dtype=torch.float32)
		yy, xx = torch.meshgrid(y, x, indexing='ij')
	else:
		xx = xx.to(device=device, dtype=torch.float32)
		yy = yy.to(device=device, dtype=torch.float32)

	# img = img - (pA * i_coords + pB * j_coords + pC)

	x = xx.reshape(-1)
	y = yy.reshape(-1)
	z = img64.reshape(-1)
	w = torch.ones_like(z, dtype=torch.float32)

	W2 = w * w

	sw2   = torch.sum(W2)
	sw2x  = torch.sum(W2 * x)
	sw2y  = torch.sum(W2 * y)
	sw2z  = torch.sum(W2 * z)
	sw2xx = torch.sum(W2 * x * x)
	sw2xy = torch.sum(W2 * x * y)
	sw2xz = torch.sum(W2 * x * z)
	sw2yy = torch.sum(W2 * y * y)
	sw2yz = torch.sum(W2 * y * z)

	A = torch.stack([
		torch.stack([sw2xx, sw2xy, sw2x]),
		torch.stack([sw2xy, sw2yy, sw2y]),
		torch.stack([sw2x,  sw2y,  sw2 ])
	])
	b = torch.stack([sw2xz, sw2yz, sw2z])

	coeff = torch.linalg.solve(A, b)
	pA, pB, pC = coeff[0], coeff[1], coeff[2]

	img64 = img64 - (pA * xx + pB * yy + pC)

	ave = img64.mean()
	std = img64.std(unbiased=False)

	if std > eps:
		img64 = (img64 - ave) / std
	else:
		img64 = img64 - ave
	return img64

def fftshift_2d(input: torch.Tensor, rfft: bool):
	if rfft is False:
		output = torch.fft.fftshift(input, dim=(-2, -1))
	else:
		output = torch.fft.fftshift(input, dim=(-2,))
	return output
def rfft_shape(input_shape: Sequence[int]) -> Tuple[int]:
	"""Get the output shape of an rfft on an input with input_shape."""
	rfft_shape = list(input_shape)
	rfft_shape[-1] = int((rfft_shape[-1] / 2) + 1)
	return tuple(rfft_shape)
def fftfreq_grid(
	image_shape: tuple[int, int] | tuple[int, int, int],
	rfft: bool,
	fftshift: bool = False,
	spacing: float | tuple[float, float] | tuple[float, float, float] = 1,
	norm: bool = False,
	device: torch.device | None = None,
):
	"""Construct a 2D or 3D grid of DFT sample frequencies.

	For a 2D image with shape `(h, w)` and `rfft=False` this function will produce
	a `(h, w, 2)` array of DFT sample frequencies in the `h` and `w` dimensions.
	If `norm` is True the Euclidean norm will be calculated over the last dimension
	leaving a `(h, w)` grid.

	Parameters
	----------
	image_shape: tuple[int, int] | tuple[int, int, int]
		Shape of the 2D or 3D image before computing the DFT.
	rfft: bool
		Whether the output should contain frequencies for a real-valued DFT.
	fftshift: bool
		Whether to fftshift the output grid.
	spacing: float | tuple[float, float] | tuple[float, float, float]
		Spacing between samples in each dimension. Sampling is considered to be
		isotropic if a single value is passed.
	norm: bool
		Whether to compute the Euclidean norm over the last dimension.
	device: torch.device | None
		PyTorch device on which the returned grid will be stored.

	Returns
	-------
	frequency_grid: torch.Tensor
		`(*image_shape, ndim)` array of DFT sample frequencies in each
		image dimension if `norm` is `False` else `(*image_shape, )`.
	"""
	if len(image_shape) == 2:
		frequency_grid = _construct_fftfreq_grid_2d(
			image_shape=image_shape,
			rfft=rfft,
			spacing=spacing,
			device=device,
		)
		if fftshift is True:
			frequency_grid = einops.rearrange(frequency_grid, '... freq -> freq ...')
			frequency_grid = fftshift_2d(frequency_grid, rfft=rfft)
			frequency_grid = einops.rearrange(frequency_grid, 'freq ... -> ... freq')
	elif len(image_shape) == 3:
		frequency_grid = _construct_fftfreq_grid_3d(
			image_shape=image_shape,
			rfft=rfft,
			spacing=spacing,
			device=device,
		)
		if fftshift is True:
			frequency_grid = einops.rearrange(frequency_grid, '... freq -> freq ...')
			frequency_grid = fftshift_3d(frequency_grid, rfft=rfft)
			frequency_grid = einops.rearrange(frequency_grid, 'freq ... -> ... freq')
	else:
		raise NotImplementedError(
			"Construction of fftfreq grids is currently only supported for "
			"2D and 3D images."
		)
	if norm is True:
		frequency_grid = einops.reduce(
			frequency_grid ** 2, '... d -> ...', reduction='sum'
		) ** 0.5
	return frequency_grid
def _construct_fftfreq_grid_2d(
	image_shape: Tuple[int, int],
	rfft: bool,
	spacing: float | tuple[float, float] = 1,
	device: torch.device = None
) -> torch.Tensor:
	"""Construct a grid of DFT sample freqs for a 2D image.

	Parameters
	----------
	image_shape: Sequence[int]
		A 2D shape `(h, w)` of the input image for which a grid of DFT sample freqs
		should be calculated.
	rfft: bool
		Whether the frequency grid is for a real fft (rfft).
	spacing: float | Tuple[float, float]
		Sample spacing in `h` and `w` dimensions of the grid.
	device: torch.device
		Torch device for the resulting grid.

	Returns
	-------
	frequency_grid: torch.Tensor
		`(h, w, 2)` array of DFT sample freqs.
		Order of freqs in the last dimension corresponds to the order of
		the two dimensions of the grid.
	"""
	dh, dw = spacing if isinstance(spacing, Sequence) else [spacing] * 2
	last_axis_frequency_func = torch.fft.rfftfreq if rfft is True else torch.fft.fftfreq
	h, w = image_shape
	freq_y = torch.fft.fftfreq(h, d=dh, device=device)
	freq_x = last_axis_frequency_func(w, d=dw, device=device)
	h, w = rfft_shape(image_shape) if rfft is True else image_shape
	freq_yy = einops.repeat(freq_y, 'h -> h w', w=w)
	freq_xx = einops.repeat(freq_x, 'w -> h w', h=h)
	return einops.rearrange([freq_yy, freq_xx], 'freq h w -> h w freq')
def _construct_fftfreq_grid_3d(
	image_shape: Sequence[int],
	rfft: bool,
	spacing: float | Tuple[float, float, float] = 1,
	device: torch.device = None
) -> torch.Tensor:
	"""Construct a grid of DFT sample freqs for a 3D image.

	Parameters
	----------
	image_shape: Sequence[int]
		A 3D shape `(d, h, w)` of the input image for which a grid of DFT sample freqs
		should be calculated.
	rfft: bool
		Controls Whether the frequency grid is for a real fft (rfft).
	spacing: float | Tuple[float, float, float]
		Sample spacing in `d`, `h` and `w` dimensions of the grid.
	device: torch.device
		Torch device for the resulting grid.

	Returns
	-------
	frequency_grid: torch.Tensor
		`(h, w, 3)` array of DFT sample freqs.
		Order of freqs in the last dimension corresponds to the order of dimensions
		of the grid.
	"""
	dd, dh, dw = spacing if isinstance(spacing, Sequence) else [spacing] * 3
	last_axis_frequency_func = torch.fft.rfftfreq if rfft is True else torch.fft.fftfreq
	d, h, w = image_shape
	freq_z = torch.fft.fftfreq(d, d=dd, device=device)
	freq_y = torch.fft.fftfreq(h, d=dh, device=device)
	freq_x = last_axis_frequency_func(w, d=dw, device=device)
	d, h, w = rfft_shape(image_shape) if rfft is True else image_shape
	freq_zz = einops.repeat(freq_z, 'd -> d h w', h=h, w=w)
	freq_yy = einops.repeat(freq_y, 'h -> d h w', d=d, w=w)
	freq_xx = einops.repeat(freq_x, 'w -> d h w', d=d, h=h)
	return einops.rearrange([freq_zz, freq_yy, freq_xx], 'freq ... -> ... freq')
def fftshift_3d(input: torch.Tensor, rfft: bool):
	if rfft is False:
		output = torch.fft.fftshift(input, dim=(-3, -2, -1))
	else:
		output = torch.fft.fftshift(input, dim=(-3, -2,))
	return output
def central_slice_grid(
	image_shape: tuple[int, int, int],
	rfft: bool,
	fftshift: bool = False,
	device: torch.device | None = None,
) -> torch.Tensor:
	h, w = image_shape[-2:]
	slice_hw = _construct_fftfreq_grid_2d(
		image_shape=(h, w),
		rfft=rfft,
		device=device
	)  # (h, w, 2)
	if rfft is True:
		h, w = rfft_shape((h, w))
	slice_d = torch.zeros(size=(h, w), dtype=slice_hw.dtype, device=device)
	central_slice, _ = einops.pack([slice_d, slice_hw], pattern='h w *')  # (h, w, 3)
	if fftshift is True:
		central_slice = einops.rearrange(central_slice, 'h w freq -> freq h w')
		central_slice = fftshift_2d(central_slice, rfft=rfft)
		central_slice = einops.rearrange(central_slice, 'freq h w -> h w freq')
	return central_slice	
def rotated_central_slice_grid(
	image_shape: tuple[int, int, int],
	rotation_matrices: torch.Tensor,
	rotation_matrix_zyx: bool,
	rfft: bool,
	fftshift: bool = False,
	device: torch.device | None = None,
):
	grid = central_slice_grid(
		image_shape=image_shape,
		rfft=rfft,
		fftshift=fftshift,
		device=device,
	)  # (h, w, 3)
	if rotation_matrix_zyx is False:
		grid = torch.flip(grid, dims=(-1,))
	rotation_matrices = einops.rearrange(rotation_matrices, '... i j -> ... 1 1 i j')
	grid = einops.rearrange(grid, 'h w coords -> h w coords 1')
	grid = rotation_matrices @ grid
	grid = einops.rearrange(grid, '... h w coords 1 -> ... h w coords')
	if rotation_matrix_zyx is False:  # back to zyx if currently xyz
		grid = torch.flip(grid, dims=(-1,))
	return grid	

def extract_central_slices_rfft(
	dft: torch.Tensor,
	image_shape: tuple[int, int, int],
	rotation_matrices: torch.Tensor,
	rotation_matrix_zyx: bool,
):
	"""Extract central slice from an fftshifted rfft."""
	# generate grid of DFT sample frequencies for a central slice in the xy-plane
	# these are a coordinate grid for the DFT
	grid = rotated_central_slice_grid(
		image_shape=image_shape,
		rotation_matrices=rotation_matrices,
		rotation_matrix_zyx=rotation_matrix_zyx,
		rfft=True,
		fftshift=True,
		device=dft.device,
	)  # (..., h, w, 3)

	# flip coordinates in redundant half transform
	conjugate_mask = grid[..., 2] < 0
	conjugate_mask = einops.repeat(conjugate_mask, '... -> ... 3')
	grid[conjugate_mask] *= -1
	conjugate_mask = conjugate_mask[..., 0]  # un-repeat

	# convert frequencies to array coordinates and sample from DFT
	grid = fftfreq_to_dft_coordinates(
		frequencies=grid,
		image_shape=image_shape,
		rfft=True
	)
	projections = sample_dft_3d(dft=dft, coordinates=grid)  # (..., h, w) rfft

	# take complex conjugate of values from redundant half transform
	projections[conjugate_mask] = torch.conj(projections[conjugate_mask])
	return projections
def dft_center(
	image_shape: Tuple[int, ...],
	rfft: bool,
	fftshifted: bool,
	device: torch.device | None = None,
) -> torch.LongTensor:
	"""Return the position of the DFT center for a given input shape."""
	fft_center = torch.zeros(size=(len(image_shape),), device=device)
	image_shape = torch.as_tensor(image_shape).float()
	if rfft is True:
		image_shape = torch.tensor(rfft_shape(image_shape)).to(device)
	if fftshifted is True:
		fft_center = torch.divide(image_shape, 2, rounding_mode='floor')
	if rfft is True:
		fft_center[-1] = 0
	return fft_center.long()	
def fftfreq_to_dft_coordinates(
	frequencies: torch.Tensor, image_shape: tuple[int, ...], rfft: bool
):
	"""Convert DFT sample frequencies into array coordinates in a fftshifted DFT.

	Parameters
	----------
	frequencies: torch.Tensor
		`(..., d)` array of multidimensional DFT sample frequencies
	image_shape: tuple[int, ...]
		Length `d` array of image dimensions.
	rfft: bool
		Whether output should be compatible with an rfft (`True`) or a
		full DFT (`False`)

	Returns
	-------
	coordinates: torch.Tensor
		`(..., d)` array of coordinates into a fftshifted DFT.
	"""
	image_shape = torch.as_tensor(
		image_shape, device=frequencies.device, dtype=frequencies.dtype
	)
	_rfft_shape = torch.as_tensor(
		rfft_shape(image_shape), device=frequencies.device, dtype=frequencies.dtype
	)
	coordinates = torch.empty_like(frequencies,device=frequencies.device)
	coordinates[..., :-1] = frequencies[..., :-1] * image_shape[:-1]
	if rfft is True:
		coordinates[..., -1] = frequencies[..., -1] * 2 * (_rfft_shape[-1] - 1)
	else:
		coordinates[..., -1] = frequencies[..., -1] * image_shape[-1]
	dc = dft_center(image_shape, rfft=rfft, fftshifted=True, device=frequencies.device)
	return coordinates + dc	
	
def array_to_grid_sample(
	array_coordinates: torch.Tensor, array_shape: Sequence[int]
) -> torch.Tensor:
	"""Generate grids for `torch.nn.functional.grid_sample` from array coordinates.

	These coordinates should be used with `align_corners=True` in
	`torch.nn.functional.grid_sample`.


	Parameters
	----------
	array_coordinates: torch.Tensor
		`(..., d)` array of d-dimensional coordinates.
		Coordinates are in the range `[0, N-1]` for the `N` elements in each dimension.
	array_shape: Sequence[int]
		shape of the array being sampled at `array_coordinates`.
	"""
	dtype, device = array_coordinates.dtype, array_coordinates.device
	array_shape = torch.as_tensor(array_shape, dtype=dtype, device=device)
	grid_sample_coordinates = (array_coordinates / (0.5 * array_shape - 0.5)) - 1
	grid_sample_coordinates = torch.flip(grid_sample_coordinates, dims=(-1,))
	return grid_sample_coordinates	
def sample_dft_3d(
	dft: torch.Tensor,
	coordinates: torch.Tensor
) -> torch.Tensor:
	"""Sample a complex volume with linear interpolation.


	Parameters
	----------
	dft: torch.Tensor
		`(d, h, w)` complex valued volume.
	coordinates: torch.Tensor
		`(..., zyx)` array of coordinates at which `dft` should be sampled.
		Coordinates should be ordered zyx, aligned with image dimensions `(d, h, w)`.
		Coordinates should be array coordinates, spanning `[0, N-1]` for a
		dimension of length N.
	Returns
	-------
	samples: torch.Tensor
		`(..., )` array of complex valued samples from `dft`.
	"""
	coordinates, ps = einops.pack([coordinates], pattern='* zyx')
	n_samples = coordinates.shape[0]

	# cannot sample complex tensors directly with grid_sample
	# c.f. https://github.com/pytorch/pytorch/issues/67634
	# workaround: treat real and imaginary parts as separate channels
	dft = einops.rearrange(torch.view_as_real(dft), 'd h w complex -> complex d h w')
	dft = einops.repeat(dft, 'complex d h w -> b complex d h w', b=n_samples)
	coordinates = einops.rearrange(coordinates, 'b zyx -> b 1 1 1 zyx')  # b d h w zyx

	samples = torch.nn.functional.grid_sample(
		input=dft,
		grid=array_to_grid_sample(coordinates, array_shape=dft.shape[-3:]),
		mode='bilinear',  # this is trilinear when input is volumetric
		padding_mode='border',  # this increases sampling fidelity at nyquist
		align_corners=True,
	)
	samples = einops.rearrange(samples, 'b complex 1 1 1 -> b complex')
	samples = torch.view_as_complex(samples.contiguous())  # (b, )

	# pack data back up and return
	[samples] = einops.unpack(samples, pattern='*', packed_shapes=ps)
	return samples  # (...)	
	
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
def extract_filename(file_path):
	base = os.path.basename(file_path)
	split_base = base.split('.')
	if len(split_base) > 1:
		return '.'.join(split_base[:-1])
	return base
#if __name__== "__main__":
#	main()
