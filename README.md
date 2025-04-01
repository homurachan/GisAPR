# GisAPR
GPU-accelerated in situ Atomic Perturbation Refinement

Requirements: Python 3, CUDA and compiler, some python libs. See requirements.txt for detailed python libs.

Has been tested under Ubuntu 22.04 and Windows 10.

I suggest to install cupy with this command `pip install cupy-cuda12x` . As for the pytorch, you don't need to install the cuda version.

If you are going to run my project on RTX 5000 series GPU, please note that CUDA 12.8 must be installed. Otherwise cupy will report compiler.py error.

You also need to install nvidia-graphics-drivers-570-open from ppa:graphics-drivers/ppa if you plan to run it on a RTX 5070 Ti.

# Steps
1. Run `python generate_healpix_order_and_relion_star.py` to generate angle starfiles.

Example: `python generate_healpix_order_and_relion_star.py --o c1_3deg_rot_removeLzero.star --discardPositiveRot --EQPSangleDegree 3.0 --apix $APIX` , or `python generate_healpix_order_and_relion_star.py -h` for help.

This will produce a starfile called "c1_3deg_rot_removeLzero.star", which stores the angular sampling points of Euler angle ROT and TILT with stepsize of 3.0 degrees.

2. Run `python GUI_v205.py` to show the GUI.

To be continued.
