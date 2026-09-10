"""Pixel-size adapter for the single shared refinement pipeline."""
from copy import copy
from func import convert_number_to_filename, func_gpuid


def func_for_PixelSize_search_gpuid(x, args):
    values, gpuid = x
    apix_pdb = float(values[0])
    if apix_pdb <= 0:
        raise ValueError('Model pixel size must be positive')
    job = copy(args)
    job.apix_PDB = apix_pdb
    job.output_name_root = str(args.output_name_root) + convert_number_to_filename(apix_pdb, 3) + '_'
    # v209 used CC/simple sum here and ignored the geometry result.
    job.do_run_CC = True
    job.do_simple_sum = True
    job.skip_geometric_restraint = getattr(args, 'skip_geometric_restraint', True)
    return func_gpuid(([0.] * 6, str(gpuid)), job)


def func_for_PixelSize_search(x, args):
    return func_for_PixelSize_search_gpuid((x, str(args.gpuid).split(':')[0]), args)
