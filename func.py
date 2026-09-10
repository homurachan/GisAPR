"""Shared refinement objective, backed by permanent optimized-search workers.

The historical search/result/PDB naming and negative-score objective are kept.
Geometry remains a hard rejection gate; its old unused bias is not added.
"""
import argparse
import math
from pathlib import Path
import shlex
import subprocess
import sys
import threading

from gisapr_runtime import (device_ids, get_worker, prepare_workers, resolve_script,
                            shutdown_workers)
from pdb_text import write_transformed_chain
from read_pdb_index_generate_sh_3DEG_local_v37 import build_commands
from wrap_to_search_v2 import create_SEARCH_parser, run_searches

_SCRIPT_DIR = Path(__file__).resolve().parent
_path_locks = {}
_path_locks_guard = threading.Lock()


def convert_number_to_filename(number, decimals=1):
    # Keep v209 filename encoding, including its historical rounding behavior.
    number = float(number)
    integer = int(abs(number))
    fraction = f'{abs(number - integer):.{decimals}f}'.split('.')[1]
    return ('N' if number < 0 else '') + str(integer) + 'p' + fraction


def pose_token(pose):
    r, t, p, x, y, z = map(convert_number_to_filename, pose)
    return f'rot{r}_tilt{t}_psi{p}deg_trans{x}_{y}_{z}ANG'


def prefixed(path, prefix, suffix=''):
    path = Path(path)
    return path.with_name(prefix + path.name + suffix)


def _path_lock(path):
    with _path_locks_guard:
        return _path_locks.setdefault(str(Path(path).resolve()), threading.RLock())


def func_gpuid(x, args):
    pose, gpuid = x
    pose = [float(value) for value in pose]
    if len(pose) != 6 or not all(math.isfinite(v) for v in pose):
        raise ValueError('A conformation must contain six finite pose values')
    root = Path(str(args.output_name_root))
    root.parent.mkdir(parents=True, exist_ok=True)
    rotated = root.with_name(root.name + str(args.rotate_chain) + '_' + pose_token(pose) + '.pdb')
    # Multiple rounded poses can share legacy filenames. Serialize their complete
    # evaluations rather than letting concurrently running workers corrupt them.
    with _path_lock(rotated):
        return _evaluate(pose, str(gpuid), args, rotated)


def _evaluate(pose, gpuid, args, rotated):
    worker = get_worker(gpuid, getattr(args, 'search_script', None))
    if not getattr(args, 'skip_geometric_restraint', False):
        overlap, distance = worker.request('geometry', geometry_kwargs={
            'pdb_path': str(Path(args.PDB_NAME).resolve()),
            'chain_id': str(args.rotate_chain),
            'boxsize': int(getattr(args, 'geometry_boxsize', 256)),
            'apix': float(getattr(args, 'geometry_apix', 1.5)),
            'threshold': float(getattr(args, 'geometry_threshold', 1.0)),
        }, pose=pose)
        geometry_output = rotated.with_name(rotated.stem.removesuffix('ANG') + '_GeometricRestrain_Result.txt')
        with open(geometry_output, 'w') as output:
            fields = [str(args.rotate_chain), *map(convert_number_to_filename, pose), str(overlap), str(distance)]
            output.write('\t'.join(fields) + '\n')
        print('debug, Overlapped pixels, Min_Distance =', overlap, distance, flush=True)
        max_overlap = getattr(args, 'MAXIUM_ALLOWED_overlapped_pixels', 300)
        max_distance = getattr(args, 'MAX_MinDistance_Allowed', 30.)
        if overlap >= max_overlap or distance >= max_distance:
            print('Search on this conformation is skipped. A preset likelihood value will be set.', flush=True)
            return float(overlap + max_distance)
    write_transformed_chain(args.PDB_NAME, str(args.rotate_chain), pose, str(rotated))
    generator = argparse.Namespace(
        i=str(rotated), ang=args.ang, p=args.STAR_NAME, o=str(prefixed(rotated, 'RUN01_')),
        searchScript=str(resolve_script(getattr(args, 'search_script', None))),
        fsc=getattr(args, 'fsc_file', None), kk=args.kk, gpuid=gpuid,
        oriboxsize=args.boxsize, newboxsize=args.newboxsize, apix=args.apix,
        apix_PDB=args.apix_PDB, transRange=args.transRange, voltage=args.voltage,
        cs=args.cs, maskRadius=args.maskRadius, maskEdge=getattr(args, 'maskEdge', 6),
        ignoreFSC=getattr(args, 'do_ignoreFSC', False),
        discardMask=getattr(args, 'discardMask', False), psiStep=args.psiStep,
        doLocalSearch=getattr(args, 'do_local_search', False),
        localRange=getattr(args, 'local_stepsize', 30.), yflip=getattr(args, 'yflip', False),
        SplitParticles=getattr(args, 'SplitParticles', 1),
        doSplitDiffGpu=getattr(args, 'doSplitDiffGpu', False),
    )
    mrc_command, search_command = build_commands(generator)
    search_tokens = shlex.split(search_command)
    search_args = create_SEARCH_parser().parse_args(search_tokens[2:])
    # All artifacts follow the output root directory; basenames remain unchanged.
    search_args.o = str(rotated.parent / Path(search_args.o).name)
    search_tokens[search_tokens.index('--o') + 1] = search_args.o
    # Record the actual replay command, including the output directory.
    Path(generator.o + '_generate_models_from_pdb.sh').write_text(mrc_command)
    Path(generator.o + '_search_script.sh').write_text(shlex.join(search_tokens) + '\n')
    worker.request('generate_mrc', mrc_kwargs={
        'pdb_path': str(rotated), 'output': search_args.mrc,
        'boxsize': int(args.boxsize), 'apix': float(args.apix_PDB),
        'res': 2.0 * float(args.apix_PDB), 'do_yflip': bool(getattr(args, 'yflip', False)),
    })
    run_searches(search_args, available_gpu_ids=device_ids(args.gpuid))
    index = prefixed(rotated, 'index_for_result_', '.txt')
    index.write_text(Path(search_args.o).name + '\n')
    result = prefixed(rotated, 'ReSuLt_', '.txt')
    subprocess.run([
        sys.executable, str(_SCRIPT_DIR / 'new_method_to_fit_the_2nd_Gaussian_PEAK_v4.py'),
        index.name, str(int(bool(getattr(args, 'do_run_CC', False)))),
        str(int(bool(getattr(args, 'do_simple_sum', False)))), result.name,
    ], check=True, cwd=rotated.parent)
    lines = result.read_text().splitlines()
    if len(lines) < 2:
        raise RuntimeError(f'Incomplete score result: {result}')
    value = float(lines[1].split()[0])
    if not math.isfinite(value):
        raise RuntimeError(f'Nonfinite score in {result}: {value}')
    return -value


def finalize_run(args):
    from gisapr_results import finalize_run as finalize
    return finalize(args)
