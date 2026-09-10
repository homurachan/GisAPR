"""Split particle searches while reusing one permanent process per device."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


def create_SEARCH_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('script', 'i', 'ang', 'mrc', 'p', 'o'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--FSC', default=None, help='Optional; omitted means unit FSC weights')
    parser.add_argument('--gpuid', default='0')
    for name, default in [('kk', 0.), ('apix', 1.42), ('voltage', 300.), ('cs', 2.7),
                          ('psiStep', 15.), ('localRange', 20.)]:
        parser.add_argument('--' + name, type=float, default=default)
    for name, default in [('oriboxsize', 256), ('newboxsize', 256), ('transRange', 0),
                          ('maskRadius', 110), ('maskEdge', 6), ('SplitParticles', 1)]:
        parser.add_argument('--' + name, type=int, default=default)
    for name in ('ignoreFSC', 'discardMask', 'doLocalSearch', 'doSplitDiffGpu'):
        parser.add_argument('--' + name, action='store_true')
    return parser


def test_length(starfile_name):
    # start/end refer to original STAR line indices, exactly as in v209.
    with open(starfile_name) as handle:
        return sum(1 for _ in handle)


def search_argv(args, start, end, serial_number):
    argv = []
    for name in ('i', 'p', 'oriboxsize', 'newboxsize', 'apix', 'transRange', 'voltage',
                 'cs', 'maskRadius', 'maskEdge', 'psiStep', 'kk', 'mrc', 'ang'):
        argv.extend(['--' + name, str(getattr(args, name))])
    if args.FSC and str(args.FSC).strip():
        argv.extend(['--FSC', str(args.FSC)])
    for name in ('ignoreFSC', 'discardMask', 'doLocalSearch'):
        if getattr(args, name):
            argv.append('--' + name)
    if args.doLocalSearch:
        argv.extend(['--localRange', str(args.localRange)])
    # The worker assigns the actual device after parsing this compatibility value.
    argv.extend(['--gpuid', '0', '--start', str(start), '--end', str(end)])
    filename = str(args.o) + '_tmp' + str(serial_number)
    argv.extend(['--o', filename])
    return argv, filename


def generate_command(args, start, end, serial_number):
    argv, filename = search_argv(args, start, end, serial_number)
    from gisapr_runtime import resolve_script
    gpuid = str(args.gpuid).split(':')[0]
    if args.doSplitDiffGpu and gpuid != 'cpu':
        gpuid = str(int(gpuid) + serial_number)
    argv[argv.index('--gpuid') + 1] = '0' if gpuid == 'cpu' else gpuid
    return shlex.join([sys.executable, str(resolve_script(args.script)), *argv]), filename


def run_command(command):
    # Kept as a public helper for callers of earlier versions.
    subprocess.run(shlex.split(command), check=True)
    return 0


def run_searches(args, available_gpu_ids=None):
    from gisapr_runtime import device_ids, get_worker
    count = int(args.SplitParticles)
    if count < 1:
        raise ValueError('SplitParticles must be >= 1')
    base_ids = available_gpu_ids or device_ids(args.gpuid)
    if args.doSplitDiffGpu:
        ids = list(base_ids)
        if len(ids) == 1 and ids[0] != 'cpu':
            ids = [str(int(ids[0]) + i) for i in range(count)]
    else:
        ids = [str(args.gpuid).split(':')[0]]
    workers = {gpu: get_worker(gpu, args.script) for gpu in ids}
    split_num = test_length(args.p) // count + 1
    jobs = []
    filenames = []
    for i in range(count):
        argv, filename = search_argv(args, i * split_num, (i + 1) * split_num, i)
        jobs.append((workers[ids[i % len(ids)]], argv))
        filenames.append(filename)
    def execute(job):
        worker, argv = job
        worker.request('search', argv=argv)
    # The per-worker lock serializes jobs sharing a device.
    with ThreadPoolExecutor(max_workers=len(ids)) as executor:
        list(executor.map(execute, jobs))
    Path(args.o).parent.mkdir(parents=True, exist_ok=True)
    with open(args.o, 'w') as output:
        for filename in filenames:
            with open(filename) as source:
                shutil.copyfileobj(source, output)
            os.remove(filename)
    return args.o


def main():
    from gisapr_runtime import shutdown_workers
    args = create_SEARCH_parser().parse_args()
    try:
        run_searches(args)
    finally:
        shutdown_workers()


if __name__ == '__main__':
    main()
