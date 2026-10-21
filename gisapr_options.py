"""Shared optional controls; existing optimization commands remain valid."""


def nonnegative_mb(value):
    import argparse
    import math
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError('cache size must be finite and >= 0 MiB')
    return value


def add_split_gpu_options(parser):
    parser.add_argument('--split_gpu_mode', choices=('consecutive', 'pool'), default='consecutive',
                        help='With --doSplitDiffGpu: consecutive uses g, g+1, ... for each candidate starting GPU (default); pool restores opt2 shared-pool routing.')
    return parser


def add_runtime_options(parser):
    add_split_gpu_options(parser)
    help_text = {
        '--gpuid': 'Colon-separated candidate starting CUDA IDs. With splitting, each g uses g, g+1, ... by default. Repeats create independent workers.',
        '--SplitParticles': 'Number of particle shards per candidate, balanced by particle count (default: 1). Merge before scoring.',
        '--doSplitDiffGpu': 'Distribute each candidate over SplitParticles consecutive GPUs from its assigned starting ID; use --split_gpu_mode pool for opt2 routing.',
    }
    for action in parser._actions:
        for option in action.option_strings:
            if option in help_text:
                action.help = help_text[option]
    parser.add_argument('--verbose', action='store_true', help='Show detailed per-candidate diagnostics.')
    parser.add_argument('--write_model_mrc', action='store_true',
                        help='Use the original disk MRC path and retain model MRC files (default: GPU tensor path).')
    parser.add_argument('--no_result_cache', action='store_true',
                        help='Re-evaluate even if the one-decimal legacy result file already exists.')
    parser.add_argument('--ctf_cache_mb', type=nonnegative_mb, default=256.,
                        help='Maximum host RAM MiB for CTF batches per worker; 0 disables (default: 256).')
    parser.add_argument('--search_cache_mb', type=nonnegative_mb, default=64.,
                        help='Maximum host RAM MiB for STAR/local-map metadata per worker; 0 disables (default: 64).')
    return parser
