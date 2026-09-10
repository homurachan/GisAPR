"""
Drop-in successor to read_pdb_index_generate_sh_3DEG_local_v38.py for the
standalone optimized v606 search script.

It keeps the existing wrap_to_search_v2.py command-line contract, keeps the
large model MRC on disk, and no longer creates or references a whitened
projection STAR/MRCS pair.  The angle STAR supplied by --ang is passed as both
--i (compatibility placeholder) and --ang to the search script.
"""
# rename to read_pdb_index_generate_sh_3DEG_local_v37.py for dropin replacement.
import argparse
import os
import shlex
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PDB2MRC_PATH = shlex.join([sys.executable, str(SCRIPT_DIR / "pdb2mrc_gpu_ver_fp32_v3.py")])
WRAP_TO_SEARCH_PATH = shlex.join([sys.executable, str(SCRIPT_DIR / "wrap_to_search_v2.py")])


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate PDB2MRC and optimized standalone search commands"
    )
    parser.add_argument("--searchScript", type=str, required=True)
    parser.add_argument("--i", type=str, required=True, help="Input PDB file")
    parser.add_argument("--ang", type=str, required=True, help="RELION angle STAR")
    parser.add_argument("--p", type=str, required=True, help="Particle STAR")
    parser.add_argument("--o", type=str, required=True, help="Output root")
    parser.add_argument("--fsc", type=str, default=None, help="Optional FSC; omitted means unit weights")
    parser.add_argument("--kk", type=float, default=3.0)
    parser.add_argument("--gpuid", type=str, default="0")
    parser.add_argument("--oriboxsize", type=int, default=256)
    parser.add_argument("--newboxsize", type=int, default=160)
    parser.add_argument("--apix", type=float, default=1.42)
    parser.add_argument("--apix_PDB", type=float, default=1.42)
    parser.add_argument("--transRange", type=int, default=0)
    parser.add_argument("--voltage", type=float, default=300.0)
    parser.add_argument("--cs", type=float, default=2.7)
    parser.add_argument("--maskRadius", type=int, default=110)
    parser.add_argument("--maskEdge", type=int, default=6)
    parser.add_argument("--ignoreFSC", action="store_true")
    parser.add_argument("--discardMask", action="store_true")
    parser.add_argument("--psiStep", type=float, default=3.0)
    parser.add_argument("--doLocalSearch", action="store_true")
    parser.add_argument("--localRange", type=float, default=30.0)
    parser.add_argument("--yflip", action="store_true")
    parser.add_argument(
        "--doEnableGpuProj",
        action="store_true",
        help="Compatibility flag; projection is integrated into the search script",
    )
    parser.add_argument("--doSplitDiffGpu", action="store_true")
    parser.add_argument("--SplitParticles", type=int, default=1)
    return parser


def convert_number_to_filename_2(number: float) -> str:
    if number < 0:
        formatted = f"N{abs(number):.2f}".replace(".", "p")
    else:
        formatted = f"{number:.2f}".replace(".", "p")
    if formatted.endswith("p0"):
        return formatted[:-2]
    return formatted


def quote(value: object) -> str:
    return shlex.quote(str(value))


def read_gpuid(value: str):
    table = value.split(":")
    return len(table), table


def build_commands(args: argparse.Namespace) -> tuple[str, str]:
    pdb_filename = args.i
    angle_star = args.ang
    particle_star = args.p
    boxsize = int(args.oriboxsize)
    newboxsize = int(args.newboxsize)
    apix_pdb_code = convert_number_to_filename_2(args.apix_PDB)
    resolution = 2.0 * float(args.apix_PDB)

    pdb_prefix = pdb_filename.split(".pdb")[0]
    pdb_label = os.path.basename(pdb_prefix)
    mrc_filename = (
        pdb_prefix
        + "_box"
        + str(boxsize)
        + "_apix"
        + apix_pdb_code
        + ".mrc"
    )

    pdb2mrc = (
        PDB2MRC_PATH
        + " --i "
        + quote(pdb_filename)
        + " --o "
        + quote(mrc_filename)
        + " --box "
        + str(boxsize)
        + " --apix "
        + str(args.apix_PDB)
        + " --res "
        + str(resolution)
    )
    model_gpu = args.gpuid.split(":")[0]
    pdb2mrc += " --device cpu" if model_gpu == "cpu" else " --gpuid " + quote(model_gpu)
    if args.yflip:
        pdb2mrc += " --yflip"
    pdb2mrc += "\n"

    ngpus, gpu_table = read_gpuid(args.gpuid)
    gpu_index = 0
    gpuid = gpu_table[gpu_index]

    # --i remains present because wrap_to_search_v2.py requires it.  The new
    # standalone search does not read it; using --ang here removes the need for
    # any generated *_whitened.star file.
    output_name = (
        "search_v606opt_"
        + pdb_label
        + "_orient3degree_trans"
        + str(args.transRange)
        + "pixels_psi"
        + str(args.psiStep).replace(".0", "")
        + "_kk"
        + str(args.kk).replace(".0", "")
        + "_Correct_run1.txt"
    )

    search = (
        WRAP_TO_SEARCH_PATH
        + " --script "
        + quote(args.searchScript)
        + " --i "
        + quote(angle_star)
        + " --p "
        + quote(particle_star)
        + " --oriboxsize "
        + str(boxsize)
        + " --newboxsize "
        + str(newboxsize)
        + " --apix "
        + str(args.apix)
        + " --transRange "
        + str(args.transRange)
        + " --voltage "
        + str(args.voltage)
        + " --cs "
        + str(args.cs)
        + " --maskRadius "
        + str(args.maskRadius)
        + " --maskEdge "
        + str(args.maskEdge)
        + " --psiStep "
        + str(args.psiStep)
        + " --kk "
        + str(args.kk)
        + " --mrc "
        + quote(mrc_filename)
        + " --ang "
        + quote(angle_star)
    )
    if args.fsc and str(args.fsc).strip():
        search += " --FSC " + quote(args.fsc)
    if args.ignoreFSC:
        search += " --ignoreFSC"
    if args.discardMask:
        search += " --discardMask"
    search += " --gpuid " + str(gpuid)
    if args.doLocalSearch:
        search += " --doLocalSearch --localRange " + str(args.localRange)
    search += " --SplitParticles " + str(args.SplitParticles)
    if args.doSplitDiffGpu:
        search += " --doSplitDiffGpu"
    search += " --o " + quote(output_name) + "\n"

    # Keep the old GPU-table cycling behavior explicit for future NN > 1 use.
    gpu_index += 1
    if gpu_index >= ngpus:
        gpu_index = 0

    return pdb2mrc, search


def main() -> None:
    args = create_parser().parse_args()
    pdb2mrc_command, search_command = build_commands(args)
    generate_filename = args.o + "_generate_models_from_pdb.sh"
    search_filename = args.o + "_search_script.sh"
    with open(generate_filename, "w", encoding="utf-8") as handle:
        handle.write(pdb2mrc_command)
    with open(search_filename, "w", encoding="utf-8") as handle:
        handle.write(search_command)
    try:
        os.chmod(generate_filename, 0o755)
        os.chmod(search_filename, 0o755)
    except OSError:
        pass


if __name__ == "__main__":
    main()
