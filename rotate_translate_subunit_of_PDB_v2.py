#!/usr/bin/env python3
"""Apply a GisAPR rotvec/translation pose without interpreting PDB numbering."""

import argparse
import math

from pdb_text import write_transformed_chain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdb_filename")
    parser.add_argument("subunit_serial", help="One-character chain ID")
    for name in ("rot", "tilt", "psi", "xshift", "yshift", "zshift"):
        parser.add_argument(name, type=float)
    parser.add_argument("output")
    parser.add_argument("--pivot", nargs=3, type=float, metavar=("X", "Y", "Z"),
                        help="Rotation pivot in angstrom; default: selected-chain geometric center")
    args = parser.parse_args()
    pose = [args.rot, args.tilt, args.psi, args.xshift, args.yshift, args.zshift]
    write_transformed_chain(args.pdb_filename, args.subunit_serial, pose, args.output, pivot=args.pivot)


def DEG2RAD(x):
    return x / 180.0 * 3.14159265359


def Euler_angles2matrix(alpha, beta, gamma):
    """Retained for external callers; the command itself uses rotation vectors."""
    alpha, beta, gamma = map(DEG2RAD, (alpha, beta, gamma))
    ca, cb, cg = math.cos(alpha), math.cos(beta), math.cos(gamma)
    sa, sb, sg = math.sin(alpha), math.sin(beta), math.sin(gamma)
    return [[cg * cb * ca - sg * sa, cg * cb * sa + sg * ca, -cg * sb],
            [-sg * cb * ca - cg * sa, -sg * cb * sa + cg * ca, sg * sb],
            [sb * ca, sb * sa, cb]]


if __name__ == "__main__":
    main()
