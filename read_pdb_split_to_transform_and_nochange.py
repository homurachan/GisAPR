#!/usr/bin/env python3
"""Split a first-model PDB by chain using only the original text records."""

import argparse

from pdb_text import split_pdb_by_chain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdb_filename")
    parser.add_argument("subunit_serial", help="One-character chain ID")
    parser.add_argument("output", help="Output PDB rootname")
    args = parser.parse_args()
    split_pdb_by_chain(args.pdb_filename, args.subunit_serial,
                       args.output + "_selected_chain_" + args.subunit_serial + ".pdb",
                       args.output + "_the_Main_Body.pdb")


if __name__ == "__main__":
    main()
