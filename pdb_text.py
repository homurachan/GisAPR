"""PDB coordinates without a structural parser or serial/residue renumbering.

All ATOM records (and, by default, HETATM records) in the selected MODELs are
retained, including repeated identifiers and alternate-location records.
Reading defaults to the first MODEL; callers can request all MODELs.
Files without MODEL records are treated as a single model. Atom and residue
identifiers are never interpreted. Element columns 77--78 take precedence;
otherwise the aligned atom name supplies the element. Coordinates are in A.
"""

from dataclasses import dataclass
import os

import numpy as np


_ELEMENTS = set((
    "H HE LI BE B C N O F NE NA MG AL SI P S CL AR K CA SC TI V CR MN FE "
    "CO NI CU ZN GA GE AS SE BR KR RB SR Y ZR NB MO TC RU RH PD AG CD IN "
    "SN SB TE I XE CS BA LA CE PR ND PM SM EU GD TB DY HO ER TM YB LU HF "
    "TA W RE OS IR PT AU HG TL PB BI PO AT RN FR RA AC TH PA U NP PU AM "
    "CM BK CF ES FM MD NO LR RF DB SG BH HS MT DS RG CN NH FL MC LV TS OG D"
).split())


def infer_element(line):
    """Return an uppercase element, respecting PDB atom-name alignment.

    Protein `` CA `` is carbon, whereas a calcium ion named ``CA  `` is
    calcium. Unaligned ATOM names starting C, N, O, H, P or S are interpreted
    as those elements. An unrecognised name returns an empty string.
    """
    explicit = line[76:78].strip().upper()
    if explicit in _ELEMENTS:
        return explicit
    name = line[12:16].ljust(4).upper()
    letters = "".join(char for char in name if char.isalpha())
    if not letters:
        return ""
    if name[0].isspace() or name[0].isdigit():
        return letters[0] if letters[0] in _ELEMENTS else ""
    if line[:6] == "ATOM  " and letters[0] in "CNOHPS":
        return letters[0]
    if letters[:2] in _ELEMENTS:
        return letters[:2]
    return letters[0] if letters[0] in _ELEMENTS else ""


@dataclass
class PDBText:
    """Raw PDB text and selected coordinate rows in their original order."""

    lines: list
    coords: np.ndarray
    elements: np.ndarray
    chains: np.ndarray
    atom_line_indices: np.ndarray

    def write(self, output, coords=None):
        """Write all original lines, replacing only supplied xyz columns."""
        lines = list(self.lines)
        if coords is not None:
            coords = np.asarray(coords)
            if coords.shape != self.coords.shape:
                raise ValueError("Replacement coordinates must have shape %s" % (self.coords.shape,))
            if not np.all(np.isfinite(coords)):
                raise ValueError("PDB coordinates must be finite")
            for line_index, xyz in zip(self.atom_line_indices, coords):
                fields = [f"{float(value):8.3f}" for value in xyz]
                if any(len(field) != 8 for field in fields):
                    raise ValueError("Coordinate exceeds the PDB 8.3 field width")
                line = lines[int(line_index)]
                lines[int(line_index)] = line[:30] + "".join(fields) + line[54:]
        with open(output, "w", encoding="utf-8", errors="surrogateescape", newline="") as handle:
            handle.writelines(lines)


def read_pdb(path, include_hetatm=True, first_model_only=True):
    """Read xyz, element and chain fields; never parse identifiers.

    ``first_model_only=False`` concatenates all models in their file order.
    The default retains the original transform/split first-model convention;
    density and geometric-restraint callers explicitly request all models.
    """
    with open(path, "r", encoding="utf-8", errors="surrogateescape", newline="") as handle:
        lines = handle.readlines()
    first_model = next((i for i, line in enumerate(lines) if line[:6].strip() == "MODEL"), None)
    active = first_model is None
    coords, elements, chains, indices = [], [], [], []
    for index, line in enumerate(lines):
        record = line[:6].strip()
        if record == "MODEL":
            if first_model_only and index != first_model:
                break
            active = True
            continue
        if record == "ENDMDL" and first_model is not None:
            if first_model_only:
                break
            active = False
            continue
        if not active or record not in (("ATOM", "HETATM") if include_hetatm else ("ATOM",)):
            continue
        try:
            xyz = [float(line[start:start + 8]) for start in (30, 38, 46)]
        except ValueError as exc:
            raise ValueError("Invalid PDB coordinates in %s, line %d" % (os.fspath(path), index + 1)) from exc
        if not np.all(np.isfinite(xyz)):
            raise ValueError("Non-finite PDB coordinates in %s, line %d" % (os.fspath(path), index + 1))
        coords.append(xyz)
        elements.append(infer_element(line))
        chains.append(line[21:22] or " ")
        indices.append(index)
    return PDBText(
        lines=lines,
        coords=np.asarray(coords, dtype=np.float32).reshape((-1, 3)),
        elements=np.asarray(elements, dtype="U2"),
        chains=np.asarray(chains, dtype="U1"),
        atom_line_indices=np.asarray(indices, dtype=np.int64),
    )


def transform_coordinates(coords, pose, pivot=None):
    """Apply a six-value degree/angstrom pose using GisAPR's row convention.

    Rotation is a rotation vector, not three successive Euler rotations:
    ``(coords - pivot) @ Rotation.from_rotvec(deg2rad(pose[:3])).as_matrix()``.
    The default pivot is the arithmetic mean of the supplied coordinates.
    """
    from scipy.spatial.transform import Rotation

    coords = np.asarray(coords)
    pose = np.asarray(pose, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3 or not len(coords):
        raise ValueError("A nonempty (N, 3) coordinate array is required")
    if pose.shape != (6,) or not np.all(np.isfinite(pose)):
        raise ValueError("Pose must contain six finite rotation/translation values")
    center = np.mean(coords, axis=0) if pivot is None else np.asarray(pivot, dtype=np.float64)
    if np.shape(center) != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("Pivot must contain three finite coordinates")
    matrix = Rotation.from_rotvec(np.deg2rad(pose[:3])).as_matrix()
    return (coords - center) @ matrix + center + pose[3:]


def transform_chain_coordinates(pdb, chain_id, pose, pivot=None):
    """Return transformed coordinates of one first-model chain only."""
    selected = pdb.chains == str(chain_id)
    if not np.any(selected):
        raise ValueError("Chain %r has no atoms in the first PDB model" % chain_id)
    return transform_coordinates(pdb.coords[selected], pose, pivot=pivot)


def write_transformed_chain(path, chain_id, pose, output, pivot=None):
    """Retain the full PDB and transform only the selected first-model chain."""
    pdb = read_pdb(path)
    selected = pdb.chains == str(chain_id)
    transformed = transform_chain_coordinates(pdb, chain_id, pose, pivot=pivot)
    # Modify only selected rows: unselected coordinate formatting also survives.
    subset = PDBText(pdb.lines, pdb.coords[selected], pdb.elements[selected],
                     pdb.chains[selected], pdb.atom_line_indices[selected])
    subset.write(output, transformed)
    return transformed


def split_pdb_by_chain(input_pdb, chain_id, output_chain_file, output_rest_file):
    """Write selected-chain and remaining first-model atom records verbatim.

    As in the original splitter, these are coordinate-only working PDBs with
    an END record. Serial numbers, residue fields, alternate locations, chain
    IDs and HETATM records are preserved without interpretation.
    """
    pdb = read_pdb(input_pdb)
    selected = pdb.chains == str(chain_id)
    if not np.any(selected):
        raise ValueError("Chain %r has no atoms in the first PDB model" % chain_id)
    for output, mask in ((output_chain_file, selected), (output_rest_file, ~selected)):
        with open(output, "w", encoding="utf-8", errors="surrogateescape", newline="") as handle:
            for index in pdb.atom_line_indices[mask]:
                line = pdb.lines[int(index)]
                handle.write(line)
                if not line.endswith(("\n", "\r")):
                    handle.write("\n")
            handle.write("END\n")
