"""
Standalone optimized successor to
`test1_with_isspa_weight_varingKK_search_translation_also_v605_torch.py`.

Compatibility goals
-------------------
* Keeps the command-line interface used by wrap_to_search_v2.py.
* Keeps the three output records per particle:
    ['cc: ', ...]
    ['psi: ', ...]
    ['rawCC: ', ...]
* Keeps --transRange as a compatibility argument, but does not randomly
  translate particles. Translation is obtained from the CCG peak exactly as
  in v605.
* --i is retained for wrapper compatibility. It is no longer read or written.
  Model orientations are read directly from --ang.
* The input model MRC supplied by --mrc is never deleted or overwritten.

Main changes from v605
----------------------
* Native PyTorch; no CuPy compatibility layer and no wildcard imports.
* Integrated 3-D projector and whitening pipeline. Projections stay on GPU and
  are converted directly to final packed Fourier templates.
* Integrated packed CTF implementations:
    - batched PyTorch backend (default, fast)
    - exact-layout NumPy backend (reference/validation)
* Cached in-plane-rotation sampling grids.
* One-particle batched grid_sample and FFT across all searched psi angles.
* Correlation inputs are pre-shifted once. The CCG search reads only the four
  unshifted corner blocks corresponding to the centered crop.
* Combined max/argmax reductions and reusable result workspaces.

The numerical path is float32/complex64. Batched GPU execution can change the
last few floating-point digits relative to v605, while preserving the search
and output contract.
"""
# This version is a standalone version of the optimized v605_torch.py. No need for the project3d and ctf_cupy
# It combines the project3d and ctf. Besides the whole architecture is re-written by chatgpt 5.6 sol Pro.
# There is no writting star file in this version.
# In my workstation, it is 3x faster than the old version.
# The main bottleneck is the IO.

from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import mrcfile
import numpy as np
import torch
import torch.nn.functional as F


# Avoid silent TF32 changes in the small matrix/grid calculations.
torch.backends.cuda.matmul.allow_tf32 = False
if hasattr(torch.backends, "cudnn"):
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

torch.set_grad_enabled(False)


# ---------------------------------------------------------------------------
# STAR parsing
# ---------------------------------------------------------------------------


@dataclass
class StarLoop:
    lines: List[str]
    labels: Dict[str, int]
    data_line_indices: List[int]
    data_start: int

    def tokens(self, line_index: int) -> List[str]:
        return self.lines[line_index].split()


def _parse_label_index(tokens: Sequence[str], fallback: int) -> int:
    for token in tokens[1:]:
        if token.startswith("#"):
            try:
                return int(token[1:]) - 1
            except ValueError:
                pass
    return fallback


def find_star_loop(lines: List[str], required_labels: Sequence[str]) -> StarLoop:
    """Find the loop containing all requested RELION labels."""
    required = set(required_labels)
    n = len(lines)
    i = 0
    while i < n:
        stripped = lines[i].strip()
        if stripped.lower() != "loop_":
            i += 1
            continue

        labels: Dict[str, int] = {}
        j = i + 1
        fallback = 0
        while j < n:
            s = lines[j].strip()
            if not s or s.startswith("#"):
                j += 1
                continue
            if not s.startswith("_"):
                break
            tokens = s.split()
            labels[tokens[0]] = _parse_label_index(tokens, fallback)
            fallback += 1
            j += 1

        if not required.issubset(labels):
            i = max(j, i + 1)
            continue

        data_indices: List[int] = []
        k = j
        data_start = -1
        while k < n:
            s = lines[k].strip()
            lower = s.lower()
            if lower == "loop_" or lower.startswith("data_") or s.startswith("_"):
                if data_indices:
                    break
                k += 1
                continue
            if not s or s.startswith("#"):
                k += 1
                continue
            if data_start < 0:
                data_start = k
            data_indices.append(k)
            k += 1

        if data_start < 0:
            raise ValueError(
                "Found a STAR loop with the required labels but no data rows: "
                + ", ".join(required_labels)
            )
        return StarLoop(lines=lines, labels=labels, data_line_indices=data_indices, data_start=data_start)

    raise ValueError("Cannot find a STAR loop containing: " + ", ".join(required_labels))


def read_text_lines(filename: str) -> List[str]:
    with open(filename, "r", encoding="utf-8", errors="replace") as handle:
        return handle.readlines()


@dataclass
class ModelAngles:
    rot: List[float]
    tilt: List[float]
    psi: List[float]
    rotation_matrices_t: np.ndarray  # (M, 3, 3), transposed as used by v605 projector


def euler_angles_to_matrix(alpha: float, beta: float, gamma: float) -> List[List[float]]:
    """Original v605/project3d Euler convention."""
    alpha = alpha / 180.0 * 3.14159265359
    beta = beta / 180.0 * 3.14159265359
    gamma = gamma / 180.0 * 3.14159265359
    ca = math.cos(alpha)
    cb = math.cos(beta)
    cg = math.cos(gamma)
    sa = math.sin(alpha)
    sb = math.sin(beta)
    sg = math.sin(gamma)
    cc = cb * ca
    cs = cb * sa
    sc = sb * ca
    ss = sb * sa
    return [
        [cg * cc - sg * sa, cg * cs + sg * ca, -cg * sb],
        [-sg * cc - cg * sa, -sg * cs + cg * ca, sg * sb],
        [sc, ss, cb],
    ]


def read_model_angles(angle_star: str) -> ModelAngles:
    lines = read_text_lines(angle_star)
    loop = find_star_loop(lines, ["_rlnAngleRot", "_rlnAngleTilt"])
    rot_idx = loop.labels["_rlnAngleRot"]
    tilt_idx = loop.labels["_rlnAngleTilt"]
    psi_idx = loop.labels.get("_rlnAnglePsi", -1)

    rot: List[float] = []
    tilt: List[float] = []
    psi: List[float] = []
    matrices: List[np.ndarray] = []

    for line_index in loop.data_line_indices:
        tokens = loop.tokens(line_index)
        if max(rot_idx, tilt_idx) >= len(tokens):
            continue
        r = float(tokens[rot_idx])
        t = float(tokens[tilt_idx])
        p = float(tokens[psi_idx]) if 0 <= psi_idx < len(tokens) else 0.0
        rot.append(r)
        tilt.append(t)
        psi.append(p)
        # The old projector ignores STAR psi and projects with gamma=0.
        matrix = np.asarray(euler_angles_to_matrix(r, t, 0.0), dtype=np.float32).T
        matrices.append(matrix)

    if not rot:
        raise ValueError(f"No model orientations found in {angle_star}")

    return ModelAngles(
        rot=rot,
        tilt=tilt,
        psi=psi,
        rotation_matrices_t=np.stack(matrices, axis=0).astype(np.float32, copy=False),
    )


@dataclass
class ParticleRecord:
    line_index: int
    image_name: str
    rot: float
    tilt: float
    psi: float
    defocus_u: float = 0.0
    defocus_v: float = 0.0
    defocus_angle: float = 0.0


@dataclass
class ParticleTable:
    lines: List[str]
    loop: StarLoop
    records: List[ParticleRecord]
    records_by_line: Dict[int, ParticleRecord]
    has_ctf: bool


def read_particle_table(particle_star: str) -> ParticleTable:
    lines = read_text_lines(particle_star)
    loop = find_star_loop(
        lines,
        ["_rlnImageName", "_rlnAngleRot", "_rlnAngleTilt"],
    )
    image_idx = loop.labels["_rlnImageName"]
    rot_idx = loop.labels["_rlnAngleRot"]
    tilt_idx = loop.labels["_rlnAngleTilt"]
    psi_idx = loop.labels.get("_rlnAnglePsi", -1)
    dfu_idx = loop.labels.get("_rlnDefocusU", -1)
    dfv_idx = loop.labels.get("_rlnDefocusV", -1)
    dfa_idx = loop.labels.get("_rlnDefocusAngle", -1)
    has_ctf = min(dfu_idx, dfv_idx, dfa_idx) >= 0

    records: List[ParticleRecord] = []
    records_by_line: Dict[int, ParticleRecord] = {}
    required_max = max(image_idx, rot_idx, tilt_idx)

    for line_index in loop.data_line_indices:
        tokens = loop.tokens(line_index)
        if required_max >= len(tokens):
            continue
        rec = ParticleRecord(
            line_index=line_index,
            image_name=tokens[image_idx],
            rot=float(tokens[rot_idx]),
            tilt=float(tokens[tilt_idx]),
            psi=float(tokens[psi_idx]) if 0 <= psi_idx < len(tokens) else 0.0,
        )
        if has_ctf:
            rec.defocus_u = float(tokens[dfu_idx])
            rec.defocus_v = float(tokens[dfv_idx])
            rec.defocus_angle = float(tokens[dfa_idx])
        records.append(rec)
        records_by_line[line_index] = rec

    if not records:
        raise ValueError(f"No particle rows found in {particle_star}")

    return ParticleTable(
        lines=lines,
        loop=loop,
        records=records,
        records_by_line=records_by_line,
        has_ctf=has_ctf,
    )


def select_particle_records(
    table: ParticleTable,
    start_line: int,
    end_line: int,
) -> List[ParticleRecord]:
    start = max(start_line, table.loop.data_start) if start_line >= 0 else table.loop.data_start
    end = len(table.lines) if end_line < 0 else min(end_line, len(table.lines))
    if end < table.loop.data_start:
        end = table.loop.data_start
    return [rec for rec in table.records if start <= rec.line_index < end]


# ---------------------------------------------------------------------------
# Original angular-distance convention and local-orientation cache
# ---------------------------------------------------------------------------


def clip_value(a: float, b: float, c: float) -> float:
    if float(a) < float(b):
        return float(b)
    if float(a) > float(c):
        return float(c)
    return float(a)


def dot_product(v1: Sequence[float], v2: Sequence[float]) -> float:
    if len(v1) != len(v2):
        return -9999999.0
    total = 0.0
    for a, b in zip(v1, v2):
        total += float(a) * float(b)
    return total


def calculate_angular_distance(
    rot1: float,
    tilt1: float,
    psi1: float,
    rot2: float,
    tilt2: float,
    psi2: float,
) -> float:
    e1 = euler_angles_to_matrix(rot1, tilt1, psi1)
    e2 = euler_angles_to_matrix(rot2, tilt2, psi2)
    axes_dist = 0.0
    for i in range(3):
        axes_dist += (
            math.acos(clip_value(dot_product(e1[i], e2[i]), -1.0, 1.0))
            * 180.0
            / 3.14159265359
        )
    return axes_dist / 3.0


def build_local_orientation_entries(
    particle_records: Sequence[ParticleRecord],
    model_rot: Sequence[float],
    model_tilt: Sequence[float],
    local_range: float,
) -> List[List[int]]:
    model_rot_v = np.asarray(model_rot)
    model_tilt_v = np.asarray(model_tilt)
    entries: List[List[int]] = []

    for rec in particle_records:
        entry = [rec.line_index]
        rot_diff = np.fabs((model_rot_v - rec.rot + 180.0) % 360.0 - 180.0)
        tilt_diff = np.fabs((model_tilt_v - rec.tilt + 180.0) % 360.0 - 180.0)
        valid_indices = np.where((rot_diff <= local_range) & (tilt_diff <= local_range))[0]
        for model_index in valid_indices:
            distance = calculate_angular_distance(
                float(model_rot_v[model_index]),
                float(model_tilt_v[model_index]),
                0.0,
                rec.rot,
                rec.tilt,
                0.0,
            )
            if abs(distance) <= local_range:
                entry.append(int(model_index))
        entries.append(entry)
    return entries


def _local_cache_valid(
    entries: object,
    particle_records: Sequence[ParticleRecord],
    n_models: int,
) -> bool:
    if not isinstance(entries, list) or len(entries) != len(particle_records):
        return False
    for entry, rec in zip(entries, particle_records):
        if not isinstance(entry, list) or not entry or entry[0] != rec.line_index:
            return False
        if any((not isinstance(idx, (int, np.integer))) or idx < 0 or idx >= n_models for idx in entry[1:]):
            return False
    return True


def load_or_build_local_orientation_map(
    particle_star: str,
    particle_records: Sequence[ParticleRecord],
    model_angles: ModelAngles,
    local_range: float,
) -> Dict[int, List[int]]:
    # Keep the v605 cache filename so existing valid caches can be reused.
    cache_filename = particle_star + "_LocalSearch_Angle_" + str(int(local_range)) + "_data.pkl"
    entries: Optional[List[List[int]]] = None

    if os.path.exists(cache_filename):
        try:
            with open(cache_filename, "rb") as handle:
                loaded = pickle.load(handle)
            if _local_cache_valid(loaded, particle_records, len(model_angles.rot)):
                entries = loaded
                print("Data loaded from file:", cache_filename)
            else:
                print("Existing local-search cache is incompatible; rebuilding:", cache_filename)
        except Exception as exc:
            print(f"Could not read local-search cache ({exc}); rebuilding: {cache_filename}")

    if entries is None:
        entries = build_local_orientation_entries(
            particle_records,
            model_angles.rot,
            model_angles.tilt,
            local_range,
        )
        tmp_filename = cache_filename + f".tmp.{os.getpid()}"
        try:
            with open(tmp_filename, "wb") as handle:
                pickle.dump(entries, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_filename, cache_filename)
            print("Data generated and saved to file:", cache_filename)
        finally:
            if os.path.exists(tmp_filename):
                try:
                    os.remove(tmp_filename)
                except OSError:
                    pass

    return {entry[0]: [int(x) for x in entry[1:]] for entry in entries}


# ---------------------------------------------------------------------------
# Masks, FSC and packed Fourier layout
# ---------------------------------------------------------------------------


def generate_soft_mask(shape: Tuple[int, int], radius: int, edge_width: int) -> np.ndarray:
    center = np.asarray(shape) // 2
    y, x = np.indices(shape)
    distances = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    mask = np.zeros_like(distances)
    mask[distances <= radius] = 1.0
    if edge_width > 0:
        edge = (distances > radius) & (distances <= radius + edge_width)
        mask[edge] = 0.5 * (
            1.0 + np.cos(np.pi * (distances[edge] - radius) / edge_width)
        )
    return mask.astype(np.float32, copy=False)


def expand_radial_values(values: np.ndarray, size: int) -> np.ndarray:
    values_1d = np.asarray(values, dtype=np.float32).reshape(-1)
    center = size // 2
    y, x = np.indices((size, size))
    radius = np.sqrt((y - center) ** 2 + (x - center) ** 2).astype(np.int64)
    output = np.zeros((size, size), dtype=np.float32)
    valid = radius < values_1d.shape[0]
    output[valid] = values_1d[radius[valid]]
    return output


def read_fsc_map(
    fsc_filename: str,
    original_box: int,
    new_box: int,
    ignore_fsc: bool,
) -> np.ndarray:
    if not fsc_filename:
        values = np.ones(original_box, dtype=np.float32)
    else:
        parsed: List[float] = []
        with open(fsc_filename, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                tokens = line.split()
                if not tokens:
                    continue
                value = 1.0 if ignore_fsc else float(tokens[1])
                parsed.append(value)
        if not parsed:
            raise ValueError(f"No FSC values found in {fsc_filename}")
        values = np.asarray(parsed, dtype=np.float32)
    return expand_radial_values(values, new_box)


class PackedFourierLayout:
    """Indices reproducing fftshift(full_fft)[:, :N//2+1] without a full shift."""

    def __init__(self, size: int, crop_size: int, device: torch.device):
        if size <= 0 or size % 2 != 0:
            raise ValueError("newboxsize must be a positive even integer")
        if crop_size <= 0 or crop_size % 2 != 0 or crop_size > size:
            raise ValueError("CCG crop size must be positive, even and <= newboxsize")
        self.size = int(size)
        self.rfft_size = self.size // 2 + 1
        self.crop_size = int(crop_size)
        half = self.size // 2
        self.centered_row_to_packed = torch.cat(
            (
                torch.arange(half, self.size, device=device, dtype=torch.long),
                torch.arange(0, half, device=device, dtype=torch.long),
            )
        )
        self.centered_col_to_packed = torch.cat(
            (
                torch.arange(half, self.size, device=device, dtype=torch.long),
                torch.zeros(1, device=device, dtype=torch.long),
            )
        )
        crop_half = self.crop_size // 2
        self.unshifted_ccg_crop_indices = torch.cat(
            (
                torch.arange(
                    self.size - crop_half,
                    self.size,
                    device=device,
                    dtype=torch.long,
                ),
                torch.arange(0, crop_half, device=device, dtype=torch.long),
            )
        )

    def centered_full_to_packed(self, centered: torch.Tensor) -> torch.Tensor:
        packed = centered.index_select(-2, self.centered_row_to_packed)
        return packed.index_select(-1, self.centered_col_to_packed)

    @staticmethod
    def pre_shift_for_correlation(packed: torch.Tensor) -> torch.Tensor:
        return torch.fft.fftshift(packed, dim=(-2, -1))

    def centered_ccg_crop_from_unshifted(self, ccg: torch.Tensor) -> torch.Tensor:
        cropped = ccg.index_select(-2, self.unshifted_ccg_crop_indices)
        return cropped.index_select(-1, self.unshifted_ccg_crop_indices)


# ---------------------------------------------------------------------------
# Packed CTF: exact-layout NumPy reference and batched PyTorch fast backend
# ---------------------------------------------------------------------------


class PackedCTFNumPy:
    """NumPy CTF evaluated directly in the v605 packed 2-D frequency layout."""

    _basis_cache: Dict[Tuple[int, float], Tuple[np.ndarray, ...]] = {}
    _weight_cache: Dict[Tuple[int, float, float], np.ndarray] = {}

    def __init__(
        self,
        size: int,
        pixel_size: float,
        cs: float,
        voltage: float,
        kk: float,
        amp_const: float = 0.1,
        bfactor: float = 0.0,
        scale: float = 1.0,
        phase_shift: float = 0.0,
    ) -> None:
        self.size = int(size)
        self.pixel_size = np.float32(pixel_size)
        self.cs = np.float32(cs)
        self.voltage = np.float32(voltage)
        self.kk = np.float32(kk)
        self.amp_const = np.float32(amp_const)
        self.bfactor = np.float32(bfactor)
        self.scale = np.float32(scale)
        self.phase_shift = np.float32(phase_shift)
        if not 0.0 <= float(self.amp_const) <= 1.0:
            raise ValueError("Amplitude contrast must be in [0, 1]")

        local_cs = np.float32(self.cs * np.float32(1e7))
        local_kv = np.float32(self.voltage * np.float32(1e3))
        self.lambda_value = np.float32(12.2643247) / np.sqrt(
            local_kv * (np.float32(1.0) + local_kv * np.float32(0.978466e-6))
        )
        self.k1 = np.float32(np.pi) * self.lambda_value
        self.k2 = np.float32(np.pi / 2.0) * local_cs * self.lambda_value ** np.float32(3.0)
        self.k3 = np.arctan(
            self.amp_const
            / np.sqrt(np.float32(1.0) - self.amp_const * self.amp_const)
        ).astype(np.float32)
        self.k4 = np.float32(-self.bfactor / np.float32(4.0))
        self.k5 = np.radians(self.phase_shift).astype(np.float32)

        basis_key = (self.size, float(self.pixel_size))
        cached = self._basis_cache.get(basis_key)
        if cached is None:
            n = self.size
            # Build the same centered float32 frequency vector as the original
            # CTF class, then apply only the row/column permutation needed by
            # the packed representation.  This avoids both a full N x N CTF
            # and the small rounding difference introduced by np.fft.fftfreq.
            centered_freq = (
                np.arange(n, dtype=np.float32)
                - np.float32(n) / np.float32(2.0)
            ) / (np.float32(n) * self.pixel_size)
            y_freq = np.concatenate(
                (centered_freq[n // 2 :], centered_freq[: n // 2])
            )
            x_freq = np.concatenate(
                (centered_freq[n // 2 :], centered_freq[:1])
            )
            x, y = np.meshgrid(x_freq, y_freq, indexing="xy")
            x = x.astype(np.float32, copy=False)
            y = y.astype(np.float32, copy=False)
            x2 = x * x
            y2 = y * y
            xy = x * y
            u2 = x2 + y2
            u4 = u2 * u2
            ss = np.sqrt(u2).astype(np.float32, copy=False)
            cached = (x, y, x2, y2, xy, u2, u4, ss)
            self._basis_cache[basis_key] = cached
        self.x, self.y, self.x2, self.y2, self.xy, self.u2, self.u4, self.ss = cached

        weight_key = (self.size, float(self.pixel_size), float(self.kk))
        ncurve = self._weight_cache.get(weight_key)
        if ncurve is None:
            a = np.float32(-9.32)
            b = np.float32(2.65)
            b2 = np.float32(0.01908)
            bfactor1 = np.float32(-78.7757)
            bfactor2 = np.float32(-12.9121)
            bfactor3 = np.float32(1.28732)
            signal = np.exp(
                bfactor1 * self.ss * self.ss + bfactor2 * self.ss + bfactor3,
                dtype=np.float32,
            ) / (self.kk + np.float32(1.0))
            ncurve = np.exp(
                a * self.ss * self.ss + b * self.ss + b2,
                dtype=np.float32,
            ) / signal
            ncurve = ncurve.astype(np.float32, copy=False)
            self._weight_cache[weight_key] = ncurve
        self.ncurve = ncurve

    def one(self, defocus_u: float, defocus_v: float, defocus_angle: float) -> np.ndarray:
        dfu = np.float32(defocus_u)
        dfv = np.float32(defocus_v)
        angle = np.radians(np.float32(defocus_angle)).astype(np.float32)
        sin_az = np.sin(angle).astype(np.float32)
        cos_az = np.cos(angle).astype(np.float32)
        axx = -dfu * cos_az * cos_az - dfv * sin_az * sin_az
        ayy = -dfu * sin_az * sin_az - dfv * cos_az * cos_az
        axy = -dfu * sin_az * cos_az + dfv * sin_az * cos_az
        # Keep the original operation order (Axx * X * X rather than
        # Axx * X2) so the NumPy reference backend matches the old full CTF
        # before permutation down to the last float32 bit.
        gamma = (
            self.k1
            * (
                axx * self.x * self.x
                + 2.0 * axy * self.x * self.y
                + ayy * self.y * self.y
            )
            + self.k2 * self.u4
            - self.k5
            - self.k3
        )
        value = -np.sin(gamma, dtype=np.float32)
        if self.bfactor != 0.0:
            value *= np.exp(self.k4 * self.u2, dtype=np.float32)
        value *= self.scale
        value = np.where(
            np.abs(value) < np.float32(1e-8),
            np.sign(value) * np.float32(1e-8),
            value,
        ).astype(np.float32, copy=False)
        weighted = value * np.sqrt(
            np.float32(1.0) / (self.ncurve + self.kk * value * value),
            dtype=np.float32,
        )
        return weighted.astype(np.float32, copy=False)

    def batch(self, records: Sequence[ParticleRecord], has_ctf: bool) -> np.ndarray:
        if not has_ctf:
            return np.ones(
                (len(records), self.size, self.size // 2 + 1),
                dtype=np.float32,
            )
        return np.stack(
            [self.one(r.defocus_u, r.defocus_v, r.defocus_angle) for r in records],
            axis=0,
        )


class PackedCTFTorch:
    """Batched packed CTF implementation residing entirely on the selected device."""

    def __init__(
        self,
        size: int,
        pixel_size: float,
        cs: float,
        voltage: float,
        kk: float,
        device: torch.device,
        amp_const: float = 0.1,
        bfactor: float = 0.0,
        scale: float = 1.0,
        phase_shift: float = 0.0,
    ) -> None:
        self.size = int(size)
        self.device = device
        self.kk = float(kk)
        self.bfactor = float(bfactor)
        self.scale = float(scale)
        if not 0.0 <= amp_const <= 1.0:
            raise ValueError("Amplitude contrast must be in [0, 1]")

        local_cs = np.float32(cs) * np.float32(1e7)
        local_kv = np.float32(voltage) * np.float32(1e3)
        lambda_value = np.float32(12.2643247) / np.sqrt(
            local_kv * (np.float32(1.0) + local_kv * np.float32(0.978466e-6))
        )
        self.k1 = float(np.float32(np.pi) * lambda_value)
        self.k2 = float(
            np.float32(np.pi / 2.0)
            * local_cs
            * lambda_value ** np.float32(3.0)
        )
        self.k3 = float(
            np.arctan(
                np.float32(amp_const)
                / np.sqrt(np.float32(1.0) - np.float32(amp_const) ** 2)
            ).astype(np.float32)
        )
        self.k4 = float(np.float32(-bfactor / 4.0))
        self.k5 = float(np.radians(np.float32(phase_shift)).astype(np.float32))

        # Static frequency/weight terms are constructed once with the exact
        # float32 NumPy layout used by the reference backend, then transferred
        # to the device.  Only the per-particle CTF trigonometry is batched in
        # PyTorch.
        n = self.size
        pixel_size32 = np.float32(pixel_size)
        centered_freq = (
            np.arange(n, dtype=np.float32)
            - np.float32(n) / np.float32(2.0)
        ) / (np.float32(n) * pixel_size32)
        y_freq = np.concatenate(
            (centered_freq[n // 2 :], centered_freq[: n // 2])
        )
        x_freq = np.concatenate(
            (centered_freq[n // 2 :], centered_freq[:1])
        )
        x_np, y_np = np.meshgrid(x_freq, y_freq, indexing="xy")
        x_np = x_np.astype(np.float32, copy=False)
        y_np = y_np.astype(np.float32, copy=False)
        u2_np = x_np * x_np + y_np * y_np
        u4_np = u2_np * u2_np
        ss_np = np.sqrt(u2_np).astype(np.float32, copy=False)
        signal_np = np.exp(
            np.float32(-78.7757) * ss_np * ss_np
            + np.float32(-12.9121) * ss_np
            + np.float32(1.28732),
            dtype=np.float32,
        ) / (np.float32(self.kk) + np.float32(1.0))
        ncurve_np = np.exp(
            np.float32(-9.32) * ss_np * ss_np
            + np.float32(2.65) * ss_np
            + np.float32(0.01908),
            dtype=np.float32,
        ) / signal_np
        self.x = torch.from_numpy(x_np).to(device)
        self.y = torch.from_numpy(y_np).to(device)
        self.u4 = torch.from_numpy(u4_np).to(device)
        self.ncurve = torch.from_numpy(ncurve_np.astype(np.float32, copy=False)).to(device)

    def batch(self, records: Sequence[ParticleRecord], has_ctf: bool) -> torch.Tensor:
        batch_size = len(records)
        if not has_ctf:
            return torch.ones(
                (batch_size, self.size, self.size // 2 + 1),
                dtype=torch.float32,
                device=self.device,
            )
        dfu = torch.tensor(
            [r.defocus_u for r in records],
            dtype=torch.float32,
            device=self.device,
        ).view(-1, 1, 1)
        dfv = torch.tensor(
            [r.defocus_v for r in records],
            dtype=torch.float32,
            device=self.device,
        ).view(-1, 1, 1)
        angle = torch.deg2rad(
            torch.tensor(
                [r.defocus_angle for r in records],
                dtype=torch.float32,
                device=self.device,
            )
        ).view(-1, 1, 1)
        sin_az = torch.sin(angle)
        cos_az = torch.cos(angle)
        axx = -dfu * cos_az * cos_az - dfv * sin_az * sin_az
        ayy = -dfu * sin_az * sin_az - dfv * cos_az * cos_az
        axy = -dfu * sin_az * cos_az + dfv * sin_az * cos_az
        gamma = (
            self.k1
            * (
                axx * self.x * self.x
                + 2.0 * axy * self.x * self.y
                + ayy * self.y * self.y
            )
            + self.k2 * self.u4
            - self.k5
            - self.k3
        )
        value = -torch.sin(gamma)
        if self.bfactor != 0.0:
            value = value * torch.exp(self.k4 * (self.x * self.x + self.y * self.y))
        if self.scale != 1.0:
            value = value * self.scale
        value = torch.where(
            torch.abs(value) < 1e-8,
            torch.sign(value) * 1e-8,
            value,
        )
        return value * torch.sqrt(1.0 / (self.ncurve + self.kk * value * value))


# ---------------------------------------------------------------------------
# Integrated projector and whitening
# ---------------------------------------------------------------------------


def apply_sinc2_filter_inplace_3d(volume: torch.Tensor, z_chunk: int = 8) -> None:
    if volume.ndim != 3:
        raise ValueError("volume must be 3-D")
    depth, height, width = volume.shape
    device = volume.device
    fz = torch.fft.fftshift(torch.fft.fftfreq(depth, device=device, dtype=torch.float32))
    fy = torch.fft.fftshift(torch.fft.fftfreq(height, device=device, dtype=torch.float32))
    fx = torch.fft.fftshift(torch.fft.fftfreq(width, device=device, dtype=torch.float32))
    fy2 = fy.view(1, height, 1) ** 2
    fx2 = fx.view(1, 1, width) ** 2
    for z0 in range(0, depth, z_chunk):
        z1 = min(z0 + z_chunk, depth)
        fz2 = fz[z0:z1].view(z1 - z0, 1, 1) ** 2
        radius = torch.sqrt(fz2 + fy2 + fx2)
        filt = torch.sinc(radius)
        filt.square_()
        volume[z0:z1].mul_(filt)


def build_projection_base_grid(image_shape: Tuple[int, int, int], device: torch.device) -> torch.Tensor:
    _, height, width = image_shape
    freq_y = torch.fft.fftfreq(height, device=device, dtype=torch.float32)
    freq_y = torch.fft.fftshift(freq_y)
    freq_x = torch.fft.rfftfreq(width, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(freq_y, freq_x, indexing="ij")
    zz = torch.zeros_like(yy)
    return torch.stack((zz, yy, xx), dim=-1)  # z, y, x


def rotated_grid_to_grid_sample(
    base_grid_zyx: torch.Tensor,
    rotation_matrices_xyz: torch.Tensor,
    image_shape: Tuple[int, int, int],
    dft_shape: Tuple[int, int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    # v605/libtilt path flips zyx -> xyz, applies R @ vector, then flips back.
    base_xyz = torch.flip(base_grid_zyx, dims=(-1,))
    rotated_xyz = torch.matmul(
        rotation_matrices_xyz[:, None, None, :, :],
        base_xyz[None, :, :, :, None],
    ).squeeze(-1)
    grid_zyx = torch.flip(rotated_xyz, dims=(-1,))

    conjugate_mask = grid_zyx[..., 2] < 0
    grid_zyx = torch.where(conjugate_mask.unsqueeze(-1), -grid_zyx, grid_zyx)

    depth, height, width = image_shape
    coordinates = torch.empty_like(grid_zyx)
    coordinates[..., 0] = grid_zyx[..., 0] * depth + depth // 2
    coordinates[..., 1] = grid_zyx[..., 1] * height + height // 2
    coordinates[..., 2] = grid_zyx[..., 2] * width

    dft_shape_t = torch.tensor(
        dft_shape,
        dtype=coordinates.dtype,
        device=coordinates.device,
    )
    normalized = coordinates / (0.5 * dft_shape_t - 0.5) - 1.0
    normalized = torch.flip(normalized, dims=(-1,))  # grid_sample expects x, y, z
    return normalized, conjugate_mask


class ProjectionWhitening:
    def __init__(self, size: int, device: torch.device) -> None:
        self.size = int(size)
        y = torch.arange(size, device=device, dtype=torch.float32)
        x = torch.arange(size, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        center = size // 2
        self.radius_map = torch.round(
            torch.sqrt((yy - center) ** 2 + (xx - center) ** 2)
        ).long()
        self.radius_flat = self.radius_map.reshape(-1)
        self.max_radius = int(self.radius_map.max().item()) + 1
        self.radius_counts = torch.bincount(
            self.radius_flat,
            minlength=self.max_radius,
        ).to(torch.float32)

        # Match the old call-site swap: xx_torch receives yy and yy_torch receives xx.
        self.coord_x = yy
        self.coord_y = xx
        x_flat = self.coord_x.reshape(-1)
        y_flat = self.coord_y.reshape(-1)
        ones = torch.ones_like(x_flat)
        self.plane_matrix = torch.stack(
            (
                torch.stack((torch.sum(x_flat * x_flat), torch.sum(x_flat * y_flat), torch.sum(x_flat))),
                torch.stack((torch.sum(x_flat * y_flat), torch.sum(y_flat * y_flat), torch.sum(y_flat))),
                torch.stack((torch.sum(x_flat), torch.sum(y_flat), torch.sum(ones))),
            )
        )

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 3:
            raise ValueError("Projection whitening expects (B,H,W)")
        batch = images.shape[0]

        # The integrated projector deliberately calls whitening one projection
        # at a time.  Use the same tensor ranks, multiplication order and
        # reductions as the old project3d code so its float32 output is retained.
        if batch == 1:
            image = images[0]
            image_fft = torch.fft.fftshift(torch.fft.fft2(image))
            dist_flat = self.radius_map.reshape(-1)
            fft_abs_flat = torch.abs(image_fft).reshape(-1)
            spectrum = torch.zeros(
                self.max_radius,
                device=image.device,
                dtype=fft_abs_flat.dtype,
            )
            count_freq = torch.zeros_like(spectrum)
            spectrum.scatter_add_(0, dist_flat, fft_abs_flat)
            count_freq.scatter_add_(0, dist_flat, torch.ones_like(fft_abs_flat))
            div_spec = torch.ones_like(spectrum)
            nonzero = count_freq > 0
            div_spec[nonzero] = count_freq[nonzero] / spectrum[nonzero]
            div_spec[0] = 1.0
            image_fft = image_fft * div_spec[self.radius_map]
            image = torch.fft.ifft2(torch.fft.ifftshift(image_fft)).real

            x = self.coord_x.reshape(-1)
            y = self.coord_y.reshape(-1)
            z = image.reshape(-1)
            w = torch.ones_like(z, dtype=torch.float32)
            w2 = w * w
            sw2 = torch.sum(w2)
            sw2x = torch.sum(w2 * x)
            sw2y = torch.sum(w2 * y)
            sw2z = torch.sum(w2 * z)
            sw2xx = torch.sum(w2 * x * x)
            sw2xy = torch.sum(w2 * x * y)
            sw2xz = torch.sum(w2 * x * z)
            sw2yy = torch.sum(w2 * y * y)
            sw2yz = torch.sum(w2 * y * z)
            matrix = torch.stack(
                (
                    torch.stack((sw2xx, sw2xy, sw2x)),
                    torch.stack((sw2xy, sw2yy, sw2y)),
                    torch.stack((sw2x, sw2y, sw2)),
                )
            )
            vector = torch.stack((sw2xz, sw2yz, sw2z))
            coeff = torch.linalg.solve(matrix, vector)
            image = image - (
                coeff[0] * self.coord_x
                + coeff[1] * self.coord_y
                + coeff[2]
            )
            mean = image.mean()
            std = image.std(unbiased=False)
            if bool((std > 1e-12).item()):
                image = (image - mean) / std
            else:
                image = image - mean
            return image.unsqueeze(0)

        image_fft = torch.fft.fftshift(torch.fft.fft2(images, dim=(-2, -1)), dim=(-2, -1))
        abs_flat = torch.abs(image_fft).reshape(batch, -1)
        indices = self.radius_flat.view(1, -1).expand(batch, -1)
        spectrum = torch.zeros(
            (batch, self.max_radius),
            dtype=abs_flat.dtype,
            device=images.device,
        )
        spectrum.scatter_add_(1, indices, abs_flat)
        div_spec = self.radius_counts.view(1, -1) / spectrum
        div_spec[:, 0] = 1.0
        image_fft = image_fft * div_spec[:, self.radius_map]
        images = torch.fft.ifft2(
            torch.fft.ifftshift(image_fft, dim=(-2, -1)),
            dim=(-2, -1),
        ).real

        flat = images.reshape(batch, -1)
        bvec = torch.stack(
            (
                torch.sum(flat * self.coord_x.reshape(1, -1), dim=1),
                torch.sum(flat * self.coord_y.reshape(1, -1), dim=1),
                torch.sum(flat, dim=1),
            ),
            dim=1,
        )
        matrix = self.plane_matrix.unsqueeze(0).expand(batch, -1, -1)
        coeff = torch.linalg.solve(matrix, bvec.unsqueeze(-1)).squeeze(-1)
        plane = (
            coeff[:, 0, None, None] * self.coord_x
            + coeff[:, 1, None, None] * self.coord_y
            + coeff[:, 2, None, None]
        )
        images = images - plane
        mean = images.mean(dim=(-2, -1), keepdim=True)
        std = images.std(dim=(-2, -1), correction=0, keepdim=True)
        centered = images - mean
        return torch.where(std > 1e-12, centered / std, centered)


class IntegratedProjector:
    def __init__(
        self,
        device: torch.device,
        original_box: int,
        new_box: int,
        layout: PackedFourierLayout,
        real_space_mask: Optional[torch.Tensor],
        packed_fsc: torch.Tensor,
        projection_batch_size: int,
    ) -> None:
        self.device = device
        self.original_box = int(original_box)
        self.new_box = int(new_box)
        self.layout = layout
        self.real_space_mask = real_space_mask
        self.packed_fsc = packed_fsc
        self.batch_size = max(1, int(projection_batch_size))

    def build_templates(self, mrc_volume: str, angles: ModelAngles) -> torch.Tensor:
        with mrcfile.mmap(mrc_volume, mode="r") as handle:
            volume_np = np.array(handle.data, dtype=np.float32, copy=True, order="C")
        if volume_np.ndim != 3 or len(set(volume_np.shape)) != 1:
            raise ValueError(f"Model MRC must be cubic; got shape {volume_np.shape}")
        if volume_np.shape[0] != self.original_box:
            raise ValueError(
                f"Model MRC box {volume_np.shape[0]} does not match --oriboxsize {self.original_box}"
            )

        volume = torch.from_numpy(volume_np).to(self.device)
        del volume_np
        n = self.original_box
        pad_length = n // 2
        volume = F.pad(volume, [pad_length] * 6, mode="constant", value=0.0)
        apply_sinc2_filter_inplace_3d(volume, z_chunk=8)

        image_shape = tuple(int(x) for x in volume.shape)
        dft = torch.fft.fftshift(volume, dim=(-3, -2, -1))
        dft = torch.fft.rfftn(dft, dim=(-3, -2, -1))
        dft = torch.fft.fftshift(dft, dim=(-3, -2))
        dft_real = torch.view_as_real(dft).permute(3, 0, 1, 2).unsqueeze(0)
        dft_shape = tuple(int(x) for x in dft.shape)
        base_grid = build_projection_base_grid(image_shape, self.device)
        whitening = ProjectionWhitening(n, self.device)

        templates = torch.empty(
            (len(angles.rot), self.new_box, self.layout.rfft_size),
            dtype=torch.complex64,
            device=self.device,
        )
        crop_low = n // 2 - self.new_box // 2
        crop_high = n // 2 + self.new_box // 2

        for start in range(0, len(angles.rot), self.batch_size):
            end = min(start + self.batch_size, len(angles.rot))
            matrices = torch.from_numpy(
                angles.rotation_matrices_t[start:end]
            ).to(self.device)
            sample_grid, conjugate_mask = rotated_grid_to_grid_sample(
                base_grid,
                matrices,
                image_shape=image_shape,
                dft_shape=dft_shape,
            )
            # Treat the projection batch as D_out in one 5-D grid_sample call.
            sampled = F.grid_sample(
                dft_real,
                sample_grid.unsqueeze(0),
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            sampled_complex = torch.view_as_complex(
                sampled.squeeze(0).permute(1, 2, 3, 0).contiguous()
            )
            sampled_complex = torch.where(
                conjugate_mask,
                torch.conj(sampled_complex),
                sampled_complex,
            )

            # Keep the inverse projection FFT, whitening, template FFT and
            # normalization one orientation at a time.  Batched irFFT/FFT plans
            # can change the last float32 bits; whitening amplifies those small
            # differences.  Central-slice sampling remains batched, while this
            # per-orientation tail reproduces the v605 projector path much more
            # closely and still avoids every GPU->CPU->GPU transfer.
            for local_index in range(end - start):
                projection_fourier = sampled_complex[local_index : local_index + 1]
                projection = torch.fft.ifftshift(projection_fourier, dim=(-2,))
                projection = torch.fft.irfftn(projection, dim=(-2, -1))
                projection = torch.fft.ifftshift(projection, dim=(-2, -1))
                projection = projection[
                    :,
                    pad_length:-pad_length,
                    pad_length:-pad_length,
                ].to(torch.float32)
                projection = whitening(projection)
                if self.real_space_mask is not None:
                    projection = projection * self.real_space_mask

                full_fft = torch.fft.fft2(projection, dim=(-2, -1))
                centered_fft = torch.fft.fftshift(full_fft, dim=(-2, -1))
                if self.new_box != n:
                    centered_fft = centered_fft[
                        :,
                        crop_low:crop_high,
                        crop_low:crop_high,
                    ]
                packed = self.layout.centered_full_to_packed(centered_fft)
                packed = packed * self.packed_fsc
                norm = torch.linalg.vector_norm(packed.reshape(1, -1), dim=1)
                packed = packed / norm[:, None, None]
                templates[start + local_index] = self.layout.pre_shift_for_correlation(packed)[0]

            del matrices, sample_grid, conjugate_mask, sampled, sampled_complex
            del projection_fourier, projection, full_fft, centered_fft, packed, norm

        del dft_real, dft, volume, base_grid, whitening
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return templates


# ---------------------------------------------------------------------------
# Cached psi grids and one-particle batched preparation
# ---------------------------------------------------------------------------


def psi_sampling_for_record(
    particle_psi: float,
    psi_step: float,
    do_local_search: bool,
    local_range: float,
) -> Tuple[List[float], List[float]]:
    num_psi = int(360 // psi_step)
    if num_psi < 1:
        num_psi = 1
    tmp = 999999.0
    if do_local_search:
        tmp = local_range // psi_step
        local_count = int(tmp * 2 + 1)
        if local_count < num_psi:
            num_psi = local_count

    fake_angles: List[float] = []
    effective_angles: List[float] = []
    nearest = round(((particle_psi + 360.0) % 360.0) / psi_step) * psi_step
    nearest %= 360.0
    for psi_index in range(int(num_psi)):
        angle = psi_step * psi_index
        if do_local_search:
            angle = nearest - tmp * psi_step + psi_index * psi_step
        fake_angles.append(float(angle))
        effective = 0.5 if abs(angle) < 0.5 else angle
        effective_angles.append(float(effective))
    return fake_angles, effective_angles


class PsiGridBank:
    def __init__(
        self,
        size: int,
        device: torch.device,
        unique_angles: Sequence[float],
        build_batch_size: int = 8,
    ) -> None:
        self.size = int(size)
        self.device = device
        self.angle_keys = [float(np.round(a, 6)) for a in unique_angles]
        self.angle_to_index = {angle: i for i, angle in enumerate(self.angle_keys)}
        self.grids = torch.empty(
            (len(self.angle_keys), size, size, 2),
            dtype=torch.float32,
            device=device,
        )
        if not self.angle_keys:
            return

        yy, xx = torch.meshgrid(
            torch.arange(size, device=device, dtype=torch.float32),
            torch.arange(size, device=device, dtype=torch.float32),
            indexing="ij",
        )
        coords = torch.stack((yy, xx), dim=-1).reshape(-1, 2)
        center = np.asarray((size, size), dtype=np.float32) / np.float32(2.0)
        build_batch_size = max(1, int(build_batch_size))

        for start in range(0, len(self.angle_keys), build_batch_size):
            end = min(start + build_batch_size, len(self.angle_keys))
            matrices_np = []
            offsets_np = []
            for angle in self.angle_keys[start:end]:
                theta = np.deg2rad(angle)
                c = np.float32(np.cos(theta))
                s = np.float32(np.sin(theta))
                matrix = np.asarray([[c, -s], [s, c]], dtype=np.float32)
                offset = center - matrix @ center
                matrices_np.append(matrix)
                offsets_np.append(offset.astype(np.float32, copy=False))
            matrices = torch.from_numpy(np.stack(matrices_np)).to(device)
            offsets = torch.from_numpy(np.stack(offsets_np)).to(device)
            source = torch.matmul(coords.unsqueeze(0), matrices.transpose(1, 2))
            source = source + offsets[:, None, :]
            y = source[..., 0]
            x = source[..., 1]
            y_norm = 2.0 * y / (size - 1) - 1.0 if size > 1 else torch.zeros_like(y)
            x_norm = 2.0 * x / (size - 1) - 1.0 if size > 1 else torch.zeros_like(x)
            grid = torch.stack((x_norm, y_norm), dim=-1).reshape(-1, size, size, 2)
            self.grids[start:end].copy_(grid)

    def indices_for_angles(self, effective_angles: Sequence[float]) -> torch.Tensor:
        indices = []
        for angle in effective_angles:
            key = float(np.round(angle, 6))
            try:
                indices.append(self.angle_to_index[key])
            except KeyError as exc:
                raise KeyError(f"Psi grid angle {key} was not precomputed") from exc
        return torch.tensor(indices, dtype=torch.long, device=self.device)

    def get(self, effective_angles: Sequence[float]) -> torch.Tensor:
        return self.grids.index_select(0, self.indices_for_angles(effective_angles))


class ParticleMRCReader:
    """Sequential mmap reader with one reusable pinned host staging image."""

    def __init__(self, expected_size: int, device: torch.device) -> None:
        self.expected_size = int(expected_size)
        self.device = device
        self.last_filename: Optional[str] = None
        self.last_mrc: Optional[mrcfile.mrcfile.MrcFile] = None
        self.last_data: Optional[np.ndarray] = None
        self.host_staging = torch.empty(
            (expected_size, expected_size),
            dtype=torch.float32,
            pin_memory=(device.type == "cuda"),
        )
        self.host_staging_np = self.host_staging.numpy()
        self.device_staging = torch.empty(
            (expected_size, expected_size),
            dtype=torch.float32,
            device=device,
        )

    @staticmethod
    def resolve_filename(filename: str) -> str:
        return os.path.realpath(filename) if os.path.islink(filename) else filename

    def _open_if_needed(self, filename: str) -> None:
        filename = self.resolve_filename(filename)
        if filename == self.last_filename:
            return
        if self.last_mrc is not None:
            self.last_mrc.close()
        self.last_mrc = mrcfile.mmap(filename, mode="r")
        self.last_data = self.last_mrc.data
        self.last_filename = filename

    def read(self, image_name: str) -> torch.Tensor:
        image_serial, filename = image_name.split("@", 1)
        image_index = int(image_serial) - 1
        self._open_if_needed(filename)
        assert self.last_data is not None
        data = self.last_data
        if data.ndim < 3:
            image = data
        else:
            image = data[image_index]
        if image.shape != (self.expected_size, self.expected_size):
            raise ValueError(
                f"Particle image shape {image.shape} does not match --oriboxsize "
                f"{self.expected_size} in {image_name}"
            )
        np.copyto(self.host_staging_np, np.asarray(image, dtype=np.float32), casting="unsafe")
        self.device_staging.copy_(self.host_staging, non_blocking=False)
        return self.device_staging

    def close(self) -> None:
        if self.last_mrc is not None:
            self.last_mrc.close()
        self.last_mrc = None
        self.last_data = None
        self.last_filename = None


class ParticleFourierPreparer:
    def __init__(
        self,
        original_box: int,
        new_box: int,
        device: torch.device,
        layout: PackedFourierLayout,
        real_space_mask: Optional[torch.Tensor],
        psi_grid_bank: PsiGridBank,
    ) -> None:
        self.original_box = int(original_box)
        self.new_box = int(new_box)
        self.device = device
        self.layout = layout
        self.real_space_mask = real_space_mask
        self.psi_grid_bank = psi_grid_bank
        self.crop_low = original_box // 2 - new_box // 2
        self.crop_high = original_box // 2 + new_box // 2

    def prepare(
        self,
        image: torch.Tensor,
        effective_angles: Sequence[float],
        packed_ctf: torch.Tensor,
    ) -> torch.Tensor:
        grids = self.psi_grid_bank.get(effective_angles)
        n_psi = grids.shape[0]
        image_batch = image.view(1, 1, self.original_box, self.original_box).expand(
            n_psi, -1, -1, -1
        )
        rotated = F.grid_sample(
            image_batch,
            grids,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(1)
        if self.real_space_mask is not None:
            rotated = rotated * self.real_space_mask

        full_fft = torch.fft.fft2(rotated, dim=(-2, -1))
        centered_fft = torch.fft.fftshift(full_fft, dim=(-2, -1))
        if self.new_box != self.original_box:
            centered_fft = centered_fft[
                :,
                self.crop_low:self.crop_high,
                self.crop_low:self.crop_high,
            ]
        packed = self.layout.centered_full_to_packed(centered_fft)
        packed = packed * packed_ctf.unsqueeze(0)
        packed = torch.conj(packed)
        norms = torch.linalg.vector_norm(packed.reshape(n_psi, -1), dim=1)
        packed = packed / norms[:, None, None]
        return self.layout.pre_shift_for_correlation(packed)


# ---------------------------------------------------------------------------
# Correlation search
# ---------------------------------------------------------------------------


class CorrelationWorkspace:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.capacity_a = 0
        self.capacity_b = 0
        self.scores: Optional[torch.Tensor] = None
        self.y_positions: Optional[torch.Tensor] = None
        self.x_positions: Optional[torch.Tensor] = None

    def ensure(self, n_a: int, n_b: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if n_a > self.capacity_a or n_b > self.capacity_b or self.scores is None:
            self.capacity_a = max(n_a, self.capacity_a)
            self.capacity_b = max(n_b, self.capacity_b)
            self.scores = torch.empty(
                (self.capacity_a, self.capacity_b),
                dtype=torch.float32,
                device=self.device,
            )
            self.y_positions = torch.empty(
                (self.capacity_a, self.capacity_b),
                dtype=torch.long,
                device=self.device,
            )
            self.x_positions = torch.empty(
                (self.capacity_a, self.capacity_b),
                dtype=torch.long,
                device=self.device,
            )
        assert self.y_positions is not None and self.x_positions is not None
        return (
            self.scores[:n_a, :n_b],
            self.y_positions[:n_a, :n_b],
            self.x_positions[:n_a, :n_b],
        )


def correlate_templates_and_particle(
    templates_pre_shifted: torch.Tensor,
    particle_pre_shifted: torch.Tensor,
    layout: PackedFourierLayout,
    workspace: CorrelationWorkspace,
    chunk_size_a: int = 16,
    chunk_size_b: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_a = templates_pre_shifted.shape[0]
    n_b = particle_pre_shifted.shape[0]
    scores, y_positions, x_positions = workspace.ensure(n_a, n_b)
    multiplier = float(1024 * layout.size * layout.size)

    for i in range(0, n_a, chunk_size_a):
        end_a = min(i + chunk_size_a, n_a)
        chunk_a = templates_pre_shifted[i:end_a]
        for j in range(0, n_b, chunk_size_b):
            end_b = min(j + chunk_size_b, n_b)
            chunk_b = particle_pre_shifted[j:end_b]
            product = chunk_a[:, None, :, :] * chunk_b[None, :, :, :]
            ccg = torch.fft.irfft2(
                product,
                s=(layout.size, layout.size),
                dim=(-2, -1),
            )
            crop = layout.centered_ccg_crop_from_unshifted(ccg)
            flat = crop.reshape(crop.shape[0], crop.shape[1], -1)
            max_value, max_index = torch.max(flat, dim=-1)
            scores[i:end_a, j:end_b].copy_(max_value * multiplier)
            y_positions[i:end_a, j:end_b].copy_(max_index // layout.crop_size)
            x_positions[i:end_a, j:end_b].copy_(max_index % layout.crop_size)
    return scores, y_positions, x_positions


def make_result_records(
    particle: ParticleRecord,
    fake_psi: Sequence[float],
    model_rot: Sequence[float],
    model_tilt: Sequence[float],
    model_indices: Sequence[int],
    scores: torch.Tensor,
    y_positions: torch.Tensor,
    x_positions: torch.Tensor,
    crop_size: int,
    trans_scale_factor: float,
    cc_threshold: float = 3.0,
    accuracy: float = 11.0,
) -> List[List[object]]:
    flat_scores = scores.reshape(-1)
    if flat_scores.numel() == 0:
        return []
    flat_peak_index = int(torch.argmax(flat_scores).item())
    n_psi = scores.shape[1]
    local_model_index = flat_peak_index // n_psi
    psi_index = flat_peak_index % n_psi
    peak_value = scores[local_model_index, psi_index]
    peak_y = int(y_positions[local_model_index, psi_index].item())
    peak_x = int(x_positions[local_model_index, psi_index].item())

    rem_xshift = float(crop_size // 2 - peak_x) / trans_scale_factor
    rem_yshift = float(crop_size // 2 - peak_y) / trans_scale_factor
    global_model_index = int(model_indices[local_model_index])
    angle_distance = calculate_angular_distance(
        model_rot[global_model_index],
        model_tilt[global_model_index],
        0.0,
        particle.rot,
        particle.tilt,
        0.0,
    )
    cc_mean = flat_scores.mean()
    cc_sigma = flat_scores.std(correction=0)
    cc_t1 = (peak_value - cc_mean) / cc_sigma
    cc_l3 = 1 if bool((cc_t1 > cc_threshold).item()) else 0
    cc_dis_accu = 1 if angle_distance < accuracy else 0

    return [
        [
            "cc: ",
            particle.line_index,
            float(cc_t1.item()),
            cc_l3,
            cc_dis_accu,
            model_rot[global_model_index],
            model_tilt[global_model_index],
            angle_distance,
            round(rem_xshift),
            round(rem_yshift),
            0,
            0,
        ],
        [
            "psi: ",
            particle.line_index,
            float(fake_psi[psi_index]),
            float((particle.psi + 360.0) % 360.0),
        ],
        ["rawCC: ", particle.line_index, float(peak_value.item())],
    ]


# ---------------------------------------------------------------------------
# CLI and main search
# ---------------------------------------------------------------------------


def create_search_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimized native-PyTorch cryo-EM model/particle local search"
    )
    parser.add_argument("--i", type=str, required=True, help="Compatibility argument; model angles are read from --ang")
    parser.add_argument("--p", type=str, required=True, help="Input particle STAR")
    parser.add_argument("--FSC", type=str, default="", help="Input FSC file")
    parser.add_argument("--o", type=str, required=True, help="Output search text file")
    parser.add_argument("--kk", type=float, default=0.0)
    parser.add_argument("--gpuid", type=int, default=0)
    parser.add_argument("--oriboxsize", type=int, default=256)
    parser.add_argument("--newboxsize", type=int, default=256)
    parser.add_argument("--apix", type=float, default=1.42, help="Original particle pixel size")
    parser.add_argument("--transRange", type=int, default=0, help="Compatibility argument; CCG peak determines translation")
    parser.add_argument("--voltage", type=float, default=300.0)
    parser.add_argument("--cs", type=float, default=2.7)
    parser.add_argument("--maskRadius", type=int, default=110)
    parser.add_argument("--maskEdge", type=int, default=6)
    parser.add_argument("--ignoreFSC", action="store_true")
    parser.add_argument("--discardMask", action="store_true")
    parser.add_argument("--psiStep", type=float, default=15.0)
    parser.add_argument("--start", type=int, default=-1)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--doLocalSearch", action="store_true")
    parser.add_argument("--localRange", type=float, default=20.0)
    parser.add_argument("--doContinue", action="store_true")
    parser.add_argument("--ang", type=str, required=True, help="RELION angle STAR used directly for projections and model orientations")
    parser.add_argument("--mrc", type=str, required=True, help="Input 3-D model MRC; retained on disk")

    # Optional tuning switches. Existing wrappers need not pass any of these.
    parser.add_argument(
        "--ctfBackend",
        choices=("torch", "numpy"),
        default=os.environ.get("V606_CTF_BACKEND", "torch"),
    )
    parser.add_argument("--ctfBatchSize", type=int, default=256)
    parser.add_argument("--projectionBatchSize", type=int, default=8)
    parser.add_argument("--psiGridBuildBatch", type=int, default=8)
    parser.add_argument("--ccgCropSize", type=int, default=32)
    parser.add_argument("--chunkSizeA", type=int, default=16)
    parser.add_argument("--chunkSizeB", type=int, default=32)
    parser.add_argument("--outputBufferParticles", type=int, default=96)
    parser.add_argument("--torchThreads", type=int, default=4)
    parser.add_argument("--profileStages", action="store_true")
    return parser


def choose_device(gpuid: int) -> torch.device:
    if torch.cuda.is_available():
        torch.cuda.set_device(gpuid)
        return torch.device(f"cuda:{gpuid}")
    print("WARNING: CUDA is unavailable; running on CPU.", file=sys.stderr)
    return torch.device("cpu")


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def chunks(sequence: Sequence[ParticleRecord], size: int) -> Iterable[Sequence[ParticleRecord]]:
    size = max(1, int(size))
    for start in range(0, len(sequence), size):
        yield sequence[start:start + size]


def run_search(args: argparse.Namespace) -> None:
    total_start = time.time()
    torch.set_num_threads(max(1, int(args.torchThreads)))
    device = choose_device(args.gpuid)

    original_box = int(args.oriboxsize)
    new_box = int(args.newboxsize)
    if original_box % 2 or new_box % 2:
        raise ValueError("oriboxsize and newboxsize must be even")
    if new_box > original_box:
        raise ValueError("newboxsize cannot exceed oriboxsize")
    if args.psiStep <= 0:
        raise ValueError("psiStep must be positive")

    layout = PackedFourierLayout(new_box, int(args.ccgCropSize), device)
    new_apix = original_box / new_box * float(args.apix)
    trans_scale_factor = float(args.apix) / new_apix

    print(f"Device: {device}")
    print(f"--i retained for compatibility and not read: {args.i}")
    print(f"Model orientations read directly from: {args.ang}")

    model_angles = read_model_angles(args.ang)
    particle_table = read_particle_table(args.p)
    selected_particles = select_particle_records(particle_table, args.start, args.end)
    print("Model orientation count:", len(model_angles.rot))
    print("Selected particle count:", len(selected_particles))
    print("Particle STAR data start line:", particle_table.loop.data_start)

    if args.doLocalSearch:
        local_start = time.time()
        local_map = load_or_build_local_orientation_map(
            args.p,
            particle_table.records,
            model_angles,
            float(args.localRange),
        )
        print("Local orientation table time =", round(time.time() - local_start, 4), "seconds")
    else:
        all_indices = list(range(len(model_angles.rot)))
        local_map = {rec.line_index: all_indices for rec in selected_particles}

    mask_np = generate_soft_mask(
        (original_box, original_box),
        int(args.maskRadius),
        int(args.maskEdge),
    )
    real_space_mask = None if args.discardMask else torch.from_numpy(mask_np).to(device)

    fsc_centered_np = read_fsc_map(
        args.FSC,
        original_box,
        new_box,
        bool(args.ignoreFSC),
    )
    fsc_centered = torch.from_numpy(fsc_centered_np).to(device)
    packed_fsc = layout.centered_full_to_packed(fsc_centered)

    projection_start = time.time()
    print(f"Time before projections: {projection_start - total_start}")
    projector = IntegratedProjector(
        device=device,
        original_box=original_box,
        new_box=new_box,
        layout=layout,
        real_space_mask=real_space_mask,
        packed_fsc=packed_fsc,
        projection_batch_size=args.projectionBatchSize,
    )
    template_bank = projector.build_templates(args.mrc, model_angles)
    sync_if_cuda(device)
    print(f"Time after generating final GPU templates: {time.time() - total_start}")
    print("Shape of template bank:", tuple(template_bank.shape))

    # The model projection DFT has now been released. Build all psi grids that
    # will be reused across the selected particles.
    psi_metadata: Dict[int, Tuple[List[float], List[float]]] = {}
    unique_angle_keys: Dict[float, None] = {}
    for rec in selected_particles:
        fake, effective = psi_sampling_for_record(
            rec.psi,
            float(args.psiStep),
            bool(args.doLocalSearch),
            float(args.localRange),
        )
        psi_metadata[rec.line_index] = (fake, effective)
        for angle in effective:
            unique_angle_keys[float(np.round(angle, 6))] = None
    psi_grid_start = time.time()
    psi_grid_bank = PsiGridBank(
        size=original_box,
        device=device,
        unique_angles=list(unique_angle_keys.keys()),
        build_batch_size=args.psiGridBuildBatch,
    )
    sync_if_cuda(device)
    print(
        "Psi grid bank:",
        len(unique_angle_keys),
        "unique grids; build time =",
        round(time.time() - psi_grid_start, 4),
        "seconds",
    )

    particle_reader = ParticleMRCReader(original_box, device)
    particle_preparer = ParticleFourierPreparer(
        original_box=original_box,
        new_box=new_box,
        device=device,
        layout=layout,
        real_space_mask=real_space_mask,
        psi_grid_bank=psi_grid_bank,
    )
    workspace = CorrelationWorkspace(device)

    if args.ctfBackend == "torch":
        ctf_generator: object = PackedCTFTorch(
            size=new_box,
            pixel_size=new_apix,
            cs=float(args.cs),
            voltage=float(args.voltage),
            kk=float(args.kk),
            device=device,
        )
    else:
        ctf_generator = PackedCTFNumPy(
            size=new_box,
            pixel_size=new_apix,
            cs=float(args.cs),
            voltage=float(args.voltage),
            kk=float(args.kk),
        )

    output_mode = "a" if args.doContinue else "w"
    output_buffer: List[str] = []
    buffered_particles = 0
    stage_rotation = 0.0
    stage_correlation = 0.0
    search_start = time.time()

    try:
        with open(args.o, output_mode, encoding="utf-8") as output_handle:
            for ctf_records in chunks(selected_particles, int(args.ctfBatchSize)):
                ctf_start = time.time()
                if isinstance(ctf_generator, PackedCTFTorch):
                    ctf_batch = ctf_generator.batch(ctf_records, particle_table.has_ctf)
                else:
                    ctf_np = ctf_generator.batch(ctf_records, particle_table.has_ctf)
                    ctf_batch = torch.from_numpy(ctf_np).to(device)
                if args.profileStages:
                    sync_if_cuda(device)
                    print(
                        f"CTF batch {len(ctf_records)} time = {time.time() - ctf_start:.6f} s"
                    )

                for batch_index, rec in enumerate(ctf_records):
                    model_indices = local_map.get(rec.line_index, [])
                    if not model_indices:
                        print(
                            f"WARNING: no model orientations for particle STAR line {rec.line_index}; skipping",
                            file=sys.stderr,
                        )
                        continue
                    fake_psi, effective_psi = psi_metadata[rec.line_index]

                    t0 = time.time()
                    image = particle_reader.read(rec.image_name)
                    particle_fourier = particle_preparer.prepare(
                        image,
                        effective_psi,
                        ctf_batch[batch_index],
                    )
                    if args.profileStages:
                        sync_if_cuda(device)
                        stage_rotation += time.time() - t0

                    index_tensor = torch.tensor(
                        model_indices,
                        dtype=torch.long,
                        device=device,
                    )
                    local_templates = template_bank.index_select(0, index_tensor)
                    t1 = time.time()
                    scores, y_positions, x_positions = correlate_templates_and_particle(
                        local_templates,
                        particle_fourier,
                        layout,
                        workspace,
                        chunk_size_a=max(1, int(args.chunkSizeA)),
                        chunk_size_b=max(1, int(args.chunkSizeB)),
                    )
                    results = make_result_records(
                        particle=rec,
                        fake_psi=fake_psi,
                        model_rot=model_angles.rot,
                        model_tilt=model_angles.tilt,
                        model_indices=model_indices,
                        scores=scores,
                        y_positions=y_positions,
                        x_positions=x_positions,
                        crop_size=layout.crop_size,
                        trans_scale_factor=trans_scale_factor,
                    )
                    if args.profileStages:
                        sync_if_cuda(device)
                        stage_correlation += time.time() - t1

                    for result in results:
                        output_buffer.append(str(result) + "\n")
                    buffered_particles += 1
                    if buffered_particles >= max(1, int(args.outputBufferParticles)):
                        output_handle.writelines(output_buffer)
                        output_handle.flush()
                        output_buffer.clear()
                        buffered_particles = 0

                    del image, particle_fourier, index_tensor, local_templates

                del ctf_batch

            if output_buffer:
                output_handle.writelines(output_buffer)
                output_handle.flush()
    finally:
        particle_reader.close()

    sync_if_cuda(device)
    print("Total search time =", round(time.time() - search_start, 4), "seconds")
    print("Total execution time =", round(time.time() - total_start, 4), "seconds")
    if args.profileStages:
        print("Accumulated particle rotate/FFT time =", round(stage_rotation, 6), "seconds")
        print("Accumulated correlation/result time =", round(stage_correlation, 6), "seconds")


def main() -> None:
    parser = create_search_parser()
    args = parser.parse_args()
    with torch.inference_mode():
        run_search(args)


if __name__ == "__main__":
    main()
