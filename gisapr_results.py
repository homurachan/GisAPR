"""Collect per-sample scores and export the matching best-fit PDB."""
import argparse
import glob
import json
import math
from pathlib import Path
import re
import shutil

_POSE = re.compile(r'rotN?\d+p\d+_tiltN?\d+p\d+_psiN?\d+p\d+deg_transN?\d+p\d+_N?\d+p\d+_N?\d+p\d+ANG')


def read_scores(path):
    lines = Path(path).read_text().splitlines()
    if len(lines) % 2:
        raise ValueError(f'Incomplete two-line score record: {path}')
    for i in range(0, len(lines), 2):
        name = lines[i].strip()
        try:
            score = float(lines[i + 1].split()[0])
        except (IndexError, ValueError) as exc:
            raise ValueError(f'Invalid score at line {i + 2} in {path}') from exc
        if not name or not math.isfinite(score):
            raise ValueError(f'Invalid result record in {path} at line {i + 1}')
        yield name, score


def find_pdb(search_name, result_path, output_dir):
    # Prefer the exact originating PDB; rotation tokens alone are not unique
    # when several chains/runs/pixel sizes share a working directory.
    result_path = Path(result_path)
    prefix, suffix = 'ReSuLt_', '.txt'
    if result_path.name.startswith(prefix) and result_path.name.endswith(suffix):
        candidate = result_path.with_name(result_path.name[len(prefix):-len(suffix)])
        if candidate.is_file() and candidate.suffix.lower() == '.pdb':
            return candidate
    name = Path(search_name).name
    if name.startswith('search_v606opt_') and '_orient3degree_' in name:
        label = name[len('search_v606opt_'):].rsplit('_orient3degree_', 1)[0]
        candidate = Path(output_dir) / (label + '.pdb')
        if candidate.is_file():
            return candidate
    match = _POSE.search(name)
    if not match:
        raise FileNotFoundError(f'Cannot find a pose token in search filename: {search_name}')
    matches = sorted(p for p in Path(output_dir).glob('*' + match.group(0) + '*.pdb')
                     if not p.name.endswith('_BEST_FIT.pdb'))
    if len(matches) != 1:
        raise FileNotFoundError(f'Expected one PDB for {search_name}; found {len(matches)}')
    return matches[0]


def finalize_run(args):
    root = Path(str(args.output_name_root))
    directory = root.parent
    prefix = 'ReSuLt_' + root.name
    if getattr(args, 'rotate_chain', None) is not None:
        pattern = re.compile(re.escape(prefix) + r'(?:N?\d+p\d+_)?'
                             + re.escape(str(args.rotate_chain)) + r'_rot')
        files = sorted(p for p in directory.glob('ReSuLt_*.txt') if pattern.match(p.name))
    else:
        files = sorted(p for p in directory.glob('ReSuLt_*.txt') if p.name.startswith(prefix))
    if not files:
        print('No completed search scores for this run; no BEST_FIT PDB was created.')
        return None
    records = []
    for path in files:
        records.extend((name, score, path) for name, score in read_scores(path))
    if not records:
        raise ValueError('No score records found for finalization')
    # Strict > and sorted input preserve the manual helper's tie behavior.
    name, score, result_path = max(records, key=lambda item: item[1])
    pdb = find_pdb(name, result_path, directory)
    match = _POSE.search(pdb.name)
    if not match:
        raise ValueError(f'The winning PDB has no conformation token: {pdb}')
    best = pdb.with_name(pdb.stem + '_BEST_FIT.pdb')
    shutil.copy2(pdb, best)
    log = directory / 'result.log'
    with log.open('w') as output:
        for path in files:
            text = path.read_text()
            output.write(text)
            if text and not text.endswith('\n'):
                output.write('\n')
    summary = directory / (root.name + 'BEST_FIT.json')
    summary.write_text(json.dumps({
        'score': score, 'search_file': name, 'result_file': str(result_path),
        'source_pdb': str(pdb), 'best_fit_pdb': str(best), 'result_log': str(log),
        'result_count': len(records),
    }, indent=2) + '\n')
    print(f'Greatest score: {score}\nFilename: {name}\nBEST_FIT PDB: {best}', flush=True)
    return str(best)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_name_root', required=True, help='The root used by test_op')
    finalize_run(parser.parse_args())
