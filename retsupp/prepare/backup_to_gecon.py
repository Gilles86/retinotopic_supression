"""Back up the modelling inputs of ds-retsupp from the cluster to the g_econ department share (SMB), add-only.

What is copied (user decision, 2026-11-09):
  * raw BIDS: per subject anat, fmap and the magnitude + phase BOLD with sidecars (``rec-NORDIC`` BOLD is left out:
    it is regenerated from magnitude + phase), plus top-level ``dataset_description.json``
  * ``sourcedata`` (behaviour, logs, settings)
  * ``derivatives/fmriprep`` without the surface BOLD (``*func.gii``, never used; the analyses sample themselves):
    T1w-space BOLD, confounds, masks, boldrefs, transforms, anat, FreeSurfer, reports, logs

Rules: the share is append-only. This script only creates directories and puts files that are not yet in its own
done-list; it never deletes or renames anything on the share. Each unit (``sub-XX``, ``sourcedata``,
``fmriprep-sub-XX``, ``fmriprep-top``) writes a manifest (local path, size, local md5, remote path) and, with
``--verify``, re-downloads every remote file and compares md5.

Needs a Kerberos ticket made by the user: ``kinit -c /scratch/gdehol/krb5cc gdehol@D.UZH.CH``.

Usage::

    python -m retsupp.prepare.backup_to_gecon --unit sub-03            # copy one unit
    python -m retsupp.prepare.backup_to_gecon --unit sub-03 --verify   # checksum the remote copies
    python -m retsupp.prepare.backup_to_gecon --list-units
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
from pathlib import Path

SRC = Path('/shares/zne.uzh/gdehol/ds-retsupp')
SHARE = '//idnas32.uzh.ch/g_econ_department$'
DEST = 'projects/2026/dehollander_ruff_retsupp_amsterdam/data/ds-retsupp'
STATE = Path('/shares/zne.uzh/gdehol/ds-retsupp_gecon_backup_2026-11')
SUBJECTS = [f'sub-{i:02d}' for i in range(1, 31)]


def units():
    return ['top', 'sourcedata', 'fmriprep-top'] + SUBJECTS + [f'fmriprep-{s}' for s in SUBJECTS]


def files_for(unit):
    if unit == 'top':
        return [SRC / 'dataset_description.json'] + [p for p in SRC.glob('*.tsv')] + [p for p in SRC.glob('*.json')
                                                                                      if p.name != 'dataset_description.json']
    if unit == 'sourcedata':
        return sorted(p for p in (SRC / 'sourcedata').rglob('*') if p.is_file())
    if unit == 'fmriprep-top':
        fp = SRC / 'derivatives' / 'fmriprep'
        out = [p for p in fp.iterdir() if p.is_file()]
        out += [p for d in ('logs', 'sourcedata') for p in (fp / d).rglob('*') if p.is_file()]
        return sorted(out)
    if unit.startswith('fmriprep-sub-'):
        fp = SRC / 'derivatives' / 'fmriprep'
        sub = unit[len('fmriprep-'):]
        out = [p for p in (fp / sub).rglob('*') if p.is_file() and not p.name.endswith('.func.gii')]
        return sorted(out)
    if unit.startswith('sub-'):
        return sorted(p for p in (SRC / unit).rglob('*') if p.is_file() and 'rec-NORDIC' not in p.name)
    raise ValueError(unit)


def smb(commands, env, check=True):
    """Run smbclient with commands on stdin; returns its output. With check, raises on any NT_STATUS error."""
    # -N: never prompt for a password (otherwise smbclient reads stdin as the password); commands via -c.
    r = subprocess.run(['smbclient', '-N', '--use-kerberos=required', SHARE, '-c', '; '.join(commands)],
                       capture_output=True, text=True, env=env)
    text = r.stdout + r.stderr
    if check and (r.returncode != 0 or 'NT_STATUS' in text):
        bad = [line for line in text.splitlines() if 'NT_STATUS' in line or not line.startswith('putting file')]
        raise RuntimeError(f'smbclient exit {r.returncode}:\n' + '\n'.join(bad[-40:]))
    return text


def md5(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def copy(unit, env, batch=50):
    files = files_for(unit)
    done_fn = STATE / f'{unit}.done.tsv'
    done = set()
    if done_fn.exists():
        done = {line.split('\t')[0] for line in done_fn.read_text().splitlines() if line}
    todo = [p for p in files if str(p.relative_to(SRC)) not in done]
    print(f'{unit}: {len(files)} files, {sum(p.stat().st_size for p in files) / 1e9:.1f} GB; {len(todo)} to copy', flush=True)
    dirs = sorted({str(p.relative_to(SRC).parent) for p in todo} - {'.'})
    mk = []
    for d in dirs:
        parts = d.split('/')
        mk += [f'mkdir "{DEST}/{"/".join(parts[:i + 1])}"' for i in range(len(parts))]
    if True:
        # mkdir of an existing directory reports NAME_COLLISION; that is fine, so no check here.
        smb([f'mkdir "{"/".join(DEST.split("/")[:i + 1])}"' for i in range(len(DEST.split('/')))] + list(dict.fromkeys(mk)),
            env, check=False)
    with open(done_fn, 'a') as log:
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            smb([f'put "{p}" "{DEST}/{p.relative_to(SRC)}"' for p in chunk], env)
            for p in chunk:
                log.write(f'{p.relative_to(SRC)}\t{p.stat().st_size}\n')
            log.flush()
            print(f'  {min(i + batch, len(todo))}/{len(todo)}', flush=True)


def verify(unit, env):
    files = files_for(unit)
    out = STATE / f'{unit}.verify.tsv'
    bad = 0
    with open(out, 'w') as f:
        f.write('path\tsize\tmd5_local\tmd5_remote\tok\n')
        for p in files:
            rel = p.relative_to(SRC)
            r = subprocess.run(['smbclient', '-N', '--use-kerberos=required', SHARE, '-c', f'get "{DEST}/{rel}" -'],
                               capture_output=True, env=env)
            remote = hashlib.md5(r.stdout).hexdigest() if r.returncode == 0 else 'missing'
            local = md5(p)
            ok = remote == local
            bad += not ok
            f.write(f'{rel}\t{p.stat().st_size}\t{local}\t{remote}\t{ok}\n')
    print(f'{unit}: {len(files)} files verified, {bad} mismatching or missing -> {out}', flush=True)
    return bad


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--unit')
    ap.add_argument('--index', type=int, help='1-based index into --list-units (for SLURM arrays)')
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--list-units', action='store_true')
    a = ap.parse_args()
    if a.list_units:
        print('\n'.join(f'{i + 1}\t{u}' for i, u in enumerate(units())))
        raise SystemExit
    unit = a.unit or units()[a.index - 1]
    STATE.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, KRB5CCNAME=os.environ.get('KRB5CCNAME', '/scratch/gdehol/krb5cc'))
    raise SystemExit(1 if (verify(unit, env) if a.verify else copy(unit, env)) else 0)
