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
import tempfile
import time
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


INTERVAL = 1.0  # seconds between file operations on the share (its security checks blocked the account at ~6 parallel streams)


def smb_session(commands, env, interval=None):
    """One smbclient connection; commands are fed one per `interval` seconds. Returns the output."""
    interval = INTERVAL if interval is None else interval
    p = subprocess.Popen(['smbclient', '-N', '--use-kerberos=required', SHARE], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    for c in commands:
        p.stdin.write(c + '\n')
        p.stdin.flush()
        time.sleep(interval)
    out, _ = p.communicate('quit\n')
    return out


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
    todo = reconcile(unit, todo[:batch], env) + todo[batch:]
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
    blocked_fn = STATE / f'{unit}.blocked.tsv'
    with open(done_fn, 'a') as log, open(blocked_fn, 'a') as blog:
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            text = smb_session([f'put "{p}" "{DEST}/{p.relative_to(SRC)}"' for p in chunk], env)
            put_ok = {line.split(' as ')[0][len('putting file '):] for line in text.splitlines()
                      if line.startswith('putting file ')}
            refused = [p for p in chunk if str(p) not in put_ok]
            if len(refused) > 5:
                # Not a refusal by name: the share is refusing everything (seen 2026-11-09 ~17:45, reads and
                # writes ACCESS_DENIED for all files). Stop instead of tarring the whole unit.
                raise RuntimeError(f'{len(refused)} of {len(chunk)} puts refused; stopping:\n' + text[-2000:])
            for p in chunk:
                if str(p) in put_ok:
                    log.write(f'{p.relative_to(SRC)}\t{p.stat().st_size}\n')
                else:
                    # The share refuses some file names (e.g. ACCESS_DENIED on creating them); these go into a tar.
                    blog.write(f'{p.relative_to(SRC)}\t{p.stat().st_size}\n')
            log.flush()
            blog.flush()
            print(f'  {min(i + batch, len(todo))}/{len(todo)}', flush=True)
    put_blocked(unit, env)


def reconcile(unit, first, env):
    """After an interrupted batch, files may be on the share but not in the done-list. Mark those with the same
    size as done instead of putting them again (the share is append-only: never overwrite)."""
    remote = {}
    for d in sorted({str(p.relative_to(SRC).parent) for p in first}):
        text = smb([f'ls "{DEST}/{d}/*"'], env, check=False)
        for line in text.splitlines():
            parts = line.split()
            # "  name   A   size  Mon Nov  9 17:40:31 2026"
            if len(parts) >= 8 and parts[-6].isdigit():
                remote[f'{d}/{" ".join(parts[:-7])}'] = int(parts[-6])
    keep = []
    with open(STATE / f'{unit}.done.tsv', 'a') as log:
        for p in first:
            rel = str(p.relative_to(SRC))
            if remote.get(rel) == p.stat().st_size:
                log.write(f'{rel}\t{p.stat().st_size}\n')
            else:
                keep.append(p)
    if len(keep) < len(first):
        print(f'{unit}: {len(first) - len(keep)} files already on the share, marked done', flush=True)
    return keep


def blocked_files(unit):
    fn = STATE / f'{unit}.blocked.tsv'
    if not fn.exists():
        return []
    done = {line.split('\t')[0] for line in (STATE / f'{unit}.done.tsv').read_text().splitlines() if line}
    rels = dict.fromkeys(line.split('\t')[0] for line in fn.read_text().splitlines() if line)
    return [r for r in rels if r not in done]


def put_blocked(unit, env):
    """Files the share refused by name go into one tar (paths relative to the dataset root), put once."""
    rels = blocked_files(unit)
    if not rels:
        return
    tar = STATE / f'{unit}_refused_by_share.tar'
    remote = f'{DEST}/refused_by_share/{tar.name}'
    marker = STATE / f'{unit}_refused_by_share.put'
    if marker.exists():
        print(f'{unit}: {tar.name} already put', flush=True)
        return
    subprocess.run(['tar', '-cf', str(tar), '-C', str(SRC)] + rels, check=True)
    smb([f'mkdir "{DEST}/refused_by_share"'], env, check=False)
    smb([f'put "{tar}" "{remote}"'], env)
    marker.write_text(remote + '\n')
    print(f'{unit}: {len(rels)} files refused by the share by name, put as {remote}', flush=True)


def remote_md5(remote, env):
    r = subprocess.run(['smbclient', '-N', '--use-kerberos=required', SHARE, '-c', f'get "{remote}" -'],
                       capture_output=True, env=env)
    return hashlib.md5(r.stdout).hexdigest() if r.returncode == 0 and r.stdout else 'missing'


def verify(unit, env):
    files = files_for(unit)
    refused = set(blocked_files(unit))
    tar = STATE / f'{unit}_refused_by_share.tar'
    tar_members = {}
    if refused:
        import tarfile
        with tarfile.open(tar) as t:
            for m in t.getmembers():
                if m.isfile():
                    tar_members[m.name] = hashlib.md5(t.extractfile(m).read()).hexdigest()
    out = STATE / f'{unit}.verify.tsv'
    bad = 0
    with open(out, 'w') as f:
        f.write('path\tsize\tmd5_local\tmd5_remote\tok\n')
        todo = [p for p in files if str(p.relative_to(SRC)) not in refused]
        remote_sum = {}
        tmp = Path(tempfile.mkdtemp(dir=os.environ.get('TMPDIR', '/scratch/gdehol')))
        for i in range(0, len(todo), 20):
            chunk = todo[i:i + 20]
            smb_session([f'get "{DEST}/{p.relative_to(SRC)}" "{tmp}/{k}"' for k, p in enumerate(chunk)], env)
            for k, p in enumerate(chunk):
                t = tmp / str(k)
                remote_sum[str(p.relative_to(SRC))] = md5(t) if t.exists() else 'missing'
                t.unlink(missing_ok=True)
        tmp.rmdir()
        for p in files:
            rel = str(p.relative_to(SRC))
            local = md5(p)
            remote = tar_members.get(rel, 'missing') if rel in refused else remote_sum[rel]
            ok = remote == local
            bad += not ok
            f.write(f'{rel}\t{p.stat().st_size}\t{local}\t{remote}\t{ok}\n')
        if refused:
            local, remote = md5(tar), remote_md5(f'{DEST}/refused_by_share/{tar.name}', env)
            ok = local == remote
            bad += not ok
            f.write(f'refused_by_share/{tar.name}\t{tar.stat().st_size}\t{local}\t{remote}\t{ok}\n')
    print(f'{unit}: {len(files)} files verified ({len(refused)} via the tar), {bad} mismatching or missing -> {out}',
          flush=True)
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
