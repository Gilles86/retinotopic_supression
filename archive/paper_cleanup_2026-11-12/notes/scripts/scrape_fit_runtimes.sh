#!/bin/bash
# Pull SLURM accounting (sacct) for all PRF / GLMsingle / bench / surface
# / neuropythy jobs since 2026-05-01 and write a long-format TSV
# with one row per array task: subject, model, GPU, elapsed seconds,
# state, host, start/end times.
#
# Run on the cluster (read access to sacct).
# Output: derivatives/fit_runtimes.tsv  (auto-overwritten on each run).

set -e
OUT=/shares/zne.uzh/gdehol/ds-retsupp/derivatives/fit_runtimes.tsv
START="${1:-2026-05-01}"

sacct --user=$(whoami) -X --starttime="$START" \
  --format=JobID,JobName%-30,Account,State,Elapsed,NodeList%-30,ReqTRES%-50,Submit,Start,End \
  --parsable2 > /tmp/fit_times_raw.tsv

$HOME/data/conda/envs/retsupp_cuda/bin/python <<'PY'
import pandas as pd, re
df = pd.read_csv("/tmp/fit_times_raw.tsv", sep="|")
mask = (df["JobName"].str.match(r"prf_", na=False) |
        df["JobName"].str.startswith("voxel_traces", na=False) |
        df["JobName"].str.startswith("neuropythy", na=False) |
        df["JobName"].str.startswith("retsupp_", na=False) |
        df["JobName"].str.startswith("bench_", na=False))
df = df[mask].copy()

def model_id(jn):
    m = re.search(r"_m(\d+)", jn or "")
    return int(m.group(1)) if m else None

def sub_id(jn):
    m = re.search(r"sub-(\d+)", jn or "")
    return int(m.group(1)) if m else None

def gpu_kind(req):
    if not isinstance(req, str): return None
    m = re.search(r"gres/gpu:([A-Za-z0-9]+)", req)
    return m.group(1) if m else ("any" if "gres/gpu=" in req else None)

def to_sec(s):
    if not isinstance(s, str): return None
    parts = s.split("-")
    days = int(parts[0]) if len(parts) > 1 else 0
    hms = parts[-1].split(":")
    h = int(hms[0]) if len(hms) > 2 else (0 if len(hms) < 2 else int(hms[0]))
    m = int(hms[-2]) if len(hms) >= 2 else 0
    s_ = int(hms[-1])
    return days * 86400 + h * 3600 + m * 60 + s_

df["model"]     = df["JobName"].apply(model_id)
df["subject"]   = df["JobName"].apply(sub_id)
df["gpu_req"]   = df["ReqTRES"].apply(gpu_kind)
df["elapsed_s"] = df["Elapsed"].apply(to_sec)

import os
out = os.environ.get("OUT", "/shares/zne.uzh/gdehol/ds-retsupp/derivatives/fit_runtimes.tsv")
df[["JobID","JobName","model","subject","State","elapsed_s","Elapsed",
    "NodeList","gpu_req","Start","End"]].to_csv(out, sep="\t", index=False)
print(f"wrote {len(df)} rows -> {out}")
PY
