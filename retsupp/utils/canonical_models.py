"""Resolve canonical-model analysis names to BIDS derivatives subdirs.

The registry lives at ``notes/data/canonical_models.yml`` (project
root). Each analysis short-name maps to a ``fits_dir`` (relative to
the BIDS root) plus optional config metadata. Aggregators / plot
scripts should accept ``--analysis <name>`` and call
:func:`resolve_analysis` to get the absolute path.

Why this matters
----------------

The AF / PRF dir tree has many sibling output dirs whose names
encode all the config knobs (``pSig0.5``, ``allSharedSigma``,
``sharedDynGain``, ``apt0.5``, ...). When the canonical config
evolves, the "obvious" dir name silently becomes legacy. Hand-typing
``--fits-dir`` is fragile.

This registry is the project-wide source of truth for which dir
holds the current canonical fits, and points at retired ones with
``*_legacy_<date>`` keys so they can't be confused.

Source of truth for each individual fit dir's config is its own
``dataset_description.json`` (BIDS-mandated). The registry is the
*pointer*; the JSON is the *receipt*.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REGISTRY_REL_PATH = Path('notes') / 'data' / 'canonical_models.yml'


def _find_repo_root(start: Path | None = None) -> Path:
    """Walk up from ``start`` (or this file) until we find the retsupp repo root."""
    p = (start or Path(__file__)).resolve()
    for parent in [p, *p.parents]:
        if (parent / REGISTRY_REL_PATH).exists():
            return parent
    raise FileNotFoundError(
        f"Could not locate {REGISTRY_REL_PATH} starting from {p}"
    )


def load_registry(repo_root: Path | None = None) -> dict[str, dict[str, Any]]:
    """Load the canonical-model registry as a dict."""
    root = repo_root or _find_repo_root()
    with open(root / REGISTRY_REL_PATH) as f:
        return yaml.safe_load(f)


def resolve_analysis(name: str, bids_folder: Path | str,
                     repo_root: Path | None = None) -> Path:
    """Resolve a canonical-model analysis name to its absolute fits dir.

    Args:
        name: registry key (e.g. ``'af_dynamic_sharedSigma'``).
        bids_folder: BIDS dataset root (e.g. ``/data/ds-retsupp`` locally
                     or ``/shares/zne.uzh/gdehol/ds-retsupp`` on cluster).
        repo_root: optional override; auto-detected from this file's
                   location otherwise.

    Returns:
        Absolute :class:`Path` to the fits directory.

    Raises:
        KeyError: ``name`` is not in the registry.
        FileNotFoundError: the resolved path does not exist on disk.
    """
    reg = load_registry(repo_root)
    if name not in reg:
        keys = ', '.join(sorted(reg.keys()))
        raise KeyError(f"Unknown analysis {name!r}. Registered: {keys}")
    entry = reg[name]
    rel = entry['fits_dir']
    path = Path(bids_folder) / rel
    if not path.exists():
        raise FileNotFoundError(
            f"Registry says {name!r} lives at {path}, but it doesn't exist. "
            f"Edit {REGISTRY_REL_PATH} if the canonical dir has moved."
        )
    return path
