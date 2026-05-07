#!/usr/bin/env python3
"""
Build a focused 1-document summary for the meeting with Dock and Jan.

Pulls selected pages from existing PDFs (`notes/meeting_report.pdf`,
`notes/projection_clusters.pdf`, `notes/gam_clusters.pdf` if present),
prepends a cover with the talk narrative, and writes
`notes/meeting_summary.pdf`.

Usage:
    python -m retsupp.visualize.build_meeting_summary
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from pypdf import PdfReader, PdfWriter


def make_cover_pdf(out_path: Path):
    """Write a one-page cover PDF with the meeting narrative."""
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111); ax.axis("off")
    ax.text(0.5, 0.96, "retsupp — meeting summary",
            ha="center", va="top", fontsize=22, weight="bold",
            transform=ax.transAxes)
    ax.text(0.5, 0.91, "for discussion with Dock Duncan & Jan Theeuwes",
            ha="center", va="top", fontsize=12, style="italic",
            transform=ax.transAxes)

    body = (
        "WHAT WE FOUND (model-free, robust to multiple analyses)\n"
        "─────────────────────────────────────────────────────────\n"
        "• HP-specific PRF push-away in V3AB and hV4, paired HP−Opposite metric (no Jensen):\n"
        "    V3AB  p_FDR = 0.016    hV4  p_FDR = 0.008\n"
        "    Combined V3AB+hV4 a priori test:  t = 4.18,  p = 0.0003\n"
        "\n"
        "• Direction asymmetry rules out Jensen's-inequality bias:\n"
        "    in V3AB and hV4 the OPPOSITE-anchored shift goes NEGATIVE while HP-anchored is positive.\n"
        "    Pure noise/Jensen would put both positive.\n"
        "\n"
        "• Projection-vs-distance with kernel/LOWESS smoother + multi-R² thresholds:\n"
        "    V3AB:  negative cluster at d ≈ 4-6° (relative attraction at far distance)\n"
        "           p = 0.018, 0.020, 0.050, 0.043 across r²>{0.10, 0.20, 0.30, 0.40}\n"
        "    VO:    positive cluster at d ≈ 2-3° (suppression at intermediate distance)\n"
        "           p = 0.003 at r²>0.10 and r²>0.30\n"
        "    TO:    trending negative cluster at d ≈ 4-5° (matches V3AB direction)\n"
        "    hV4:   strong near-HP suppression visible in paired metric (p<0.01) but\n"
        "           localized at d<1° so dispersed in coarser smoothings\n"
        "\n"
        "• None of this depends on aperture filter or R² threshold — survives sensitivity sweep.\n"
        "\n"
        "WHAT'S NEXT (cluster analyses, all submitted/staged today)\n"
        "─────────────────────────────────────────────────────────\n"
        "1.  GLMsingle single-trial betas — replicating Richter 2025 BOLD-suppression\n"
        "    pattern at HPDL using location-localizer ROIs.\n"
        "    SLURM array 2747605 (subjects 1-30, running NOW).\n"
        "\n"
        "2.  Extended-design PRF fits with distractors as additional stimulus inputs.\n"
        "    Tests whether the bar-aperture-edge signal is real or a fitting artefact.\n"
        "    SLURM array 2748981 (subjects 1-30, queued).\n"
        "\n"
        "3.  Per-ROI hierarchical Bayesian GAM (bambi/HSGP):\n"
        "        proj ~ s(distance) + (1 | subject)\n"
        "    Posterior credible intervals, no cluster-permutation gymnastics.\n"
        "    Currently running locally — included in summary if finished.\n"
        "\n"
        "4.  GAM with σ and R² interactions: tests whether the shift magnitude\n"
        "    scales with PRF size (AF+ prediction) vs fit quality (artefact).\n"
        "    Script ready — will run after #3.\n"
        "\n"
        "WHAT DIDN'T WORK (and why)\n"
        "─────────────────────────────────────────────────────────\n"
        "• Post-hoc 4-AF parametric model: log(g_HP/g_LP) is bimodal at the parameter\n"
        "  bounds (±5), R² ≈ 0.005. Underdetermined fit at this SNR. The signal is real\n"
        "  (model-free tests confirm) but the parametric model can't recover stable\n"
        "  parameters from per-condition PRF center estimates alone.\n"
        "  → Joint AF+PRF model in braincoder (cluster plan #4) is the proper fix.\n"
        "  → Methodological-paper opportunity.\n"
        "\n"
        "OPEN QUESTIONS FOR THE MEETING\n"
        "─────────────────────────────────────────────────────────\n"
        "• Tunçok 2025 has a true neutral baseline (distributed attention); we don't.\n"
        "  Worth adding a neutral condition in future scans?\n"
        "• Richter 2025 finds broad EVC suppression (HPDL + nearby neutrals). Does our\n"
        "  PRF-shift result reflect the same proactive priority map?\n"
        "• Joint AF+PRF in braincoder — methodological paper independent of retsupp?\n"
    )
    ax.text(0.04, 0.86, body, ha="left", va="top",
            fontsize=9, family="monospace", transform=ax.transAxes)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def section_divider_pdf(out_path: Path, title: str, subtitle: str = ""):
    """Write a one-page section divider PDF."""
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111); ax.axis("off")
    ax.text(0.5, 0.55, title, ha="center", va="center",
            fontsize=28, weight="bold", transform=ax.transAxes)
    if subtitle:
        ax.text(0.5, 0.46, subtitle, ha="center", va="center",
                fontsize=12, style="italic", color="0.4",
                transform=ax.transAxes)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def append_pages(writer: PdfWriter, pdf_path: Path, page_idxs: list[int]):
    """Append specific 0-indexed pages from `pdf_path` to the writer."""
    if not pdf_path.exists():
        print(f"  WARN: {pdf_path} missing — skipping")
        return
    reader = PdfReader(str(pdf_path))
    for i in page_idxs:
        if i < len(reader.pages):
            writer.add_page(reader.pages[i])
        else:
            print(f"  WARN: {pdf_path} has only {len(reader.pages)} pages — skipped index {i}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path("notes/meeting_summary.pdf"))
    parser.add_argument("--meeting-report", type=Path,
                        default=Path("notes/meeting_report.pdf"))
    parser.add_argument("--projection-clusters", type=Path,
                        default=Path("notes/projection_clusters.pdf"))
    parser.add_argument("--gam-clusters", type=Path,
                        default=Path("notes/gam_clusters.pdf"))
    parser.add_argument("--gam-size-r2", type=Path,
                        default=Path("notes/gam_size_r2_interaction.pdf"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.parent / "_tmp_meeting_pages"
    tmp.mkdir(exist_ok=True)

    writer = PdfWriter()

    # 1. Cover.
    cover = tmp / "cover.pdf"
    make_cover_pdf(cover)
    append_pages(writer, cover, [0])

    # 2. Headline model-free result: HP vs Opposite paired test (p<0.01).
    div = tmp / "div_model_free.pdf"
    section_divider_pdf(div, "1. Model-free results", "robust across analyses")
    append_pages(writer, div, [0])
    # meeting_report.pdf page 19 = HP vs Opposite paired test per ROI (0-indexed: 18)
    # page 20 = combined V3AB+hV4 a priori test (0-indexed: 19)
    append_pages(writer, args.meeting_report, [18, 19])

    # 3. Per-anchor HP-vs-LP at multiple distances (Fig 1b).
    # meeting_report.pdf page 3 = Fig 1b (0-indexed: 2)
    append_pages(writer, args.meeting_report, [2])

    # 4. Projection-vs-distance: hierarchy overview + per-ROI multi-R² panels.
    div = tmp / "div_proj_dist.pdf"
    section_divider_pdf(div, "2. Projection vs distance",
                        "kernel-smoothed; multi-R² thresholds")
    append_pages(writer, div, [0])
    # projection_clusters.pdf:
    # page 2 = hierarchy overview (0-indexed: 1)
    # pages 3-10 = per-ROI panels (V1, V2, V3, V3AB, hV4, LO, TO, VO)
    append_pages(writer, args.projection_clusters, [1])
    # Highlight V3AB (index 5), hV4 (6), VO (9):
    append_pages(writer, args.projection_clusters, [5, 6, 9])

    # 5. Bambi GAM if available.
    if args.gam_clusters.exists():
        div = tmp / "div_gam.pdf"
        section_divider_pdf(div, "3. Hierarchical Bayesian GAM",
                            "bambi/HSGP per ROI")
        append_pages(writer, div, [0])
        # All pages of gam_clusters.pdf.
        reader = PdfReader(str(args.gam_clusters))
        append_pages(writer, args.gam_clusters,
                     list(range(len(reader.pages))))

    # 6. GAM size/R² interaction if available.
    if args.gam_size_r2.exists():
        div = tmp / "div_gam_int.pdf"
        section_divider_pdf(div, "4. Does shift scale with PRF size?",
                            "GAM with σ × distance and R² × distance interactions")
        append_pages(writer, div, [0])
        reader = PdfReader(str(args.gam_size_r2))
        append_pages(writer, args.gam_size_r2,
                     list(range(len(reader.pages))))

    # 7. AF fit limitations + cluster plan.
    div = tmp / "div_af.pdf"
    section_divider_pdf(div, "5. AF parametric fit limitations",
                        "→ braincoder joint model (cluster plan #4)")
    append_pages(writer, div, [0])
    # meeting_report.pdf has the 4-AF page (typically 8 in current run, 0-indexed: 7)
    append_pages(writer, args.meeting_report, [7])

    # 8. DN model 6 parameter hierarchy as supporting evidence.
    div = tmp / "div_dn.pdf"
    section_divider_pdf(div, "Supporting: DN model parameters",
                        "size hierarchy in independent (non-conditionwise) fits")
    append_pages(writer, div, [0])
    # meeting_report.pdf DN page (typically 6, 0-indexed: 5)
    append_pages(writer, args.meeting_report, [5])

    with open(args.out, "wb") as f:
        writer.write(f)
    print(f"Wrote {args.out} ({len(writer.pages)} pages)")

    # Cleanup temp dir.
    for f in tmp.glob("*.pdf"):
        f.unlink()
    tmp.rmdir()


if __name__ == "__main__":
    main()
