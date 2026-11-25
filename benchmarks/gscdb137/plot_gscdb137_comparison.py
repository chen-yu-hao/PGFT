#!/usr/bin/env python3
"""
Publication-Grade Comparison Plots: PGFT v2 (C++20) vs CF22D vs PBE on GSCDB137
===============================================================================
Produces two alignment-gated, PDF-text-audited, collision-free plots --
  - fig1_scaling_and_timing.{pdf,svg,tiff,png}: Empirical O(N^alpha) scaling,
    basis scaling, 802-molecule GSCDB137 timing scatter, and CPU time anatomy.
  - fig2_gscdb137_accuracy.{pdf,svg,tiff,png}: 20-dataset GSCDB137 relative-metric
    profile, SIE4x4 stretch error curve, and basis-set convergence (SVP to TZVP/aug-DZ).
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

# Optional render-time panel alignment gate (available when alignment checker is installed)
_ALIGN_DIR = os.environ.get("PGFT_ALIGNMENT_AUDIT_DIR", "")
if _ALIGN_DIR and _ALIGN_DIR not in sys.path:
    sys.path.insert(0, _ALIGN_DIR)
try:
    from audit_panel_alignment import require_matplotlib_panel_alignment  # type: ignore
except ImportError:
    def require_matplotlib_panel_alignment(*args: Any, **kwargs: Any) -> None:  # type: ignore
        return None

# Publication styling (Nature double-column width = 183 mm = 7.205 in, >= 5.5 pt glyph floor)
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 6.8,
    "axes.titlesize": 7.4,
    "axes.labelsize": 7.0,
    "xtick.labelsize": 6.2,
    "ytick.labelsize": 6.2,
    "legend.fontsize": 5.8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.65,
    "ytick.major.width": 0.65,
    "xtick.major.size": 2.8,
    "ytick.major.size": 2.8,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# Unified restrained palette: PGFT v2 (hero vermilion), CF22D (deep blue), PBE (teal)
C_PGFT = "#C84B31"
C_PGFT_TINT = "#E9A99B"
C_PGFT_DYN = "#E08E45"
C_CF22D = "#2B5B84"
C_CF22D_TINT = "#9CB8D0"
C_PBE = "#2E8B77"
C_PBE_TINT = "#99CCC0"

BASE_DIR = Path(__file__).resolve().parent
RES_DIR = BASE_DIR / "results"
FIG_DIR = BASE_DIR / "figures"
GSCDB_DIR = Path(os.environ.get("GSCDB_DIR", str(BASE_DIR / "GSCDB")))
HARTREE_TO_KCAL = 627.50947406


def place_panel_label(ax: plt.Axes, letter: str, dx_pt: float = -24.0, dy_pt: float = 5.0) -> None:
    """Place bold lowercase panel letter at a fixed physical point offset from axes top-left."""
    ax.annotate(
        letter,
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(dx_pt, dy_pt),
        textcoords="offset points",
        fontsize=8.5,
        fontweight="bold",
        ha="left",
        va="bottom",
        color="#111111",
    )


def format_plain_log_ticks(ax: plt.Axes, axis: str, ticks: List[float]) -> None:
    """Apply plain-text tick labels on a log axis to eliminate sub-5pt mathtext superscripts."""
    labels = []
    for t in ticks:
        if t >= 1.0:
            labels.append(f"{int(round(t))}")
        elif t >= 0.1:
            labels.append(f"{t:.1f}")
        else:
            labels.append(f"{t:.2f}")
    if axis == "y":
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
        ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    else:
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())


def load_all_svp_mols() -> Dict[str, Dict[str, Any]]:
    """Load all unique molecules in the exact first-seen order matching analyze_gscdb137.py."""
    mols: Dict[str, Dict[str, Any]] = {}
    for fname in [
        "gscdb_shard_w1_svp.jsonl",
        "gscdb_shard_w2_svp.jsonl",
        "gscdb_o24_svp.jsonl",
        "gscdb_bde99nonmr_svp.jsonl",
        "gscdb_orbh35_svp.jsonl",
    ]:
        p = RES_DIR / fname
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            for ln in f:
                if ln.strip():
                    rec = json.loads(ln)
                    if rec.get("status") == "ok" and rec["mol_key"] not in mols:
                        mols[rec["mol_key"]] = rec
    return mols


def make_figure_1_scaling_and_timing() -> None:
    """Figure 1: Scaling exponents, basis scaling, 802-molecule CPU time, and PGFT component anatomy."""
    sc = json.loads((RES_DIR / "scaling_time_results.json").read_text(encoding="utf-8"))
    s_svp = json.loads((RES_DIR / "summary_20sets_svp.json").read_text(encoding="utf-8"))
    mols = load_all_svp_mols()

    fig, axes = plt.subplots(
        2, 2,
        figsize=(7.205, 5.55),
        gridspec_kw={"wspace": 0.34, "hspace": 0.42},
    )
    fig.subplots_adjust(left=0.095, right=0.975, bottom=0.09, top=0.925)
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    # -------------------------------------------------------------------------
    # Panel a: 1D (H2)_k chain & 3D (H2O)_n cluster wall-time scaling (6-31G)
    # -------------------------------------------------------------------------
    h2 = sc["families"]["h2_chain_631g"]
    w3 = sc["families"]["water_clusters_631g"]
    fits_h2 = sc["scaling_fits"]["h2_chain_631g"]
    fits_w3 = sc["scaling_fits"]["water_clusters_631g"]

    nao_h2 = np.array([p["nao"] for p in h2], dtype=float)
    t_h2_pbe = np.array([p["PBE_exact"]["t_tot_s"] for p in h2], dtype=float)
    t_h2_cf = np.array([p["CF22D_exact"]["t_tot_s"] for p in h2], dtype=float)
    t_h2_pg_c10 = np.array([p["PGFT_v2_c10"]["t_tot_s"] for p in h2], dtype=float)
    t_h2_pg_dyn = np.array([p["PGFT_v2_c10"]["t_dyn_s"] for p in h2], dtype=float)

    nao_w3 = np.array([p["nao"] for p in w3], dtype=float)
    t_w3_cf = np.array([p["CF22D_exact"]["t_tot_s"] for p in w3], dtype=float)
    w3_pg = [p for p in w3 if "PGFT_v2_default" in p]
    nao_w3_pg = np.array([p["nao"] for p in w3_pg], dtype=float)
    t_w3_pg = np.array([p["PGFT_v2_default"]["t_tot_s"] for p in w3_pg], dtype=float)

    for arr in (nao_h2, t_h2_pg_c10, nao_w3_pg, t_w3_pg, t_h2_pg_dyn, t_h2_cf, t_h2_pbe):
        assert np.all(np.asarray(arr) > 0.0), "Non-positive values before loglog plot"

    ax_a.loglog(nao_h2, t_h2_pg_c10, "o-", color=C_PGFT, lw=1.6, ms=3.8,
                label=f"PGFT v2 (H₂)ₖ (α={fits_h2['PGFT_v2_c10']['alpha']:.2f})")
    ax_a.loglog(nao_w3_pg, t_w3_pg, "s-", color="#8E2817", lw=1.4, ms=3.5,
                label=f"PGFT v2 (H₂O)ₙ (α={fits_w3['PGFT_v2_default']['alpha']:.2f})")
    ax_a.loglog(nao_h2, t_h2_pg_dyn, "^--", color=C_PGFT_DYN, lw=1.2, ms=3.3,
                label=f"PGFT Dyn. ACFD (α={fits_h2['PGFT_v2_c10_dyn']['alpha']:.2f})")
    ax_a.loglog(nao_h2, t_h2_cf, "D-", color=C_CF22D, lw=1.35, ms=3.2,
                label=f"CF22D (H₂)ₖ (α={fits_h2['CF22D_exact']['alpha']:.2f})")
    ax_a.loglog(nao_w3, t_w3_cf, "d:", color="#6B9AC4", lw=1.1, ms=3.0,
                label=f"CF22D (H₂O)ₙ (α={fits_w3['CF22D_exact']['alpha']:.2f})")
    ax_a.loglog(nao_h2, t_h2_pbe, "v-", color=C_PBE, lw=1.25, ms=3.2,
                label=f"PBE (H₂)ₖ (α={fits_h2['PBE_exact']['alpha']:.2f})")

    # Reference O(N^3) guide line well separated from text
    x_ref = np.array([28.0, 95.0])
    y_ref = 0.00042 * (x_ref ** 3)
    ax_a.loglog(x_ref, y_ref, "--", color="#666666", lw=0.85, alpha=0.85)
    ax_a.text(72.0, 420.0, "O(N³) ref.", fontsize=5.8, color="#444444", ha="center", va="bottom")

    ax_a.set_xlabel("Basis dimension N_AO (6-31G, single core)")
    ax_a.set_ylabel("Wall-clock time (s)")
    ax_a.set_xlim(7.0, 125.0)
    ax_a.set_ylim(0.012, 1800.0)
    format_plain_log_ticks(ax_a, "x", [8, 16, 32, 64, 100])
    format_plain_log_ticks(ax_a, "y", [0.1, 1, 10, 100, 1000])
    ax_a.legend(loc="upper left", ncol=1, handlelength=1.7, labelspacing=0.22, fontsize=5.4)
    place_panel_label(ax_a, "a")

    # -------------------------------------------------------------------------
    # Panel b: Basis-set scaling at fixed molecule (H2O: STO-3G .. def2-QZVPP)
    # -------------------------------------------------------------------------
    h2o = sc["families"]["h2o_basis_scaling"]
    fits_b = sc["scaling_fits"]["h2o_basis_scaling"]
    nao_b = np.array([p["nao"] for p in h2o], dtype=float)
    t_b_pbe = np.array([p["PBE_exact"]["t_tot_s"] for p in h2o], dtype=float)
    t_b_cf = np.array([p["CF22D_exact"]["t_tot_s"] for p in h2o], dtype=float)
    t_b_cf_fine = np.array([p["CF22D_exact_99_590"]["t_tot_s"] for p in h2o], dtype=float)
    b_pg = [p for p in h2o if "PGFT_v2_default" in p]
    nao_b_pg = np.array([p["nao"] for p in b_pg], dtype=float)
    t_b_pg = np.array([p["PGFT_v2_default"]["t_tot_s"] for p in b_pg], dtype=float)
    t_b_pg_dyn = np.array([p["PGFT_v2_default"]["t_dyn_s"] for p in b_pg], dtype=float)

    ax_b.loglog(nao_b_pg, t_b_pg, "o-", color=C_PGFT, lw=1.6, ms=3.8,
                label=f"PGFT v2 total (α={fits_b['PGFT_v2_default']['alpha']:.2f})")
    ax_b.loglog(nao_b_pg, t_b_pg_dyn, "^--", color=C_PGFT_DYN, lw=1.2, ms=3.3,
                label=f"PGFT Dyn. ACFD (α={fits_b['PGFT_v2_default_dyn']['alpha']:.2f})")
    ax_b.loglog(nao_b, t_b_cf_fine, "s:", color="#5B88B2", lw=1.2, ms=3.1,
                label="CF22D (99,590 grid)")
    ax_b.loglog(nao_b, t_b_cf, "D-", color=C_CF22D, lw=1.35, ms=3.2,
                label=f"CF22D default (α={fits_b['CF22D_exact']['alpha']:.2f})")
    ax_b.loglog(nao_b, t_b_pbe, "v-", color=C_PBE, lw=1.25, ms=3.2,
                label=f"PBE default (α={fits_b['PBE_exact']['alpha']:.2f})")

    # Clean annotations in open whitespace
    ax_b.annotate("def2-TZVP", xy=(43, 103.65), xytext=(15.5, 220.0), fontsize=5.6, color=C_PGFT,
                  arrowprops={"arrowstyle": "-", "color": C_PGFT, "lw": 0.55})
    ax_b.annotate("def2-QZVPP", xy=(117, 13.385), xytext=(64.0, 48.0), fontsize=5.6, color=C_CF22D,
                  arrowprops={"arrowstyle": "-", "color": C_CF22D, "lw": 0.55})

    ax_b.set_xlabel("H₂O basis dimension N_AO (STO-3G to def2-QZVPP)")
    ax_b.set_ylabel("Wall-clock time (s)")
    ax_b.set_xlim(6.0, 140.0)
    ax_b.set_ylim(0.012, 1800.0)
    format_plain_log_ticks(ax_b, "x", [7, 13, 24, 43, 74, 117])
    format_plain_log_ticks(ax_b, "y", [0.1, 1, 10, 100, 1000])
    ax_b.legend(loc="lower right", ncol=1, handlelength=1.7, labelspacing=0.22, fontsize=5.4)
    place_panel_label(ax_b, "b")

    # -------------------------------------------------------------------------
    # Panel c: Per-molecule single-core CPU time across all 802 GSCDB137 molecules
    # -------------------------------------------------------------------------
    naos = np.array([r["nao"] for r in mols.values()], dtype=float)
    t_pg = np.array([r["PGFT"]["t_tot_s"] for r in mols.values()], dtype=float)
    t_cf = np.array([r["CF22D"]["t_tot_s"] for r in mols.values()], dtype=float)
    t_pb = np.array([r["PBE"]["t_tot_s"] for r in mols.values()], dtype=float)
    n_tot_mols = len(naos)

    assert np.all(t_pg > 0.0) and np.all(t_cf > 0.0) and np.all(t_pb > 0.0), "Non-positive timings before log y-scale"
    ax_c.scatter(naos, t_pg, s=5.5, color=C_PGFT, alpha=0.34, linewidths=0, label=f"PGFT v2 (n={n_tot_mols})")
    ax_c.scatter(naos, t_cf, s=5.0, color=C_CF22D, alpha=0.32, linewidths=0, label="CF22D (99,590)")
    ax_c.scatter(naos, t_pb, s=5.0, color=C_PBE, alpha=0.30, linewidths=0, label="PBE (99,590)")

    bins = np.array([4, 12, 20, 30, 42, 55, 70, 95], dtype=float)
    for arr, col, mk in [(t_pg, C_PGFT, "o"), (t_cf, C_CF22D, "D"), (t_pb, C_PBE, "v")]:
        bx, by = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (naos >= lo) & (naos < hi)
            if np.sum(mask) >= 5:
                bx.append(float(np.median(naos[mask])))
                by.append(float(np.median(arr[mask])))
        ax_c.plot(bx, by, f"{mk}-", color=col, lw=1.45, ms=3.4, markeredgecolor="white", markeredgewidth=0.4)

    ax_c.set_yscale("log")
    ax_c.set_xlim(2, 96)
    ax_c.set_ylim(0.006, 6500.0)
    ax_c.set_xticks([10, 20, 30, 40, 50, 60, 70, 80, 90])
    format_plain_log_ticks(ax_c, "y", [0.01, 0.1, 1, 10, 100, 1000])
    ax_c.set_xlabel("Molecular basis size N_AO (def2-SVP)")
    ax_c.set_ylabel("Single-core CPU time per molecule (s)")
    ax_c.legend(loc="lower right", ncol=1, markerscale=1.4, fontsize=5.6)
    place_panel_label(ax_c, "c")

    # -------------------------------------------------------------------------
    # Panel d: Exact total CPU time anatomy (hours) from summary_20sets_svp.json
    # -------------------------------------------------------------------------
    tg = s_svp["timing_global"]
    tot_pbe_h = tg["PBE"]["total_cpu_s"] / 3600.0
    tot_cf_h = tg["CF22D"]["total_cpu_s"] / 3600.0
    tot_pg_h = tg["PGFT"]["total_cpu_s"] / 3600.0
    bk = tg["PGFT_breakdown_total_s"]
    tot_hf_thc_h = (bk["hf"] + bk["thc"]) / 3600.0
    tot_dyn_h = bk["dyn"] / 3600.0
    tot_stat_h = bk["stat"] / 3600.0
    non_stat_h = tot_hf_thc_h + tot_dyn_h
    stat_pct = 100.0 * tot_stat_h / tot_pg_h

    y_pos = np.array([0.0, 1.0, 2.0, 3.0])
    labels_d = [
        "PBE\n(99,590)",
        "CF22D\n(99,590+D3)",
        "PGFT v2\n(HF+THC+ACFD)",
        "PGFT v2\n(Full Total)",
    ]
    bar_h = 0.54
    ax_d.barh(y_pos[0], tot_pbe_h, height=bar_h, color=C_PBE, edgecolor="none")
    ax_d.barh(y_pos[1], tot_cf_h, height=bar_h, color=C_CF22D, edgecolor="none")
    ax_d.barh(y_pos[2], tot_hf_thc_h, height=bar_h, color="#7A8B99", label="HF + ISDF-THC")
    ax_d.barh(y_pos[2], tot_dyn_h, left=tot_hf_thc_h, height=bar_h, color=C_PGFT_DYN, label="Dynamic ACFD")
    ax_d.barh(y_pos[3], tot_hf_thc_h, height=bar_h, color="#7A8B99")
    ax_d.barh(y_pos[3], tot_dyn_h, left=tot_hf_thc_h, height=bar_h, color=C_PGFT_DYN)
    ax_d.barh(y_pos[3], tot_stat_h, left=non_stat_h, height=bar_h, color=C_PGFT, label="Static L-BFGS SCF")

    ax_d.text(tot_pbe_h + 0.8, y_pos[0], f"{tot_pbe_h:.2f} h (1.0×)", va="center", ha="left", fontsize=5.8, color="#222222")
    ax_d.text(tot_cf_h + 0.8, y_pos[1], f"{tot_cf_h:.2f} h ({tot_cf_h/tot_pbe_h:.1f}×)", va="center", ha="left", fontsize=5.8, color="#222222")
    ax_d.text(non_stat_h + 0.8, y_pos[2], f"{non_stat_h:.2f} h ({100.0 - stat_pct:.1f}%)", va="center", ha="left", fontsize=5.8, color="#222222")
    ax_d.text(tot_pg_h + 0.8, y_pos[3], f"{tot_pg_h:.1f} h ({stat_pct:.1f}% SCF)", va="center", ha="left", fontsize=5.8, fontweight="bold", color=C_PGFT)

    ax_d.set_yticks(y_pos)
    ax_d.set_yticklabels(labels_d)
    ax_d.set_xlim(0, 56.0)
    ax_d.set_xticks([0, 10, 20, 30, 40, 50])
    ax_d.set_xlabel(f"Cumulative single-core CPU time across {n_tot_mols} molecules (hours)")
    ax_d.legend(loc="lower right", fontsize=5.5, handlelength=1.4)
    place_panel_label(ax_d, "d")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    stem = FIG_DIR / "fig1_scaling_and_timing"
    require_matplotlib_panel_alignment(
        fig,
        json_out=str(stem) + ".alignment.json",
        overlay_svg=str(stem) + ".alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight")
    fig.savefig(str(stem) + ".svg", bbox_inches="tight")
    fig.savefig(str(stem) + ".tiff", dpi=600, bbox_inches="tight")
    fig.savefig(str(stem) + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 1] Saved {stem}.{{pdf,svg,tiff,png}} cleanly!", flush=True)


def make_figure_2_gscdb137_accuracy() -> None:
    """Figure 2: 20-dataset GSCDB137 accuracy, SIE4x4 stretch curve, and basis-set convergence."""
    s_svp = json.loads((RES_DIR / "summary_20sets_svp.json").read_text(encoding="utf-8"))
    s_qz = json.loads((RES_DIR / "summary_20sets_qzvppd_dfpbe.json").read_text(encoding="utf-8"))
    s_tz = json.loads((RES_DIR / "summary_4sets_tzvp.json").read_text(encoding="utf-8"))
    s_ad = json.loads((RES_DIR / "summary_augdz.json").read_text(encoding="utf-8"))

    fig = plt.figure(figsize=(7.205, 6.05))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1.22, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.34,
        hspace=0.40,
        left=0.145, right=0.975, bottom=0.085, top=0.875,
    )
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    # -------------------------------------------------------------------------
    # Panel a (Hero, left spanning both rows): 20-dataset Relative Metric profile
    # -------------------------------------------------------------------------
    cat_order = {"TC": 0, "BH": 1, "ISO": 2, "NC": 3, "TM": 4}
    ds_items = sorted(
        s_svp["per_dataset"].values(),
        key=lambda d: (cat_order.get(d["datatype_short"], 9), d["methods"]["PGFT"]["RelMetric"]),
    )
    y_idx = np.arange(len(ds_items), dtype=float)
    names = [f"{d['dataset']} ({d['datatype_short']})" for d in ds_items]
    rel_pg = np.array([d["methods"]["PGFT"]["RelMetric"] for d in ds_items], dtype=float)
    rel_pb = np.array([d["methods"]["PBE"]["RelMetric"] for d in ds_items], dtype=float)
    rel_cf = np.array([d["methods"]["CF22D"]["RelMetric"] for d in ds_items], dtype=float)
    rel_cf_cbs = np.array([d["qchem_ref_rel"]["CF22D_QChem"] for d in ds_items], dtype=float)
    rel_pb_qz = np.array([s_qz["per_dataset"][d["dataset"]]["methods"]["PBE"]["RelMetric"] for d in ds_items], dtype=float)

    ax_a.axvline(1.0, color="#333333", ls="--", lw=0.75, zorder=1)
    for yi, p1, p2, p3 in zip(y_idx, rel_pg, rel_pb, rel_cf):
        ax_a.plot([min(p1, p2, p3), max(p1, p2, p3)], [yi, yi], color="#D6DBE0", lw=0.85, zorder=1)

    ax_a.scatter(rel_cf_cbs, y_idx, marker="|", s=36, color="#1D3D59", linewidths=1.2, zorder=3, label="CF22D (CBS ref.)")
    ax_a.scatter(rel_pb_qz, y_idx, marker="|", s=32, color="#1D6354", linewidths=1.0, zorder=3, label="DF-PBE (QZVPPD)")
    ax_a.scatter(rel_cf, y_idx, marker="D", s=19, color=C_CF22D, zorder=4, label="CF22D (def2-SVP)")
    ax_a.scatter(rel_pb, y_idx, marker="s", s=18, color=C_PBE, zorder=4, label="PBE (def2-SVP)")
    ax_a.scatter(rel_pg, y_idx, marker="o", s=25, color=C_PGFT, edgecolor="white", linewidth=0.4, zorder=5, label="PGFT v2 (def2-SVP)")

    prev_cat = ds_items[0]["datatype_short"]
    for i, d in enumerate(ds_items):
        if d["datatype_short"] != prev_cat:
            ax_a.axhline(i - 0.5, color="#CFD5DC", ls=":", lw=0.6)
            prev_cat = d["datatype_short"]

    assert np.all(rel_pg > 0.0) and np.all(rel_pb > 0.0) and np.all(rel_cf > 0.0), "Non-positive metrics before log x-scale"
    ax_a.set_xscale("log")
    ax_a.set_xlim(0.45, 110.0)
    format_plain_log_ticks(ax_a, "x", [0.5, 1, 2, 5, 10, 20, 50, 100])
    ax_a.set_yticks(y_idx)
    ax_a.set_yticklabels(names, fontsize=5.8)
    ax_a.invert_yaxis()
    ax_a.set_xlabel("Relative error metric (Metric / σ_std; lower is better)")
    # Place legend cleanly ABOVE panel a so zero data points are obscured
    ax_a.legend(
        loc="lower left",
        bbox_to_anchor=(-0.02, 1.015),
        ncol=2,
        fontsize=5.4,
        handletextpad=0.35,
        columnspacing=0.9,
        labelspacing=0.25,
    )
    place_panel_label(ax_a, "a", dx_pt=-48.0, dy_pt=24.0)

    # -------------------------------------------------------------------------
    # Panel b (top-right): SIE4x4 stretch profile (16 radical-cation dissociations)
    # -------------------------------------------------------------------------
    eval_rows = []
    with open(GSCDB_DIR / "Info" / "DatasetEval.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["Dataset"] == "SIE4x4":
                parts = [p.strip() for p in r["Stoichiometry"].split(",")]
                st = [(float(parts[2 * i]), parts[2 * i + 1]) for i in range(len(parts) // 2)]
                eval_rows.append((r["Reaction"], float(r["Reference"]) * HARTREE_TO_KCAL, st))

    tz_mols: Dict[str, Dict[str, Any]] = {}
    with open(RES_DIR / "gscdb_4sets_tzvp.jsonl", encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                r = json.loads(ln)
                if r.get("status") == "ok":
                    tz_mols[r["mol_key"]] = r

    ratios = np.array([1.0, 1.25, 1.5, 1.75])
    err_pg_tz, err_pb_tz, err_cf_tz = np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((4, 4))
    for idx, (_, ref, st) in enumerate(eval_rows):
        dim_i = idx // 4
        r_i = idx % 4
        err_pg_tz[dim_i, r_i] = sum(c * tz_mols[mk]["PGFT"]["e_tot"] for c, mk in st) * HARTREE_TO_KCAL - ref
        err_pb_tz[dim_i, r_i] = sum(c * tz_mols[mk]["PBE"]["e_tot"] for c, mk in st) * HARTREE_TO_KCAL - ref
        err_cf_tz[dim_i, r_i] = sum(c * tz_mols[mk]["CF22D"]["e_tot"] for c, mk in st) * HARTREE_TO_KCAL - ref

    ax_b.axhline(0.0, color="#444444", ls="--", lw=0.75)
    for arr, col, mk, lbl in [
        (err_pb_tz, C_PBE, "s", "PBE (def2-TZVP)"),
        (err_cf_tz, C_CF22D, "D", "CF22D (def2-TZVP)"),
        (err_pg_tz, C_PGFT, "o", "PGFT v2 (def2-TZVP)"),
    ]:
        mean_e = np.mean(arr, axis=0)
        sd_e = np.std(arr, axis=0, ddof=1)
        ax_b.fill_between(ratios, mean_e - sd_e, mean_e + sd_e, color=col, alpha=0.14, linewidth=0)
        ax_b.plot(ratios, mean_e, f"{mk}-", color=col, lw=1.5, ms=3.8, label=lbl)

    ax_b.set_xticks(ratios)
    ax_b.set_xticklabels(["1.00", "1.25", "1.50", "1.75"])
    ax_b.set_ylim(-20.0, 56.0)
    ax_b.set_yticks([-20, -10, 0, 10, 20, 30, 40, 50])
    ax_b.set_xlabel("Bond stretch ratio R / Rₑ (SIE4x4, n=4 dimers)")
    ax_b.set_ylabel("Signed error (kcal/mol, mean ± SD)")
    ax_b.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.015),
        ncol=1,
        fontsize=5.4,
        labelspacing=0.22,
    )
    place_panel_label(ax_b, "b", dx_pt=-26.0, dy_pt=24.0)

    # -------------------------------------------------------------------------
    # Panel c (bottom-right): Basis-set convergence MAE (side-by-side clean bars, no hatch)
    # -------------------------------------------------------------------------
    sets_conv = [
        ("SIE4x4\n(TZVP)", s_svp["per_dataset"]["SIE4x4"], s_tz["per_dataset"]["SIE4x4"]),
        ("G21EA\n(aug-DZ)", s_svp["per_dataset"]["G21EA"], s_ad["per_dataset"]["G21EA"]),
        ("He3×20\n(TZVP)", s_svp["per_dataset"]["He3"], s_tz["per_dataset"]["He3"]),
        ("ALKBDE10\n(TZVP)", s_svp["per_dataset"]["ALKBDE10"], s_tz["per_dataset"]["ALKBDE10"]),
    ]
    x_c = np.arange(len(sets_conv), dtype=float)
    w_pair = 0.26
    w_sub = 0.115
    for m_idx, (mkey, col_hi, col_lo, lbl) in enumerate([
        ("PGFT", C_PGFT, C_PGFT_TINT, "PGFT v2"),
        ("PBE", C_PBE, C_PBE_TINT, "PBE"),
        ("CF22D", C_CF22D, C_CF22D_TINT, "CF22D"),
    ]):
        sv_vals, hi_vals = [], []
        for s_lbl, d_sv, d_hi in sets_conv:
            scale = 20.0 if s_lbl.startswith("He3") else 1.0
            sv_vals.append(d_sv["methods"][mkey]["MAE"] * scale)
            hi_vals.append(d_hi["methods"][mkey]["MAE"] * scale)
        x_center = x_c + (m_idx - 1.0) * w_pair
        ax_c.bar(x_center - 0.52 * w_sub, sv_vals, width=w_sub, color=col_lo, edgecolor="none")
        ax_c.bar(x_center + 0.52 * w_sub, hi_vals, width=w_sub, color=col_hi, edgecolor="none", label=lbl)

    handles, labels = ax_c.get_legend_handles_labels()
    handles.append(Patch(facecolor="#CCCCCC", edgecolor="none"))
    labels.append("Left bar: def2-SVP")
    ax_c.legend(
        handles, labels,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=2,
        fontsize=5.3,
        handlelength=1.1,
        columnspacing=0.8,
        labelspacing=0.22,
    )

    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels([x[0] for x in sets_conv], fontsize=5.8)
    ax_c.set_ylim(0.0, 38.0)
    ax_c.set_yticks([0, 10, 20, 30])
    ax_c.set_ylabel("MAE (kcal/mol; He3 scaled 20×)")
    place_panel_label(ax_c, "c", dx_pt=-26.0, dy_pt=20.0)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    stem = FIG_DIR / "fig2_gscdb137_accuracy"
    require_matplotlib_panel_alignment(
        fig,
        json_out=str(stem) + ".alignment.json",
        overlay_svg=str(stem) + ".alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight")
    fig.savefig(str(stem) + ".svg", bbox_inches="tight")
    fig.savefig(str(stem) + ".tiff", dpi=600, bbox_inches="tight")
    fig.savefig(str(stem) + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 2] Saved {stem}.{{pdf,svg,tiff,png}} cleanly!", flush=True)


if __name__ == "__main__":
    make_figure_1_scaling_and_timing()
    make_figure_2_gscdb137_accuracy()
