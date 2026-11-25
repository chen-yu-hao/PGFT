#!/usr/bin/env python3
"""
GSCDB137 Official-Metric Analyzer: PGFT v2 (C++20) vs CF22D vs PBE
==================================================================
Computes MAE, RMSE, MSE, official GSCDB137 specialized metrics (including
O24 and TMD10 weights), Standard_errors-normalized relative metrics, and
single-core wall-clock timing statistics from sharded JSONL worker outputs.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

HARTREE_TO_KCAL = 627.50947406


def load_molecule_records(jsonl_paths: List[Path]) -> Dict[str, Dict[str, Any]]:
    mols: Dict[str, Dict[str, Any]] = {}
    for p in jsonl_paths:
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            for ln in f:
                if not ln.strip():
                    continue
                rec = json.loads(ln)
                if rec.get("status") == "ok":
                    mk = rec["mol_key"]
                    if mk not in mols:
                        mols[mk] = rec
                    else:
                        # Merge methods if split across files
                        for m in ("PGFT", "PBE", "CF22D"):
                            if m in rec and m not in mols[mk]:
                                mols[mk][m] = rec[m]
    return mols


def load_gscdb_metadata(gscdb_root: Path) -> Dict[str, Any]:
    datasets: Dict[str, Dict[str, str]] = {}
    with open(gscdb_root / "Info" / "Datasets.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("Datatype_Short"):
                datasets[r["Name"]] = r

    std_err: Dict[str, Dict[str, float]] = {}
    with open(gscdb_root / "Info" / "Standard_errors.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            std_err[r["Dataset"]] = {
                "Metric": float(r["Metric"]),
                "RMSE": float(r["RMSE"]),
                "MAE": float(r["MAE"]),
            }

    o24_w: Dict[str, float] = {}
    with open(gscdb_root / "Info" / "DatasetEval_O24x5_weight.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            o24_w[r["Reaction"]] = float(r["Weight"])

    tmc_w: Dict[str, List[Tuple[str, float]]] = collections.defaultdict(list)
    with open(gscdb_root / "Info" / "DatasetEval_TMC34_weight.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            tmc_w[r["Dataset"]].append((r["Reaction"], float(r["Weight"])))

    qchem_mae: Dict[str, Dict[str, float]] = {}
    with open(gscdb_root / "Analysis" / "Errors_per_set_MAE.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            qchem_mae[r["Dataset"]] = {
                "CF22D_QChem": float(r["CF22D"]),
                "PBE0_QChem": float(r["PBE0"]),
                "B3LYP_QChem": float(r["B3LYP"]),
            }

    qchem_rel: Dict[str, Dict[str, float]] = {}
    with open(gscdb_root / "Analysis" / "Relative_metric_per_set.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            qchem_rel[r["Dataset"]] = {
                "CF22D_QChem": float(r["CF22D"]),
                "PBE0_QChem": float(r["PBE0"]),
                "B3LYP_QChem": float(r["B3LYP"]),
            }

    eval_by_ds: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    with open(gscdb_root / "Info" / "DatasetEval.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            parts = [p.strip() for p in r["Stoichiometry"].split(",")]
            stoich = [(float(parts[2 * i]), parts[2 * i + 1]) for i in range(len(parts) // 2)]
            eval_by_ds[r["Dataset"]].append({
                "reaction": r["Reaction"],
                "dataset": r["Dataset"],
                "ref_kcal": float(r["Reference"]) * HARTREE_TO_KCAL,
                "stoich": stoich,
            })

    return {
        "datasets": datasets,
        "std_err": std_err,
        "o24_w": o24_w,
        "tmc_w": tmc_w,
        "qchem_mae": qchem_mae,
        "qchem_rel": qchem_rel,
        "eval_by_ds": eval_by_ds,
    }


def evaluate_suite(
    mols: Dict[str, Dict[str, Any]],
    meta: Dict[str, Any],
    methods: List[str] = ["PGFT", "PBE", "CF22D"],
    require_complete: bool = True,
) -> Dict[str, Any]:
    eval_by_ds = meta["eval_by_ds"]
    datasets_info = meta["datasets"]
    std_err = meta["std_err"]
    o24_w = meta["o24_w"]
    tmc_w = meta["tmc_w"]

    per_dataset: Dict[str, Any] = {}
    all_reactions: List[Dict[str, Any]] = []

    for dname, rxns in sorted(eval_by_ds.items()):
        if dname not in datasets_info:
            continue
        # Check coverage across requested methods
        complete_rxns = []
        for rxn in rxns:
            ok = True
            for _, mk in rxn["stoich"]:
                if mk not in mols:
                    ok = False
                    break
                for m in methods:
                    if m not in mols[mk]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                complete_rxns.append(rxn)

        if not complete_rxns:
            continue
        if require_complete and len(complete_rxns) < len(rxns):
            continue

        dt_short = datasets_info[dname]["Datatype_Short"]
        dt_full = datasets_info[dname]["Datatype"]
        ds_res: Dict[str, Any] = {
            "dataset": dname,
            "datatype_short": dt_short,
            "datatype": dt_full,
            "n_total": len(rxns),
            "n_eval": len(complete_rxns),
            "standard_metric": std_err.get(dname, {}).get("Metric"),
            "qchem_ref_mae": meta["qchem_mae"].get(dname, {}),
            "qchem_ref_rel": meta["qchem_rel"].get(dname, {}),
            "methods": {},
        }

        errs_by_m: Dict[str, List[float]] = {m: [] for m in methods}
        rxn_names: List[str] = []
        for rxn in complete_rxns:
            rxn_names.append(rxn["reaction"])
            r_rec = {
                "reaction": rxn["reaction"],
                "dataset": dname,
                "datatype_short": dt_short,
                "ref_kcal": rxn["ref_kcal"],
            }
            for m in methods:
                val = sum(c * mols[mk][m]["e_tot"] for c, mk in rxn["stoich"]) * HARTREE_TO_KCAL
                err = val - rxn["ref_kcal"]
                errs_by_m[m].append(err)
                r_rec[f"{m}_kcal"] = val
                r_rec[f"{m}_err"] = err
            all_reactions.append(r_rec)

        for m in methods:
            es = errs_by_m[m]
            mae = sum(abs(e) for e in es) / len(es)
            rmse = math.sqrt(sum(e * e for e in es) / len(es))
            mse = sum(es) / len(es)
            max_abs = max(abs(e) for e in es)

            # Official GSCDB137 Metric calculation
            metric = mae
            if dname in ("O24", "O24x4") and len(complete_rxns) == len(rxns):
                metric = sum(abs(e) * o24_w[rn] for e, rn in zip(es, rxn_names))
            elif dname in ("TMD10", "MOR13", "TMB11") and len(complete_rxns) == len(rxns):
                w_list = tmc_w[dname]
                w_map = {k: v for k, v in w_list}
                resid = w_list[-1][1]
                metric = sum(abs(e) * w_map[rn] for e, rn in zip(es, rxn_names)) + resid

            std_m = std_err.get(dname, {}).get("Metric")
            rel_metric = (metric / std_m) if std_m else None

            ds_res["methods"][m] = {
                "MAE": mae,
                "RMSE": rmse,
                "MSE": mse,
                "MaxAbs": max_abs,
                "Metric": metric,
                "RelMetric": rel_metric,
            }

        # Timing statistics on unique molecules in this dataset
        ds_mkeys = sorted({mk for rxn in complete_rxns for _, mk in rxn["stoich"]})
        ds_res["n_unique_mols"] = len(ds_mkeys)
        ds_res["max_nao"] = max(mols[mk]["nao"] for mk in ds_mkeys)
        ds_res["median_nao"] = sorted(mols[mk]["nao"] for mk in ds_mkeys)[len(ds_mkeys) // 2]
        ds_res["timing_s"] = {}
        for m in methods:
            ts = sorted(mols[mk][m]["t_tot_s"] for mk in ds_mkeys)
            ds_res["timing_s"][m] = {
                "sum_s": sum(ts),
                "mean_s": sum(ts) / len(ts),
                "median_s": ts[len(ts) // 2],
                "max_s": ts[-1],
            }
        per_dataset[dname] = ds_res

    # Category & overall summaries (following GSCDB137 equal-weight per dataset!)
    by_cat: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for d in per_dataset.values():
        by_cat[d["datatype"]].append(d)

    cat_summary: Dict[str, Any] = {}
    for cat, dlist in sorted(by_cat.items()):
        cat_summary[cat] = {
            "n_datasets": len(dlist),
            "n_reactions": sum(d["n_eval"] for d in dlist),
            "methods": {},
        }
        for m in methods:
            cat_summary[cat]["methods"][m] = {
                "mean_MAE": sum(d["methods"][m]["MAE"] for d in dlist) / len(dlist),
                "mean_RMSE": sum(d["methods"][m]["RMSE"] for d in dlist) / len(dlist),
                "mean_RelMetric": sum(d["methods"][m]["RelMetric"] for d in dlist) / len(dlist),
            }
        cat_summary[cat]["qchem_CF22D_mean_RelMetric"] = (
            sum(d["qchem_ref_rel"]["CF22D_QChem"] for d in dlist) / len(dlist)
        )

    # Overall across all evaluated datasets
    dlist_all = list(per_dataset.values())
    overall: Dict[str, Any] = {
        "n_datasets": len(dlist_all),
        "n_reactions": sum(d["n_eval"] for d in dlist_all),
        "n_unique_mols": len(mols),
        "methods": {},
    }
    if dlist_all:
        for m in methods:
            overall["methods"][m] = {
                "mean_MAE": sum(d["methods"][m]["MAE"] for d in dlist_all) / len(dlist_all),
                "mean_RMSE": sum(d["methods"][m]["RMSE"] for d in dlist_all) / len(dlist_all),
                "mean_RelMetric": sum(d["methods"][m]["RelMetric"] for d in dlist_all) / len(dlist_all),
                "rxn_weighted_MAE": sum(abs(r[f"{m}_err"]) for r in all_reactions) / len(all_reactions),
            }
        overall["qchem_CF22D_mean_RelMetric"] = (
            sum(d["qchem_ref_rel"]["CF22D_QChem"] for d in dlist_all) / len(dlist_all)
        )

    # Global per-molecule timing breakdown
    timing_global: Dict[str, Any] = {}
    for m in methods:
        ts = sorted(rec[m]["t_tot_s"] for rec in mols.values() if m in rec)
        if ts:
            timing_global[m] = {
                "n_mols": len(ts),
                "total_cpu_s": sum(ts),
                "mean_s": sum(ts) / len(ts),
                "median_s": ts[len(ts) // 2],
                "p90_s": ts[int(0.9 * len(ts))],
                "max_s": ts[-1],
            }
            if m == "PGFT":
                timing_global["PGFT_breakdown_total_s"] = {
                    "hf": sum(rec["PGFT"]["t_hf_s"] for rec in mols.values() if "PGFT" in rec),
                    "thc": sum(rec["PGFT"]["t_thc_s"] for rec in mols.values() if "PGFT" in rec),
                    "stat": sum(rec["PGFT"]["t_stat_s"] for rec in mols.values() if "PGFT" in rec),
                    "dyn": sum(rec["PGFT"]["t_dyn_s"] for rec in mols.values() if "PGFT" in rec),
                }

    return {
        "overall": overall,
        "by_category": cat_summary,
        "timing_global": timing_global,
        "per_dataset": per_dataset,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gscdb", required=True)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()

    mols = load_molecule_records([Path(p) for p in args.inputs])
    meta = load_gscdb_metadata(Path(args.gscdb))
    summary = evaluate_suite(mols, meta, require_complete=(not args.allow_partial))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n=== GSCDB137 Summary ({summary['overall']['n_datasets']} datasets, {summary['overall']['n_reactions']} reactions, {summary['overall']['n_unique_mols']} mols) ===")
    print(f"{'Dataset':<17} {'Cat':<4} {'N':>3} {'MaxN':>4} | {'PGFT MAE':>8} {'PBE MAE':>8} {'CF22D MAE':>9} {'QChemCF':>8} | {'PGFT Rel':>8} {'PBE Rel':>7} {'CF Rel':>7}")
    print("-" * 105)
    for dname, d in sorted(summary["per_dataset"].items(), key=lambda kv: (kv[1]["datatype_short"], kv[0])):
        pm = d["methods"]["PGFT"]
        pb = d["methods"]["PBE"]
        cf = d["methods"]["CF22D"]
        qc = d["qchem_ref_mae"].get("CF22D_QChem", float("nan"))
        print(
            f"{dname:<17} {d['datatype_short']:<4} {d['n_eval']:>3d} {d['max_nao']:>4d} | "
            f"{pm['MAE']:>8.3f} {pb['MAE']:>8.3f} {cf['MAE']:>9.3f} {qc:>8.3f} | "
            f"{pm['RelMetric']:>8.2f} {pb['RelMetric']:>7.2f} {cf['RelMetric']:>7.2f}"
        )

    print("\n=== By Category (Mean Relative Metric = Metric / Standard_Error; lower is better) ===")
    for cat, c in summary["by_category"].items():
        print(
            f"  {cat:<26s} ({c['n_datasets']:2d} sets, {c['n_reactions']:3d} rxns) | "
            f"PGFT={c['methods']['PGFT']['mean_RelMetric']:6.2f}  "
            f"PBE={c['methods']['PBE']['mean_RelMetric']:6.2f}  "
            f"CF22D={c['methods']['CF22D']['mean_RelMetric']:6.2f}  "
            f"(QChem_CF22D={c['qchem_CF22D_mean_RelMetric']:5.2f})"
        )
    ov = summary["overall"]
    if ov["n_datasets"] > 0:
        print(
            f"\n  OVERALL ({ov['n_datasets']} sets, {ov['n_reactions']} rxns) | "
            f"Mean RelMetric: PGFT={ov['methods']['PGFT']['mean_RelMetric']:.3f}, "
            f"PBE={ov['methods']['PBE']['mean_RelMetric']:.3f}, "
            f"CF22D={ov['methods']['CF22D']['mean_RelMetric']:.3f} "
            f"(QChem CF22D={ov['qchem_CF22D_mean_RelMetric']:.3f})"
        )
    print("\n=== Global Single-Core CPU Timing ===")
    print(json.dumps(summary["timing_global"], indent=2))


if __name__ == "__main__":
    main()
