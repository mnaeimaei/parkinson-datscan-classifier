#!/usr/bin/env python3
"""Aggregate Step35A matched baseline-vs-robust E2/P8 stress predictions."""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, balanced_accuracy_score

EPS = 1e-6

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--input-dir", type=Path, default=Path(
        "data/generalization_validation_data/step35a_e2p8_acquisition_robust_domain_stress"))
    p.add_argument("--bootstrap-reps", type=int, default=5000)
    p.add_argument("--seed", type=int, default=3501)
    return p.parse_args()

def resolve(root,p): return p.resolve() if p.is_absolute() else (root/p).resolve()

def find_col(df,names,label):
    for n in names:
        if n in df.columns: return n
    raise ValueError(f"Missing {label}; columns={list(df.columns)}")

def load_arm(base, arm):
    parts=[]
    for fold in range(3):
        p=base/arm/f"fold_{fold}"/"predictions.csv"
        if not p.is_file(): raise FileNotFoundError(p)
        d=pd.read_csv(p)
        u=find_col(d,("uid","UID","subject_uid","subject_id"),"uid")
        y=find_col(d,("is_pathologic","label","target","y_true"),"label")
        pr=find_col(d,("probability","prob","y_prob","positive_probability"),"probability")
        out=pd.DataFrame({
            "uid":d[u].astype(str),
            "is_pathologic":pd.to_numeric(d[y],errors="raise").astype(int),
            "probability":pd.to_numeric(d[pr],errors="raise").astype(float),
            "fold":fold,
        })
        parts.append(out)
    x=pd.concat(parts,ignore_index=True)
    if len(x)!=1362: raise ValueError(f"{arm}: expected 1362 OOF rows, got {len(x)}")
    if x.uid.duplicated().any(): raise ValueError(f"{arm}: duplicate OOF UIDs")
    return x

def ece(y,p,bins=15):
    edges=np.linspace(0,1,bins+1); total=0.
    for i in range(bins):
        m=(p>=edges[i]) & ((p<=edges[i+1]) if i==bins-1 else (p<edges[i+1]))
        if m.sum(): total += m.mean()*abs(p[m].mean()-y[m].mean())
    return float(total)

def metrics(df):
    y=df.is_pathologic.to_numpy(int)
    p=np.clip(df.probability.to_numpy(float),EPS,1-EPS)
    pred=(p>=.5).astype(int)
    return {
        "n":len(df),
        "log_loss":float(log_loss(y,p,labels=[0,1])),
        "auroc":float(roc_auc_score(y,p)),
        "auprc":float(average_precision_score(y,p)),
        "brier":float(np.mean((p-y)**2)),
        "ece":ece(y,p),
        "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
    }

def subj_ll(y,p):
    p=np.clip(p,EPS,1-EPS)
    return -(y*np.log(p)+(1-y)*np.log(1-p))

def main():
    a=parse_args(); root=a.project_root.expanduser().resolve(); base=resolve(root,a.input_dir)
    b=load_arm(base,"baseline"); r=load_arm(base,"robust")
    m=b.merge(r,on="uid",suffixes=("_baseline","_robust"),validate="one_to_one")
    if not np.array_equal(m.is_pathologic_baseline.values,m.is_pathologic_robust.values):
        raise ValueError("Label mismatch baseline vs robust")
    if not np.array_equal(m.fold_baseline.values,m.fold_robust.values):
        raise ValueError("Fold mismatch baseline vs robust")

    mb=metrics(b); mr=metrics(r)
    rows=[]
    for fold in range(3):
        xb=metrics(b[b.fold==fold]); xr=metrics(r[r.fold==fold])
        rows.append({
            "fold":fold,
            "n":xb["n"],
            "baseline_log_loss":xb["log_loss"],
            "robust_log_loss":xr["log_loss"],
            "delta_robust_minus_baseline_log_loss":xr["log_loss"]-xb["log_loss"],
            "baseline_auroc":xb["auroc"], "robust_auroc":xr["auroc"],
            "baseline_auprc":xb["auprc"], "robust_auprc":xr["auprc"],
        })
    fold_df=pd.DataFrame(rows)
    fold_df.to_csv(base/"step35a_fold_comparison.csv",index=False)

    y=m.is_pathologic_baseline.to_numpy(int)
    lb=subj_ll(y,m.probability_baseline.to_numpy(float))
    lr=subj_ll(y,m.probability_robust.to_numpy(float))
    delta=lr-lb
    rng=np.random.default_rng(a.seed); n=len(m)
    boot=np.empty(a.bootstrap_reps)
    for i in range(a.bootstrap_reps):
        idx=rng.integers(0,n,size=n); boot[i]=delta[idx].mean()
    ci=[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))]
    obs=float(delta.mean())
    wins=int((fold_df.delta_robust_minus_baseline_log_loss<0).sum())

    if obs < 0 and wins >= 2:
        status="PROMISING_PASS_TO_STEP35B"
    else:
        status="DO_NOT_ADVANCE"

    summary={
        "step":"35A",
        "status":status,
        "comparison":"E2/P8-like baseline vs acquisition-robust-only",
        "baseline":mb,
        "robust":mr,
        "delta_robust_minus_baseline":{
            "log_loss":mr["log_loss"]-mb["log_loss"],
            "auroc":mr["auroc"]-mb["auroc"],
            "auprc":mr["auprc"]-mb["auprc"],
            "brier":mr["brier"]-mb["brier"],
            "ece":mr["ece"]-mb["ece"],
            "balanced_accuracy":mr["balanced_accuracy"]-mb["balanced_accuracy"],
        },
        "folds_robust_better_log_loss":wins,
        "paired_bootstrap":{
            "repetitions":a.bootstrap_reps,
            "observed_delta_log_loss":obs,
            "ci95":ci,
            "fraction_robust_better":float(np.mean(boot<0)),
        },
        "decision_rule":"Advance only if robust global LL is lower and robust wins >=2/3 stress folds. Step35B ordinary Step11 CV is still mandatory before integration.",
        "hidden_competition_data_used":False,
    }
    (base/"step35a_summary.json").write_text(json.dumps(summary,indent=2))

    report=[
        "STEP 35A — E2/P8 ACQUISITION-ROBUST DOMAIN-STRESS COMPARISON",
        "="*86,
        f"Status: {status}",
        "",
        f"Baseline LL : {mb['log_loss']:.6f}",
        f"Robust LL   : {mr['log_loss']:.6f}",
        f"Delta       : {mr['log_loss']-mb['log_loss']:+.6f}",
        f"Baseline AUROC: {mb['auroc']:.6f}",
        f"Robust AUROC  : {mr['auroc']:.6f}",
        f"Baseline AUPRC: {mb['auprc']:.6f}",
        f"Robust AUPRC  : {mr['auprc']:.6f}",
        f"Baseline Brier: {mb['brier']:.6f}",
        f"Robust Brier  : {mr['brier']:.6f}",
        f"Baseline ECE  : {mb['ece']:.6f}",
        f"Robust ECE    : {mr['ece']:.6f}",
        "",
        f"Stress folds improved: {wins}/3",
        f"Bootstrap 95% CI robust-baseline LL: [{ci[0]:+.6f}, {ci[1]:+.6f}]",
        f"Bootstrap fraction robust better: {float(np.mean(boot<0)):.4f}",
        "",
        "Decision rule:",
        "  robust global LL < baseline AND robust wins >=2/3 folds -> advance to Step35B",
        "  otherwise -> do not advance this E2 robust candidate",
        "",
        "No hidden competition labels were used.",
    ]
    (base/"step35a_report.txt").write_text("\n".join(report)+"\n")

    paired=m.copy()
    paired["subject_log_loss_baseline"]=lb
    paired["subject_log_loss_robust"]=lr
    paired["delta_robust_minus_baseline_log_loss"]=delta
    paired.to_csv(base/"step35a_paired_oof_predictions.csv",index=False)

    print("\n".join(report))

if __name__=="__main__": main()
