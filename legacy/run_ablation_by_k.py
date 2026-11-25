"""
run_ablation_by_k.py
--------------------
Extends the ablation study across K: for each feature category, how much
does performance improve as you add 1, 2, 3 personal nights?

This answers: which kinds of features actually benefit from subject-specific
training data?

Feature categories:
  - Personal features only (17)
  - Non-personal features only (51)
  - Heart rate only (all HR features)
  - Motion only (all motion features)
  - HR + motion (no time)
  - All features (68)
  - Walch features (4, if available)

For each category, runs personalization K=0-3 with the clean control design
(donor removed, stranger arm). Then plots the gap (personal - stranger) by K
for each category on one figure.

    python run_ablation_by_k.py                    # all categories
    python run_ablation_by_k.py --quick            # 8 subjects, fast test
    python run_ablation_by_k.py --donors 2         # fewer donors for speed

Outputs:
    ablation_by_k_results.csv    all per-subject results
    ablation_by_k_gaps.csv       gap summary per category per K
    ablation_by_k.png            the main figure
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

META = ("subject", "night", "epoch", "stage3", "stage5")
PERSONAL_BASES = {"hr_zscore", "hr_relative", "hr_vs_baseline",
                  "epochs_since_move", "epochs_until_move"}


# ============================================================================
# FEATURE CLASSIFICATION
# ============================================================================
def is_personal(f):
    base = f
    for suf in ("_prev1", "_prev2", "_prev4", "_next1", "_next2", "_next4"):
        base = base.replace(suf, "")
    return base in PERSONAL_BASES


def classify(f):
    """Which signal source does this feature come from?"""
    base = f
    for suf in ("_prev1", "_prev2", "_prev4", "_next1", "_next2", "_next4"):
        base = base.replace(suf, "")
    if base.startswith("hr_"):
        return "hr"
    if base.startswith(("mag_", "jerk_", "move_", "orient_", "accel_",
                        "epochs_since", "epochs_until")):
        return "motion"
    if base in ("minutes_elapsed", "frac_through_night", "hours_elapsed",
                "hour_of_day", "clock_cosine", "clock_sine"):
        return "time"
    return "other"


def build_feature_sets(columns):
    feats = [c for c in columns if c not in META]
    by_source = {"hr": [], "motion": [], "time": []}
    personal, nonpersonal = [], []

    for f in feats:
        src = classify(f)
        if src in by_source:
            by_source[src].append(f)
        if is_personal(f):
            personal.append(f)
        else:
            nonpersonal.append(f)

    sets = {
        "personal only": personal,
        "non-personal only": nonpersonal,
        "HR only": by_source["hr"],
        "motion only": by_source["motion"],
        "HR + motion": by_source["hr"] + by_source["motion"],
        "all features": feats,
    }
    return {k: v for k, v in sets.items() if v}


# ============================================================================
# PERSONALIZATION PER FEATURE SET
# ============================================================================
def load_params(cap=150):
    if os.path.exists("xgb_best_params.json"):
        p = json.load(open("xgb_best_params.json"))
        p["n_estimators"] = min(p.get("n_estimators", 150), cap)
        return p
    return {"n_estimators": cap, "learning_rate": 0.1, "random_state": 0}


def renormalize(df, visible_nights, subject):
    out = df.copy()
    if "hr_mean" not in out.columns:
        return out
    mask = out["subject"] == subject
    if len(visible_nights) == 0:
        ref = out.loc[~mask, "hr_mean"].median()
        sd = out.loc[~mask, "hr_mean"].std()
    else:
        vis = out[mask & out["night"].isin(visible_nights)]
        ref = vis["hr_mean"].median()
        sd = vis["hr_mean"].std()
    if not np.isfinite(sd) or sd == 0:
        sd = out["hr_mean"].std()
    for col in ["hr_relative", "hr_zscore"]:
        if col in out:
            if col == "hr_relative":
                out.loc[mask, col] = out.loc[mask, "hr_mean"] - ref
            else:
                out.loc[mask, col] = (out.loc[mask, "hr_mean"] - ref) / (sd + 1e-6)
    return out


def run_category(df, feats, label, eligible, donors_pool, ks, n_donors,
                 params, rng, seed=0, needs_renorm=True):
    """Run personalization for one feature category."""
    rows = []
    for subj in eligible:
        subj_nights = sorted(df.loc[df.subject == subj, "night"].unique())
        pool = [d for d in donors_pool if d != subj]
        donors = list(rng.choice(pool, size=min(n_donors, len(pool)), replace=False))

        for donor in donors:
            donor_nights = sorted(df.loc[df.subject == donor, "night"].unique())
            base_mask = ~df["subject"].isin([subj, donor])

            for k in ks:
                if len(subj_nights) <= k or len(donor_nights) < k:
                    continue
                train_nights = list(subj_nights[:k])
                test_nights = [n for n in subj_nights if n not in train_nights]
                test_mask = (df.subject == subj) & (df.night.isin(test_nights))

                # personal arm
                if needs_renorm:
                    work = renormalize(df, train_nights, subj)
                else:
                    work = df
                extra = work[(work.subject == subj) & (work.night.isin(train_nights))]
                train = pd.concat([work[base_mask], extra], ignore_index=True)
                m = XGBClassifier(**params, use_label_encoder=False,
                                  eval_metric="mlogloss", verbosity=0)
                m.fit(train[feats], train[label])
                p = m.predict(work.loc[test_mask, feats])
                y = work.loc[test_mask, label].values
                rows.append({"arm": "personal", "k": k, "subject": subj,
                             "macro_f1": f1_score(y, p, average="macro")})

                # stranger arm
                if needs_renorm:
                    work = renormalize(df, [], subj)
                else:
                    work = df
                dk = list(donor_nights[:k])
                extra = work[(work.subject == donor) & (work.night.isin(dk))]
                train = pd.concat([work[base_mask], extra], ignore_index=True)
                m = XGBClassifier(**params, use_label_encoder=False,
                                  eval_metric="mlogloss", verbosity=0)
                m.fit(train[feats], train[label])
                p = m.predict(work.loc[test_mask, feats])
                y = work.loc[test_mask, label].values
                rows.append({"arm": "stranger", "k": k, "subject": subj,
                             "macro_f1": f1_score(y, p, average="macro")})

    res = pd.DataFrame(rows)
    per_subj = res.groupby(["arm", "k", "subject"])["macro_f1"].mean().reset_index()
    return per_subj


# ============================================================================
# ANALYSIS AND PLOTTING
# ============================================================================
def compute_gaps(all_results):
    """For each category and K, compute the personal-stranger gap."""
    rows = []
    for cat, per_subj in all_results.items():
        for k in sorted(per_subj["k"].unique()):
            if k == 0:
                continue
            p = per_subj[(per_subj.arm == "personal") & (per_subj.k == k)].set_index("subject")["macro_f1"]
            s = per_subj[(per_subj.arm == "stranger") & (per_subj.k == k)].set_index("subject")["macro_f1"]
            common = p.index.intersection(s.index)
            if len(common) < 5:
                continue
            gap = p[common] - s[common]
            _, pv = stats.ttest_1samp(gap, 0)
            rows.append({
                "category": cat, "k": k, "n": len(common),
                "gap_mean": gap.mean(), "gap_sem": gap.sem(),
                "p": pv,
            })
    return pd.DataFrame(rows)


def plot_results(all_results, gaps_df, path="ablation_by_k.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "personal only": "#e74c3c",
        "non-personal only": "#3498db",
        "HR only": "#c0392b",
        "motion only": "#2980b9",
        "HR + motion": "#8e44ad",
        "all features": "#2c3e50",
        "walch features": "#f39c12",
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Panel 1: absolute macro-F1 (personal arm) by K
    ax = axes[0]
    for cat, per_subj in all_results.items():
        sub = per_subj[per_subj.arm == "personal"].groupby("k")["macro_f1"].agg(["mean", "sem"])
        ax.errorbar(sub.index, sub["mean"], yerr=sub["sem"],
                    marker="o", lw=1.8, capsize=4,
                    color=colors.get(cat, "#7f8c8d"), label=cat)
    ax.set_xlabel("K = personal nights added", fontsize=11)
    ax.set_ylabel("macro-F1 (personal arm)", fontsize=11)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_title("Absolute performance by K", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)

    # Panel 2: gap (personal - stranger) by K — this is the key plot
    ax = axes[1]
    for cat in all_results:
        sub = gaps_df[gaps_df.category == cat].sort_values("k")
        if sub.empty:
            continue
        marker_style = "o"
        ax.errorbar(sub["k"], sub["gap_mean"], yerr=sub["gap_sem"],
                    marker=marker_style, lw=1.8, capsize=4,
                    color=colors.get(cat, "#7f8c8d"), label=cat)

        # annotate significance
        for _, r in sub.iterrows():
            star = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else ""
            if star:
                ax.text(r["k"] + 0.08, r["gap_mean"], star, fontsize=10,
                        fontweight="bold", color=colors.get(cat, "#7f8c8d"),
                        va="center")

    ax.axhline(0, color="#e74c3c", ls="--", lw=1)
    ax.set_xlabel("K = personal nights added", fontsize=11)
    ax.set_ylabel("Δ macro-F1 (personal − stranger)", fontsize=11)
    ax.set_xticks([1, 2, 3])
    ax.set_title("Personalization gap by K\n"
                 "Which feature categories benefit from personal data?",
                 fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def report(gaps_df):
    line = "=" * 66
    print(f"\n{line}")
    print("WHICH FEATURE CATEGORIES BENEFIT FROM PERSONALIZATION?")
    print(line)
    print(f"\n  {'category':<22}{'K=1 gap':>10}{'K=2 gap':>10}{'K=3 gap':>10}{'K=3 p':>10}")
    print("  " + "-" * 62)

    for cat in gaps_df["category"].unique():
        sub = gaps_df[gaps_df.category == cat].set_index("k")
        vals = []
        for k in [1, 2, 3]:
            if k in sub.index:
                vals.append(f"{sub.loc[k, 'gap_mean']:+.4f}")
            else:
                vals.append("    —")
        p3 = sub.loc[3, "p"] if 3 in sub.index else np.nan
        star = " *" if p3 < 0.05 else "  "
        print(f" {star}{cat:<22}{vals[0]:>10}{vals[1]:>10}{vals[2]:>10}"
              f"{p3:>10.4f}" if np.isfinite(p3) else
              f"  {cat:<22}{vals[0]:>10}{vals[1]:>10}{vals[2]:>10}{'—':>10}")

    print(f"\n  * = K=3 gap significant at p < 0.05")
    print(f"\n  Categories whose gap GROWS with K benefit from more personal data.")
    print(f"  Categories whose gap stays flat or near zero do not.")
    print(line + "\n")


# ============================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="features_v2.csv")
    p.add_argument("--walch", default="walch_features.csv")
    p.add_argument("--classes", type=int, default=3)
    p.add_argument("--k-max", type=int, default=3)
    p.add_argument("--donors", type=int, default=2)
    p.add_argument("--cap-trees", type=int, default=100)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    label = "stage3" if args.classes == 3 else "stage5"
    params = load_params(args.cap_trees)
    ks = list(range(0, args.k_max + 1))

    # ---- Load and classify ----
    print("Loading...")
    df = pd.read_csv(args.features)
    df = df[df[label] >= 0].reset_index(drop=True)
    feature_sets = build_feature_sets(df.columns)

    counts = df.groupby("subject")["night"].nunique()
    eligible = sorted(counts[counts > args.k_max].index)
    donors_pool = sorted(counts[counts >= args.k_max].index)
    if args.quick:
        eligible = eligible[:8]

    n_donors = args.donors
    print(f"  {len(df):,} epochs, {df['subject'].nunique()} subjects")
    print(f"  {len(eligible)} eligible, {n_donors} donors, trees capped at {args.cap_trees}")

    for name, feats in feature_sets.items():
        print(f"  {name:<22} {len(feats):>3} features")

    total_fits = len(feature_sets) * len(eligible) * n_donors * len(ks) * 2
    print(f"\n  ~{total_fits} total model fits")
    print(f"  estimated time: {total_fits * 0.5 / 60:.0f}–{total_fits * 1.5 / 60:.0f} min\n")

    # ---- Run each category ----
    all_results = {}
    t0 = time.time()

    for cat_name, feats in feature_sets.items():
        print(f"  Running: {cat_name} ({len(feats)} features)...")
        t1 = time.time()
        needs_renorm = any(is_personal(f) for f in feats)
        per_subj = run_category(df, feats, label, eligible, donors_pool, ks,
                                n_donors, params, rng, args.seed, needs_renorm)
        all_results[cat_name] = per_subj
        per_subj.to_csv(f"abk_{cat_name.replace(' ', '_')}.csv", index=False)
        elapsed = (time.time() - t1) / 60
        print(f"    done ({elapsed:.1f} min)")

    # ---- Add Walch if available ----
    if os.path.exists(args.walch):
        print(f"  Running: walch features...")
        t1 = time.time()
        wdf = pd.read_csv(args.walch)
        wdf = wdf[wdf[label] >= 0].reset_index(drop=True)
        wfeats = [c for c in wdf.columns if c not in META]
        wps = run_category(wdf, wfeats, label, eligible, donors_pool, ks,
                           n_donors, params, rng, args.seed, needs_renorm=False)
        all_results["walch features"] = wps
        wps.to_csv("abk_walch_features.csv", index=False)
        print(f"    done ({(time.time()-t1)/60:.1f} min)")

    print(f"\n  total: {(time.time()-t0)/60:.1f} min")

    # ---- Compute gaps and plot ----
    gaps_df = compute_gaps(all_results)
    gaps_df.to_csv("ablation_by_k_gaps.csv", index=False)

    # Save all results
    all_rows = []
    for cat, ps in all_results.items():
        ps = ps.copy()
        ps["category"] = cat
        all_rows.append(ps)
    pd.concat(all_rows, ignore_index=True).to_csv("ablation_by_k_results.csv", index=False)

    report(gaps_df)
    plot_results(all_results, gaps_df)
    print("Wrote ablation_by_k_results.csv, ablation_by_k_gaps.csv, ablation_by_k.png\n")


if __name__ == "__main__":
    main()