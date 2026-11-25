"""
run_feature_split.py
--------------------
The experiment your PI asked for: split features into "personal" (encode
individual baselines) versus "non-personal" (population-level summaries).
Then run personalization on each group to see which one responds to adding K.

DEFINITION OF "PERSONAL FEATURES"
---------------------------------
A feature is personal if computing it for one epoch requires knowing something
specific to THIS subject — their median heart rate, their typical stillness
pattern — that differs meaningfully across people.

  Personal:
    hr_zscore       - heart rate relative to THIS person's median (in SDs)
    hr_relative     - heart rate minus THIS person's median (in bpm)
    hr_vs_baseline  - heart rate minus THIS person's 30-min rolling baseline
    epochs_since_move - how long since THIS person last moved (individual rhythm)
    epochs_until_move - how long until THIS person next moves
    + all _prev/_next context copies of the above

  Non-personal:
    hr_mean, hr_std, hr_min, hr_max, hr_range, hr_baseline_30min
    mag_mean, mag_std, jerk_mean, jerk_max, move_frac, orient_change
    move_frac_5min, move_frac_15min
    minutes_elapsed, frac_through_night, hours_elapsed
    hour_of_day, clock_cosine, clock_sine
    + all _prev/_next context copies of these

This definition can be stated in one sentence: "Features that subtract or
reference a per-subject statistic (median HR, last movement time) are personal;
all others are non-personal."

OUTPUTS
-------
1. Feature CSVs: features_personal.csv, features_nonpersonal.csv
2. Ablation with personal as a category
3. Personalization × feature-type interaction (the K × category plot)
4. Combined 4-line plot (your model + Walch, personal + stranger)
5. Random night order figure

    python run_feature_split.py --features features_v2.csv
    python run_feature_split.py --features features_v2.csv --skip-personalization
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, cohen_kappa_score
from xgboost import XGBClassifier

META = ("subject", "night", "epoch", "stage3", "stage5")
NAMES = {0: "Wake", 1: "NREM", 2: "REM"}

# ============================================================================
# FEATURE CLASSIFICATION
# ============================================================================
PERSONAL_BASES = {"hr_zscore", "hr_relative", "hr_vs_baseline",
                  "epochs_since_move", "epochs_until_move"}


def is_personal(feature_name):
    """Is this feature personal (references an individual baseline)?"""
    base = feature_name
    for suf in ("_prev1", "_prev2", "_prev4", "_next1", "_next2", "_next4"):
        base = base.replace(suf, "")
    return base in PERSONAL_BASES


def classify_features(columns):
    """Split feature columns into personal and non-personal."""
    feats = [c for c in columns if c not in META]
    personal = [f for f in feats if is_personal(f)]
    nonpersonal = [f for f in feats if not is_personal(f)]
    return feats, personal, nonpersonal


def print_classification(personal, nonpersonal):
    print(f"\n  PERSONAL features ({len(personal)}):")
    for f in personal:
        print(f"    {f}")
    print(f"\n  NON-PERSONAL features ({len(nonpersonal)}):")
    for f in nonpersonal:
        print(f"    {f}")
    print()


# ============================================================================
# MODEL HELPERS
# ============================================================================
def load_params():
    if os.path.exists("xgb_best_params.json"):
        with open("xgb_best_params.json") as f:
            return json.load(f)
    return {"n_estimators": 150, "learning_rate": 0.1, "random_state": 0}


def make_model(params, cap_trees=150):
    p = {**params, "n_estimators": min(params.get("n_estimators", 150), cap_trees)}
    return XGBClassifier(**p, use_label_encoder=False,
                         eval_metric="mlogloss", verbosity=0)


# ============================================================================
# ABLATION WITH PERSONAL CATEGORY
# ============================================================================
def run_ablation(df, feature_sets, label, subjects, cap_trees):
    """LOSO for each feature set."""
    params = load_params()
    results = {}
    for name, feats in feature_sets.items():
        if not feats:
            continue
        rows = []
        t0 = time.time()
        for held in subjects:
            tr, te = df[df.subject != held], df[df.subject == held]
            m = make_model(params, cap_trees)
            m.fit(tr[feats], tr[label])
            p = m.predict(te[feats])
            y = te[label].values
            per = f1_score(y, p, average=None, labels=sorted(NAMES), zero_division=0)
            rows.append({"subject": held,
                         "macro_f1": f1_score(y, p, average="macro"),
                         "kappa": cohen_kappa_score(y, p),
                         **{f"f1_{NAMES[i]}": v for i, v in enumerate(per)}})
        res = pd.DataFrame(rows)
        results[name] = res
        print(f"  {name:<24} {len(feats):>3} feat  "
              f"macro-F1 {res['macro_f1'].mean():.3f}  "
              f"({(time.time()-t0)/60:.1f} min)")
    return results


# ============================================================================
# PERSONALIZATION × FEATURE TYPE (K × category)
# ============================================================================
LEAKY = ["hr_relative", "hr_zscore"]
SOURCE = "hr_mean"


def renormalize(df, visible_nights, subject):
    out = df.copy()
    if SOURCE not in out.columns:
        return out
    mask = out["subject"] == subject
    if len(visible_nights) == 0:
        ref = out.loc[~mask, SOURCE].median()
        sd = out.loc[~mask, SOURCE].std()
    else:
        vis = out[mask & out["night"].isin(visible_nights)]
        ref = vis[SOURCE].median()
        sd = vis[SOURCE].std()
    if not np.isfinite(sd) or sd == 0:
        sd = out[SOURCE].std()
    if "hr_relative" in out:
        out.loc[mask, "hr_relative"] = out.loc[mask, SOURCE] - ref
    if "hr_zscore" in out:
        out.loc[mask, "hr_zscore"] = (out.loc[mask, SOURCE] - ref) / (sd + 1e-6)
    return out


def run_personalization_for_feats(df, feats, label, subjects, donors_pool,
                                  ks, n_donors, cap_trees, rng, seed=0):
    """Run personalization on a specific feature subset."""
    params = load_params()
    rows = []
    for subj in subjects:
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

                # personal
                work = renormalize(df, train_nights, subj)
                extra = work[(work.subject == subj) & (work.night.isin(train_nights))]
                train = pd.concat([work[base_mask], extra], ignore_index=True)
                m = make_model(params, cap_trees)
                m.fit(train[feats], train[label])
                p = m.predict(work.loc[test_mask, feats])
                y = work.loc[test_mask, label].values
                rows.append({"arm": "personal", "k": k, "subject": subj, "donor": donor,
                             "macro_f1": f1_score(y, p, average="macro")})

                # stranger
                work = renormalize(df, [], subj)
                dk = list(donor_nights[:k])
                extra = work[(work.subject == donor) & (work.night.isin(dk))]
                train = pd.concat([work[base_mask], extra], ignore_index=True)
                m = make_model(params, cap_trees)
                m.fit(train[feats], train[label])
                p = m.predict(work.loc[test_mask, feats])
                y = work.loc[test_mask, label].values
                rows.append({"arm": "stranger", "k": k, "subject": subj, "donor": donor,
                             "macro_f1": f1_score(y, p, average="macro")})

    res = pd.DataFrame(rows)
    # average across donors
    per_subj = res.groupby(["arm", "k", "subject"])["macro_f1"].mean().reset_index()
    return per_subj


# ============================================================================
# PLOTS
# ============================================================================
def plot_ablation_personal(abl_results, path="ablation_personal.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(abl_results.keys())
    means = [abl_results[n]["macro_f1"].mean() for n in names]
    sds = [abl_results[n]["macro_f1"].std() for n in names]

    colors = {"personal only": "#e74c3c", "non-personal only": "#3498db",
              "all features": "#2c3e50", "hr only": "#c0392b",
              "motion only": "#2980b9", "time only": "#27ae60"}

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(names, means, yerr=sds, capsize=4,
                  color=[colors.get(n, "#7f8c8d") for n in names])

    for i, (n, m, s) in enumerate(zip(names, means, sds)):
        ax.text(i, m + s + 0.008, f"{m:.3f}", ha="center", fontsize=9, fontweight="bold")

    ax.set_ylabel("macro-F1", fontsize=11)
    ax.set_title("Ablation with personal feature category\n"
                 "Red = personal features only. Blue = non-personal only.", fontsize=11)
    ax.tick_params(axis="x", rotation=25, labelsize=9)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_k_by_category(category_results, path="personalization_by_category.png"):
    """The K × category interaction plot your PI asked for."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"personal only": "#e74c3c", "non-personal only": "#3498db",
              "all features": "#2c3e50", "walch features": "#f39c12"}

    for cat_name, per_subj in category_results.items():
        for arm, ls, alpha in [("personal", "-", 1.0), ("stranger", "--", 0.5)]:
            sub = per_subj[per_subj.arm == arm].groupby("k")["macro_f1"].agg(["mean", "sem"])
            ax.errorbar(sub.index, sub["mean"], yerr=sub["sem"],
                        marker="o" if arm == "personal" else "s",
                        ls=ls, lw=1.8, alpha=alpha, capsize=4,
                        color=colors.get(cat_name, "#7f8c8d"),
                        label=f"{cat_name} ({arm})")

    ax.set_xlabel("K = personal nights added", fontsize=11)
    ax.set_ylabel("macro-F1", fontsize=11)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_title("Which feature category benefits from personalization?\n"
                 "Solid = personal arm. Dashed = stranger arm.\n"
                 "A widening gap means personalization helps that category.",
                 fontsize=10)
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_combined_4line(your_results, walch_results, path="combined_personalization.png"):
    """4 lines on one plot: your personal, your stranger, Walch personal, Walch stranger."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))

    configs = [
        (your_results, "personal", "Our features (personal)", "#27ae60", "-", "o"),
        (your_results, "stranger", "Our features (stranger)", "#27ae60", "--", "s"),
        (walch_results, "personal", "Walch features (personal)", "#e67e22", "-", "o"),
        (walch_results, "stranger", "Walch features (stranger)", "#e67e22", "--", "s"),
    ]

    for per_subj, arm, label, color, ls, marker in configs:
        sub = per_subj[per_subj.arm == arm].groupby("k")["macro_f1"].agg(["mean", "sem"])
        ax.errorbar(sub.index, sub["mean"], yerr=sub["sem"],
                    marker=marker, ls=ls, lw=1.8, capsize=4,
                    color=color, label=label)

    ax.set_xlabel("K = nights added to training", fontsize=11)
    ax.set_ylabel("macro-F1", fontsize=11)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_title("Our features vs Walch features\n"
                 "Solid = personal arm. Dashed = stranger arm.", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_gap_by_category(category_results, path="gap_by_category.png"):
    """Bar chart: personalization gap at K=3 for each feature category."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"personal only": "#e74c3c", "non-personal only": "#3498db",
              "all features": "#2c3e50", "walch features": "#f39c12"}

    names, gaps, pvals = [], [], []
    for cat_name, per_subj in category_results.items():
        p = per_subj[(per_subj.arm == "personal") & (per_subj.k == 3)].set_index("subject")["macro_f1"]
        s = per_subj[(per_subj.arm == "stranger") & (per_subj.k == 3)].set_index("subject")["macro_f1"]
        common = p.index.intersection(s.index)
        if len(common) < 5:
            continue
        gap = (p[common] - s[common])
        _, pv = stats.ttest_1samp(gap, 0)
        names.append(cat_name)
        gaps.append(gap.mean())
        pvals.append(pv)

    bars = ax.bar(names, gaps, color=[colors.get(n, "#7f8c8d") for n in names])
    ax.axhline(0, color="0.5", lw=0.8)

    for i, (g, p) in enumerate(zip(gaps, pvals)):
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        color = "#27ae60" if p < 0.05 else "#7f8c8d"
        ax.text(i, g + 0.001 if g >= 0 else g - 0.002,
                f"{g:+.3f}\np={p:.3f}\n{star}",
                ha="center", fontsize=9, fontweight="bold", color=color)

    ax.set_ylabel("Δ macro-F1 (personal − stranger) at K=3", fontsize=10)
    ax.set_title("Which feature category benefits from personalization?", fontsize=11)
    ax.tick_params(axis="x", rotation=15, labelsize=10)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# ============================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="features_v2.csv")
    p.add_argument("--walch", default="walch_features.csv")
    p.add_argument("--walch-pers", default="personalization_v2_walch_per_subject.csv")
    p.add_argument("--your-pers", default="personalization_v2_full_per_subject.csv")
    p.add_argument("--classes", type=int, default=3)
    p.add_argument("--k-max", type=int, default=3)
    p.add_argument("--donors", type=int, default=3)
    p.add_argument("--cap-trees", type=int, default=150)
    p.add_argument("--skip-personalization", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    label = "stage3" if args.classes == 3 else "stage5"

    # ---- Load and classify ----
    print("Loading...")
    df = pd.read_csv(args.features)
    df = df[df[label] >= 0].reset_index(drop=True)
    all_feats, personal, nonpersonal = classify_features(df.columns)
    print_classification(personal, nonpersonal)

    subjects = sorted(df["subject"].unique())
    counts = df.groupby("subject")["night"].nunique()
    eligible = sorted(counts[counts > args.k_max].index)
    donors_pool = sorted(counts[counts >= args.k_max].index)

    # ---- Step 1: Ablation with personal category ----
    print("=" * 60)
    print("ABLATION WITH PERSONAL FEATURE CATEGORY")
    print("=" * 60)
    feature_sets = {
        "personal only": personal,
        "non-personal only": nonpersonal,
        "all features": all_feats,
    }
    abl = run_ablation(df, feature_sets, label, subjects, args.cap_trees)
    for name, res in abl.items():
        res.to_csv(f"ablation_{name.replace(' ', '_')}.csv", index=False)
    plot_ablation_personal(abl)

    if args.skip_personalization:
        print("\nSkipping personalization runs (--skip-personalization).")
        print("Run without that flag for the full K × category interaction.\n")
        return

    # ---- Step 2: Personalization × feature type ----
    print("\n" + "=" * 60)
    print("PERSONALIZATION BY FEATURE CATEGORY")
    print("=" * 60)
    ks = list(range(0, args.k_max + 1))
    category_results = {}

    for cat_name, feats in feature_sets.items():
        print(f"\n  Running: {cat_name} ({len(feats)} features)...")
        t0 = time.time()
        per_subj = run_personalization_for_feats(
            df, feats, label, eligible, donors_pool, ks,
            args.donors, args.cap_trees, rng, args.seed)
        per_subj.to_csv(f"pers_split_{cat_name.replace(' ', '_')}.csv", index=False)
        category_results[cat_name] = per_subj

        # quick summary
        for k in ks:
            if k == 0:
                continue
            pa = per_subj[(per_subj.arm == "personal") & (per_subj.k == k)].set_index("subject")["macro_f1"]
            sa = per_subj[(per_subj.arm == "stranger") & (per_subj.k == k)].set_index("subject")["macro_f1"]
            common = pa.index.intersection(sa.index)
            if len(common) > 5:
                gap = (pa[common] - sa[common]).mean()
                _, pv = stats.ttest_1samp(pa[common] - sa[common], 0)
                print(f"    K={k}: gap {gap:+.4f}  p={pv:.4f}")
        print(f"    ({(time.time()-t0)/60:.1f} min)")

    # Add Walch if available
    if os.path.exists(args.walch_pers):
        walch_ps = pd.read_csv(args.walch_pers)
        category_results["walch features"] = walch_ps

    # ---- Step 3: Plots ----
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    plot_k_by_category(category_results)
    plot_gap_by_category(category_results)

    # Combined 4-line plot
    if os.path.exists(args.your_pers) and os.path.exists(args.walch_pers):
        your_ps = pd.read_csv(args.your_pers)
        walch_ps = pd.read_csv(args.walch_pers)
        plot_combined_4line(your_ps, walch_ps)

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY: PERSONALIZATION GAP AT K=3 BY FEATURE TYPE")
    print("=" * 60)
    print(f"\n  {'category':<24}{'gap':>9}{'p':>10}{'significant':>13}")
    print("  " + "-" * 56)
    for cat_name, per_subj in category_results.items():
        pa = per_subj[(per_subj.arm == "personal") & (per_subj.k == 3)].set_index("subject")["macro_f1"]
        sa = per_subj[(per_subj.arm == "stranger") & (per_subj.k == 3)].set_index("subject")["macro_f1"]
        common = pa.index.intersection(sa.index)
        if len(common) < 5:
            continue
        gap = (pa[common] - sa[common])
        _, pv = stats.ttest_1samp(gap, 0)
        sig = "YES" if pv < 0.05 else "no"
        print(f"  {cat_name:<24}{gap.mean():>+9.4f}{pv:>10.4f}{sig:>13}")

    print("\n  If 'personal only' shows a significant gap and 'non-personal only'")
    print("  does not, the mechanism is confirmed: personalization works through")
    print("  features that encode individual baselines.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()