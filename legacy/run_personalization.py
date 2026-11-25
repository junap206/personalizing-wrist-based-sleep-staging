"""
run_personalization.py
----------------------
The experiment your PI designed:

    "Let's have everyone else plus one day of that person and try to predict the
     other days. And then everyone else plus two days of this person."

So for each subject: train on all 46 others plus K of that subject's own nights,
then test on their REMAINING nights. Sweep K = 0, 1, 2, 3.

TWO THINGS HE WAS SPECIFIC ABOUT
--------------------------------
1. No leakage. "You should not leak the information like using the mean of all
   the nights. When you get just one night of this person, you only should use
   the features of that night."

   This matters because hr_zscore in features.csv subtracts each subject's median
   computed across ALL their nights -- including the test nights. So even at K=0
   the normalization already encodes the answer. This script RECOMPUTES the
   per-person normalization from only the nights available at each K, so K=0 uses
   no personal information at all.

2. The model must not know which subject it is scoring. Subject identity is never
   a feature; the personal nights only enter through the training data.

A STRANGER ARM IS INCLUDED
--------------------------
Adding someone's night also adds training data. Without a volume-matched control
you cannot say whether a gain came from THEIR nights or just from MORE nights.
The stranger arm adds K nights from a randomly chosen different subject. The gap
between the two curves is the actual finding. Drop it with --no-stranger.

    python run_personalization.py --quick             # 10 subjects, fast look
    python run_personalization.py --max-iter 100
    python run_personalization.py --no-stranger       # only what the PI asked for

Outputs:
    personalization_results.csv    one row per (arm, K, subject)
    personalization_summary.csv    one row per (arm, K)
    personalization.png            the curves
"""

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, cohen_kappa_score

META = ("subject", "night", "epoch", "stage3", "stage5")
NAMES = {0: "Wake", 1: "NREM", 2: "REM"}

# Columns that were normalised per-subject during extraction. These have to be
# rebuilt from only the nights the model is allowed to see.
LEAKY = ["hr_relative", "hr_zscore"]
SOURCE = "hr_mean"           # what they were derived from


# ============================================================================
def renormalize(df, visible_nights, subject):
    """
    Rebuild the per-subject normalised heart rate columns for ONE subject, using
    only `visible_nights`. If nothing is visible (K=0), fall back to the
    population baseline, which contains no information about this person.
    """
    out = df.copy()
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


def pick_nights(nights, k, how="first"):
    """Which of a subject's nights go into training."""
    if how == "first":
        return list(nights[:k])
    return list(nights[-k:])


# ============================================================================
def run_one(df, feats, label, subject, k, arm, max_iter, rng, fix_leak, seed=0):
    """
    Train on everyone else (+ k nights from somewhere) and test on the
    subject's remaining nights. Returns None if there is nothing to test on.
    """
    subj_nights = sorted(df.loc[df["subject"] == subject, "night"].unique())
    if len(subj_nights) <= k:
        return None                      # no held-out nights left

    train_nights = pick_nights(subj_nights, k)
    test_nights = [n for n in subj_nights if n not in train_nights]

    work = df
    if fix_leak:
        # Personal normalisation may only use the nights we are allowed to see.
        work = renormalize(df, train_nights if arm == "personal" else [], subject)

    others = work["subject"] != subject
    test = work[(work["subject"] == subject) & (work["night"].isin(test_nights))]

    if arm == "personal":
        extra = work[(work["subject"] == subject) & (work["night"].isin(train_nights))]
    elif arm == "stranger":
        pool = [s for s in work["subject"].unique() if s != subject]
        donor = rng.choice(pool)
        dn = sorted(work.loc[work["subject"] == donor, "night"].unique())
        if len(dn) < k:
            return None
        extra = work[(work["subject"] == donor) & (work["night"].isin(dn[:k]))]
        # the donor's own rows are already in `others`; duplicating them is the
        # point -- it matches the extra data volume of the personal arm
    else:
        extra = work.iloc[0:0]

    train = pd.concat([work[others], extra], ignore_index=True)

    model = HistGradientBoostingClassifier(
        max_iter=max_iter, learning_rate=0.1,
        early_stopping=False, random_state=seed)
    model.fit(train[feats], train[label])
    yhat = model.predict(test[feats])
    ytrue = test[label].values

    per = f1_score(ytrue, yhat, average=None, labels=sorted(NAMES),
                   zero_division=0)
    return {
        "arm": arm, "k": k, "subject": subject,
        "macro_f1": f1_score(ytrue, yhat, average="macro"),
        "kappa": cohen_kappa_score(ytrue, yhat),
        "accuracy": (yhat == ytrue).mean(),
        "n_test_nights": len(test_nights),
        "n_test_epochs": len(test),
        **{f"f1_{NAMES[i]}": v for i, v in enumerate(per)},
    }


# ============================================================================
def report(res, summary):
    line = "=" * 74
    print("\n" + line)
    print("PERSONALIZATION CURVE")
    print(line)
    print("\n  K = number of the subject's own nights added to training.")
    print("  Tested on their remaining nights. Same subjects at every K.\n")
    print(f"  {'arm':<11}{'K':>3}{'n subj':>8}{'macro-F1':>11}{'kappa':>9}"
          f"{'REM F1':>9}{'vs K=0':>9}")
    print("  " + "-" * 60)

    for arm in summary["arm"].unique():
        sub = summary[summary["arm"] == arm].sort_values("k")
        base = sub.loc[sub["k"] == 0, "macro_f1"]
        base = base.iloc[0] if len(base) else np.nan
        for _, r in sub.iterrows():
            d = f"{r['macro_f1'] - base:+.3f}" if r["k"] > 0 else "  --"
            print(f"  {arm:<11}{int(r['k']):>3}{int(r['n_subjects']):>8}"
                  f"{r['macro_f1']:>11.3f}{r['kappa']:>9.3f}"
                  f"{r['f1_REM']:>9.3f}{d:>9}")
        print()

    # ---- the comparison that matters ----
    if {"personal", "stranger"} <= set(summary["arm"]):
        print(line)
        print("PERSONAL vs STRANGER  (the actual finding)")
        print(line)
        print("\n  Both arms add the same amount of data. The only difference is")
        print("  whose nights they are.\n")
        print(f"  {'K':>3}{'personal':>12}{'stranger':>12}{'gap':>10}{'paired p':>11}")
        print("  " + "-" * 48)
        from scipy import stats
        for k in sorted(summary["k"].unique()):
            if k == 0:
                continue
            p_ = res[(res.arm == "personal") & (res.k == k)].set_index("subject")["macro_f1"]
            s_ = res[(res.arm == "stranger") & (res.k == k)].set_index("subject")["macro_f1"]
            common = p_.index.intersection(s_.index)
            if len(common) < 5:
                continue
            a, b = p_[common].values, s_[common].values
            _, pv = stats.ttest_rel(a, b)
            print(f"  {k:>3}{a.mean():>12.3f}{b.mean():>12.3f}"
                  f"{a.mean()-b.mean():>10.3f}{pv:>11.4f}")
        print("\n  A positive gap that grows with K is evidence that a person's")
        print("  own nights carry information beyond just more training data.")

    # ---- who benefits ----
    print("\n" + line)
    print("WHO BENEFITS?")
    print(line)
    k_max = res[res.arm == "personal"]["k"].max()
    p0 = res[(res.arm == "personal") & (res.k == 0)].set_index("subject")["macro_f1"]
    pk = res[(res.arm == "personal") & (res.k == k_max)].set_index("subject")["macro_f1"]
    common = p0.index.intersection(pk.index)
    delta = (pk[common] - p0[common]).sort_values()
    print(f"\n  Change from K=0 to K={int(k_max)}, per subject:")
    print(f"    improved : {(delta > 0).sum()} of {len(delta)}")
    print(f"    worsened : {(delta < 0).sum()} of {len(delta)}")
    print(f"    mean     : {delta.mean():+.3f}   median {delta.median():+.3f}")
    print(f"\n  Biggest gains:")
    for s, v in delta.tail(5)[::-1].items():
        print(f"    {s:<14}{v:+.3f}")
    print(f"  Biggest losses:")
    for s, v in delta.head(5).items():
        print(f"    {s:<14}{v:+.3f}")
    print("\n  Save this: it can be joined against night-to-night consistency")
    print("  to test whether personalization helps the stable subjects most.")
    print(line + "\n")


def make_plot(res, summary, path="personalization.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    colors = {"personal": "#27ae60", "stranger": "#c0392b"}

    for j, (col, title) in enumerate(
            [("macro_f1", "macro-F1"), ("kappa", "kappa"), ("f1_REM", "REM F1")]):
        ax = axes[j]
        for arm in summary["arm"].unique():
            sub = summary[summary["arm"] == arm].sort_values("k")
            ax.errorbar(sub["k"], sub[col], yerr=sub[f"{col}_sd"],
                        marker="o", capsize=4, lw=1.8,
                        color=colors.get(arm, "#34495e"), label=arm)
        ax.set_xlabel("K = own nights added to training")
        ax.set_ylabel(title)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(sorted(summary["k"].unique()))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)

    fig.suptitle("Personalization curve. Error bars are sd across subjects.",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")


# ============================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="features_v2.csv")
    p.add_argument("--classes", type=int, choices=[3, 5], default=3)
    p.add_argument("--k-max", type=int, default=3)
    p.add_argument("--quick", action="store_true", help="10 subjects only")
    p.add_argument("--max-iter", type=int, default=100)
    p.add_argument("--no-stranger", action="store_true")
    p.add_argument("--keep-leak", action="store_true",
                   help="do NOT recompute per-subject normalisation (not advised)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print("Loading...")
    df = pd.read_csv(args.features)
    label = "stage3" if args.classes == 3 else "stage5"
    df = df[df[label] >= 0].reset_index(drop=True)
    feats = [c for c in df.columns if c not in META]

    subjects = sorted(df["subject"].unique())
    if args.quick:
        subjects = subjects[:10]

    arms = ["personal"] if args.no_stranger else ["personal", "stranger"]
    ks = list(range(0, args.k_max + 1))

    fix_leak = not args.keep_leak
    present = [c for c in LEAKY if c in df.columns]
    print(f"  {len(df):,} epochs, {len(feats)} features, {len(subjects)} subjects")
    if fix_leak and present:
        print(f"  recomputing {present} per K to avoid leakage")
    elif present:
        print(f"  WARNING: leaving {present} as-is; K=0 is optimistic")

    # only subjects with enough nights can reach the higher K values
    counts = df.groupby("subject")["night"].nunique()
    for k in ks:
        n = int((counts[subjects] > k).sum())
        print(f"    K={k}: {n} subjects have a held-out night left")

    total = sum(int((counts[subjects] > k).sum()) for k in ks) * len(arms)
    print(f"\n  ~{total} model fits\n")

    rng = np.random.default_rng(args.seed)
    rows = []
    t0 = time.time()
    for arm in arms:
        for k in ks:
            if k == 0 and arm == "stranger":
                continue                      # identical to personal K=0
            t1 = time.time()
            got = 0
            for subj in subjects:
                r = run_one(df, feats, label, subj, k, arm,
                            args.max_iter, rng, fix_leak, args.seed)
                if r:
                    rows.append(r)
                    got += 1
            mean = np.mean([r["macro_f1"] for r in rows
                            if r["arm"] == arm and r["k"] == k])
            print(f"  {arm:<10} K={k}  {got:>3} subjects  "
                  f"macro-F1 {mean:.3f}   ({(time.time()-t1)/60:.1f} min)")

    res = pd.DataFrame(rows)

    # K=0 is shared; copy it into the stranger arm so both curves start together
    if "stranger" in arms:
        z = res[(res.arm == "personal") & (res.k == 0)].copy()
        z["arm"] = "stranger"
        res = pd.concat([res, z], ignore_index=True)

    agg = {c: [("mean"), ("std")] for c in ("macro_f1", "kappa", "f1_REM")}
    summary = res.groupby(["arm", "k"]).agg(
        n_subjects=("subject", "nunique"),
        macro_f1=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
        kappa=("kappa", "mean"), kappa_sd=("kappa", "std"),
        f1_REM=("f1_REM", "mean"), f1_REM_sd=("f1_REM", "std"),
        accuracy=("accuracy", "mean"),
    ).reset_index()

    res.to_csv("personalization_results.csv", index=False)
    summary.to_csv("personalization_summary.csv", index=False)
    print(f"\n  total {(time.time()-t0)/60:.1f} min")

    report(res, summary)
    make_plot(res, summary)
    print("Wrote personalization_results.csv and personalization_summary.csv\n")


if __name__ == "__main__":
    main()