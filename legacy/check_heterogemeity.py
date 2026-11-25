"""
check_heterogeneity.py
----------------------
Task 1. Is "personalization helps the hard subjects most" real, or is it
regression to the mean?

The trap: a subject who scored badly at K=0 will tend to score better at K=3
just by chance, whether or not personalization did anything. So a negative
correlation between baseline and improvement proves nothing on its own.

The test: the STRANGER arm has the same regression-to-the-mean pressure but no
personal information. If both arms show the same slope, it is regression to the
mean. If only the personal arm shows it, the heterogeneity is real.

    python check_heterogeneity.py
    python check_heterogeneity.py --results personalization_results.csv --k 3

Needs: personalization_results.csv, night_scores.csv (optional), and
subject_profile.csv (optional) for fragmentation and consistency.
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


# ============================================================================
def build_delta_table(res, k):
    """One row per subject per arm: baseline, K score, and the change."""
    base = res[(res.arm == "personal") & (res.k == 0)].set_index("subject")["macro_f1"]
    rows = []
    for arm in ("personal", "stranger"):
        at_k = res[(res.arm == arm) & (res.k == k)].set_index("subject")["macro_f1"]
        common = base.index.intersection(at_k.index)
        for s in common:
            rows.append({"subject": s, "arm": arm, "baseline": base[s],
                         "at_k": at_k[s], "delta": at_k[s] - base[s]})
    return pd.DataFrame(rows)


def slope(df_arm):
    r = stats.linregress(df_arm["baseline"], df_arm["delta"])
    return r.slope, r.intercept, r.rvalue, r.pvalue


def interaction_test(tab):
    """
    delta ~ baseline * arm. The arm:baseline coefficient asks whether the
    slope differs between arms. That is the whole question.
    """
    d = tab.copy()
    d["arm"] = (d["arm"] == "personal").astype(int)
    m = smf.ols("delta ~ baseline * arm", data=d).fit()
    return m


def paired_by_baseline_half(tab):
    """
    Simpler version a reader can hold in their head: split subjects at the
    median baseline. Is the personal-minus-stranger gap bigger in the hard half?
    """
    p = tab[tab.arm == "personal"].set_index("subject")
    s = tab[tab.arm == "stranger"].set_index("subject")
    common = p.index.intersection(s.index)
    gap = (p.loc[common, "at_k"] - s.loc[common, "at_k"])
    base = p.loc[common, "baseline"]
    med = base.median()
    hard, easy = gap[base <= med], gap[base > med]
    t, pv = stats.ttest_ind(hard, easy, equal_var=False)
    return hard, easy, pv


# ============================================================================
def report(tab, k, model, halves, extra):
    line = "=" * 72
    print("\n" + line)
    print(f"IS THE HETEROGENEITY REAL?  (K={k})")
    print(line)

    print("\n  Change from K=0, regressed on K=0 score, per arm:\n")
    print(f"  {'arm':<10}{'n':>4}{'slope':>9}{'r':>8}{'p':>10}")
    print("  " + "-" * 42)
    out = {}
    for arm in ("personal", "stranger"):
        sub = tab[tab.arm == arm]
        sl, ic, r, p = slope(sub)
        out[arm] = sl
        print(f"  {arm:<10}{len(sub):>4}{sl:>9.3f}{r:>8.3f}{p:>10.4f}")

    print("\n  A negative slope means low-baseline subjects gained more.")
    print("  Regression to the mean produces that in BOTH arms.")

    coef = model.params.get("baseline:arm", np.nan)
    pval = model.pvalues.get("baseline:arm", np.nan)
    print("\n" + line)
    print("THE TEST THAT MATTERS: does the slope differ between arms?")
    print(line)
    print(f"\n  interaction coefficient (baseline x arm) = {coef:+.3f}")
    print(f"  p = {pval:.4f}")
    print(f"  slope difference (personal - stranger)  = {out['personal'] - out['stranger']:+.3f}")

    if pval < 0.05 and coef < 0:
        print("\n  -> Personal arm slope is reliably MORE negative than stranger.")
        print("     The extra benefit for hard subjects is NOT just regression")
        print("     to the mean. 'Personalization helps hard subjects most' holds.")
    elif pval < 0.05 and coef > 0:
        print("\n  -> Personal arm helps EASY subjects more than stranger does.")
        print("     Opposite of the hypothesis. Report as such.")
    else:
        print("\n  -> No reliable difference in slope. Whatever baseline")
        print("     correlation exists appears in both arms, so it is consistent")
        print("     with regression to the mean. DROP the 'who benefits' claim")
        print("     or state it only as exploratory.")

    hard, easy, pv = halves
    print("\n" + line)
    print("SAME QUESTION, SIMPLER FORM")
    print(line)
    print("\n  Personal-minus-stranger gap at K, split at median baseline:\n")
    print(f"  hard half (low K=0):  gap {hard.mean():+.3f}  (n={len(hard)})")
    print(f"  easy half (high K=0): gap {easy.mean():+.3f}  (n={len(easy)})")
    print(f"  difference: {hard.mean() - easy.mean():+.3f}   Welch p = {pv:.4f}")

    if extra is not None and len(extra):
        print("\n" + line)
        print("OTHER CANDIDATE MODERATORS  (personal gap vs. subject property)")
        print(line)
        print(f"\n  {'property':<24}{'r':>8}{'p':>10}")
        print("  " + "-" * 42)
        for _, r in extra.iterrows():
            print(f"  {r['property']:<24}{r['r']:>8.3f}{r['p']:>10.4f}")
        print("\n  These are exploratory. Nothing here is corrected for multiple")
        print("  testing; treat p < 0.01 as worth a follow-up, not as a finding.")

    print("\n" + line)
    print("WHAT TO WRITE")
    print(line)
    if pval < 0.05 and coef < 0:
        print("  'The benefit of personalization was larger for subjects the")
        print("   population model scored poorly, and this exceeded the")
        print("   regression-to-the-mean effect observed in the control arm")
        print(f"   (interaction p = {pval:.3f}).'")
    else:
        print("  'Improvement was negatively associated with baseline performance")
        print("   in both arms, consistent with regression to the mean; we found")
        print("   no evidence that personalization preferentially helped")
        print("   poorly-scored subjects.'")
    print(line + "\n")


def moderators(tab, k):
    """Correlate the personal-minus-stranger gap with subject properties."""
    p = tab[tab.arm == "personal"].set_index("subject")
    s = tab[tab.arm == "stranger"].set_index("subject")
    common = p.index.intersection(s.index)
    gap = (p.loc[common, "at_k"] - s.loc[common, "at_k"]).rename("gap")

    props = pd.DataFrame(index=common)
    if os.path.exists("night_scores.csv"):
        ns = pd.read_csv("night_scores.csv")
        props["consistency_sd"] = ns.groupby("subject")["macro_f1"].std()
        props["n_nights"] = ns.groupby("subject")["night"].nunique()
    if os.path.exists("subject_profile.csv"):
        sp = pd.read_csv("subject_profile.csv").set_index("subject")
        for c in ("transitions_per_hour", "hr_mean", "frac_rem", "frac_wake",
                  "stillness", "frag_sd"):
            if c in sp:
                props[c] = sp[c]
    props = props.join(gap).dropna(subset=["gap"])

    rows = []
    for c in props.columns:
        if c == "gap":
            continue
        d = props[[c, "gap"]].dropna()
        if len(d) < 10 or d[c].std() == 0:
            continue
        r, pv = stats.pearsonr(d[c], d["gap"])
        rows.append({"property": c, "r": r, "p": pv, "n": len(d)})
    return pd.DataFrame(rows).sort_values("p") if rows else None


def make_plot(tab, k, path="heterogeneity.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = {"personal": "#27ae60", "stranger": "#c0392b"}
    for arm in ("personal", "stranger"):
        sub = tab[tab.arm == arm]
        ax.scatter(sub["baseline"], sub["delta"], s=28, alpha=0.7,
                   color=colors[arm], label=f"{arm} (n={len(sub)})")
        sl, ic, r, p = slope(sub)
        xs = np.linspace(sub["baseline"].min(), sub["baseline"].max(), 50)
        ax.plot(xs, ic + sl * xs, color=colors[arm], lw=2,
                label=f"{arm} slope {sl:+.2f}, p={p:.3f}")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel("macro-F1 at K=0 (population model)")
    ax.set_ylabel(f"change in macro-F1 at K={k}")
    ax.set_title("If both slopes match, it's regression to the mean.\n"
                 "If the personal slope is steeper, hard subjects genuinely gain more.",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")


# ============================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="personalization_results.csv")
    p.add_argument("--k", type=int, default=3)
    args = p.parse_args()

    res = pd.read_csv(args.results)
    if args.k not in res["k"].unique():
        raise SystemExit(f"K={args.k} not in results; available: {sorted(res.k.unique())}")

    tab = build_delta_table(res, args.k)
    tab.to_csv("heterogeneity_table.csv", index=False)
    model = interaction_test(tab)
    halves = paired_by_baseline_half(tab)
    extra = moderators(tab, args.k)

    report(tab, args.k, model, halves, extra)
    make_plot(tab, args.k)
    print("Wrote heterogeneity_table.csv\n")


if __name__ == "__main__":
    main()