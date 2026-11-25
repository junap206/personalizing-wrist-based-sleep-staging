"""
run_nonpersonal_only.py
-----------------------
Quick script to get JUST the non-personal personalization result.
Uses 2 donors and capped trees for speed.

    python run_nonpersonal_only.py
"""

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


def is_personal(f):
    base = f
    for suf in ("_prev1", "_prev2", "_prev4", "_next1", "_next2", "_next4"):
        base = base.replace(suf, "")
    return base in PERSONAL_BASES


def main():
    print("Loading...")
    df = pd.read_csv("features_v2.csv")
    df = df[df["stage3"] >= 0].reset_index(drop=True)

    feats = [c for c in df.columns if c not in META and not is_personal(c)]
    print(f"  {len(feats)} non-personal features")
    print(f"  {len(df):,} epochs, {df['subject'].nunique()} subjects")

    params = json.load(open("xgb_best_params.json")) if os.path.exists("xgb_best_params.json") else {}
    params = {**params, "n_estimators": 100}  # cap for speed

    counts = df.groupby("subject")["night"].nunique()
    eligible = sorted(counts[counts > 3].index)
    donors_pool = sorted(counts[counts >= 3].index)
    rng = np.random.default_rng(0)
    ks = [0, 1, 2, 3]
    n_donors = 2

    print(f"  {len(eligible)} subjects, {n_donors} donors, trees capped at 100")
    print(f"  ~{len(eligible) * n_donors * len(ks) * 2} fits\n")

    rows = []
    t0 = time.time()
    for i, subj in enumerate(eligible, 1):
        t1 = time.time()
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

                for arm in ("personal", "stranger"):
                    if arm == "personal":
                        extra = df[(df.subject == subj) & (df.night.isin(train_nights))]
                    else:
                        dk = list(donor_nights[:k])
                        extra = df[(df.subject == donor) & (df.night.isin(dk))]

                    train = pd.concat([df[base_mask], extra], ignore_index=True)
                    m = XGBClassifier(**params, use_label_encoder=False,
                                     eval_metric="mlogloss", verbosity=0)
                    m.fit(train[feats], train["stage3"])
                    p = m.predict(df.loc[test_mask, feats])
                    y = df.loc[test_mask, "stage3"].values
                    rows.append({"arm": arm, "k": k, "subject": subj,
                                 "donor": donor,
                                 "macro_f1": f1_score(y, p, average="macro")})

        print(f"  [{i:>2}/{len(eligible)}] {subj:<14} {(time.time()-t1)/60:.1f} min")

    res = pd.DataFrame(rows)
    per_subj = res.groupby(["arm", "k", "subject"])["macro_f1"].mean().reset_index()
    per_subj.to_csv("pers_split_nonpersonal_only.csv", index=False)

    print(f"\n  total {(time.time()-t0)/60:.1f} min")
    print("\n" + "=" * 50)
    print("NON-PERSONAL FEATURES — PERSONALIZATION RESULTS")
    print("=" * 50)

    for k in [1, 2, 3]:
        pa = per_subj[(per_subj.arm == "personal") & (per_subj.k == k)].set_index("subject")["macro_f1"]
        sa = per_subj[(per_subj.arm == "stranger") & (per_subj.k == k)].set_index("subject")["macro_f1"]
        common = pa.index.intersection(sa.index)
        gap = (pa[common] - sa[common])
        _, pv = stats.ttest_1samp(gap, 0)
        print(f"  K={k}: gap {gap.mean():+.4f}  p={pv:.4f}  "
              f"{'*' if pv < 0.017 else 'n.s.'}")

    print("\nCompare to personal-only features:")
    print("  K=1: gap +0.0098  p=0.0000  *")
    print("  K=2: gap +0.0099  p=0.0002  *")
    print("  K=3: gap +0.0102  p=0.0014  *")
    print("\nIf non-personal gaps are near zero and not significant,")
    print("the mechanism is confirmed.\n")


if __name__ == "__main__":
    main()