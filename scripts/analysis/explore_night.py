"""
explore_night.py

Script that inspects one night of data before it is trusted. Reads only, changes nothing.
Run: python scripts/analysis/explore_night.py data/Bidslab00/1 --start-min 120 --window-min 10

1. Loads the night
- load - cleans motion and heart rate CSVs, reads labels and recording start from labels.mat
- returns which label key was used (expert or dreem)

2. Prints a text report (print_report)
- recording start time and number of epochs
- sampling rates - median gap between samples and the implied rate for motion and heart rate
- coverage - share of samples inside the labeled window, epochs with no HR sample
- stage distribution - count and percent of epochs per stage
- two alignment checks - motion and heart rate in Wake vs N3, marked OK or PROBLEM

3. Draws a zoomed window plot (plot_window)
- raw heart rate, raw x/y/z acceleration, and sleep stage for a short slice (default 10 min)
- 30-second epoch boundaries drawn as faint vertical lines
- saved as <subject>_night<N>_window.png

4. Draws a whole-night plot (plot_full_night)
- per-epoch heart rate and motion (aggregated with per_epoch_mean) plus the hypnogram
- saved as <subject>_night<N>_full.png

5. Runs from the command line (main, argparse)
- folder is required; --start-min, --window-min, --label, --outdir are optional flags
- depends on hypnogram_service.py for STAGE_ROW and STAGE_ROW_LABELS (plot y-axis rows)
"""

import argparse
import datetime as dt
import os
import sys

import matplotlib
matplotlib.use("Agg")  # draw to files, no window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.io import loadmat  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root, so `sleep_staging` imports
from sleep_staging.alignment_service import AlignmentService  # noqa: E402
from sleep_staging.constants import Constants  # noqa: E402
from sleep_staging.hypnogram_service import STAGE_ROW, STAGE_ROW_LABELS  # noqa: E402
from sleep_staging.raw_data_service import TIMEZONE, RawDataService  # noqa: E402

EPOCH = Constants.EPOCH_SECONDS
STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM", 5: "Unknown"}
LINE = "=" * 68


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="folder containing motion.csv, hr.csv, labels.mat")
    parser.add_argument("--start-min", type=float, default=120,
                        help="where the zoomed window starts, minutes into the recording")
    parser.add_argument("--window-min", type=float, default=10, help="zoomed window length")
    parser.add_argument("--label", choices=["expert", "auto"], default="expert")
    parser.add_argument("--outdir", default=os.path.join(Constants.FIGURES_DIR, "explore"), help="where to save the figures")
    return parser.parse_args()


def load(folder, which_label):
    motion = RawDataService.read_signal(os.path.join(folder, "motion.csv"), ["x", "y", "z"])
    heart_rate = RawDataService.read_signal(os.path.join(folder, "hr.csv"), ["bpm"])
    mat = loadmat(os.path.join(folder, "labels.mat"))
    if "recStart" not in mat:
        keys = [key for key in mat if not key.startswith("__")]
        raise KeyError(f"labels.mat has no recStart; keys found: {keys}")
    recording_start = RawDataService.parse_recording_start(mat["recStart"])
    key = RawDataService.get_label_key(mat, which_label)
    labels = np.asarray(mat[key]).ravel().astype(int)
    return motion, heart_rate, labels, recording_start, key


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------
def print_report(motion, heart_rate, labels, recording_start, label_key, folder):
    print(LINE)
    print(f"NIGHT: {folder}")
    print(f"Labels used: {label_key}")
    print(LINE)
    start = dt.datetime.fromtimestamp(recording_start, TIMEZONE)
    print(f"\nRecording start : {start:%Y-%m-%d %H:%M:%S %Z}")
    print(f"Epochs          : {len(labels)}   ({len(labels) * EPOCH / 3600:.2f} hours)")

    print("\nSAMPLING RATES")
    for name, signal in (("motion", motion), ("heart rate", heart_rate)):
        rate = AlignmentService.sampling_rate(signal)
        if rate is None:
            print(f"  {name:<11}: could not estimate")
        else:
            print(f"  {name:<11}: median gap {rate[0]:.4f} s  ->  ~{rate[1]:.2f} Hz"
                  f"   (samples: {len(signal):,})")

    print("\nCOVERAGE")
    for name, signal in (("motion", motion), ("heart rate", heart_rate)):
        share = AlignmentService.share_inside_window(signal, recording_start, len(labels), EPOCH)
        print(f"  {name:<11}: {share * 100:5.1f}% of samples fall inside the labelled window")
    missing = AlignmentService.epochs_without_samples(heart_rate, labels, recording_start)
    print(f"  HR dropout : {missing} of {len(labels)} epochs have no HR sample")

    print("\nSTAGE DISTRIBUTION")
    stages, counts = np.unique(labels, return_counts=True)
    for stage, count in zip(stages, counts):
        print(f"  {STAGE_NAMES.get(int(stage), stage):<8}: {count:5d} epochs "
              f"({count / len(labels) * 100:5.1f}%)")

    print("\nALIGNMENT CHECKS")
    awake, deep = AlignmentService.motion_by_stage(motion, labels, recording_start)
    print(f"  Mean motion during Wake : {awake:.5f}")
    print(f"  Mean motion during N3   : {deep:.5f}")
    print_check(AlignmentService.check(awake, deep), "more movement awake than in deep sleep")
    awake, deep = AlignmentService.heart_rate_by_stage(heart_rate, labels, recording_start)
    print(f"  Mean HR during Wake     : {awake:.1f} bpm")
    print(f"  Mean HR during N3       : {deep:.1f} bpm")
    print_check(AlignmentService.check(awake, deep), "heart rate lower in deep sleep")
    print()


def print_check(result, expectation):
    if result is None:
        print("  -> not enough of one stage to compare")
    elif result == "ok":
        print(f"  -> OK: {expectation}, as expected")
    else:
        print(f"  -> PROBLEM: the opposite of '{expectation}'. Check the alignment.")


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def stage_steps(labels, recording_start):
    """Clock time and plot row of every epoch; Unknown epochs become gaps."""
    times = np.full(len(labels), np.nan)
    rows = np.full(len(labels), np.nan)
    for epoch, stage in enumerate(labels):
        if stage in STAGE_ROW:
            times[epoch] = recording_start + epoch * EPOCH
            rows[epoch] = STAGE_ROW[stage]
    return times, rows


def label_clock_times(axis):
    ticks = axis.get_xticks()
    axis.set_xticks(ticks)
    axis.set_xticklabels([dt.datetime.fromtimestamp(t, TIMEZONE).strftime("%H:%M")
                          for t in ticks])


def draw_stages(axis, times, rows, line_width):
    axis.step(times, rows, where="post", lw=line_width, color="black")
    axis.set_yticks(range(len(STAGE_ROW_LABELS)))
    axis.set_yticklabels(STAGE_ROW_LABELS)
    axis.set_ylabel("stage")
    axis.set_xlabel("local time")


def plot_window(motion, heart_rate, labels, recording_start, start_min, window_min, path):
    """Raw heart rate and x, y, z acceleration, with epoch boundaries marked."""
    start = recording_start + start_min * 60
    end = start + window_min * 60
    window_motion = motion[(motion["t"] >= start) & (motion["t"] < end)]
    window_hr = heart_rate[(heart_rate["t"] >= start) & (heart_rate["t"] < end)]
    times, rows = stage_steps(labels, recording_start)
    shown = (times >= start - EPOCH) & (times < end + EPOCH)

    fig, axes = plt.subplots(5, 1, figsize=(13, 9), sharex=True)
    axes[0].plot(window_hr["t"], window_hr["bpm"], marker=".", ms=4, lw=0.8, color="crimson")
    axes[0].set_ylabel("HR (bpm)")
    axes[0].set_title(f"{window_min}-minute window starting {start_min} min into the recording",
                      loc="left")
    for axis, column, color in zip(axes[1:4], ("x", "y", "z"), ("#1f77b4", "#2ca02c", "#9467bd")):
        axis.plot(window_motion["t"], window_motion[column], lw=0.5, color=color)
        axis.set_ylabel(f"accel {column}")
    draw_stages(axes[4], times[shown], rows[shown], 1.6)

    first_epoch = int(start_min * 60 / EPOCH)
    last_epoch = int(end - recording_start) // EPOCH
    for epoch in range(first_epoch, last_epoch + 1):
        boundary = recording_start + epoch * EPOCH
        if start <= boundary <= end:
            for axis in axes:
                axis.axvline(boundary, color="0.9", lw=0.5, zorder=0)
    for axis in axes:
        axis.set_xlim(start, end)
        axis.grid(alpha=0.25)
    label_clock_times(axes[4])
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  wrote {path}")


def plot_full_night(motion, heart_rate, labels, recording_start, path):
    """Per-epoch heart rate and motion over the whole night, with the hypnogram."""
    motion = motion.copy()
    motion["mag"] = np.sqrt(motion["x"] ** 2 + motion["y"] ** 2 + motion["z"] ** 2)
    motion["jerk"] = motion["mag"].diff().abs()
    n_epochs = len(labels)
    jerk = AlignmentService.per_epoch_mean(motion, "jerk", recording_start, n_epochs)
    bpm = AlignmentService.per_epoch_mean(heart_rate, "bpm", recording_start, n_epochs)
    epoch_times = recording_start + np.arange(n_epochs) * EPOCH
    times, rows = stage_steps(labels, recording_start)

    fig, axes = plt.subplots(3, 1, figsize=(13, 7), sharex=True)
    axes[0].plot(epoch_times, bpm, lw=0.9, color="crimson")
    axes[0].set_ylabel("HR (bpm)")
    axes[0].set_title("Whole night", loc="left")
    axes[1].plot(epoch_times, jerk, lw=0.7, color="#1f77b4")
    axes[1].set_ylabel("motion")
    draw_stages(axes[2], times, rows, 1.0)
    for axis in axes:
        axis.set_xlim(epoch_times[0], epoch_times[-1])
        axis.grid(alpha=0.25)
    label_clock_times(axes[2])
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    args = parse_arguments()
    if not os.path.isdir(args.folder):
        sys.exit("Not a folder: " + args.folder)
    motion, heart_rate, labels, recording_start, key = load(args.folder, args.label)
    print_report(motion, heart_rate, labels, recording_start, key, args.folder)

    os.makedirs(args.outdir, exist_ok=True)
    name = os.path.basename(os.path.normpath(args.folder))
    subject = os.path.basename(os.path.dirname(os.path.normpath(args.folder)))
    prefix = os.path.join(args.outdir, f"{subject}_night{name}")
    plot_window(motion, heart_rate, labels, recording_start,
                args.start_min, args.window_min, prefix + "_window.png")
    plot_full_night(motion, heart_rate, labels, recording_start, prefix + "_full.png")


if __name__ == "__main__":
    main()
