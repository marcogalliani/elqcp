#!/usr/bin/env python3
"""
plot_nonlinear_smoothing.py

Visualize outputs produced by examples/nonlinear_smoothing_sqp.cpp.

Expected CSV files (typically in build/ after running the executable):
  - nonlinear_smoothing_trajectory.csv
  - nonlinear_smoothing_optimization.csv
  - nonlinear_smoothing_iterates.csv

Usage:
  python3 examples/plot_nonlinear_smoothing.py
  python3 examples/plot_nonlinear_smoothing.py --data-dir build
  python3 examples/plot_nonlinear_smoothing.py --data-dir build --no-gif
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def read_trajectory_csv(path):
    t, y_true, y_obs, y_est, u_est = [], [], [], [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t.append(float(row["t"]))
            y_true.append(float(row["y_true"]))
            y_obs.append(float(row["y_obs"]))
            y_est.append(float(row["y_est"]))
            u_est.append(float(row["u_est"]))
    return {
        "t": np.array(t),
        "y_true": np.array(y_true),
        "y_obs": np.array(y_obs),
        "y_est": np.array(y_est),
        "u_est": np.array(u_est),
    }


def read_optimization_csv(path):
    cols = defaultdict(list)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cols["iter"].append(int(row["iter"]))
            cols["cost"].append(float(row["cost"]))
            cols["grad_norm"].append(float(row["grad_norm"]))
            cols["alpha"].append(float(row["alpha"]))
            cols["step_norm"].append(float(row["step_norm"]))
            cols["fd_rel_error"].append(float(row["fd_rel_error"]))
            cols["fd_dir_deriv"].append(float(row["fd_dir_deriv"]))
            cols["adj_dir_deriv"].append(float(row["adj_dir_deriv"]))
    return {k: np.array(v) for k, v in cols.items()}


def read_iterates_csv(path):
    grouped = defaultdict(lambda: {"t": [], "y": []})
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            it = int(row["iter"])
            grouped[it]["t"].append(float(row["t"]))
            grouped[it]["y"].append(float(row["y_est"]))

    iters = sorted(grouped.keys())
    t_ref = np.array(grouped[iters[0]]["t"])
    y_stack = np.array([grouped[it]["y"] for it in iters])
    return iters, t_ref, y_stack


def find_data_dir(user_dir):
    candidates = []
    if user_dir:
        candidates.append(user_dir)
    candidates.extend([".", "build", "../build"])

    for d in candidates:
        traj = os.path.join(d, "nonlinear_smoothing_trajectory.csv")
        opt = os.path.join(d, "nonlinear_smoothing_optimization.csv")
        iters = os.path.join(d, "nonlinear_smoothing_iterates.csv")
        if os.path.exists(traj) and os.path.exists(opt) and os.path.exists(iters):
            return d

    raise FileNotFoundError(
        "Could not find nonlinear smoothing CSV files. Run nonlinear_smoothing_sqp first."
    )


def save_summary_plot(data_dir, traj, opt):
    t = traj["t"]
    u_t = t[:-1]
    u = traj["u_est"][:-1]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(t, traj["y_true"], color="tab:green", linewidth=2, label="True trajectory")
    ax.scatter(t, traj["y_obs"], color="tab:gray", s=16, alpha=0.7, label="Noisy observations")
    ax.plot(t, traj["y_est"], color="tab:blue", linewidth=2, label="Smoothed estimate")
    ax.set_title("Observed vs smoothed trajectory")
    ax.set_xlabel("t")
    ax.set_ylabel("y(t)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    ax.step(u_t, u, where="post", color="tab:orange", linewidth=2)
    ax.set_title("Estimated control u(t)")
    ax.set_xlabel("t")
    ax.set_ylabel("u(t)")
    ax.grid(True, linestyle=":", alpha=0.5)

    ax = axes[1, 0]
    ax.plot(opt["iter"], opt["cost"], "o-", color="tab:blue", label="Objective")
    ax2 = ax.twinx()
    ax2.plot(opt["iter"], opt["grad_norm"], "s--", color="tab:red", label="Gradient norm")
    ax.set_title("Objective and gradient norm")
    ax.set_xlabel("SQP iteration")
    ax.set_ylabel("Cost")
    ax2.set_ylabel("Gradient norm")
    ax.grid(True, linestyle=":", alpha=0.5)

    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=9, loc="upper right")

    ax = axes[1, 1]
    ax.semilogy(opt["iter"], np.maximum(opt["fd_rel_error"], 1e-16), "d-", color="tab:purple")
    ax.set_title("Finite-difference gradient check")
    ax.set_xlabel("SQP iteration")
    ax.set_ylabel("Relative error")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)

    fig.suptitle("Nonlinear ODE smoothing via Riccati-SQP", fontsize=14)
    fig.tight_layout()

    out_png = os.path.join(data_dir, "nonlinear_smoothing_summary.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved summary plot: {out_png}")


def save_animation_gif(data_dir, traj, iter_info):
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
    except Exception as exc:
        print(f"Skipping GIF (matplotlib animation backend unavailable): {exc}")
        return

    iters, t, y_stack = iter_info
    y_obs = traj["y_obs"]
    y_true = traj["y_true"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, y_true, color="tab:green", linewidth=1.8, label="True")
    ax.scatter(t, y_obs, color="tab:gray", s=12, alpha=0.55, label="Observed")
    line_est, = ax.plot(t, y_stack[0], color="tab:blue", linewidth=2.2, label="Estimate")

    txt = ax.text(0.02, 0.95, "iter=0", transform=ax.transAxes, va="top")
    ax.set_title("SQP trajectory evolution")
    ax.set_xlabel("t")
    ax.set_ylabel("y(t)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="best")

    y_min = min(np.min(y_stack), np.min(y_obs), np.min(y_true))
    y_max = max(np.max(y_stack), np.max(y_obs), np.max(y_true))
    pad = 0.08 * (y_max - y_min + 1e-12)
    ax.set_ylim(y_min - pad, y_max + pad)

    def update(frame_idx):
        line_est.set_ydata(y_stack[frame_idx])
        txt.set_text(f"iter={iters[frame_idx]}")
        return line_est, txt

    anim = FuncAnimation(fig, update, frames=len(iters), interval=250, blit=False)
    out_gif = os.path.join(data_dir, "nonlinear_smoothing_animation.gif")
    anim.save(out_gif, writer=PillowWriter(fps=4))
    plt.close(fig)
    print(f"Saved animation: {out_gif}")


def main():
    parser = argparse.ArgumentParser(description="Plot Riccati-SQP smoothing outputs")
    parser.add_argument("--data-dir", default=None, help="Directory containing output CSV files")
    parser.add_argument("--no-gif", action="store_true", help="Disable GIF creation")
    args = parser.parse_args()

    data_dir = find_data_dir(args.data_dir)
    print(f"Using data directory: {data_dir}")

    traj_path = os.path.join(data_dir, "nonlinear_smoothing_trajectory.csv")
    opt_path = os.path.join(data_dir, "nonlinear_smoothing_optimization.csv")
    iter_path = os.path.join(data_dir, "nonlinear_smoothing_iterates.csv")

    traj = read_trajectory_csv(traj_path)
    opt = read_optimization_csv(opt_path)
    iter_info = read_iterates_csv(iter_path)

    save_summary_plot(data_dir, traj, opt)
    if not args.no_gif:
        save_animation_gif(data_dir, traj, iter_info)

    print("Done.")


if __name__ == "__main__":
    main()
