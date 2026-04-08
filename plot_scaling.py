#!/usr/bin/env python3
"""
plot_scaling.py

Draw a log-log scaling plot of solver time vs. horizon length N for the two
ELQCP solvers: Riccati recursion and sparse KKT (Eigen::SparseLU).

Usage
-----
1.  Build and run the C++ benchmark first to generate timing_results.csv:

        mkdir -p build && cd build
        cmake .. && make -j
        ./compare_solvers          # writes ../timing_results.csv

2.  Then run this script from the repository root:

        python3 plot_scaling.py

    or point it at an alternative CSV:

        python3 plot_scaling.py path/to/timing_results.csv

The script also overlays an O(N) reference line anchored to the first data
point of each solver, which makes the linear scaling visually obvious.
"""

import sys
import os
import csv
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(path: str) -> dict:
    """Return {'N': [...], 'riccati': [...], 'kkt': [...]} from the CSV."""
    data = {'N': [], 'riccati': [], 'kkt': []}
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['N'].append(int(row['N']))
            data['riccati'].append(float(row['riccati_us']))
            data['kkt'].append(float(row['kkt_sparse_us']))
    return data


def find_or_run_csv(csv_path: str) -> dict:
    """Load CSV; if it does not exist, try to run compare_solvers first."""
    if not os.path.exists(csv_path):
        # Try to locate the binary in common build directories
        candidates = [
            os.path.join('build', 'compare_solvers'),
            os.path.join('build', 'Release', 'compare_solvers'),
            os.path.join('..', 'build', 'compare_solvers'),
        ]
        binary = next((c for c in candidates if os.path.isfile(c)), None)

        if binary is None:
            raise FileNotFoundError(
                f"'{csv_path}' not found and no pre-built compare_solvers binary "
                f"could be located.  Please build with CMake and run "
                f"compare_solvers first."
            )

        print(f"Running {binary} to generate {csv_path} …")
        result = subprocess.run([binary], capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            raise RuntimeError(f"compare_solvers failed:\n{result.stderr}")

    return load_csv(csv_path)


# ---------------------------------------------------------------------------
# Reference O(N) lines
# ---------------------------------------------------------------------------

def linear_ref(N_arr, t0, N0):
    """O(N) line anchored at (N0, t0)."""
    return t0 * np.array(N_arr) / N0


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(data: dict, out_path: str = 'scaling.png') -> None:
    N        = np.array(data['N'])
    riccati  = np.array(data['riccati'])
    kkt      = np.array(data['kkt'])

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # ---- solver curves ----
    ax.loglog(N, riccati, 'o-',  color='tab:blue',   lw=2, ms=7,
              label='Riccati recursion')
    ax.loglog(N, kkt,     's--', color='tab:orange',  lw=2, ms=7,
              label='Sparse KKT (Eigen::SparseLU)')

    # ---- O(N) reference lines (anchored at first point of each solver) ----
    ref_N = np.array([N[0], N[-1]])

    riccati_ref = linear_ref(ref_N, riccati[0], N[0])
    kkt_ref     = linear_ref(ref_N, kkt[0],     N[0])

    ax.loglog(ref_N, riccati_ref, ':', color='tab:blue',   lw=1.4, alpha=0.6,
              label='O(N) reference (Riccati)')
    ax.loglog(ref_N, kkt_ref,     ':', color='tab:orange', lw=1.4, alpha=0.6,
              label='O(N) reference (KKT)')

    # ---- decoration ----
    ax.set_xlabel('Horizon length  N', fontsize=13)
    ax.set_ylabel('Solve time  [µs]',  fontsize=13)
    ax.set_title('ELQCP solver scaling vs. horizon length\n'
                 '(nx = 2, nu = 1, LTI system)', fontsize=13)

    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.grid(True, which='both', ls=':', alpha=0.4)
    ax.legend(fontsize=10, loc='upper left')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Plot saved to  {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Ratio subplot
# ---------------------------------------------------------------------------

def plot_ratio(data: dict, out_path: str = 'scaling_ratio.png') -> None:
    """Secondary figure: SpKKT / Riccati ratio vs N."""
    N       = np.array(data['N'])
    ratio   = np.array(data['kkt']) / np.array(data['riccati'])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(N, ratio, 'D-', color='tab:purple', lw=2, ms=7)
    ax.axhline(1.0, color='gray', lw=1, ls='--', label='Riccati baseline (ratio = 1)')

    ax.set_xlabel('Horizon length  N', fontsize=13)
    ax.set_ylabel('SpKKT time / Riccati time', fontsize=13)
    ax.set_title('Sparse KKT overhead relative to Riccati recursion', fontsize=12)

    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.grid(True, which='both', ls=':', alpha=0.4)
    ax.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Plot saved to  {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    csv_file = sys.argv[1] if len(sys.argv) > 1 else 'timing_results.csv'

    data = find_or_run_csv(csv_file)

    print(f"\nLoaded {len(data['N'])} data points from '{csv_file}'")
    print(f"  N range    : {data['N'][0]} … {data['N'][-1]}")
    print(f"  Riccati    : {data['riccati'][0]:.1f} µs … {data['riccati'][-1]:.1f} µs")
    print(f"  Sparse KKT : {data['kkt'][0]:.1f} µs … {data['kkt'][-1]:.1f} µs")

    plot(data)
    plot_ratio(data)
