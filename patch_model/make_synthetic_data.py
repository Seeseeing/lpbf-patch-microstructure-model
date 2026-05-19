import argparse
import csv
from pathlib import Path

import numpy as np


RUNS = [
    # run_id, P, v, h
    (2, 220, 2000, 0.050),
    (13, 220, 1429, 0.070),
    (14, 220, 1111, 0.090),
    (15, 157, 1429, 0.050),
    (16, 283, 1429, 0.090),
    (17, 283, 2000, 0.064),
    (18, 157, 1111, 0.064),
]


def make_one_patch(rng, run_id, patch_id, p, v, h):
    """Create one synthetic patch row.

    This is only a plausible demo dataset. It is not physics-validated.
    The purpose is to let the model and validation workflow run end-to-end.
    """
    # Position in an idealized melt pool section.
    x_norm = rng.uniform(-1.0, 1.0)
    y_norm = rng.uniform(0.0, 1.0)

    # A rough signed distance: positive inside, near zero at boundary.
    # Here we use an ellipse-like melt pool boundary for demonstration.
    ellipse_level = (x_norm / 1.0) ** 2 + ((y_norm - 0.25) / 0.85) ** 2
    signed_distance = 1.0 - ellipse_level

    line_energy = p / v

    # Synthetic thermal descriptors. These are constructed to vary smoothly
    # with process parameters and position.
    t_peak = (
        1200
        + 900 * line_energy
        + 350 * signed_distance
        - 120 * y_norm
        + rng.normal(0, 20)
    )
    cooling_rate = (
        2.0e5
        + 85.0 * v
        - 500.0 * p
        + 5.0e5 * y_norm
        - 1.0e5 * signed_distance
        + rng.normal(0, 2.0e4)
    )
    cooling_rate = max(cooling_rate, 1.0e4)

    # Synthetic zone labels:
    # fine grains are favored by high cooling rate and high normalized depth;
    # coarse grains are favored by high peak temperature and pool interior;
    # boundary is near the transition band.
    score = (
        1.8 * y_norm
        + 0.8 * (cooling_rate - 3.0e5) / 3.0e5
        - 0.7 * (t_peak - 1450) / 250
        - 0.5 * signed_distance
        + rng.normal(0, 0.25)
    )

    if abs(score) < 0.28 or abs(signed_distance) < 0.08:
        zone_label = 2  # boundary
    elif score > 0:
        zone_label = 1  # fine
    else:
        zone_label = 0  # coarse

    # Synthetic local grain size in micrometers.
    grain_size = (
        30
        - 10 * (cooling_rate / 8.0e5)
        + 12 * max(0, signed_distance)
        + 8 * (zone_label == 0)
        - 5 * (zone_label == 1)
        + rng.normal(0, 1.5)
    )
    grain_size = max(grain_size, 1.0)

    return {
        "run_id": run_id,
        "patch_id": f"{run_id}_{patch_id:04d}",
        "P": p,
        "v": v,
        "h": h,
        "x_norm": x_norm,
        "y_norm": y_norm,
        "signed_distance_to_boundary": signed_distance,
        "T_peak": t_peak,
        "cooling_rate": cooling_rate,
        "zone_label": zone_label,
        "local_grain_size": grain_size,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("synthetic_patch_data.csv"))
    parser.add_argument("--patches-per-run", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    for run_id, p, v, h in RUNS:
        for patch_index in range(args.patches_per_run):
            rows.append(make_one_patch(rng, run_id, patch_index, p, v, h))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
