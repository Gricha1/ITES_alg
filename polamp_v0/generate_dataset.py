#!/usr/bin/env python3
"""Generate POLAMPv0 dataset: open field, no obstacles, start/goal tasks only."""

import argparse
import math
import os
import random

X_MIN, X_MAX = 5.0, 35.0
Y_MIN, Y_MAX = 5.0, 30.0
MIN_DIST = 5.0
MAX_DIST = 25.0


def _sample_task(rng: random.Random):
    for _ in range(500):
        sx = rng.uniform(X_MIN, X_MAX)
        sy = rng.uniform(Y_MIN, Y_MAX)
        gx = rng.uniform(X_MIN, X_MAX)
        gy = rng.uniform(Y_MIN, Y_MAX)
        dist = math.hypot(gx - sx, gy - sy)
        if MIN_DIST <= dist <= MAX_DIST:
            theta_start = rng.uniform(-math.pi, math.pi)
            theta_goal = math.atan2(gy - sy, gx - sx)
            start = [sx, sy, theta_start, 0.0, 0.0]
            goal = [gx, gy, theta_goal, 0.0, 0.0]
            return start, goal
    raise RuntimeError("failed to sample a valid task")


def _write_obstacle_map(path: str):
    with open(path, "w") as f:
        f.write("0\n")


def _write_tasks(path: str, tasks):
    with open(path, "w") as f:
        f.write(f"{len(tasks)}\n")
        for start, goal in tasks:
            for value in start:
                f.write(f"{value}\t")
            for value in goal:
                f.write(f"{value}\t")
            f.write("\n")


def generate_dataset(output_dir: str, n_train: int, n_val: int, seed: int):
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(seed)

    train_tasks = [_sample_task(rng) for _ in range(n_train)]
    val_tasks = [_sample_task(rng) for _ in range(n_val)]

    _write_obstacle_map(os.path.join(output_dir, "obstacle_map0.txt"))
    _write_tasks(os.path.join(output_dir, "train_map0.txt"), train_tasks)
    _write_tasks(os.path.join(output_dir, "val_map0.txt"), val_tasks)

    print(f"Saved POLAMPv0 dataset to {output_dir}")
    print(f"  obstacles: 0")
    print(f"  train tasks: {n_train}")
    print(f"  val tasks: {n_val}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate POLAMPv0 dataset")
    parser.add_argument("--output_dir", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--n_train", type=int, default=200)
    parser.add_argument("--n_val", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_dataset(args.output_dir, args.n_train, args.n_val, args.seed)
