#!/usr/bin/env python3
"""Quick smoke test for POLAMPv0 (no obstacles)."""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")

from types import SimpleNamespace

from polamp_wrapper.create_env import create_polamp_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env_args = SimpleNamespace(
        polamp_dataset="polamp_v0",
        polamp_uniform_feasible_train_dataset=False,
        polamp_random_train_dataset=False,
        validate=False,
    )
    env, state_dim, goal_dim, action_dim, _ = create_polamp_env(env_args)
    env.seed(args.seed)

    obs = env.reset()
    print(f"state_dim={state_dim}, goal_dim={goal_dim}, action_dim={action_dim}")
    print(f"frame_size={env.frame_size}, n_beams={env.n_beams}, max_len={env.max_len}")
    print(f"obstacles on map0: {len(env.maps['map0'])}")
    print(f"train tasks: {len(env._env.trainTasks['map0'])}")
    print(f"val tasks: {len(env.valTasks['map0'])}")
    print(f"start={obs['achieved_goal']}, goal={obs['desired_goal']}")

    total_cost = 0.0
    for step in range(args.steps):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_cost += info.get("cost", 0.0)
        if done:
            print(f"episode done at step {step + 1}, reward={reward:.2f}, "
                  f"success={info.get('is_success')}, cost={info.get('cost', 0.0)}")
            obs = env.reset()

    print(f"smoke test ok ({args.steps} steps, total_cost={total_cost:.1f})")


if __name__ == "__main__":
    main()
