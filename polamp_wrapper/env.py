import gym
import numpy as np
import torch
from gym import spaces


class SafePolampEnv:
    """ITES wrapper around GCPOLAMPEnvironment."""

    GOAL_DIM = 2

    def __init__(self, env):
        self._env = env
        self.action_space = env.action_space
        self.evaluate = False
        self.safe_dataset = None
        self.val_map_key = "map0"
        self._setup_observation_space()

    def _setup_observation_space(self):
        base_obs_space = self._env.observation_space
        self.observation_space = spaces.Dict({
            "observation": base_obs_space["observation"],
            "desired_goal": spaces.Box(
                -np.inf, np.inf, shape=(self.GOAL_DIM,), dtype=np.float32),
            "achieved_goal": spaces.Box(
                -np.inf, np.inf, shape=(self.GOAL_DIM,), dtype=np.float32),
        })

    @property
    def n_beams(self):
        return self._env.n_beams

    @property
    def frame_stack(self):
        return self._env.frame_stack

    @property
    def frame_size(self):
        return 5 + self.n_beams

    @property
    def max_len(self):
        return self._env._max_episode_steps

    @property
    def environment(self):
        return self._env.environment

    @property
    def maps(self):
        return self._env.maps

    @property
    def valTasks(self):
        return self._env.valTasks

    def seed(self, seed):
        np.random.seed(seed)

    def success_fn(self, reward):
        return reward >= 0

    def _current_obstacle_map(self):
        map_key = getattr(self._env, "map_key", self.val_map_key)
        return self._env.maps.get(map_key, [])

    def _point_unsafe(self, x, y):
        safe_eps = self._env.environment.agent.dynamic_model.safe_eps
        for obstacle in self._current_obstacle_map():
            ox, oy, _, ow, ol = obstacle
            if abs(x - ox) <= ol + safe_eps and abs(y - oy) <= ow + safe_eps:
                return True
        return False

    def cost_func(self, state, hazard_poses=None):
        if isinstance(state, torch.Tensor):
            batch_size = state.shape[0]
            device = state.device
            cost = torch.zeros(batch_size, dtype=torch.float, device=device)
            for i in range(batch_size):
                cost[i] = 1.0 if self._point_unsafe(state[i, 0].item(), state[i, 1].item()) else 0.0
            return cost

        if len(state.shape) == 1:
            return 1.0 if self._point_unsafe(state[0], state[1]) else 0.0

        costs = []
        for i in range(state.shape[0]):
            costs.append(1.0 if self._point_unsafe(state[i, 0], state[i, 1]) else 0.0)
        return np.array(costs, dtype=np.float32)

    def _agent_xy(self):
        agent = self._env.environment.agent.current_state
        return np.array([agent.x, agent.y], dtype=np.float32)

    def _goal_xy(self):
        goal = self._env.goal
        return np.array([goal.x, goal.y], dtype=np.float32)

    def _wrap_obs(self, raw_obs, info=None):
        if info is not None and "agent_state" in info:
            achieved_goal = np.array(info["agent_state"][:2], dtype=np.float32)
            desired_goal = np.array(info["goal_state"][:2], dtype=np.float32)
        else:
            achieved_goal = self._agent_xy()
            desired_goal = self._goal_xy()
        return {
            "observation": raw_obs["observation"],
            "desired_goal": desired_goal,
            "achieved_goal": achieved_goal,
        }

    def reset(self, eval_idx=None, **kwargs):
        if self.evaluate and eval_idx is not None:
            val_tasks = self._env.valTasks[self.val_map_key]
            task_id = eval_idx % len(val_tasks)
            raw_obs = self._env.reset(id=task_id, val_key=self.val_map_key, **kwargs)
        else:
            raw_obs = self._env.reset(**kwargs)
        return self._wrap_obs(raw_obs)

    def step(self, action):
        raw_obs, reward, done, info = self._env.step(action)
        if "Collision" in info:
            info["cost"] = info.get("cost", 0.0) + 100.0
        info["safety_cost"] = info.get("cost", 0.0)
        info["is_success"] = float(info.get("goal_achieved", False))
        return self._wrap_obs(raw_obs, info), reward, done, info

    def render(self, mode="rgb_array", **kwargs):
        if mode == "rgb_array":
            return self._env.render(mode="human", save_image=True)
        return self._env.render(mode=mode, **kwargs)
