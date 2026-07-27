import json
import os

from goal_polamp_env.env import GCPOLAMPEnvironment
from polamp_env.lib.utils_operations import generateDataSet
from polamp_wrapper.env import SafePolampEnv


def _repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_json(relative_path):
    path = os.path.join(_repo_root(), relative_path)
    with open(path, "r") as f:
        return json.load(f)


def create_polamp_env(args):
    root = _repo_root()
    goal_our_env_config = _load_json("goal_polamp_env/goal_environment_configs.json")
    our_env_config = _load_json("polamp_env/configs/environment_configs.json")
    reward_config = _load_json("polamp_env/configs/reward_weight_configs.json")
    car_config = _load_json("polamp_env/configs/car_configs.json")

    dataset = args.polamp_dataset
    if dataset == "medium_dataset":
        total_maps = 12
    elif dataset == "test_medium_dataset":
        total_maps = 3
    elif dataset == "hard_dataset_simplified_test":
        total_maps = 2
    else:
        total_maps = 1

    dataset_path = os.path.join(root, dataset)
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"POLAMP dataset folder not found: {dataset_path}")

    data_set = generateDataSet(
        our_env_config,
        name_folder=dataset,
        total_maps=total_maps,
        dynamic=False,
    )
    maps, train_task, val_tasks = data_set["obstacles"]

    goal_our_env_config["dataset"] = dataset
    goal_our_env_config["uniform_feasible_train_dataset"] = args.polamp_uniform_feasible_train_dataset
    goal_our_env_config["random_train_dataset"] = args.polamp_random_train_dataset
    if not goal_our_env_config["static_env"]:
        maps["map0"] = []

    environment_config = {
        "vehicle_config": car_config,
        "tasks": train_task,
        "valTasks": val_tasks,
        "maps": maps,
        "our_env_config": our_env_config,
        "reward_config": reward_config,
        "evaluation": args.validate,
        "goal_our_env_config": goal_our_env_config,
    }

    base_env = GCPOLAMPEnvironment("polamp_env", environment_config)
    env = SafePolampEnv(base_env)
    env.val_map_key = list(maps.keys())[0]

    state_dim = env.observation_space["observation"].shape[0]
    goal_dim = env.observation_space["desired_goal"].shape[0]
    action_dim = env.action_space.shape[0]
    env.state_dim = state_dim

    return env, state_dim, goal_dim, action_dim, None
