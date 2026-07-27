import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np

from polamp_env.lib.structures import State


def _ensure_pil_antialias_compat():
    """TensorBoard < 2.14 uses Image.ANTIALIAS, removed in Pillow 10+."""
    try:
        from PIL import Image
        if not hasattr(Image, "ANTIALIAS"):
            resample = getattr(Image, "Resampling", Image)
            Image.ANTIALIAS = getattr(resample, "LANCZOS", Image.LANCZOS)
    except ImportError:
        pass


_ensure_pil_antialias_compat()


def _fig_to_rgb_array(fig):
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return image


def _get_polamp_bounds(base_env):
    return getattr(base_env, "dataset_info", None) or {
        "min_x": -5,
        "max_x": 40,
        "min_y": -5,
        "max_y": 36,
    }


def _draw_obstacles(ax, base_env, plot_obstacles=True):
    if not plot_obstacles:
        return
    polygon_env = base_env.environment
    if not hasattr(polygon_env, "obstacle_segments"):
        return

    if hasattr(polygon_env.agent, "safe_vehicle_array") and len(polygon_env.agent.safe_vehicle_array) > 0:
        expansion_distance = np.max(polygon_env.agent.safe_vehicle_array) / 4.0
    else:
        expansion_distance = polygon_env.agent.dynamic_model.safe_eps / 4.0

    for obstacle in polygon_env.obstacle_segments:
        corners = [
            [obstacle[0][0].x, obstacle[0][0].y],
            [obstacle[1][0].x, obstacle[1][0].y],
            [obstacle[2][0].x, obstacle[2][0].y],
            [obstacle[3][0].x, obstacle[3][0].y],
        ]
        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            ax.scatter(
                np.linspace(p1[0], p2[0], 200),
                np.linspace(p1[1], p2[1], 200),
                color="blue",
                s=1,
                zorder=2,
            )

        center_x = np.mean([c[0] for c in corners])
        center_y = np.mean([c[1] for c in corners])
        x_coords = [c[0] for c in corners]
        y_coords = [c[1] for c in corners]
        width = np.max(x_coords) - np.min(x_coords)
        height = np.max(y_coords) - np.min(y_coords)
        expanded_width = width + 2 * expansion_distance
        expanded_height = height + 2 * expansion_distance
        expanded_corners = [
            [center_x - expanded_width / 2, center_y - expanded_height / 2],
            [center_x + expanded_width / 2, center_y - expanded_height / 2],
            [center_x + expanded_width / 2, center_y + expanded_height / 2],
            [center_x - expanded_width / 2, center_y + expanded_height / 2],
        ]
        for i in range(4):
            p1 = expanded_corners[i]
            p2 = expanded_corners[(i + 1) % 4]
            ax.scatter(
                np.linspace(p1[0], p2[0], 200),
                np.linspace(p1[1], p2[1], 200),
                color="red",
                s=1,
                zorder=2,
            )


def _draw_agent(ax, base_env, x, y, theta, color="green", zorder=6, draw_bbox=True):
    if draw_bbox:
        agent_state = State([x, y, theta, 0.0, 0.0])
        center = base_env.environment.agent.dynamic_model.shift_state(agent_state)
        bbox = base_env.environment.getBB(center, ego=True)
        for i in range(4):
            ax.scatter(
                np.linspace(bbox[i][0].x, bbox[i][1].x, 200),
                np.linspace(bbox[i][0].y, bbox[i][1].y, 200),
                color=color,
                s=2,
                zorder=zorder,
            )
    ax.arrow(
        x,
        y,
        1.5 * np.cos(theta),
        1.5 * np.sin(theta),
        head_width=0.2,
        head_length=0.15,
        fc=color,
        ec=color,
        linewidth=2,
        zorder=zorder + 1,
        length_includes_head=True,
    )


def _draw_goal(ax, info):
    goal_x = info["goal_state"][0]
    goal_y = info["goal_state"][1]
    goal_theta = info["goal_state"][2]
    ax.scatter([goal_x], [goal_y], color="yellow", s=120, edgecolors="black", zorder=5, label="Goal")
    ax.arrow(
        goal_x,
        goal_y,
        1.5 * np.cos(goal_theta),
        1.5 * np.sin(goal_theta),
        head_width=0.2,
        head_length=0.15,
        fc="green",
        ec="green",
        linewidth=2,
        zorder=5,
        length_includes_head=True,
    )


def _draw_polamp_topdown(ax, base_env, bounds, trajectory_x, trajectory_y,
                         subgoals_x, subgoals_y, info, step_count=None, show_start=True):
    _draw_obstacles(ax, base_env, plot_obstacles=True)
    _draw_goal(ax, info)

    if len(trajectory_x) > 1:
        ax.scatter(trajectory_x, trajectory_y, color="orange", s=10, alpha=0.6, zorder=3, label="Trajectory")
    elif len(trajectory_x) == 1:
        ax.scatter(trajectory_x[0], trajectory_y[0], color="orange", s=20, alpha=0.8, zorder=3, label="Trajectory")

    if len(subgoals_x) > 0:
        ax.scatter(
            subgoals_x,
            subgoals_y,
            color="purple",
            s=80,
            alpha=0.85,
            marker="*",
            zorder=4,
            label="Subgoals",
        )

    if show_start and len(trajectory_x) > 0:
        start_theta = info.get("start_theta", info["agent_state"][2])
        _draw_agent(ax, base_env, trajectory_x[0], trajectory_y[0], start_theta, color="green", zorder=6)
    if len(trajectory_x) > 0:
        current_theta = info["agent_state"][2]
        _draw_agent(ax, base_env, trajectory_x[-1], trajectory_y[-1], current_theta, color="green", zorder=7)

    ax.set_xlim(bounds["min_x"], bounds["max_x"])
    ax.set_ylim(bounds["min_y"], bounds["max_y"])
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right", fontsize=8)
    title = "POLAMP validation (top-down)"
    if step_count is not None:
        title += f", t={step_count}"
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def render_polamp_topdown_frame(env, trajectory_x, trajectory_y, subgoals_x, subgoals_y, info, step_count=None):
    """SGSafe plot_full_env-style frame: bird's-eye map with agent, goal, and subgoals."""
    base_env = env._env
    bounds = _get_polamp_bounds(base_env)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    _draw_polamp_topdown(
        ax, base_env, bounds, trajectory_x, trajectory_y,
        subgoals_x, subgoals_y, info, step_count=step_count, show_start=False,
    )
    return _fig_to_rgb_array(fig)


def plot_polamp_trajectory(env, trajectory_x, trajectory_y, subgoals_x, subgoals_y, info):
    """SGSafe plot_trajectory-style static summary at episode end."""
    base_env = env._env
    bounds = _get_polamp_bounds(base_env)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    _draw_polamp_topdown(
        ax, base_env, bounds, trajectory_x, trajectory_y,
        subgoals_x, subgoals_y, info, step_count=None, show_start=True,
    )
    ax.set_title("POLAMP validation: trajectory and manager subgoals")
    return _fig_to_rgb_array(fig)


def log_polamp_eval_artifacts(writer, experiment_comet, trajectory_image, video_frames,
                              total_timesteps, eval_ep, env_render_frames=None):
    if trajectory_image is not None and writer is not None:
        _ensure_pil_antialias_compat()
        chw = trajectory_image.transpose(2, 0, 1)
        try:
            writer.add_image("eval/polamp_trajectory", chw, total_timesteps)
        except Exception as exc:
            print(f"Warning: failed to log eval/polamp_trajectory to TensorBoard: {exc}")

    if trajectory_image is not None and experiment_comet is not None:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            tmp_image_path = tmp_file.name
        try:
            plt.imsave(tmp_image_path, trajectory_image)
            experiment_comet.log_image(
                tmp_image_path,
                name=f"eval_polamp_trajectory_ep{eval_ep}",
                step=total_timesteps,
            )
        finally:
            if os.path.exists(tmp_image_path):
                os.unlink(tmp_image_path)

    if video_frames and len(video_frames) > 0:
        _log_polamp_video(
            writer, experiment_comet, video_frames, total_timesteps, eval_ep,
            tb_tag="eval/polamp_topdown_video",
            comet_name=f"eval_polamp_topdown_ep{eval_ep}",
        )

    if env_render_frames and len(env_render_frames) > 0:
        _log_polamp_video(
            writer, experiment_comet, env_render_frames, total_timesteps, eval_ep,
            tb_tag="eval/polamp_env_render_video",
            comet_name=f"eval_polamp_env_render_ep{eval_ep}",
        )


def _log_polamp_video(writer, experiment_comet, video_frames, total_timesteps, eval_ep,
                      tb_tag, comet_name):
    video = np.stack(video_frames, axis=0)
    if writer is not None:
        _ensure_pil_antialias_compat()
        try:
            writer.add_video(
                tb_tag,
                video.transpose(0, 3, 1, 2)[None, ...],
                total_timesteps,
                fps=4,
            )
        except Exception as exc:
            print(f"Warning: failed to log {tb_tag} to TensorBoard: {exc}")

    if experiment_comet is not None:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            tmp_video_path = tmp_file.name
        try:
            try:
                import imageio
                imageio.mimwrite(tmp_video_path, video, fps=4, codec="libx264")
            except Exception:
                from PIL import Image
                gif_path = tmp_video_path.replace(".mp4", ".gif")
                frames = [Image.fromarray(frame) for frame in video]
                frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=frames[1:],
                    duration=250,
                    loop=0,
                )
                tmp_video_path = gif_path
            experiment_comet.log_video(
                tmp_video_path,
                name=comet_name,
                step=total_timesteps,
            )
        finally:
            if os.path.exists(tmp_video_path):
                os.unlink(tmp_video_path)
