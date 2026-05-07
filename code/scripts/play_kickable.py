# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a trained RSL-RL policy with keyboard-driven kicks.

This is a fork of `IsaacLab/scripts/reinforcement_learning/rsl_rl/play.py`
that swaps the (broken-in-Isaac-Sim-5.x) mouse-click hook for a keyboard
hook. While the GUI is in focus you can shove the robot in any cardinal
direction to test policy robustness.

Key bindings (I/J/K/L cluster — chosen because Isaac Sim's viewport already
uses WASD for camera fly and arrow keys for navigation):

    I = kick robot FORWARD   (+X in robot frame)
    K = kick robot BACKWARD  (-X)
    J = kick robot LEFT      (+Y)
    L = kick robot RIGHT     (-Y)
    U = upward boost         (+Z) — useful for sanity-checking landing recovery
    R = manually reset env 0
    P = print current root state of env 0 (pos, vel) for debugging

Run with:
    cd C:\\Users\\jaha\\robotic\\IsaacLab
    C:\\Users\\jaha\\env_isaaclab\\Scripts\\python.exe ^
      C:\\Users\\jaha\\robotic\\quadruped_zhawild_v1\\script\\play_kickable.py ^
      --task=Isaac-Velocity-Flat-ZHAWild-Play-v0 --num_envs=1 --real-time ^
      --checkpoint logs\\rsl_rl\\zhawild_flat\\2026-04-26_17-46-09\\model_1500.pt ^
      --kick_speed 2.0
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from pathlib import Path

# The official play.py lives in IsaacLab/scripts/reinforcement_learning/rsl_rl/.
# We need its `cli_args` and `rsl_rl_patches` modules. Add that dir to sys.path
# so imports work regardless of where this script is launched from.
_ISAACLAB_RSL_RL_DIR = Path("C:/Users/jaha/robotic/IsaacLab/scripts/reinforcement_learning/rsl_rl")
sys.path.insert(0, str(_ISAACLAB_RSL_RL_DIR))

from isaaclab.app import AppLauncher

import cli_args  # isort: skip  # noqa: E402

# add argparse arguments
parser = argparse.ArgumentParser(description="Play RSL-RL policy with keyboard-driven kicks.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playback.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--kick_speed",
    type=float,
    default=2.0,
    help="Linear velocity impulse in m/s applied per keypress.",
)
parser.add_argument(
    "--kick_env_id",
    type=int,
    default=0,
    help="Index of the environment whose robot to kick. Defaults to 0.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl_patches import apply_actor_critic_std_safety_patch

apply_actor_critic_std_safety_patch()


class KickArrowVisualizer:
    """Spawns a fading arrow at the robot's body whenever a kick fires.

    Implementation notes:
    - The marker is a single cone with axis="X", so the cone's tip naturally
      points along its local +X axis. We rotate that local +X to the kick's
      world-direction by computing an axis-angle quaternion. This avoids
      needing a separate cylinder-shaft + cone-tip composite arrow.
    - "Fade" is implemented by shrinking the marker scale linearly over
      `fade_steps` physics steps. Below a small threshold the marker is hidden
      so it doesn't leave a visible dot at the robot.
    - Uses Isaac Lab's `VisualizationMarkers` (the same machinery the
      velocity-command debug arrows use), so it plays nicely with the GPU
      pipeline and doesn't conflict with the trained policy.
    """

    def __init__(self, fade_steps: int = 25, length: float = 0.5):
        cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/KickArrow",
            markers={
                "arrow": sim_utils.ConeCfg(
                    radius=0.07,
                    height=length,
                    axis="X",  # cone tip points in +X by default → easy to rotate to any direction
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.25, 0.0),
                        # Slight emissive so the arrow stays visible against any background.
                        emissive_color=(0.6, 0.15, 0.0),
                    ),
                )
            },
        )
        self._markers = VisualizationMarkers(cfg)
        self._markers.set_visibility(False)

        self._fade_steps = fade_steps
        self._frames_left = 0
        self._max_length = length
        self._anchor_pos: torch.Tensor | None = None
        self._orientation: torch.Tensor | None = None
        self._device: torch.device | str = "cpu"

    def fire(self, anchor_pos: torch.Tensor, world_direction: torch.Tensor):
        """Show an arrow at `anchor_pos` pointing along `world_direction`.

        `anchor_pos` is the robot's root position in world coordinates (3,).
        `world_direction` is a non-unit vector in world coordinates (3,) —
        we normalize it internally.
        """
        direction = world_direction[:3].clone().to(torch.float32)
        norm = torch.linalg.norm(direction)
        if norm < 1.0e-6:
            return  # zero kick, nothing meaningful to show

        direction = direction / norm
        self._anchor_pos = anchor_pos[:3].clone().to(torch.float32)
        self._orientation = self._direction_to_x_quaternion(direction)
        self._frames_left = self._fade_steps
        self._device = anchor_pos.device

    def update(self):
        """Call once per env.step() to advance fade state and update the marker."""
        if self._frames_left <= 0:
            self._markers.set_visibility(False)
            return

        # Linear fade: scale 1.0 at fire-time → 0 at end-of-life.
        fade_t = self._frames_left / self._fade_steps
        # Hide once we get close to zero so we don't render a smushed dot.
        if fade_t < 0.05:
            self._markers.set_visibility(False)
            self._frames_left = 0
            return

        # Lift the arrow slightly above the robot body so it's not buried inside the mesh.
        translation = self._anchor_pos.clone().to(self._device)
        translation[2] = translation[2] + 0.15

        translations = translation.unsqueeze(0)
        orientations = self._orientation.to(self._device).unsqueeze(0)
        scales = torch.tensor([[fade_t, fade_t, fade_t]], device=self._device, dtype=torch.float32)

        self._markers.visualize(
            translations=translations,
            orientations=orientations,
            scales=scales,
        )
        self._markers.set_visibility(True)
        self._frames_left -= 1

    @staticmethod
    def _direction_to_x_quaternion(direction: torch.Tensor) -> torch.Tensor:
        """Quaternion (w, x, y, z) that rotates the +X axis onto `direction`.

        We use the standard axis-angle construction:
          axis  = normalize(cross(+X, direction))
          angle = arccos(dot(+X, direction))
        with two degenerate fallbacks: parallel (no rotation needed) and
        antiparallel (rotate 180° around an arbitrary perpendicular axis).
        """
        device = direction.device
        x_axis = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=torch.float32)
        cross = torch.linalg.cross(x_axis, direction, dim=0)
        dot = torch.dot(x_axis, direction)
        cross_norm = torch.linalg.norm(cross)

        if cross_norm < 1e-6:
            if dot > 0:
                return torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float32)
            # Antiparallel — rotate 180° around the +Y axis so the arrow flips along X.
            return torch.tensor([0.0, 0.0, 1.0, 0.0], device=device, dtype=torch.float32)

        axis = cross / cross_norm
        angle = torch.acos(torch.clamp(dot, -1.0, 1.0))
        half = angle / 2.0
        sin_half = torch.sin(half)
        return torch.tensor(
            [
                torch.cos(half).item(),
                (axis[0] * sin_half).item(),
                (axis[1] * sin_half).item(),
                (axis[2] * sin_half).item(),
            ],
            device=device,
            dtype=torch.float32,
        )


class KeyboardKickController:
    """Listens for I/J/K/L (and friends) and queues velocity impulses on the robot.

    Why keyboard instead of mouse? In Isaac Sim 5.x the viewport-mouse-event
    API is wrapped in many layers and forwarding clicks past the selection
    handler is fragile. Carb's keyboard input subscription, on the other hand,
    is rock-solid: it fires regardless of which UI panel is focused.
    """

    # Map of carb keyboard input names -> body-frame impulse direction
    # (x_forward, y_left, z_up) in m/s scaled by `speed`.
    _KICK_MAP = {
        "I": (+1.0,  0.0,  0.0),   # forward
        "K": (-1.0,  0.0,  0.0),   # backward
        "J": ( 0.0, +1.0,  0.0),   # left
        "L": ( 0.0, -1.0,  0.0),   # right
        "U": ( 0.0,  0.0, +1.0),   # up boost
    }

    def __init__(self, env, speed: float, env_id: int = 0, visualizer: "KickArrowVisualizer | None" = None):
        self.env = env
        self.speed = speed
        self.device = env.device
        self.env_id = max(0, min(env_id, env.num_envs - 1))
        self.visualizer = visualizer
        self._pending_impulses: list[tuple[float, float, float]] = []
        self._reset_pending = False
        self._print_state_pending = False
        self._sub_keyboard = None
        self._carb = None

        try:
            import carb
            import omni.appwindow

            self._carb = carb
            self._input = carb.input.acquire_input_interface()
            self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
            self._sub_keyboard = self._input.subscribe_to_keyboard_events(
                self._keyboard, self._on_keyboard_event
            )
            print(
                "[INFO] Keyboard kick enabled.\n"
                f"       I/K = forward/back, J/L = left/right, U = up boost (speed={speed} m/s)\n"
                "       R = reset env 0, P = print env 0 root state"
            )
        except Exception as exc:
            print(f"[WARN] Keyboard kick setup failed: {exc}")

    def _on_keyboard_event(self, event):
        if self._carb is None:
            return False
        # Only react to key DOWN edges (ignore key-hold repeats and key-up).
        if event.type != self._carb.input.KeyboardEventType.KEY_PRESS:
            return True

        key_name = getattr(getattr(event, "input", None), "name", "").upper()
        if key_name in self._KICK_MAP:
            self._pending_impulses.append(self._KICK_MAP[key_name])
        elif key_name == "R":
            self._reset_pending = True
        elif key_name == "P":
            self._print_state_pending = True
        # Returning True keeps the event flowing to other listeners (camera fly etc.).
        return True

    def apply_pending(self):
        """Called once per env.step() from the main loop. Drains the input queue."""
        robot = self.env.scene["robot"]
        env_ids = torch.tensor([self.env_id], device=self.device, dtype=torch.long)

        if self._pending_impulses:
            # Sum all impulses queued since last step so a fast double-tap still works.
            total = torch.zeros(3, device=self.device)
            for dx, dy, dz in self._pending_impulses:
                total += torch.tensor([dx, dy, dz], device=self.device)
            self._pending_impulses.clear()

            # Add to current root linear velocity rather than overwriting — keeps the
            # robot's existing motion state and just nudges it.
            root_vel = robot.data.root_vel_w[self.env_id : self.env_id + 1].clone()
            root_vel[:, 0:3] += total[None, :] * self.speed
            robot.write_root_velocity_to_sim(root_vel, env_ids=env_ids)
            print(
                f"[KICK] env {self.env_id}: impulse=({total[0].item():+.1f},"
                f" {total[1].item():+.1f}, {total[2].item():+.1f}) × {self.speed:.1f} m/s"
            )

            # Spawn the fading visual arrow at the robot's current root position.
            if self.visualizer is not None:
                anchor = robot.data.root_pos_w[self.env_id].clone()
                self.visualizer.fire(anchor_pos=anchor, world_direction=total)

        # Always advance the visualizer — even on frames without a new kick — so an
        # in-flight arrow continues fading instead of freezing on screen.
        if self.visualizer is not None:
            self.visualizer.update()

        if self._reset_pending:
            self._reset_pending = False
            # Hard reset of one env: snap root pose + joints back to defaults.
            root_state = robot.data.default_root_state[self.env_id : self.env_id + 1].clone()
            # Add the env origin so the robot doesn't teleport to world origin in multi-env.
            root_state[:, 0:3] += self.env.scene.env_origins[self.env_id : self.env_id + 1]
            robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
            robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)
            joint_pos = robot.data.default_joint_pos[self.env_id : self.env_id + 1].clone()
            joint_vel = robot.data.default_joint_vel[self.env_id : self.env_id + 1].clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
            print(f"[RESET] env {self.env_id} snapped back to default pose.")

        if self._print_state_pending:
            self._print_state_pending = False
            pos = robot.data.root_pos_w[self.env_id].cpu().numpy()
            vel = robot.data.root_lin_vel_w[self.env_id].cpu().numpy()
            print(
                f"[STATE] env {self.env_id}: "
                f"pos=({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f}) m, "
                f"lin_vel=({vel[0]:+.2f}, {vel[1]:+.2f}, {vel[2]:+.2f}) m/s"
            )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent + keyboard kicks."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] No pre-trained checkpoint available for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during playback.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    # The visual marker must be created after the scene is built (i.e. after gym.make),
    # because VisualizationMarkers attaches itself to the live USD stage.
    kick_visualizer = KickArrowVisualizer(fade_steps=25, length=0.5)
    kicker = KeyboardKickController(
        env.unwrapped,
        args_cli.kick_speed,
        env_id=args_cli.kick_env_id,
        visualizer=kick_visualizer,
    )

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    obs = env.get_observations()
    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)
            kicker.apply_pending()
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
