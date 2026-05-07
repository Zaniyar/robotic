# Load the ZHAW Wild quadruped (zhawild_v1) and test physics + joints.
# Prefers the preconverted USD asset for stability and falls back to the URDF if needed.
#
# Run with:
#   cd C:/Users/jaha/robotic/IsaacLab
#   C:/Users/jaha/env_isaaclab/Scripts/python.exe C:/Users/jaha/robotic/quadruped_zhawild_v1/script/load_zhawild.py
# Quick headless check:
#   C:/Users/jaha/env_isaaclab/Scripts/python.exe C:/Users/jaha/robotic/quadruped_zhawild_v1/script/load_zhawild.py --headless --max-steps 50
#
# Robot overview (see quadruped2.urdf):
#   - 12 actuated joints: fl/fr/rl/rr × haa (hip-abduction), hfe (hip-flexion), kfe (knee-flexion)
#   - 4 fixed foot joints (no actuation — just rigid feet at the shank tip)
#   - Mass ~17.5 kg, legs ~0.66 m → spawn ~0.75 m off the ground so it doesn't clip.

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Load ZHAW Wild quadruped.")
parser.add_argument(
    "--max-steps",
    type=int,
    default=0,
    help="Stop automatically after this many simulation steps. Use 0 to run until the app closes.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg

# The script lives in .../quadruped_zhawild_v1/script/ — the asset files are one level up.
SCRIPT_DIR = Path(__file__).resolve().parent
ASSET_DIR = SCRIPT_DIR.parent
URDF_PATH = ASSET_DIR / "quadruped2.urdf"
USD_PATH = ASSET_DIR / "quadruped2" / "quadruped2.usd"


def _make_robot_spawn_cfg():
    common_kwargs = dict(
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    )

    if USD_PATH.is_file():
        return sim_utils.UsdFileCfg(usd_path=str(USD_PATH), **common_kwargs)

    if URDF_PATH.is_file():
        return sim_utils.UrdfFileCfg(
            asset_path=str(URDF_PATH),
            usd_dir=str(ASSET_DIR / "usd_generated"),
            fix_base=False,
            joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
                drive_type="force",
                target_type="position",
                gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=80.0,
                    damping=4.0,
                ),
            ),
            **common_kwargs,
        )

    raise FileNotFoundError(f"Neither USD nor URDF robot asset exists in: {ASSET_DIR}")


ROBOT_ASSET_PATH = USD_PATH if USD_PATH.is_file() else URDF_PATH


ZHAWILD_ROBOT_CFG = ArticulationCfg(
    spawn=_make_robot_spawn_cfg(),
    init_state=ArticulationCfg.InitialStateCfg(
        # spawn 75cm above ground — thigh (0.33m) + shank (0.33m) = 0.66m legs,
        # plus a bit of clearance so feet don't intersect on the first frame.
        pos=(0.0, 0.0, 0.75),
        # The URDF was exported with Y-up convention (base_link origin has y=0.52),
        # so in Isaac (Z-up) the robot lands on its side. Rotate +90° around X to
        # stand it up. Quaternion (w, x, y, z) for Rx(+90°) = (cos(+45°), sin(+45°), 0, 0).
        rot=(0.7071068, 0.7071068, 0.0, 0.0),
        joint_pos={
            # hip-abduction neutral (legs point straight down, not splayed)
            ".*_haa": 0.0,
            # hip-flexion forward ~0.5 rad so the thigh tilts slightly (a nominal standing pose)
            ".*_hfe": 0.5,
            # knee bent at -1.2 rad — within limits (-2.705 to -0.175), creates a natural stance
            ".*_kfe": -1.2,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # One group for all 12 joints. Stiffness/damping tuned conservatively for a 17.5 kg robot.
        # The URDF specifies effort 80 N·m for haa and 120 N·m for hfe/kfe — we pick 100 as a
        # safe middle value. These defaults are fine for a wiggle test; tune per-group later.
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=100.0,
            stiffness=80.0,
            damping=4.0,
        ),
    },
)


def design_scene():
    print("[INFO] Adding ground...", flush=True)
    # Use a local static collider instead of Isaac Sim's default remote ground USD.
    ground_cfg = sim_utils.CuboidCfg(
        size=(4.0, 4.0, 0.02),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.25, 0.25)),
    )
    ground_cfg.func("/World/Ground", ground_cfg, translation=(0.0, 0.0, -0.01))

    print("[INFO] Adding light...", flush=True)
    cfg_light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg_light.func("/World/Light", cfg_light)

    print("[INFO] Creating articulation...", flush=True)
    robot = Articulation(ZHAWILD_ROBOT_CFG.replace(prim_path="/World/Robot"))
    return robot


def run_simulator(sim, robot):
    sim_dt = sim.get_physics_dt()
    steps_since_reset = 0
    total_steps = 0

    while simulation_app.is_running():
        if args_cli.max_steps > 0 and total_steps >= args_cli.max_steps:
            print(f"[INFO] Reached max steps ({args_cli.max_steps}). Exiting.")
            break

        # Reset every 200 steps
        if steps_since_reset % 200 == 0:
            steps_since_reset = 0
            root_state = robot.data.default_root_state.clone()
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            print(f"[INFO] Reset. {robot.num_joints} joints: {robot.joint_names}")

        # Random small wiggle (RL will replace this line)
        joint_targets = robot.data.default_joint_pos + torch.randn_like(robot.data.joint_pos) * 0.1
        robot.set_joint_position_target(joint_targets)
        robot.write_data_to_sim()

        sim.step()
        steps_since_reset += 1
        total_steps += 1
        robot.update(sim_dt)


def main():
    if not ROBOT_ASSET_PATH.is_file():
        raise FileNotFoundError(f"Robot asset not found: {ROBOT_ASSET_PATH}")

    print(f"[INFO] Using robot asset: {ROBOT_ASSET_PATH}", flush=True)
    print("[INFO] Creating simulation context...", flush=True)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01))
    # Camera pulled further back than the GoodLegRobot — this bot is ~10x bigger.
    sim.set_camera_view(eye=[2.5, 2.5, 1.5], target=[0.0, 0.0, 0.5])

    print("[INFO] Spawning scene...", flush=True)
    robot = design_scene()
    print("[INFO] Resetting simulation...", flush=True)
    sim.reset()

    print(f"[INFO] Robot loaded: {robot.num_joints} joints", flush=True)
    print(f"[INFO] Joint names: {robot.joint_names}", flush=True)
    print("[INFO] Running physics with random wiggle...", flush=True)

    run_simulator(sim, robot)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
