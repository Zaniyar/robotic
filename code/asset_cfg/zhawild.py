# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ZHAW Wild quadruped."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


ZHAWILD_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path="C:/Users/jaha/robotic/quadruped_zhawild_v1/quadruped2.urdf",
        usd_dir="C:/Users/jaha/robotic/quadruped_zhawild_v1/converted_usd",
        usd_file_name="zhawild_converted.usd",
        activate_contact_sensors=True,
        fix_base=False,
        merge_fixed_joints=False,
        joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
            target_type="position",
            drive_type="force",
            gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(
                stiffness=80.0,
                damping=4.0,
            ),
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        rot=(0.7071068, 0.7071068, 0.0, 0.0),
        joint_pos={
            ".*_haa": 0.0,
            ".*_hfe": 0.5,
            ".*_kfe": -1.2,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "hip_abduction": ImplicitActuatorCfg(
            joint_names_expr=[".*_haa"],
            effort_limit_sim=80.0,
            velocity_limit_sim=10.0,
            stiffness=80.0,
            damping=4.0,
        ),
        "hip_flexion": ImplicitActuatorCfg(
            joint_names_expr=[".*_hfe"],
            effort_limit_sim=120.0,
            velocity_limit_sim=10.0,
            stiffness=80.0,
            damping=4.0,
        ),
        "knee": ImplicitActuatorCfg(
            joint_names_expr=[".*_kfe"],
            effort_limit_sim=120.0,
            velocity_limit_sim=10.0,
            stiffness=80.0,
            damping=4.0,
        ),
    },
)
"""Configuration for the ZHAW Wild quadruped."""
