"""HU_D03 flat-terrain motion tracking environment configuration."""

from mjlab.asset_zoo.robots import (
  HU_D03_ACTION_SCALE,
  get_hu_d03_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import BuiltinSensorCfg, ContactMatch, ContactSensorCfg, ObjRef
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg


def hu_d03_flat_tracking_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_hu_d03_robot_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  imu_lin_vel_cfg = BuiltinSensorCfg(
    name="imu_lin_vel",
    sensor_type="velocimeter",
    obj=ObjRef(type="site", name="imu", entity="robot"),
  )
  imu_ang_vel_cfg = BuiltinSensorCfg(
    name="imu_ang_vel",
    sensor_type="gyro",
    obj=ObjRef(type="site", name="imu", entity="robot"),
  )
  cfg.scene.sensors = (self_collision_cfg, imu_lin_vel_cfg, imu_ang_vel_cfg)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = HU_D03_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "waist_pitch_link"
  motion_cmd.body_names = (
    "base_link",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "waist_pitch_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
  )

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
    "left_foot",
    "right_foot",
  )
  cfg.events["base_com"].params["asset_cfg"].body_names = ("waist_pitch_link",)

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )

  cfg.viewer.body_name = "waist_pitch_link"
  cfg.sim.nconmax = 64
  cfg.sim.njmax = 512

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.sampling_mode = "start"

  return cfg
