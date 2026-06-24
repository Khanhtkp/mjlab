"""LimX Dynamics HUD03 velocity environment configurations."""

from mjlab.asset_zoo.robots import (
  HUD03_ACTION_SCALE,
  get_hud03_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def limx_hud03_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the HUD03 rough-terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  # HUD03 has more collision primitives and roughly 1.5x G1's mass.
  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 750
  cfg.sim.nconmax = 100
  cfg.sim.njmax = 2500

  cfg.scene.entities = {"robot": get_hud03_robot_cfg()}

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "base_link"

  site_names = ("left_foot", "right_foot")
  geom_names = ("left_foot_collision", "right_foot_collision")

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=site_name, entity="robot") for site_name in site_names
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.04, num_samples=8)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = HUD03_ACTION_SCALE

  cfg.viewer.body_name = "waist_pitch_link"
  cfg.viewer.distance = 3.5
  cfg.viewer.elevation = -5.0

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.25

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("waist_pitch_link",)

  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    r".*_hip_pitch_joint": 0.3,
    r".*_(hip_roll|hip_yaw)_joint": 0.14,
    r".*_knee_joint": 0.38,
    r".*_ankle_pitch_joint": 0.25,
    r".*_ankle_roll_joint": 0.1,
    r"waist_yaw_joint": 0.18,
    r"waist_roll_joint": 0.07,
    r"waist_pitch_joint": 0.1,
    r"head_(yaw|pitch)_joint": 0.06,
    r".*_shoulder_pitch_joint": 0.25,
    r".*_shoulder_roll_joint": 0.16,
    r".*_shoulder_yaw_joint": 0.12,
    r".*_elbow_joint": 0.22,
    r".*_(wrist_yaw|wrist_pitch|hand_yaw)_joint": 0.25,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*_hip_pitch_joint": 0.5,
    r".*_(hip_roll|hip_yaw)_joint": 0.2,
    r".*_knee_joint": 0.62,
    r".*_ankle_pitch_joint": 0.36,
    r".*_ankle_roll_joint": 0.15,
    r"waist_yaw_joint": 0.28,
    r"waist_roll_joint": 0.09,
    r"waist_pitch_joint": 0.16,
    r"head_(yaw|pitch)_joint": 0.08,
    r".*_shoulder_pitch_joint": 0.5,
    r".*_shoulder_roll_joint": 0.25,
    r".*_shoulder_yaw_joint": 0.18,
    r".*_elbow_joint": 0.4,
    r".*_(wrist_yaw|wrist_pitch|hand_yaw)_joint": 0.3,
  }

  torso_cfg = cfg.rewards["upright"].params["asset_cfg"]
  torso_cfg.body_names = ("waist_pitch_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("waist_pitch_link",)

  for reward_name in ("foot_clearance", "foot_slip"):
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["foot_clearance"].params["target_height"] = 0.12
  cfg.rewards["foot_swing_height"].params["target_height"] = 0.12
  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.015
  cfg.rewards["air_time"].weight = 0.0
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 15.0},
  )

  # Begin below the final command envelope, then smoothly reach the same terminal
  # range used by G1 once basic balance and stepping have converged.
  cfg.curriculum["command_vel"].params["velocity_stages"] = [
    {
      "step": 0,
      "lin_vel_x": (-0.8, 1.0),
      "lin_vel_y": (-0.6, 0.6),
      "ang_vel_z": (-0.4, 0.4),
    },
    {
      "step": 5000 * 24,
      "lin_vel_x": (-1.2, 1.8),
      "lin_vel_y": (-0.8, 0.8),
      "ang_vel_z": (-0.6, 0.6),
    },
    {
      "step": 10_000 * 24,
      "lin_vel_x": (-1.8, 2.5),
      "lin_vel_y": (-1.0, 1.0),
      "ang_vel_z": (-0.7, 0.7),
    },
    {
      "step": 15_000 * 24,
      "lin_vel_x": (-2.0, 3.0),
      "lin_vel_y": (-1.0, 1.0),
      "ang_vel_z": (-0.7, 0.7),
    },
  ]

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def limx_hud03_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the HUD03 flat-terrain velocity configuration."""
  cfg = limx_hud03_rough_env_cfg(play=play)

  cfg.sim.njmax = 400
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 96
  cfg.sim.nconmax = None

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  cfg.scene.sensors = tuple(
    sensor for sensor in (cfg.scene.sensors or ()) if sensor.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg
