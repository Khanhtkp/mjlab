"""HU_D03 flat-terrain velocity environment configuration."""

from mjlab.asset_zoo.robots import (
  HU_D03_ACTION_SCALE,
  get_hu_d03_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import (
  BuiltinSensorCfg,
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


def hu_d03_rough_velocity_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create LimX HU_D03 rough-terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 128
  cfg.sim.njmax = 1500

  cfg.scene.entities = {"robot": get_hu_d03_robot_cfg()}

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "waist_pitch_link"

  foot_site_names = ("left_foot", "right_foot")
  foot_geom_names = ("left_foot", "right_foot")

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in foot_site_names
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
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
  root_angmom_cfg = BuiltinSensorCfg(
    name="root_angmom",
    sensor_type="subtreeangmom",
    obj=ObjRef(type="body", name="base_link", entity="robot"),
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
    imu_lin_vel_cfg,
    imu_ang_vel_cfg,
    root_angmom_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = HU_D03_ACTION_SCALE

  cfg.viewer.body_name = "waist_pitch_link"
  cfg.viewer.distance = 3.0
  cfg.viewer.elevation = -5.0

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.25

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("waist_pitch_link",)

  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    r".*hip_pitch.*": 0.3,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.35,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    r".*waist_yaw.*": 0.2,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.1,
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.15,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.15,
    r".*wrist.*": 0.3,
    r".*hand.*": 0.3,
    r".*head.*": 0.2,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.2,
    r".*hip_yaw.*": 0.2,
    r".*knee.*": 0.6,
    r".*ankle_pitch.*": 0.35,
    r".*ankle_roll.*": 0.15,
    r".*waist_yaw.*": 0.3,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.2,
    r".*shoulder_pitch.*": 0.5,
    r".*shoulder_roll.*": 0.2,
    r".*shoulder_yaw.*": 0.15,
    r".*elbow.*": 0.35,
    r".*wrist.*": 0.3,
    r".*hand.*": 0.3,
    r".*head.*": 0.2,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("waist_pitch_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("waist_pitch_link",)

  for reward_name in ("foot_clearance", "foot_slip"):
    cfg.rewards[reward_name].params["asset_cfg"].site_names = foot_site_names

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["air_time"].weight = 0.0

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

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


def hu_d03_flat_velocity_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create LimX HU_D03 flat-terrain velocity configuration."""
  cfg = hu_d03_rough_velocity_env_cfg(play=play)

  cfg.sim.njmax = 512
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.curriculum.pop("terrain_levels", None)

  # Keep the G1 flat command distribution/curriculum intact, but bias HU_D03 away
  # from the stable "standing shuffle" local optimum seen with this robot model.
  cfg.rewards["track_linear_velocity"].weight = 3.0
  cfg.rewards["track_angular_velocity"].weight = 1.5
  cfg.rewards["pose"].weight = 0.7
  cfg.rewards["action_rate_l2"].weight = -0.06
  cfg.rewards["air_time"].weight = 0.35
  cfg.rewards["air_time"].params["command_threshold"] = 0.15
  cfg.rewards["foot_slip"].weight = -0.25
  cfg.rewards["soft_landing"].weight = -5.0e-5

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg


def hu_d03_flat_forward_velocity_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create an easier forward-only HU_D03 velocity configuration."""
  cfg = hu_d03_flat_velocity_env_cfg(play=play)

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.heading_command = False
  twist_cmd.ranges.heading = None
  twist_cmd.rel_standing_envs = 0.0
  twist_cmd.rel_heading_envs = 0.0
  twist_cmd.rel_forward_envs = 1.0
  twist_cmd.ranges.lin_vel_x = (0.2, 0.6)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (0.0, 0.0)

  cfg.curriculum.pop("command_vel", None)

  return cfg


def hu_d03_flat_forward_tuned_velocity_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a forward-only HU_D03 config with gentler locomotion rewards."""
  cfg = hu_d03_flat_forward_velocity_env_cfg(play=play)

  cfg.rewards["track_linear_velocity"].weight = 4.0
  cfg.rewards["pose"].weight = 0.5
  cfg.rewards["action_rate_l2"].weight = -0.05
  cfg.rewards["body_ang_vel"].weight = -0.03
  cfg.rewards["angular_momentum"].weight = -0.01

  cfg.rewards["pose"].params["std_walking"] |= {
    r".*hip_pitch.*": 0.45,
    r".*hip_roll.*": 0.2,
    r".*knee.*": 0.55,
    r".*ankle_pitch.*": 0.35,
    r".*ankle_roll.*": 0.15,
  }
  cfg.rewards["pose"].params["std_running"] |= {
    r".*hip_pitch.*": 0.65,
    r".*knee.*": 0.75,
    r".*ankle_pitch.*": 0.45,
  }

  cfg.rewards["foot_clearance"].params["target_height"] = 0.05
  cfg.rewards["foot_swing_height"].params["target_height"] = 0.05

  return cfg


def hu_d03_flat_forward_stable_velocity_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a smoother forward-only HU_D03 config for post-warmup tuning."""
  cfg = hu_d03_flat_forward_tuned_velocity_env_cfg(play=play)

  cfg.rewards["track_linear_velocity"].weight = 3.5
  cfg.rewards["upright"].weight = 1.2
  cfg.rewards["action_rate_l2"].weight = -0.08
  cfg.rewards["foot_slip"].weight = -0.15
  cfg.rewards["soft_landing"].weight = -3.0e-5

  return cfg


def hu_d03_flat_self_organized_velocity_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a G1-like HU_D03 task without staged command scheduling."""
  cfg = hu_d03_flat_velocity_env_cfg(play=play)

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.heading_command = False
  twist_cmd.ranges.heading = None
  twist_cmd.rel_standing_envs = 0.35
  twist_cmd.rel_heading_envs = 0.0
  twist_cmd.rel_forward_envs = 0.35
  twist_cmd.rel_world_envs = 0.0
  twist_cmd.ranges.lin_vel_x = (-0.3, 0.8)
  twist_cmd.ranges.lin_vel_y = (-0.2, 0.2)
  twist_cmd.ranges.ang_vel_z = (-0.3, 0.3)

  cfg.curriculum.pop("command_vel", None)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = {
    name: 0.75 * scale for name, scale in HU_D03_ACTION_SCALE.items()
  }

  cfg.rewards["alive"] = RewardTermCfg(func=mdp.alive, weight=0.25, params={})
  cfg.rewards["track_linear_velocity"].weight = 2.8
  cfg.rewards["track_angular_velocity"].weight = 1.5
  cfg.rewards["upright"].weight = 1.4
  cfg.rewards["pose"].weight = 0.7
  cfg.rewards["action_rate_l2"].weight = -0.12
  cfg.rewards["body_ang_vel"].weight = -0.04
  cfg.rewards["angular_momentum"].weight = -0.015
  cfg.rewards["foot_slip"].weight = -0.18
  cfg.rewards["soft_landing"].weight = -5.0e-5

  cfg.rewards["foot_clearance"].params["target_height"] = 0.05
  cfg.rewards["foot_swing_height"].params["target_height"] = 0.05
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.08}
  cfg.rewards["pose"].params["std_walking"] |= {
    r".*hip_pitch.*": 0.4,
    r".*hip_roll.*": 0.2,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.35,
    r".*ankle_roll.*": 0.15,
  }

  return cfg


def hu_d03_flat_survival_first_velocity_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a no-stage HU_D03 task that prioritizes episode length first."""
  cfg = hu_d03_flat_self_organized_velocity_env_cfg(play=play)

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.rel_standing_envs = 0.5
  twist_cmd.rel_forward_envs = 0.25
  twist_cmd.ranges.lin_vel_x = (-0.15, 0.35)
  twist_cmd.ranges.lin_vel_y = (-0.05, 0.05)
  twist_cmd.ranges.ang_vel_z = (-0.15, 0.15)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = {
    name: 0.6 * scale for name, scale in HU_D03_ACTION_SCALE.items()
  }

  cfg.rewards["alive"].weight = 2.0
  cfg.rewards["track_linear_velocity"].weight = 1.6
  cfg.rewards["track_angular_velocity"].weight = 0.8
  cfg.rewards["upright"].weight = 2.5
  cfg.rewards["pose"].weight = 1.0
  cfg.rewards["action_rate_l2"].weight = -0.2
  cfg.rewards["body_ang_vel"].weight = -0.08
  cfg.rewards["angular_momentum"].weight = -0.03
  cfg.rewards["foot_slip"].weight = -0.25
  cfg.rewards["soft_landing"].weight = -1.0e-4
  cfg.rewards["dof_pos_limits"].weight = -2.0

  if "push_robot" in cfg.events:
    cfg.events["push_robot"].interval_range_s = (3.0, 6.0)
    cfg.events["push_robot"].params["velocity_range"] = {
      "x": (-0.2, 0.2),
      "y": (-0.2, 0.2),
      "z": (-0.15, 0.15),
      "roll": (-0.2, 0.2),
      "pitch": (-0.2, 0.2),
      "yaw": (-0.3, 0.3),
    }

  cfg.events["encoder_bias"].params["bias_range"] = (-0.008, 0.008)
  cfg.events["base_com"].params["ranges"] = {
    0: (-0.012, 0.012),
    1: (-0.012, 0.012),
    2: (-0.015, 0.015),
  }

  return cfg


def hu_d03_flat_forward_survival_velocity_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a survival-first HU_D03 curriculum before forward tracking."""
  cfg = hu_d03_flat_forward_stable_velocity_env_cfg(play=play)

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.rel_standing_envs = 0.8
  twist_cmd.rel_forward_envs = 0.0
  twist_cmd.ranges.lin_vel_x = (0.0, 0.15)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (0.0, 0.0)

  cfg.rewards["alive"] = RewardTermCfg(func=mdp.alive, weight=0.5, params={})
  cfg.rewards["track_linear_velocity"].weight = 3.0
  cfg.rewards["upright"].weight = 1.5
  cfg.rewards["pose"].weight = 0.7
  cfg.rewards["action_rate_l2"].weight = -0.1
  cfg.rewards["foot_slip"].weight = -0.2
  cfg.rewards["soft_landing"].weight = -5.0e-5

  if not play:
    cfg.curriculum["command_vel"] = CurriculumTermCfg(
      func=mdp.adaptive_commands_vel,
      params={
        "command_name": "twist",
        "ema_alpha": 0.08,
        "min_updates": 10,
        "velocity_stages": [
          {
            "step": 0,
            "lin_vel_x": (0.0, 0.15),
            "lin_vel_y": (0.0, 0.0),
            "ang_vel_z": (0.0, 0.0),
            "rel_standing_envs": 0.8,
            "rel_forward_envs": 0.0,
            "advance_episode_length": 850.0,
            "advance_timeout_rate": 0.6,
          },
          {
            "step": 0,
            "lin_vel_x": (0.05, 0.3),
            "rel_standing_envs": 0.5,
            "advance_episode_length": 800.0,
            "advance_timeout_rate": 0.5,
          },
          {
            "step": 0,
            "lin_vel_x": (0.1, 0.45),
            "rel_standing_envs": 0.2,
            "advance_episode_length": 700.0,
            "advance_timeout_rate": 0.35,
          },
          {
            "step": 0,
            "lin_vel_x": (0.2, 0.6),
            "rel_standing_envs": 0.0,
          },
        ],
      },
    )

  return cfg
