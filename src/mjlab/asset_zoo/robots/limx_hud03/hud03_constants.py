"""LimX Dynamics HUD03 constants and training articulation."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import reflected_inertia
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

HUD03_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "limx_hud03" / "xmls" / "hud03.xml"
)
assert HUD03_XML.exists()

_AUXILIARY_JOINT_NAMES = (
  "left_A_achilles_joint",
  "left_A_achilles_rod_joint",
  "left_B_achilles_joint",
  "left_B_achilles_rod_joint",
  "right_A_achilles_joint",
  "right_A_achilles_rod_joint",
  "right_B_achilles_joint",
  "right_B_achilles_rod_joint",
  "waist_A_joint",
  "waist_A_rod_joint",
  "waist_B_joint",
  "waist_B_rod_joint",
)


def get_spec() -> mujoco.MjSpec:
  """Load the HUD03 training model.

  The source model contains closed-chain Achilles and waist mechanisms with six
  auxiliary ball joints. MjLab articulation state, reset, and observation APIs use
  one scalar position per joint, so retaining those ball joints would silently
  misalign keyframes and joint state. For locomotion training we collapse the
  transmission links into rigid attachments and actuate the ankle and waist output
  joints directly. This is the same reduced-transmission abstraction used by the G1
  model, while preserving HUD03's body masses, inertias, visuals, and output joint
  kinematics.
  """
  spec = mujoco.MjSpec.from_file(str(HUD03_XML))
  # Simulation timing is configured by the task. Reset the source-model override so
  # MjSpec.attach does not emit a warning for a value it intentionally discards.
  spec.option.timestep = mujoco.MjSpec().option.timestep

  # Replace the torque motors with MjLab position actuators below.
  for actuator in list(spec.actuators):
    spec.delete(actuator)

  # The source joint sensors are redundant with EntityData and add substantial sensor
  # bandwidth. Re-create only the sensors used by the velocity task.
  for sensor in list(spec.sensors):
    spec.delete(sensor)

  # Remove the closed-chain constraints and their non-scalar auxiliary joints. The
  # corresponding bodies remain rigidly attached, preserving their mass and inertia.
  for equality in list(spec.equalities):
    spec.delete(equality)
  for joint_name in _AUXILIARY_JOINT_NAMES:
    spec.delete(spec.joint(joint_name))

  # Terrain, lighting, and cameras are owned by the scene rather than the robot.
  for geom in list(spec.geoms):
    if geom.parent.name == "world":
      spec.delete(geom)
  for camera in list(spec.cameras):
    spec.delete(camera)
  for light in list(spec.lights):
    spec.delete(light)

  # Give every physical collider a stable name for collision configuration,
  # randomization, and contact sensing.
  collision_counts: dict[str, int] = {}
  for geom in spec.geoms:
    if geom.contype == 0 and geom.conaffinity == 0:
      continue
    # MuJoCo Warp's MULTICCD path requires zero explicit geom margin.
    geom.margin = 0.0
    parent_name = geom.parent.name
    if geom.name in ("left_foot", "right_foot"):
      geom.name = f"{geom.name}_collision"
      continue
    if not geom.name:
      collision_counts[parent_name] = collision_counts.get(parent_name, 0) + 1
      geom.name = f"{parent_name}_collision_{collision_counts[parent_name]}"

  # Foot sites sit on the sole plane and are used for clearance/slip measurements.
  foot_site_pos = (0.018, 0.0, -0.0635)
  spec.body("left_ankle_roll_link").add_site(
    name="left_foot", pos=foot_site_pos, size=(0.01,)
  )
  spec.body("right_ankle_roll_link").add_site(
    name="right_foot", pos=foot_site_pos, size=(0.01,)
  )

  spec.add_sensor(
    name="imu_ang_vel",
    type=mujoco.mjtSensor.mjSENS_GYRO,
    objtype=mujoco.mjtObj.mjOBJ_SITE,
    objname="imu",
  )
  spec.add_sensor(
    name="imu_lin_vel",
    type=mujoco.mjtSensor.mjSENS_VELOCIMETER,
    objtype=mujoco.mjtObj.mjOBJ_SITE,
    objname="imu",
  )
  spec.add_sensor(
    name="root_angmom",
    type=mujoco.mjtSensor.mjSENS_SUBTREEANGMOM,
    objtype=mujoco.mjtObj.mjOBJ_BODY,
    objname="base_link",
  )
  return spec


##
# Actuator config.
##

# Rotor inertias and gear ratios from HU_D03_03.srdf.
LEG_ARMATURE = reflected_inertia(0.000244114, 25.0)
LINKAGE_ARMATURE = reflected_inertia(0.000073217, 36.0)
ARM_ARMATURE = reflected_inertia(0.000073217, 25.0)
SMALL_ARMATURE = reflected_inertia(0.000017, 25.0)

NATURAL_FREQ = 10.0 * 2.0 * 3.1415926535  # 10 Hz.
DAMPING_RATIO = 2.0


def _stiffness(armature: float) -> float:
  return armature * NATURAL_FREQ**2


def _damping(armature: float) -> float:
  return 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ


HUD03_LEG_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    r".*_hip_(pitch|roll|yaw)_joint",
    r".*_knee_joint",
  ),
  stiffness=_stiffness(LEG_ARMATURE),
  damping=_damping(LEG_ARMATURE),
  effort_limit=120.0,
  armature=LEG_ARMATURE,
)

# Each ankle output is driven by the two 45 Nm Achilles motors. As with G1's
# reduced parallel linkage, use their summed nominal inertia and effort.
HUD03_ANKLE_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(r".*_ankle_(pitch|roll)_joint",),
  stiffness=_stiffness(2.0 * LINKAGE_ARMATURE),
  damping=_damping(2.0 * LINKAGE_ARMATURE),
  effort_limit=90.0,
  armature=2.0 * LINKAGE_ARMATURE,
)

HUD03_WAIST_YAW_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_yaw_joint",),
  stiffness=_stiffness(LINKAGE_ARMATURE),
  damping=_damping(LINKAGE_ARMATURE),
  effort_limit=45.0,
  armature=LINKAGE_ARMATURE,
)

HUD03_WAIST_RP_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(r"waist_(roll|pitch)_joint",),
  stiffness=_stiffness(2.0 * LINKAGE_ARMATURE),
  damping=_damping(2.0 * LINKAGE_ARMATURE),
  effort_limit=90.0,
  armature=2.0 * LINKAGE_ARMATURE,
)

HUD03_ARM_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(r".*_(shoulder_pitch|shoulder_roll|shoulder_yaw|elbow)_joint",),
  stiffness=_stiffness(ARM_ARMATURE),
  damping=_damping(ARM_ARMATURE),
  effort_limit=30.0,
  armature=ARM_ARMATURE,
)

HUD03_SMALL_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    r"head_(yaw|pitch)_joint",
    r".*_(wrist_yaw|wrist_pitch|hand_yaw)_joint",
  ),
  stiffness=_stiffness(SMALL_ARMATURE),
  damping=_damping(SMALL_ARMATURE),
  effort_limit=18.0,
  armature=SMALL_ARMATURE,
)

##
# Initial state.
##

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  # At this height the nominal sole plane is about 1 mm above flat ground.
  pos=(0.0, 0.0, 0.89),
  joint_pos={
    r".*_hip_pitch_joint": -0.25,
    r".*_knee_joint": 0.5,
    r".*_ankle_pitch_joint": -0.25,
    r".*_shoulder_pitch_joint": 0.15,
    "left_shoulder_roll_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    r".*_elbow_joint": 0.6,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_FOOT_REGEX = r"^(left|right)_foot_collision$"

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(r".*_collision(_\d+)?$",),
  condim={_FOOT_REGEX: 3, r".*_collision(_\d+)?$": 1},
  priority={_FOOT_REGEX: 1},
  friction={_FOOT_REGEX: (0.7,)},
)

##
# Final config.
##

HUD03_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    HUD03_LEG_ACTUATOR,
    HUD03_ANKLE_ACTUATOR,
    HUD03_WAIST_YAW_ACTUATOR,
    HUD03_WAIST_RP_ACTUATOR,
    HUD03_ARM_ACTUATOR,
    HUD03_SMALL_ACTUATOR,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_hud03_robot_cfg() -> EntityCfg:
  """Get a fresh HUD03 robot configuration instance."""
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=HUD03_ARTICULATION,
  )


HUD03_ACTION_SCALE: dict[str, float] = {}
for actuator_cfg in HUD03_ARTICULATION.actuators:
  assert isinstance(actuator_cfg, BuiltinPositionActuatorCfg)
  effort = actuator_cfg.effort_limit
  assert effort is not None
  for name_expr in actuator_cfg.target_names_expr:
    HUD03_ACTION_SCALE[name_expr] = 0.25 * effort / actuator_cfg.stiffness


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity import Entity

  robot = Entity(get_hud03_robot_cfg())
  viewer.launch(robot.spec.compile())
