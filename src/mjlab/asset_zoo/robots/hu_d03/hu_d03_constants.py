"""LimX Dynamics HU_D03 humanoid constants and robot config."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from math import pi

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

_REPO_ROOT = MJLAB_SRC_PATH.parent.parent
HU_D03_DESCRIPTION_DIR = _REPO_ROOT / "HU_D03_description"
HU_D03_XML = HU_D03_DESCRIPTION_DIR / "xml" / "HU_D03_03.xml"
HU_D03_MESH_DIR = HU_D03_DESCRIPTION_DIR / "meshes" / "HU_D03_03"

assert HU_D03_XML.exists(), f"HU_D03 XML not found: {HU_D03_XML}"
assert HU_D03_MESH_DIR.exists(), f"HU_D03 mesh dir not found: {HU_D03_MESH_DIR}"


HU_D03_JOINT_NAMES: tuple[str, ...] = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "head_yaw_joint",
  "head_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_yaw_joint",
  "left_wrist_pitch_joint",
  "left_hand_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_yaw_joint",
  "right_wrist_pitch_joint",
  "right_hand_yaw_joint",
)
HU_D03_MOTION_JOINT_NAMES = HU_D03_JOINT_NAMES

_LINKAGE_BODY_NAMES = {
  "left_A_achilles_link",
  "left_B_achilles_link",
  "right_A_achilles_link",
  "right_B_achilles_link",
  "waist_A_link",
  "waist_B_link",
}
_FOOT_GEOM_NAMES = {"left_foot", "right_foot"}

# HU_D03 has substantially larger reflected rotor inertia than G1. Using G1's
# 10 Hz bandwidth makes its direct-joint training model several times stiffer
# while leaving only a few degrees of policy action authority. A 4 Hz bandwidth
# keeps the same critically damped controller design while bringing lower-body
# gains and usable position-target ranges into the same regime as G1.
NATURAL_FREQ = 4.0 * 2.0 * pi
DAMPING_RATIO = 2.0

HIP_KNEE_ARMATURE = 0.15257125
LINKAGE_ARMATURE = 0.094889232
UPPER_BODY_ARMATURE = 0.045760625
SMALL_JOINT_ARMATURE = 0.010625


def _stiffness_from_armature(armature: float) -> float:
  return armature * NATURAL_FREQ**2


def _damping_from_armature(armature: float) -> float:
  return 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ


def _remove_children_by_tag(parent: ET.Element, tag: str) -> None:
  for child in list(parent):
    if child.tag == tag:
      parent.remove(child)


def _remove_named_body_subtrees(parent: ET.Element, names: set[str]) -> None:
  for child in list(parent):
    if child.tag == "body" and child.attrib.get("name") in names:
      parent.remove(child)
      continue
    _remove_named_body_subtrees(child, names)


def _find_body(parent: ET.Element, name: str) -> ET.Element | None:
  if parent.tag == "body" and parent.attrib.get("name") == name:
    return parent
  for child in parent:
    found = _find_body(child, name)
    if found is not None:
      return found
  return None


def _ensure_site(body: ET.Element, name: str, pos: str) -> None:
  for child in body:
    if child.tag == "site" and child.attrib.get("name") == name:
      return
  ET.SubElement(
    body,
    "site",
    {
      "name": name,
      "pos": pos,
      "size": "0.01",
      "rgba": "0 0 1 1",
    },
  )


def _zero_geom_margins(root: ET.Element) -> None:
  """MuJoCo Warp MULTICCD does not support non-zero geom margins."""
  for geom in root.iter("geom"):
    geom.set("margin", "0")


def _configure_training_collisions(root: ET.Element) -> None:
  """Use G1-style training contacts for the sanitized HU_D03 model.

  HU_D03's vendor MJCF gives every collision primitive condim=3 and relatively high
  friction. That makes torso, limb, and arm contacts behave like sticky ground
  contacts during RL. G1 uses rich frictional contact only for feet and simple
  condim=1 contacts for the rest of the body; mirroring that setup removes a large
  source of artificial contact constraints without disabling self-collision.
  """
  for geom in root.iter("geom"):
    name = geom.attrib.get("name", "")

    if geom.attrib.get("class") == "visual" or name.startswith("contact_"):
      geom.set("contype", "0")
      geom.set("conaffinity", "0")
      geom.set("group", "1")
      continue

    geom.set("contype", "1")
    geom.set("conaffinity", "1")
    geom.set("margin", "0")

    if name in _FOOT_GEOM_NAMES:
      geom.set("condim", "3")
      geom.set("priority", "1")
      geom.set("friction", "0.6 0.005 0.0005")
    else:
      geom.set("condim", "1")
      geom.set("priority", "0")


def _sanitize_hu_d03_xml() -> str:
  """Return an mjlab-friendly HU_D03 MJCF string.

  The vendor MJCF is a complete standalone scene. For mjlab we need only the robot:
  terrain/cameras/lights come from SceneCfg, and policy actions are position targets.
  The original ankle/waist parallel linkages include ball joints, which do not fit
  mjlab's current joint-vector assumptions, so the local training model controls the
  equivalent ankle and waist hinge joints directly.
  """
  root = ET.fromstring(HU_D03_XML.read_text())

  compiler = root.find("compiler")
  if compiler is not None:
    compiler.set("meshdir", HU_D03_MESH_DIR.as_posix())

  worldbody = root.find("worldbody")
  if worldbody is not None:
    for child in list(worldbody):
      name = child.attrib.get("name")
      if child.tag in {"light", "camera"} or name == "floor":
        worldbody.remove(child)
    _remove_named_body_subtrees(worldbody, _LINKAGE_BODY_NAMES)
    left_ankle = _find_body(worldbody, "left_ankle_roll_link")
    right_ankle = _find_body(worldbody, "right_ankle_roll_link")
    if left_ankle is not None:
      _ensure_site(left_ankle, "left_foot", "0.018 0 -0.0535")
    if right_ankle is not None:
      _ensure_site(right_ankle, "right_foot", "0.018 0 -0.0535")

  _configure_training_collisions(root)
  _zero_geom_margins(root)
  _remove_children_by_tag(root, "equality")
  _remove_children_by_tag(root, "actuator")
  _remove_children_by_tag(root, "sensor")

  return ET.tostring(root, encoding="unicode")


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_string(_sanitize_hu_d03_xml())


HU_D03_LOWER_BODY_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_pitch_joint",
    ".*_hip_roll_joint",
    ".*_hip_yaw_joint",
    ".*_knee_joint",
  ),
  stiffness=_stiffness_from_armature(HIP_KNEE_ARMATURE),
  damping=_damping_from_armature(HIP_KNEE_ARMATURE),
  effort_limit=120.0,
  armature=HIP_KNEE_ARMATURE,
)

HU_D03_ANKLE_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
  ),
  # The vendor model drives each ankle through an A/B linkage pair. The
  # simplified model exposes direct ankle hinges, so give those hinges slightly
  # more authority than a single linkage motor while keeping the reflected
  # inertia/stiffness model.
  stiffness=_stiffness_from_armature(LINKAGE_ARMATURE),
  damping=_damping_from_armature(LINKAGE_ARMATURE),
  effort_limit=60.0,
  armature=LINKAGE_ARMATURE,
)

HU_D03_WAIST_ROLL_PITCH_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "waist_roll_joint",
    "waist_pitch_joint",
  ),
  stiffness=_stiffness_from_armature(LINKAGE_ARMATURE),
  damping=_damping_from_armature(LINKAGE_ARMATURE) * 1.2,
  effort_limit=45.0,
  armature=LINKAGE_ARMATURE,
)

HU_D03_WAIST_YAW_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_yaw_joint",),
  stiffness=_stiffness_from_armature(LINKAGE_ARMATURE),
  damping=_damping_from_armature(LINKAGE_ARMATURE),
  effort_limit=45.0,
  armature=LINKAGE_ARMATURE,
)

HU_D03_UPPER_BODY_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_joint",
  ),
  stiffness=_stiffness_from_armature(UPPER_BODY_ARMATURE),
  damping=_damping_from_armature(UPPER_BODY_ARMATURE),
  effort_limit=30.0,
  armature=UPPER_BODY_ARMATURE,
)

HU_D03_SMALL_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "head_yaw_joint",
    "head_pitch_joint",
    ".*_wrist_yaw_joint",
    ".*_wrist_pitch_joint",
    ".*_hand_yaw_joint",
  ),
  stiffness=_stiffness_from_armature(SMALL_JOINT_ARMATURE),
  damping=_damping_from_armature(SMALL_JOINT_ARMATURE),
  effort_limit=18.0,
  armature=SMALL_JOINT_ARMATURE,
)

HU_D03_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    HU_D03_LOWER_BODY_ACTUATOR,
    HU_D03_ANKLE_ACTUATOR,
    HU_D03_WAIST_ROLL_PITCH_ACTUATOR,
    HU_D03_WAIST_YAW_ACTUATOR,
    HU_D03_UPPER_BODY_ACTUATOR,
    HU_D03_SMALL_ACTUATOR,
  ),
  soft_joint_pos_limit_factor=0.9,
)


HU_D03_STAND_KEYFRAME = EntityCfg.InitialStateCfg(
  # The crouch gives the policy enough knee travel for weight transfer and
  # single-support locomotion. Root height is computed from the sanitized foot
  # box geometry so both soles start on the ground.
  pos=(0.0, 0.0, 0.886854),
  joint_pos={
    ".*_hip_pitch_joint": -0.25,
    ".*_knee_joint": 0.55,
    ".*_ankle_pitch_joint": -0.3,
    "left_shoulder_roll_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    ".*_elbow_joint": -0.5,
  },
  joint_vel={".*": 0.0},
)


def get_hu_d03_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=HU_D03_STAND_KEYFRAME,
    spec_fn=get_spec,
    articulation=HU_D03_ARTICULATION,
  )


HU_D03_ACTION_SCALE: dict[str, float] = {}
for actuator in HU_D03_ARTICULATION.actuators:
  assert isinstance(actuator, BuiltinPositionActuatorCfg)
  assert actuator.effort_limit is not None
  for name in actuator.target_names_expr:
    HU_D03_ACTION_SCALE[name] = 0.25 * actuator.effort_limit / actuator.stiffness
