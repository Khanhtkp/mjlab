"""LimX Dynamics HU_D03 humanoid constants and robot config."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

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
  stiffness=300.0,
  damping=30.0,
  effort_limit=120.0,
)

HU_D03_ANKLE_WAIST_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
  ),
  stiffness=180.0,
  damping=18.0,
  effort_limit=45.0,
)

HU_D03_WAIST_YAW_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_yaw_joint",),
  stiffness=220.0,
  damping=22.0,
  effort_limit=45.0,
)

HU_D03_UPPER_BODY_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_joint",
  ),
  stiffness=120.0,
  damping=12.0,
  effort_limit=30.0,
)

HU_D03_SMALL_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "head_yaw_joint",
    "head_pitch_joint",
    ".*_wrist_yaw_joint",
    ".*_wrist_pitch_joint",
    ".*_hand_yaw_joint",
  ),
  stiffness=80.0,
  damping=8.0,
  effort_limit=18.0,
)

HU_D03_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    HU_D03_LOWER_BODY_ACTUATOR,
    HU_D03_ANKLE_WAIST_ACTUATOR,
    HU_D03_WAIST_YAW_ACTUATOR,
    HU_D03_UPPER_BODY_ACTUATOR,
    HU_D03_SMALL_ACTUATOR,
  ),
  soft_joint_pos_limit_factor=0.9,
)


HU_D03_STAND_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 1.0),
  joint_pos={
    ".*": 0.0,
    ".*_hip_pitch_joint": -0.15,
    ".*_knee_joint": 0.35,
    ".*_ankle_pitch_joint": -0.2,
    "left_shoulder_roll_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    ".*_elbow_joint": 0.5,
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
