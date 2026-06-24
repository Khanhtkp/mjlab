"""Tests for the HUD03 training articulation."""

import re

import mujoco
import numpy as np
import pytest

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots.limx_hud03 import hud03_constants
from mjlab.entity import Entity
from mjlab.utils.string import resolve_expr


@pytest.fixture(scope="module")
def hud03_entity() -> Entity:
  return Entity(hud03_constants.get_hud03_robot_cfg())


@pytest.fixture(scope="module")
def hud03_model(hud03_entity: Entity) -> mujoco.MjModel:
  return hud03_entity.spec.compile()


def test_hud03_reduced_articulation(hud03_entity: Entity, hud03_model) -> None:
  assert hud03_entity.num_joints == 31
  assert hud03_entity.num_actuators == 31
  assert hud03_entity.is_actuated
  assert not hud03_entity.is_fixed_base
  assert hud03_model.neq == 0
  assert all(
    hud03_model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
    for i in range(1, hud03_model.njnt)
  )


def test_hud03_preserves_total_mass(hud03_model) -> None:
  np.testing.assert_allclose(hud03_model.body_subtreemass[1], 53.02208, rtol=1e-6)


def test_hud03_keyframe(hud03_entity: Entity, hud03_model) -> None:
  key = hud03_model.key("init_state")
  expected = hud03_constants.KNEES_BENT_KEYFRAME.joint_pos
  assert expected is not None
  expected_values = resolve_expr(expected, hud03_entity.joint_names, 0.0)
  np.testing.assert_allclose(key.qpos[:3], (0.0, 0.0, 0.89))
  np.testing.assert_allclose(key.qpos[7:], expected_values)


def test_hud03_required_sensors_and_sites(hud03_model) -> None:
  sensor_names = {hud03_model.sensor(i).name for i in range(hud03_model.nsensor)}
  assert {"imu_ang_vel", "imu_lin_vel", "root_angmom"} <= sensor_names
  site_names = {hud03_model.site(i).name for i in range(hud03_model.nsite)}
  assert {"left_foot", "right_foot"} <= site_names


def test_hud03_foot_collisions(hud03_model) -> None:
  for side in ("left", "right"):
    geom = hud03_model.geom(f"{side}_foot_collision")
    assert geom.condim == 3
    assert geom.priority == 1
    assert geom.friction[0] == pytest.approx(0.7)


def test_hud03_actuator_parameters(hud03_model) -> None:
  cfgs = hud03_constants.HUD03_ARTICULATION.actuators
  for actuator_id in range(hud03_model.nu):
    actuator = hud03_model.actuator(actuator_id)
    matching_cfgs = [
      cfg
      for cfg in cfgs
      if any(re.fullmatch(pattern, actuator.name) for pattern in cfg.target_names_expr)
    ]
    assert len(matching_cfgs) == 1
    cfg = matching_cfgs[0]
    assert isinstance(cfg, BuiltinPositionActuatorCfg)
    assert cfg.effort_limit is not None
    assert actuator.gainprm[0] == pytest.approx(cfg.stiffness)
    assert actuator.biasprm[1] == pytest.approx(-cfg.stiffness)
    assert actuator.biasprm[2] == pytest.approx(-cfg.damping)
    assert actuator.forcerange[0] == pytest.approx(-cfg.effort_limit)
    assert actuator.forcerange[1] == pytest.approx(cfg.effort_limit)
