from __future__ import annotations

from typing import TYPE_CHECKING, NotRequired, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .velocity_command import UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


class VelocityStage(TypedDict):
  step: int
  lin_vel_x: NotRequired[tuple[float, float] | None]
  lin_vel_y: NotRequired[tuple[float, float] | None]
  ang_vel_z: NotRequired[tuple[float, float] | None]
  rel_standing_envs: NotRequired[float | None]
  rel_heading_envs: NotRequired[float | None]
  rel_forward_envs: NotRequired[float | None]
  rel_world_envs: NotRequired[float | None]
  advance_episode_length: NotRequired[float | None]
  advance_timeout_rate: NotRequired[float | None]


def _apply_velocity_stage(
  cfg: UniformVelocityCommandCfg,
  stage: VelocityStage,
) -> None:
  if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
    cfg.ranges.lin_vel_x = stage["lin_vel_x"]
  if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
    cfg.ranges.lin_vel_y = stage["lin_vel_y"]
  if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
    cfg.ranges.ang_vel_z = stage["ang_vel_z"]
  if "rel_standing_envs" in stage and stage["rel_standing_envs"] is not None:
    cfg.rel_standing_envs = stage["rel_standing_envs"]
  if "rel_heading_envs" in stage and stage["rel_heading_envs"] is not None:
    cfg.rel_heading_envs = stage["rel_heading_envs"]
  if "rel_forward_envs" in stage and stage["rel_forward_envs"] is not None:
    cfg.rel_forward_envs = stage["rel_forward_envs"]
  if "rel_world_envs" in stage and stage["rel_world_envs"] is not None:
    cfg.rel_world_envs = stage["rel_world_envs"]


def _velocity_curriculum_log(
  cfg: UniformVelocityCommandCfg,
  *,
  stage_index: int | None = None,
  ema_episode_length: float | None = None,
  ema_timeout_rate: float | None = None,
) -> dict[str, torch.Tensor]:
  result = {
    "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
    "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
    "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
    "rel_standing_envs": torch.tensor(cfg.rel_standing_envs),
    "rel_heading_envs": torch.tensor(cfg.rel_heading_envs),
    "rel_forward_envs": torch.tensor(cfg.rel_forward_envs),
    "rel_world_envs": torch.tensor(cfg.rel_world_envs),
  }
  if stage_index is not None:
    result["stage"] = torch.tensor(float(stage_index))
  if ema_episode_length is not None:
    result["ema_episode_length"] = torch.tensor(ema_episode_length)
  if ema_timeout_rate is not None:
    result["ema_timeout_rate"] = torch.tensor(ema_timeout_rate)
  return result


def terrain_levels_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> dict[str, torch.Tensor]:
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command = env.command_manager.get_command(command_name)
  assert command is not None

  # Compute the distance the robot walked.
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
    dim=1,
  )

  # Robots that walked far enough progress to harder terrains.
  move_up = distance > terrain_generator.size[0] / 2

  # Robots that walked less than half of their required distance go to
  # simpler terrains.
  move_down = (
    distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
  )
  move_down *= ~move_up

  # Update terrain levels.
  terrain.update_env_origins(env_ids, move_up, move_down)

  # Compute per-terrain-type mean levels.
  levels = terrain.terrain_levels.float()
  result: dict[str, torch.Tensor] = {
    "mean": torch.mean(levels),
    "max": torch.max(levels),
  }

  # In curriculum mode num_cols == num_terrains (one column per type),
  # so the column index directly maps to the sub-terrain name.
  sub_terrain_names = list(terrain_generator.sub_terrains.keys())
  terrain_origins = terrain.terrain_origins
  assert terrain_origins is not None
  num_cols = terrain_origins.shape[1]
  if num_cols == len(sub_terrain_names):
    types = terrain.terrain_types
    for i, name in enumerate(sub_terrain_names):
      mask = types == i
      if mask.any():
        result[name] = torch.mean(levels[mask])

  return result


def commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  for stage in velocity_stages:
    if env.common_step_counter >= stage["step"]:
      _apply_velocity_stage(cfg, stage)
  return _velocity_curriculum_log(cfg)


class adaptive_commands_vel:
  """Advance velocity command stages only after survival performance improves."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    command_name: str = cfg.params["command_name"]
    self._command_term = env.command_manager.get_term(command_name)
    assert self._command_term is not None
    self._stages: list[VelocityStage] = cfg.params["velocity_stages"]
    self._ema_alpha: float = cfg.params.get("ema_alpha", 0.05)
    self._min_updates: int = cfg.params.get("min_updates", 5)
    self._stage_index = 0
    self._update_count = 0
    self._ema_episode_length = 0.0
    self._ema_timeout_rate = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    velocity_stages: list[VelocityStage],
    ema_alpha: float = 0.05,
    min_updates: int = 5,
  ) -> dict[str, torch.Tensor]:
    del command_name, velocity_stages, ema_alpha, min_updates
    cfg = cast(UniformVelocityCommandCfg, self._command_term.cfg)

    if isinstance(env_ids, slice):
      env_ids = torch.arange(env.num_envs, device=env.device)

    if env.common_step_counter > 0 and len(env_ids) > 0:
      lengths = env.episode_length_buf[env_ids].float()
      self._update_count += 1
      episode_length = float(torch.mean(lengths).item())
      timeout_rate = float(torch.mean(env.reset_time_outs[env_ids].float()).item())
      alpha = self._ema_alpha
      if self._update_count == 1:
        self._ema_episode_length = episode_length
        self._ema_timeout_rate = timeout_rate
      else:
        self._ema_episode_length = (
          (1.0 - alpha) * self._ema_episode_length + alpha * episode_length
        )
        self._ema_timeout_rate = (
          (1.0 - alpha) * self._ema_timeout_rate + alpha * timeout_rate
        )

      if (
        self._update_count >= self._min_updates
        and self._stage_index < len(self._stages) - 1
      ):
        stage = self._stages[self._stage_index]
        target_length = stage.get("advance_episode_length")
        target_timeout = stage.get("advance_timeout_rate")
        length_ready = (
          target_length is not None and self._ema_episode_length >= target_length
        )
        timeout_ready = (
          target_timeout is not None and self._ema_timeout_rate >= target_timeout
        )
        if length_ready or timeout_ready:
          self._stage_index += 1
          self._update_count = 0

    _apply_velocity_stage(cfg, self._stages[self._stage_index])
    return _velocity_curriculum_log(
      cfg,
      stage_index=self._stage_index,
      ema_episode_length=self._ema_episode_length,
      ema_timeout_rate=self._ema_timeout_rate,
    )
