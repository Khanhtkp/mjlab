from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.metrics_manager import MetricsTermCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class command_transition_failure:
  """Measure whether a fall happens shortly after a command changes."""

  def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRlEnv):
    command_name = cfg.params["command_name"]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    self.previous_command = command.clone()
    self.steps_since_change = torch.zeros(
      env.num_envs,
      dtype=torch.long,
      device=env.device,
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    termination_name: str = "fell_over",
    transition_window_s: float = 1.0,
    change_threshold: float = 0.05,
  ) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."

    changed = torch.norm(command - self.previous_command, dim=1) > change_threshold
    self.steps_since_change += 1
    self.steps_since_change[changed] = 0
    self.previous_command.copy_(command)

    fell = env.termination_manager.get_term(termination_name)
    seconds_since_change = self.steps_since_change.float() * env.step_dt
    recent = fell & (seconds_since_change <= transition_window_s)

    fall_count = fell.float().sum()
    mean_fall_age = torch.sum(seconds_since_change * fell.float()) / torch.clamp(
      fall_count,
      min=1.0,
    )
    env.extras["log"]["Diagnostics/fall_seconds_since_command_change"] = mean_fall_age
    env.extras["log"]["Diagnostics/fall_after_command_change_fraction"] = (
      recent.float().sum() / torch.clamp(fall_count, min=1.0)
    )
    return recent.float()

  def reset(self, env_ids: torch.Tensor) -> None:
    self.steps_since_change[env_ids] = 0
