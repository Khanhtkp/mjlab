from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  limx_hud03_flat_env_cfg,
  limx_hud03_rough_env_cfg,
)
from .rl_cfg import limx_hud03_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-LimX-HUD03",
  env_cfg=limx_hud03_rough_env_cfg(),
  play_env_cfg=limx_hud03_rough_env_cfg(play=True),
  rl_cfg=limx_hud03_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-LimX-HUD03",
  env_cfg=limx_hud03_flat_env_cfg(),
  play_env_cfg=limx_hud03_flat_env_cfg(play=True),
  rl_cfg=limx_hud03_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
