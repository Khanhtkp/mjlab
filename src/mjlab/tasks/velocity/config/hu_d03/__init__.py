from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  hu_d03_flat_forward_stable_velocity_env_cfg,
  hu_d03_flat_forward_survival_velocity_env_cfg,
  hu_d03_flat_forward_tuned_velocity_env_cfg,
  hu_d03_flat_forward_velocity_env_cfg,
  hu_d03_flat_self_organized_velocity_env_cfg,
  hu_d03_flat_survival_first_velocity_env_cfg,
  hu_d03_flat_velocity_env_cfg,
  hu_d03_rough_velocity_env_cfg,
)
from .rl_cfg import (
  hu_d03_velocity_ppo_runner_cfg,
  hu_d03_velocity_self_organized_ppo_runner_cfg,
  hu_d03_velocity_survival_first_ppo_runner_cfg,
  hu_d03_velocity_survival_ppo_runner_cfg,
  hu_d03_velocity_stable_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-HU-D03",
  env_cfg=hu_d03_rough_velocity_env_cfg(),
  play_env_cfg=hu_d03_rough_velocity_env_cfg(play=True),
  rl_cfg=hu_d03_velocity_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-HU-D03",
  env_cfg=hu_d03_flat_velocity_env_cfg(),
  play_env_cfg=hu_d03_flat_velocity_env_cfg(play=True),
  rl_cfg=hu_d03_velocity_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-HU-D03-Forward",
  env_cfg=hu_d03_flat_forward_velocity_env_cfg(),
  play_env_cfg=hu_d03_flat_forward_velocity_env_cfg(play=True),
  rl_cfg=hu_d03_velocity_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-HU-D03-Forward-Tuned",
  env_cfg=hu_d03_flat_forward_tuned_velocity_env_cfg(),
  play_env_cfg=hu_d03_flat_forward_tuned_velocity_env_cfg(play=True),
  rl_cfg=hu_d03_velocity_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-HU-D03-Forward-Stable",
  env_cfg=hu_d03_flat_forward_stable_velocity_env_cfg(),
  play_env_cfg=hu_d03_flat_forward_stable_velocity_env_cfg(play=True),
  rl_cfg=hu_d03_velocity_stable_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-HU-D03-Forward-Survival",
  env_cfg=hu_d03_flat_forward_survival_velocity_env_cfg(),
  play_env_cfg=hu_d03_flat_forward_survival_velocity_env_cfg(play=True),
  rl_cfg=hu_d03_velocity_survival_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-HU-D03-Self-Organized",
  env_cfg=hu_d03_flat_self_organized_velocity_env_cfg(),
  play_env_cfg=hu_d03_flat_self_organized_velocity_env_cfg(play=True),
  rl_cfg=hu_d03_velocity_self_organized_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-HU-D03-Survival-First",
  env_cfg=hu_d03_flat_survival_first_velocity_env_cfg(),
  play_env_cfg=hu_d03_flat_survival_first_velocity_env_cfg(play=True),
  rl_cfg=hu_d03_velocity_survival_first_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
