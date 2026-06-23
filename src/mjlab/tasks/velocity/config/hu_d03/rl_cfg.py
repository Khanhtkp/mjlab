"""RL configuration for HU_D03 velocity task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def hu_d03_velocity_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for LimX HU_D03 velocity task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.7,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.004,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=7.5e-4,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.008,
      max_grad_norm=1.0,
    ),
    experiment_name="hu_d03_velocity",
    save_interval=50,
    num_steps_per_env=24,
    clip_actions=1.5,
    max_iterations=30_000,
  )


def hu_d03_velocity_stable_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create a lower-noise PPO config for stabilizing HU_D03 locomotion."""
  cfg = hu_d03_velocity_ppo_runner_cfg()
  assert cfg.actor.distribution_cfg is not None
  cfg.actor.distribution_cfg["init_std"] = 0.7
  cfg.algorithm.entropy_coef = 0.005
  cfg.algorithm.learning_rate = 7.5e-4
  cfg.clip_actions = 1.5
  cfg.experiment_name = "hu_d03_velocity_stable"
  return cfg


def hu_d03_velocity_survival_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create a low-noise PPO config for survival-first HU_D03 training."""
  cfg = hu_d03_velocity_ppo_runner_cfg()
  assert cfg.actor.distribution_cfg is not None
  cfg.actor.distribution_cfg["init_std"] = 0.5
  cfg.algorithm.entropy_coef = 0.002
  cfg.algorithm.learning_rate = 5.0e-4
  cfg.algorithm.desired_kl = 0.008
  cfg.clip_actions = 1.0
  cfg.experiment_name = "hu_d03_velocity_survival"
  cfg.max_iterations = 4_000
  return cfg


def hu_d03_velocity_self_organized_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create PPO config for G1-like self-organized HU_D03 locomotion."""
  cfg = hu_d03_velocity_ppo_runner_cfg()
  assert cfg.actor.distribution_cfg is not None
  cfg.actor.distribution_cfg["init_std"] = 0.7
  cfg.algorithm.entropy_coef = 0.005
  cfg.algorithm.learning_rate = 7.5e-4
  cfg.algorithm.desired_kl = 0.01
  cfg.clip_actions = 1.5
  cfg.experiment_name = "hu_d03_velocity_self_organized"
  cfg.max_iterations = 4_000
  return cfg


def hu_d03_velocity_survival_first_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create PPO config focused on stable HU_D03 episode length growth."""
  cfg = hu_d03_velocity_ppo_runner_cfg()
  assert cfg.actor.distribution_cfg is not None
  cfg.actor.distribution_cfg["init_std"] = 0.45
  cfg.algorithm.entropy_coef = 0.001
  cfg.algorithm.learning_rate = 5.0e-4
  cfg.algorithm.desired_kl = 0.006
  cfg.clip_actions = 1.0
  cfg.experiment_name = "hu_d03_velocity_survival_first"
  cfg.max_iterations = 2_000
  return cfg
