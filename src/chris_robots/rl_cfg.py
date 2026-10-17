"""PPO configuration (mjlab's G1 velocity recipe, logging to TensorBoard)."""

from dataclasses import dataclass, field

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from chris_robots.robots.base import RobotDef


@dataclass
class SymmetryCfg:
  """rsl-rl ``Symmetry`` extension settings (see ``chris_robots.symmetry``)."""

  data_augmentation_func: str = "chris_robots.symmetry:mirror"
  use_data_augmentation: bool = True
  use_mirror_loss: bool = True
  mirror_loss_coeff: float = 0.5


@dataclass
class RslRlPpoSymmetryAlgorithmCfg(RslRlPpoAlgorithmCfg):
  symmetry_cfg: SymmetryCfg = field(default_factory=SymmetryCfg)


def make_ppo_runner_cfg(robot: RobotDef, symmetry: bool = False) -> RslRlOnPolicyRunnerCfg:
  alg_cls = RslRlPpoSymmetryAlgorithmCfg if symmetry else RslRlPpoAlgorithmCfg
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=alg_cls(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name=f"{robot.name}_velocity",
    logger="tensorboard",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=5_000,
  )
