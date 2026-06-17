"""Convert HU_D03 retargeted CSV motions to mjlab NPZ motions."""

import torch
import tyro

import mjlab
from mjlab.asset_zoo.robots import HU_D03_MOTION_JOINT_NAMES
from mjlab.scene import Scene
from mjlab.scripts.csv_to_npz import run_sim
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.config.hu_d03.env_cfgs import hu_d03_flat_tracking_env_cfg
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig


def main(
  input_file: str,
  output_name: str,
  input_fps: float = 30.0,
  output_fps: float = 50.0,
  device: str = "cuda:0",
  render: bool = False,
  line_range: tuple[int, int] | None = None,
):
  """Replay a HU_D03 CSV motion and upload the mjlab-compatible NPZ to W&B.

  The CSV format is:
    base position xyz, base quaternion xyzw, then joint angles in
    ``HU_D03_MOTION_JOINT_NAMES`` order.
  """
  if device.startswith("cuda") and not torch.cuda.is_available():
    print("[WARNING]: CUDA is not available. Falling back to CPU. This may be slow.")
    device = "cpu"

  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps

  scene = Scene(hu_d03_flat_tracking_env_cfg().scene, device=device)
  model = scene.compile()

  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)

  scene.initialize(sim.mj_model, sim.model, sim.data)

  renderer = None
  if render:
    viewer_cfg = ViewerConfig(
      height=480,
      width=640,
      origin_type=ViewerConfig.OriginType.ASSET_ROOT,
      entity_name="robot",
      distance=2.4,
      elevation=-5.0,
      azimuth=20.0,
    )
    renderer = OffscreenRenderer(
      model=sim.mj_model,
      cfg=viewer_cfg,
      scene=scene,
    )
    renderer.initialize()

  run_sim(
    sim=sim,
    scene=scene,
    joint_names=HU_D03_MOTION_JOINT_NAMES,
    input_fps=input_fps,
    input_file=input_file,
    output_fps=output_fps,
    output_name=output_name,
    render=render,
    line_range=line_range,
    renderer=renderer,
  )


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
