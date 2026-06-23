# HU_D03 Mimic Policy Training with mjlab

This workspace contains an mjlab integration for the LimX Dynamics HU_D03
humanoid. The goal is to train a motion-imitation, or mimic, policy with the
task:

```bash
Mjlab-Tracking-Flat-HU-D03
```

The current HU_D03 integration is a practical training prototype. It uses the
HU_D03 robot description from `HU_D03_description`, registers the robot in
mjlab's asset zoo, adds a tracking task, and provides a HU_D03 CSV-to-NPZ
converter.

## What Is Included

### HU_D03 robot asset

The robot config is defined in:

```text
src/mjlab/asset_zoo/robots/hu_d03/hu_d03_constants.py
```

It loads:

```text
HU_D03_description/xml/HU_D03_03.xml
HU_D03_description/meshes/HU_D03_03
```

and defines:

- HU_D03 joint order
- HU_D03 motion joint order
- HU_D03 initial standing pose
- position actuators for lower body, ankles, waist, upper body, head, wrists,
  and hands
- action scale used by the policy

The HU_D03 model is exported through:

```text
src/mjlab/asset_zoo/robots/hu_d03/__init__.py
src/mjlab/asset_zoo/robots/__init__.py
```

### HU_D03 tracking task

The task config is defined in:

```text
src/mjlab/tasks/tracking/config/hu_d03/
```

It registers:

```bash
Mjlab-Tracking-Flat-HU-D03
```

This is the task used to train a mimic policy for HU_D03.

### HU_D03 motion converter

The converter is:

```text
src/mjlab/scripts/csv_to_npz_hu_d03.py
```

It converts a HU_D03 motion CSV into the `.npz` motion format used by mjlab
tracking tasks, and can upload the converted motion to a Weights & Biases
motion registry.

The CLI entry is:

```bash
uv run csv-to-npz-hu-d03
```

### HU_D03 velocity locomotion task

The velocity task config is defined in:

```text
src/mjlab/tasks/velocity/config/hu_d03/
```

It registers:

```bash
Mjlab-Velocity-Flat-HU-D03
```

This task trains a locomotion policy from velocity commands and does not need a
motion CSV.

## Important Limitation

`HU_D03_description` only contains the robot model:

- URDF
- MJCF/XML
- USD
- SRDF
- STL meshes

It does not contain motion data. A mimic policy cannot be trained without a
reference motion.

You need one of the following:

- a CSV motion already retargeted to HU_D03
- a local `motion.npz` compatible with HU_D03
- a motion retargeting pipeline that converts AMASS/LAFAN1/G1 motions into
  HU_D03 joint order

## Simplified Training Model

The current HU_D03 integration uses a sanitized/simplified MJCF for training.
During loading, it removes:

- standalone floor, camera, and light from the vendor XML
- vendor XML actuators and sensors
- equality constraints
- ankle and waist linkage subtrees

The simplified model directly controls the ankle and waist hinge joints with
position actuators. This is intentional: it makes the model easier to run in
mjlab and allows the mimic-training pipeline to work first. It is not a
perfect dynamics replica of the full vendor model.

## Motion CSV Format

The HU_D03 CSV must use this layout:

```text
base_x, base_y, base_z,
base_qx, base_qy, base_qz, base_qw,
joint_0, joint_1, ..., joint_30
```

The 31 joint columns must follow `HU_D03_MOTION_JOINT_NAMES` from:

```text
src/mjlab/asset_zoo/robots/hu_d03/hu_d03_constants.py
```

Current order:

```text
left_hip_pitch_joint
left_hip_roll_joint
left_hip_yaw_joint
left_knee_joint
left_ankle_pitch_joint
left_ankle_roll_joint
right_hip_pitch_joint
right_hip_roll_joint
right_hip_yaw_joint
right_knee_joint
right_ankle_pitch_joint
right_ankle_roll_joint
waist_yaw_joint
waist_roll_joint
waist_pitch_joint
head_yaw_joint
head_pitch_joint
left_shoulder_pitch_joint
left_shoulder_roll_joint
left_shoulder_yaw_joint
left_elbow_joint
left_wrist_yaw_joint
left_wrist_pitch_joint
left_hand_yaw_joint
right_shoulder_pitch_joint
right_shoulder_roll_joint
right_shoulder_yaw_joint
right_elbow_joint
right_wrist_yaw_joint
right_wrist_pitch_joint
right_hand_yaw_joint
```

If the CSV joint order is wrong, the reference motion will be wrong and the
tracking policy will not learn correctly.

## Setup

Install dependencies from the repository root:

```bash
uv sync
```

For training, use an NVIDIA GPU. CPU is only suitable for small smoke tests.

If you use Weights & Biases:

```bash
uv run wandb login
```

## Step 1: Convert HU_D03 CSV to Motion NPZ

Example:

```bash
uv run csv-to-npz-hu-d03 \
  --input-file motions/hu_d03_walk.csv \
  --output-name hu_d03_walk \
  --input-fps 30 \
  --output-fps 50 \
  --device cuda:0 \
  --render False
```

The converter replays the CSV motion in the HU_D03 mjlab scene and creates a
MuJoCo/mjlab-compatible motion file. By default, the shared converter logic
saves:

```text
/tmp/motion.npz
```

and uploads the motion artifact to W&B as:

```text
your-entity/motions/hu_d03_walk
```

If you do not want to use W&B for training, copy `/tmp/motion.npz` to a stable
location after conversion.

## Step 2A: Train from W&B Motion Registry

Use this if the converted motion was uploaded to W&B:

```bash
uv run train Mjlab-Tracking-Flat-HU-D03 \
  --registry-name your-entity/motions/hu_d03_walk \
  --env.scene.num-envs 1024 \
  --agent.logger wandb \
  --agent.upload-model True \
  --gpu-ids "[0]"
```

For two GPUs:

```bash
uv run train Mjlab-Tracking-Flat-HU-D03 \
  --registry-name your-entity/motions/hu_d03_walk \
  --env.scene.num-envs 1024 \
  --agent.logger wandb \
  --agent.upload-model True \
  --gpu-ids "[0, 1]"
```

If you run out of GPU memory, reduce `--env.scene.num-envs` to `512` or `256`.

## Step 2B: Train from Local Motion NPZ

Use this if you have a local `motion.npz`:

```bash
uv run train Mjlab-Tracking-Flat-HU-D03 \
  --env.commands.motion.motion-file motions/hu_d03_walk.npz \
  --env.scene.num-envs 1024 \
  --agent.logger tensorboard \
  --agent.upload-model False \
  --gpu-ids "[0]"
```

## Step 3: Play a Trained Policy

Using a local checkpoint and local motion file:

```bash
uv run play Mjlab-Tracking-Flat-HU-D03 \
  --checkpoint-file logs/rsl_rl/hu_d03_tracking/<run_name>/model_XXXX.pt \
  --motion-file motions/hu_d03_walk.npz \
  --num-envs 1 \
  --viewer viser
```

Using a W&B run:

```bash
uv run play Mjlab-Tracking-Flat-HU-D03 \
  --wandb-run-path your-entity/mjlab/run-id \
  --num-envs 1 \
  --viewer viser
```

The `viser` viewer opens a local web UI, usually at:

```text
http://localhost:8080
```

## Kaggle Notes

Kaggle is useful for training, but interactive `viser` viewing is limited
because `localhost:8080` belongs to the Kaggle container. For Kaggle, prefer
rendering video files or using a manual recorder script.

Example train command on Kaggle:

```python
!cd /kaggle/working/mjlab && MUJOCO_GL=egl uv run train Mjlab-Tracking-Flat-HU-D03 \
  --registry-name your-entity/motions/hu_d03_walk \
  --env.scene.num-envs 512 \
  --agent.logger wandb \
  --agent.upload-model True \
  --gpu-ids "[0, 1]"
```

## Expected Outputs

After successful training, checkpoints are saved under:

```text
logs/rsl_rl/hu_d03_tracking/<run_name>/model_*.pt
```

If W&B upload is enabled, the model checkpoints and logs are also available in
your W&B run.

## Minimal Command Summary

Convert:

```bash
uv run csv-to-npz-hu-d03 \
  --input-file motions/hu_d03_walk.csv \
  --output-name hu_d03_walk \
  --input-fps 30 \
  --output-fps 50 \
  --device cuda:0
```

Train:

```bash
uv run train Mjlab-Tracking-Flat-HU-D03 \
  --registry-name your-entity/motions/hu_d03_walk \
  --env.scene.num-envs 1024 \
  --agent.logger wandb \
  --agent.upload-model True \
  --gpu-ids "[0]"
```

Play:

```bash
uv run play Mjlab-Tracking-Flat-HU-D03 \
  --wandb-run-path your-entity/mjlab/run-id \
  --num-envs 1 \
  --viewer viser
```

## HU_D03 Locomotion Policy

To train a locomotion policy instead of a mimic policy, use the velocity task:

```bash
uv run train Mjlab-Velocity-Flat-HU-D03 \
  --env.scene.num-envs 1024 \
  --agent.logger wandb \
  --agent.upload-model True \
  --gpu-ids "[0]"
```

For a quick smoke test:

```bash
uv run train Mjlab-Velocity-Flat-HU-D03 \
  --env.scene.num-envs 64 \
  --agent.max-iterations 10 \
  --agent.logger tensorboard \
  --agent.upload-model False \
  --gpu-ids "[0]"
```

For Kaggle T4x2:

```python
!cd /kaggle/working/mjlab && MUJOCO_GL=egl uv run train Mjlab-Velocity-Flat-HU-D03 \
  --env.scene.num-envs 1024 \
  --agent.max-iterations 4000 \
  --agent.logger wandb \
  --agent.upload-model True \
  --gpu-ids "[0, 1]"
```

If the full command range is too hard at the start, train a forward-only warmup:

```bash
uv run train Mjlab-Velocity-Flat-HU-D03 \
  --env.scene.num-envs 1024 \
  --agent.max-iterations 3000 \
  --env.commands.twist.rel-standing-envs 0.0 \
  --env.commands.twist.rel-heading-envs 0.0 \
  --env.commands.twist.rel-forward-envs 1.0 \
  --env.commands.twist.ranges.lin-vel-x "(0.25, 0.6)" \
  --env.commands.twist.ranges.lin-vel-y "(0.0, 0.0)" \
  --env.commands.twist.ranges.ang-vel-z "(0.0, 0.0)" \
  --env.curriculum.command-vel.params.velocity-stages.0.lin-vel-x "(0.25, 0.6)" \
  --env.curriculum.command-vel.params.velocity-stages.0.ang-vel-z "(0.0, 0.0)" \
  --env.curriculum.command-vel.params.velocity-stages.1.lin-vel-x "(0.25, 0.6)" \
  --env.curriculum.command-vel.params.velocity-stages.1.ang-vel-z "(0.0, 0.0)" \
  --env.curriculum.command-vel.params.velocity-stages.2.lin-vel-x "(0.25, 0.6)" \
  --gpu-ids "[0]"
```

Play a trained local checkpoint:

```bash
uv run play Mjlab-Velocity-Flat-HU-D03 \
  --checkpoint-file logs/rsl_rl/hu_d03_velocity/<run_name>/model_XXXX.pt \
  --num-envs 1 \
  --viewer viser
```
