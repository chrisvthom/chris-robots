# Chris's robots

MuJoCo models of the humanoids I've built, each with a small walking policy trained
with [mjlab](https://github.com/mujocolab/mjlab) (MuJoCo Warp + PPO). The goal is
not sim-to-real: it's a lean, good-looking walk that can run in a browser
(mujoco-wasm + onnxruntime-web) on my portfolio.

| Robot | | DoF | Mass | Height | Policy |
|---|---|---|---|---|---|
| **K-Bot V2** (`kbot`) | <img src="media/kbot.png" width="120"> | 20 | 36 kg | 1.4 m | [`policies/kbot`](policies/kbot) |
| **Z-Bot V3** (`zbot_v3`) | <img src="media/zbot_v3.png" width="120"> | 20 | 3.3 kg | 0.40 m | |
| **Z-Bot V2** (`zbot_v2`) | <img src="media/zbot_v2.png" width="120"> | 18 | 3.8 kg | 0.48 m | |
| **Opus** (`opus`) | <img src="media/opus.png" width="120"> | 16 | 2.8 kg | 0.32 m | |
| **GPR / Stompy** (`gpr`) | <img src="media/gpr.png" width="120"> | 20 | 29 kg | 1.4 m | |

K-Bot V2 and GPR are full-size Robstride-actuated humanoids from my time at K-Scale;
the Z-Bots are the 2-ft Feetech-servo robots; Opus is the 3D-printed plastic one.
The GPR physics model is the K-Bot V1 skeleton (same mechanism); its head and shells
are a visual overlay still to be added.

## Layout

```
robots/<name>/          MJCF model (<name>.xml), standalone scene.xml, meshes, robot.json
src/chris_robots/
  robots/<name>.py      actuator gains, posture tolerances, speed ranges per robot
  env_cfgs.py           mjlab velocity task config, scaled by robot size
  rl_cfg.py             PPO config
  tools/                import-robots, render-robots, smoke-test, export-web, curves
policies/<name>/        exported bundle: scene.xml + assets + policy.onnx + policy.json + preview
media/                  renders
```

## Setup

Needs Python 3.12, [uv](https://docs.astral.sh/uv/) and an NVIDIA GPU (trained on an
RTX 5070 Ti; the lockfile pins CUDA 13 wheels for torch/warp).

```sh
uv sync
MJLAB_WARP_QUIET=1 uv run smoke-test kbot          # builds the env, steps it, prints stats
uv run list-envs | grep Chris                       # Chris-Velocity-Flat-<Robot>
```

## Train / play / export

```sh
uv run train Chris-Velocity-Flat-Kbot --env.scene.num-envs 4096 --agent.run-name v1
uv run curves kbot                                  # training curves as a table
uv run play Chris-Velocity-Flat-Kbot                # interactive viewer (latest checkpoint)
MUJOCO_GL=egl uv run export-web kbot --preview      # policies/kbot/, verified with plain mujoco + onnxruntime
```

A 5000-iteration run (4096 envs, 24 steps each) takes ~40 min on the 5070 Ti; the
runner exports an ONNX policy with every checkpoint.

## Policy bundle format

`export-web` produces a folder that only needs MuJoCo and an ONNX runtime, no mjlab:

* `scene.xml`: floor, lights, robot, one `<position>` actuator per policy action (in
  action order, with the trained kp/kd), and an `init_state` keyframe.
* `policy.onnx`: actor MLP with its observation normalizer baked in. `obs[1, N] -> actions[1, A]`.
* `policy.json`: everything needed to build the observation vector and apply actions:
  observation layout (`base_lin_vel`, `base_ang_vel`, `projected_gravity`,
  `joint_pos - default`, `joint_vel`, `last_action`, `command`), joint order and qpos/qvel
  addresses, default pose, gains, action scale, sim timestep (0.005 s, 4 substeps per
  50 Hz policy step), and command ranges.

`chris_robots.tools.export_web.BundleRunner` is a 60-line reference implementation of
the runtime; `export-web` runs it after every export to make sure the bundle walks.

## Modelling notes

* Models come from the K-Scale URDF/MJCF exports via `import-robots`, which strips
  actuators/sensors, renames the root to `base`, keeps only the sole boxes as collision
  geometry, adds mjlab's IMU sensors and foot sites, and derives a knees-bent standing
  pose from the joint axes.
* PD gains follow mjlab's G1 recipe (`kp = armature * (2*pi*10)^2`, `zeta = 2`), with
  action scale ~0.25 rad.
* Not sim-to-real, so a few motors are stronger than the real ones where the real
  limits make a 36 kg robot unable to balance (K-Bot/GPR ankle and hip roll); see the
  comments in `robots/kbot.py`.
