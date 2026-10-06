# Experiment log

The protocol was committed on `main` before tuning (`9a8486a`). Development records live under `results/development/`; they are not used in the README timing table. Final cohorts use the fixed seeds and budgets in [PROTOCOL.md](PROTOCOL.md). Entries below retain failed ideas as well as successful ones.

## Setup

- Pinned Python 3.12, NumPy 2.2.6, Gymnasium 1.2.1, Torch 2.8.0 CPU and Stable-Baselines3 2.7.0.
- The first isolated Box2D build failed because SWIG was not in its build environment. Adding `box2d-py = ["swig==4.3.1"]` to uv's extra build dependencies made `uv sync --python 3.12` succeed. No OS package installation or paid compute was required. This is an installation result, not a training-speed measurement.

## Candidate definitions before development measurements

- CPU PPO uses NumPy batch dynamics, one Torch thread, a 32-unit single-layer actor and critic on CartPole, and 64-by-64 separate networks on Acrobot. Rollout batch sizes are 512 and 8,192, with optimizer minibatches 512 and 1,024. Unlike the CleanRL baseline it bootstraps time-limit truncations but ends GAE propagation at every episode boundary. It uses fixed physical observation scaling on Acrobot, not evaluation-fitted statistics.
- Planned comparisons of the same PPO candidate: Gymnasium dynamics instead of vectorized dynamics; a smaller environment/optimizer batch; and wider networks. These are changes to try, not claimed improvements. Each measured attempt will be recorded below.
- ARS uses symmetric affine-policy perturbations and a standard-deviation-scaled update. CEM fits an elite Gaussian population. Both use four common-seed training episodes per candidate. Fixed velocity scales are disclosed rather than called the original ARS observation normalization.
