# Public Real-data Sim-to-Real Workflow Status

## Bootstrap identity

- Branch: `exp/public-sim2real-v1`
- HEAD: `568ebb5480b487340172230fa9490b017141d339`
- Frozen simulator: `e79e656ac00f7847b98db9e12bf43b093f3afff1`
- Genesis: `1.2.2 + HDR overlay`
- Current stage: **bootstrap only**

## Workflow goal

For the same Genesis physical trajectory and labels, generate events independently
with Ours and V2E, train with exactly the same model and training budget, and
evaluate only on public real event data.

Real test data must not be used to tune the simulator or the synthetic
distribution.

## Planned experiment layers

### Primary

- Task: event optical flow
- Protocol: synthetic training → real MVSEC evaluation
- Candidate model: STE-FlowNet
- The final model and protocol remain to be frozen by the Sim-to-Real research chat.

### Complementary

- E2VID → HS-ERGB

### Optional

- 6DoF pose → E-POSE

## Fairness principles

- same trajectory
- same labels
- same model
- same optimizer
- same training budget
- same seeds
- only the event generator differs

## Prohibitions

- No real-test tuning.
- No privileged-information leakage.
- No post-hoc synthetic-distribution tuning.
- No claim of results at bootstrap stage.

## Scope guardrails

- This workflow does not modify the frozen simulator core.
- This bootstrap does not download MVSEC, HS-ERGB, or E-POSE.
- This bootstrap does not run network training or formal real-data testing.
- `benchmark/sim2real/results/` is reserved for small summaries; large data
  belongs in an external data directory.

## Next gate

Proceed only after the Sim-to-Real research chat freezes the first formal
pipeline:

> 由 Sim-to-Real Chat 冻结第一个正式 pipeline
