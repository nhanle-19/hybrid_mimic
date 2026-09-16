# Atlas reference-conditioned contact policy

This is a preliminary continuous contact approximation. QP feasibility alone
is not tracking success; closed-loop validation remains required.

## Policy and observations

The existing candidate bodies remain `left_foot_link`, `right_foot_link`.
There are 14 raw policy outputs, seven per foot in left/right order:

- Support activation: `c = clip(u, 0, 1)`, allowing exact zero and one.
- Six motion weights: `W = diag(w_min + (w_max-w_min)*sigmoid(u[1:7]))`.
  Defaults are `w_min=0`, `w_max=100`. Components are linear XYZ, angular XYZ.

The actor sees **only reference motion** at offsets 0, 1, 5, and 10 frames:
reference joint positions/velocities and body positions, orientations, linear
and angular velocities. Future indices clamp at the last frame. At 50 Hz the
lookahead is 0, 20, 100, and 200 ms. There are no actual-state transforms,
previous actions, or sensor values in the actor input.

The critic receives that reference window plus the inherited privileged robot
state, tracking information and previous actions. Previous two- and 29-output
Atlas checkpoints are incompatible; train a new policy.

## Motion objective and dynamics

For each candidate body, including one without support, the QP adds

```
a_target_i = a_ref_i + diag(Kv)*(v_ref_i-v_i)
r_i = Ji*nudot + Jdot_i*nu - a_target_i
L_contact = 0.5 * r_i.T * Wi * r_i
```

All six stationary-contact equalities are removed. `Ji`, velocity and
acceleration are expressed in **world-aligned axes at the body-frame origin**,
with spatial ordering `[linear, angular]`. Reference acceleration is a centered
finite difference of reference body spatial velocity, using the motion's FPS;
endpoints use one-sided differences. `Kv=(10,10,10,5,5,5)` by default.
These are soft body-motion requests, not assertions of physical contact.

The QP retains centroidal momentum tracking, posture tracking, pelvis tracking,
regularization, rigid-body dynamics, and torque limits. The existing rho force
coefficients remain the force decision variables. No contact-motion slack
variables are introduced. Actuator stiffness, damping, armature and friction
are zero; nonzero actuator PD gains are rejected, so QP bounds cover the entire
commanded torque.

## Support geometry and force capacity

The previous controller used a manually specified full-sole rectangle.
Inspection of the USD found one box collision shape per foot: approximately
0.223 by 0.100 by 0.030 m, centered at (0.010, 0, -0.015) in the foot frame.
`scripts/export_atlas_contacts.py` exports its bottom-face vertices and source
hashes to `atlas_contacts.json`. The existing four-corner rho representation now
uses those collider-derived vertices; no new contact bodies or hand-placed sole
locations are added.

Actual body poses, not reference activation or reference contact labels, decide
which vertices may carry force. The floor is the existing horizontal USD box,
with top z=0 and bounds ±100 m. Vertices must be within 1 mm above the floor and
no more than 3 cm penetrating it. Of those, only vertices within 10 micrometers
of the lowest vertex height are retained. Thus a tilted foot uses its low edge
or corner, not full-foot moment arms. Side/inverted contact is unsupported
(foot normal must be within 60 degrees of upright). Contact overrides may
remove support but cannot create unavailable support.

For every supported foot:

```
0 <= Fn_i = sum(rho_i) <= c_i * Fmax_i
rho_i >= 0
```

`Fmax=(600,600)` N initially. Each friction ray has unit vertical component, so
the normal-force expression uses the existing rho mapping exactly. The friction
pyramid remains inside the Coulomb cone (`mu=0.6`). Moments arise only from
forces at available collider vertices. Unsupported physical coefficients are masked to zero;
`c=0` or unavailable geometry gives exactly zero wrench. This preserves the
admissible moment restrictions while reducing edge/corner support capability.

The batched backend uses one fixed-size QP batch across all environments,
including different support activations and vertex masks. Each problem has
29 acceleration variables and 32 latent force coefficients (61 total), six
centroidal-balance equalities, and 80 inequality rows. Batched QR eliminates
the six equalities, giving a 55-variable solve with 80 rows for every environment.
There are no Python loops over contact patterns or environments in this backend.

Physical force coefficients equal the latent coefficients multiplied by the
availability mask. Disabled columns have no physical effect and retain an
independent quadratic regularizer; their optimum is zero. Inactive inequality
rows become `0 <= 1`, preserving a strict interior rather than imposing opposing
zero-capacity bounds. Zero rows retain unit scaling in the solver. Torque,
friction, support-force capacity, and physical residual checks remain in force.
The OSQP reference backend remains a sequential CPU solve.

Limitations: flat terrain, a conservative rigid box bottom face, no side/rolling
contact, finite contact-gap tolerance, no impulse/contact-transition model,
and no guarantee that PhysX's distributed contact matches the predicted patch.
The geometry mask itself changes discretely. This is not validated general
slipping or rolling control.

## Rollout and rewards

Defaults remain **1,024 environments**, batched CUDA dynamics/QP, 500 Hz physics
and 50 Hz policy/QP. Each new policy action triggers one QP solve; its torque
is held for all ten physics substeps. Dynamics and QP assembly are skipped
during those held substeps. Resets clear the affected torque buffers and
request a fresh solve. This applies to both batched and OSQP backends.
`backend=osqp` uses the same assembly with an explicit CPU reference solve.

All six inherited motion-tracking rewards and fall terminations are restored.
Existing penalties remain: action changes -0.1, joint limits -10, undesired
contacts -0.1, and foot-to-foot collision -0.2. Excessive slip adds -1 times
squared tangential body-origin speed above 0.05 m/s, gated by measured ground
normal force above 1 N. There is no feasibility-only reward.

On a failed numerical solve, the action applies joint damping torque
`clip(-2*qdot, -tau_max, tau_max)` and records the failure. This fallback is
bounded but does not promise balance or dynamically feasible contact forces.
Failures are counted in training logs and latched per environment until reset.
Any infeasible, nonconverged, nonfinite, or residual-invalid solve terminates
that environment at the next policy boundary, even if a later decimation solve
succeeds. Bounded damping remains active until that reset.

Atlas rewards retain body pose/velocity imitation, joint-position limits, and
the separate foot-to-foot collision penalty. Global trunk pose rewards remain
intentional: body pose targets are anchor-aligned and do not fully constrain
global translation/heading. Action smoothness is the mean squared change of
the two clipped support activations and twelve motion weights normalized to
their configured ranges. Undesired contacts above 1 N apply only to Trunk, H1,
and H2; limb contacts may provide support. This reward allowance does not add
hand/knee contacts to the QP's foot-only support model.

Slip uses rigid-body velocity at measured foot-ground contact points, projected
onto each contact tangent plane. Per-foot excess speed above 0.05 m/s is squared
and averaged by normal force, gated at 1 N total support force. A stationary
toe pivot is therefore unpenalized. Non-timeout terminations receive a -10
event penalty, with reward-manager timestep scaling canceled explicitly.

Diagnostics record predicted and actual contact wrenches in world axes at the
foot origin, tracking RMSE, support activation, geometry masks, all weights,
solver failures, commanded/applied torque, and torque saturation. Actual wrenches
sum normal and friction forces and their measured contact-point moments. A
custom ground-only contact sensor allocates detailed contact buffers on the
installed Isaac Lab version. Measurements are paired with the preceding physics
step prediction; reset-crossing samples are dropped.

## Validation and commands

Offline manual standing and four-second slow right-foot-lift checks:

```bash
python scripts/check_atlas_contact_policy.py
python -m pytest tests/test_atlas_gpu.py tests/test_atlas_qp.py tests/test_atlas_contact.py tests/test_atlas_standing.py tests/test_exporter_metadata.py -q
```

These use prescribed kinematic snapshots, not realized simulator trajectories.
The validation report is `eval_data/atlas/reference_contact_validation/`.

On a host with a working GPU, test manual standing before training:

```bash
python scripts/rsl_rl/eval_atlas.py --task Standing-Atlas-T1-v0 \
  --motion_file retargeted_motion/g18_push_kick_right_t1_training.npz \
  --qp_only --manual_support 1 1 --manual_weight 50 \
  --num_envs 1 --steps 200 --device cuda:0 --headless
```

For a slow-lift rollout supply a validated slow-lift reference clip to
`Tracking-Atlas-T1-Eval-v0`; the QP still follows actual geometry. The offline
slow-lift scenario above is not a substitute for this closed-loop test.

Short training smoke command:

```bash
python scripts/rsl_rl/train.py --task Tracking-Atlas-T1-v0 \
  --motion_file retargeted_motion/g18_push_kick_right_t1_training.npz \
  --num_envs 32 --max_iterations 2 --device cuda:0 --headless \
  --logger tensorboard --run_name reference_contact_smoke
```

The current host has no working NVIDIA driver. The smoke attempt failed during
Isaac Sim startup; no successful training rollout or GPU throughput result is
claimed. Do not launch a long run until manual closed-loop standing/lift and the
short training test pass on a working simulator host.
