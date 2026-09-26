# Using choice-eval-toolkit in embodied-AI evaluation

The toolkit is domain-agnostic by design — it audits *slots, labels and
orders*, never content — so it plugs into any embodied evaluation that can be
expressed as (a) a multiple-choice answer or (b) a pairwise preference. This
page maps the common embodied evaluation formats onto the two input schemas
and points out which audits matter most in each.

## Format 1 — MCQ-style embodied evaluation

**Formats covered:** embodied QA (OpenEQA-style), spatial reasoning probes,
VLM/VLA action selection among discretized candidates, "which step went
wrong?" trajectory debugging, any rollout-expensive decision approximated by
a cheap multiple-choice probe.

Schema mapping (`question_id` … `variant_id`, see the README):

| Embodied concept | Field | Notes |
|---|---|---|
| Candidate actions / objects / steps | `option_ids` | stable per-content id, e.g. `q12#action_grasp_red` — never the presentation letter |
| Action description length | `option_lengths` | `len(description)` — length bias is realistic here: longer action texts often sound more competent |
| Correct next action | `gold_index` | position **in this presentation order**; recompute after every shuffle |
| Policy's chosen action | `selected_index` | `null` if the policy failed to answer |
| Policy confidence | `confidence` | logprob-derived or verbalized, [0, 1] |
| Candidate-order replicate | `variant_id` | `canonical`, `shuffle_1`, … |

Which audits matter most here:

- **Position bias (marginal + gold-offset).** Candidate orders come from
  samplers, so they are arbitrary — a slot preference in the policy directly
  distorts which action is "chosen", and with an imbalanced answer key the
  marginal test alone can mislead (the report says *inconclusive* rather
  than guessing).
- **Ordering consistency.** Reshuffle the candidates, ask again: a policy
  that picks a different action when the same menu is reordered is not
  executing a stable decision — on hardware that is a safety property, not
  a curiosity.
- **Selective prediction with the risk ladder.** Embodied errors have
  asymmetric costs. The abstention audit reports operating points at several
  risk levels (5/10/15/20/30%); pick the one your cost structure implies,
  and route low-confidence decisions to human teleoperation or a safe
  fallback policy.
- **Gold-length artifact.** If the correct action tends to be the longest
  description (common when annotators elaborate the right answer), models
  can score well on the heuristic alone. The dataset-side audit catches it
  before you ship the benchmark.

## Format 2 — Pairwise trajectory preferences (arena-style)

**Formats covered:** RoboArena-style A/B judgments of two trajectories or
policies, LLM-judged plan comparisons, human preference studies over
demonstrations.

Schema (`pair_id`, `content_a_id`, `content_b_id`, `slot_of_a`,
`selected_slot`, …, see the README):

```json
{"pair_id": "task17", "content_a_id": "policy_v3_traj9",
 "content_b_id": "policy_v4_traj2",
 "slot_of_a": 0, "selected_slot": 1, "confidence": 0.7,
 "context_lengths": [220, 185], "variant_id": "ba"}
```

- `slot_of_a`: which slot (0 = shown first) content A occupied in this
  judgment; `selected_slot`: slot of the winner, `null` = tie/undecided.
- Judge each pair under **both** presentation orders to unlock the swap audit.

Which audits matter most here:

- **Slot preference** — do judges pick whatever is shown first? This is the
  embodied analog of first-answer bias and it is well documented in human and
  LLM judging. The slot-balance check tells you whether the preference is
  confounded by a non-randomized design.
- **Swap consistency** — present the same pair with A/B swapped: does the
  judge flip? Unstable judging means your leaderboard ordering is partly an
  artifact of presentation order.
- **Corrected leaderboard** — per-content win rates split by as-first /
  as-second; with a balanced design the overall rate is order-corrected, and
  large first/second gaps flag residual order effects per content.
- **Length preference** — judges (human and LLM) favor longer trajectory
  descriptions; the report quantifies P(chosen is longer) against the coin
  flip.

## What the toolkit does NOT audit in embodied settings

- Rollout success rates / SPL / continuous control metrics — no slots, no
  choices; the tool has no object to audit there. (Predicted-success-probability
  vs outcome *is* a calibration question, but the tool's data model is
  choice-shaped — don't shoehorn it.)
- Sim2real gaps, hardware variance, physics randomness — different noise
  sources entirely.
- Temporal-step bias *within* a free-form trajectory (as opposed to
  "which step?" MCQs, where the steps are the options and are fully covered).

## Quick start with synthetic embodied-flavored data

```bash
python -m choice_eval demo-pairwise --out examples --n 1000
# compare examples/clean/report.md vs examples/biased/report.md
```

The biased run injects a first-slot preference (20%), a length preference
(30% of undecided judgments), order-sensitive verdicts (40% re-rolled on
swap) and an 8% tie rate — every number is recoverable against the closed
forms in `choice_eval.generators`.
