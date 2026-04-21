# Noise-Dependent Planning for Discrete dLLMs
## Experiment Plan v1: Phase 0 Parallel Tracks -> Pilot -> Scale-Up

## 0. Decision Summary

This v1 plan keeps the core thesis fixed and only sharpens execution.

- **core method:** `C + C-inf`
- **optional strengthening:** `+A2` only if needed
- **must-have evidence:** diagnostics, falsification, one Block Diffusion reference comparison, one synthetic anchor task
- **primary goal:** generate a fast, interpretable go/no-go signal for whether the project has a realistic path to a solid NeurIPS submission

Relative to v0, this version makes six practical upgrades:

1. the task suite is now more concrete,
2. the `outline -> story` pipeline has an explicit prototype path,
3. the synthetic topic-skeleton task has a tighter prototype specification,
4. Phase 0 is split into two **parallel tracks**,
5. Phase 0 exit criteria are now quantified,
6. seed policy and Block Diffusion codebase evaluation are made explicit.

---

## 1. Strategy

### 1.1 Governing principle

Do **not** optimize for breadth. Optimize for a clean causal story.

The fastest way to kill the project is to spread it across too many domains, too many tasks, or too many degrees of freedom before the core signal is known.

### 1.2 Recommended strategic choice

Use a **single text-generation domain** for the first serious pilot rather than mixing text generation and code generation immediately.

Reason:

- the scientific claim is about **planning induction in denoising dynamics**, not about cross-domain generality;
- a single-domain setup makes diagnostics comparable across tasks;
- it avoids turning the project into two separate engineering pipelines;
- it makes the synthetic benchmark easier to align with the naturalistic tasks.

### 1.3 Paper-level bet

The paper should win by showing the following package:

1. `C + C-inf` improves at least one planning-sensitive naturalistic task;
2. denoising diagnostics move in the intended direction;
3. matched `C-inf` beats mismatched or perturbed inference;
4. the effect is very clear on a synthetic topic-skeleton task;
5. the method recovers a nontrivial fraction of the Block Diffusion gain at matched training FLOPs.

That package is enough. More benchmark sprawl is not necessary for the first serious submission attempt.

---

## 2. Locked Task Suite (3 + 1)

## 2.1 Primary naturalistic task: WritingPrompts prompt -> story

**Choice:** WritingPrompts story generation.

**Why this task:**

- it is naturally long-form;
- it has an obvious global-to-local writing process;
- it was originally introduced in a hierarchical story-generation setting, which is exactly the kind of planning-sensitive regime this paper wants to target.

**Operational use:**

- train / finetune on WritingPrompts-style prompt-story pairs;
- evaluate on long story generation;
- use it as the main task for the Block Diffusion reference comparison.

### Official pilot metric stack

Use a **co-primary** metric stack rather than relying on MAUVE alone.

- **co-primary metric 1:** MAUVE
- **co-primary metric 2:** cross-segment coherence score
- **secondary:** prompt-story embedding relevance
- **secondary:** entity / topic consistency proxy over segments

### Cross-segment coherence score

This metric is meant to capture the paper's actual thesis more directly than MAUVE.

Suggested implementation:

1. split each generated story into 4-6 macro-segments,
2. encode each segment with a fixed sentence/document encoder,
3. compute average adjacent-segment similarity and non-adjacent consistency,
4. optionally normalize by the same quantity on reference stories.

This should serve as the main coherence-oriented metric for pilot decisions.

## 2.2 Structured generation task: outline -> story (derived from WritingPrompts)

**Choice:** derived outline-to-story task built from the same WritingPrompts corpus.

**Why not code for the first pilot:**

- code introduces a second domain, separate tokenizer / data assumptions, and different evaluation infrastructure;
- if the signal is weak, you will not know whether the method failed or the cross-domain setup diluted the story;
- a same-domain structured task is cleaner for the paper's planning claim.

### Construction pipeline: explicit prototype path

The extraction step is not assumed trivial. Phase 0 must prototype it.

For each story:

1. split the story into **4-6 contiguous macro-segments**;
2. derive one short beat / segment summary per segment;
3. concatenate the beats into an outline;
4. use `prompt + outline` as input and the full story as target.

### Default prototype policy

Use a **two-stage practical pipeline**:

- **Stage A: simple deterministic prototype**
  - segment by paragraph boundaries if available, otherwise by equal-length chunking,
  - use the **first sentence** of each segment as a crude beat,
  - filter obvious failures manually on a small batch.

- **Stage B: improved automatic outline extraction**
  - after the deterministic prototype is working, optionally replace the beat extractor with an LLM summarizer or a better extractive rule,
  - but do not block Phase 1 on this improvement.

### Phase 0 deliverable for outline pipeline

Generate and inspect at least **100 outline-story pairs**.

Pass if:

- most outlines preserve correct segment order,
- most outlines contain semantically recognizable beats,
- the task is neither trivial copying nor hopelessly lossy.

### Primary evaluation

- outline coverage / adherence,
- final story coherence,
- segment order preservation,
- the same long-form metric stack used above.

## 2.3 Synthetic anchor task: topic-skeleton completion

**Choice:** topic-skeleton completion.

**Why this one:**

- easiest to generate at scale;
- easiest to define a latent global skeleton;
- easiest to test whether coherent visible spans reveal more about that skeleton than scattered random tokens;
- easiest to interpret if the method works.

### Prototype specification: explicit default instantiation

Each example is generated from a latent topic sequence:

- number of topics per example: **4-8**,
- each topic controls one contiguous span,
- span length per topic: **80-160 tokens** by default,
- total sequence length: start with **512-1024 tokens** in pilot-scale setups.

Each topic has:

- a topic-specific lexical pool,
- a small set of topic-specific event templates,
- optional shared background vocabulary,
- stochastic surface realization.

### Lexical pool construction

Use a **hand-designed pool** in Phase 0, not corpus clustering.

Rationale:

- faster to prototype,
- easier to inspect,
- easier to ensure the latent skeleton is actually clean.

Each topic should include:

- 15-30 topic-indicative content words,
- 5-10 optional stylistic or role words,
- some overlap across topics so the task is not degenerate.

### Transition template policy

Use **hard-coded but varied** transition templates in Phase 0.

For each topic boundary, sample from a small library of transition forms such as:

- continuation,
- contrast,
- escalation,
- scene shift,
- return / callback.

This keeps the generator controllable while avoiding a single rigid pattern.

### Phase 0 deliverable for synthetic task

Generate **100 synthetic examples** and manually inspect them.

Pass if:

- the latent topic skeleton is clearly visible to a human reader,
- span boundaries are meaningful rather than arbitrary,
- random-token visibility is obviously weaker than coherent-span visibility,
- the examples do not collapse into trivial bag-of-words recognition.

### Evaluation

- topic-skeleton recovery accuracy from partially denoised states,
- final generation topic-order correctness,
- segment-level coherence,
- robustness gap between coherent-span visibility and random-token visibility.

## 2.4 Control task

Use one lightweight short-text control:

- short continuation or standard perplexity-like evaluation on short slices.

Purpose:

- verify the method does not obviously help where planning is weak;
- support the claim that gains are specific to planning-sensitive regimes.

---

## 3. Core Experimental Matrix

## 3.1 Must-run systems

Run these four first:

1. **Baseline** = uniform train + uniform inference
2. **C-only-train** = structured train + uniform inference
3. **Main** = structured train + matched `C-inf`
4. **Perturbed main** = structured train + perturbed `C-inf`

This exactly matches the paper's core scientific question.

## 3.2 Perturbed `C-inf`: lock two must-run perturbations

Do **not** run all candidate perturbations in the first pilot.

Lock these two:

### Perturbation A: forced tokenwise early commitment

- in the high-noise regime, revert to tokenwise unmasking;
- keep the rest unchanged.

**What it tests:** granularity mismatch.

### Perturbation B: shuffled span grouping

- keep span-level confidence aggregation,
- but compute span groups using boundaries inconsistent with training-time grouping.

**What it tests:** grouping mismatch.

These two are enough for the first pilot.

## 3.3 Optional system, only if needed

5. **Main + A2** = structured train + matched `C-inf` + semantic span auxiliary loss

Only activate this arm if the main system shows the right diagnostics but insufficient task-level effect size.

---

## 4. `C-inf` Sensitivity Plan

The goal is not exhaustive tuning. The goal is to test whether the method has a broad reasonable basin.

## 4.1 Lock the default `C-inf`

Use this as the default decode rule:

- **span confidence aggregation:** mean
- **commitment style:** mixed span-token
- **threshold rule:** fixed reveal ratio per step

This is the reference implementation.

## 4.2 Minimal sensitivity checks

Run one-factor changes around the default:

1. aggregation: `mean -> top-k mean`
2. commitment: `mixed span-token -> hard span`
3. thresholding: `fixed reveal ratio -> adaptive confidence threshold`

No full `2 x 2 x 2` grid is needed in the first pilot.

---

## 5. Block Diffusion Comparison Plan

## 5.1 What “matched” means

Use **training FLOPs** as the primary matching criterion.

Also report, as secondary descriptors:

- parameter count,
- denoising / generation steps,
- inference wall-clock for a fixed output length,
- approximate memory footprint if available.

This is the cleanest comparison rule and the least vulnerable to reviewer complaints about cherry-picking.

## 5.2 Minimum Block Diffusion comparison

Do the Block Diffusion reference comparison on **one task only** in the first serious pilot:

- **WritingPrompts prompt -> story**

Report:

- baseline dLLM,
- `C + C-inf`,
- small Block Diffusion reference.

Do **not** require the full task suite for Block Diffusion at the pilot stage.

## 5.3 Block Diffusion implementation path: resolve in Phase 0

Phase 0 must explicitly evaluate the Block Diffusion engineering path.

Check, in order:

1. whether there is usable public code,
2. whether there is a pretrained checkpoint suitable for quick inspection,
3. whether finetuning on WritingPrompts is realistic,
4. whether training from scratch is infeasible for the pilot budget.

### Default policy

Prefer, in order:

1. **existing code + finetune**, if available and stable,
2. **existing code + reduced-scale retraining**, if finetune is not enough,
3. **defer full BD comparison to Phase 2 only if codebase inspection is clearly negative**.

Do **not** commit blindly to reimplementing a full two-stage pipeline from scratch before the codebase check is done.

## 5.4 What counts as success

The project does not need to beat Block Diffusion.

A strong enough result is:

- `C + C-inf` materially beats the baseline,
- closes a nontrivial fraction of the BD gap,
- and preserves a simpler single-tier architecture.

---

## 6. Quantitative Go / No-Go Bars

These thresholds are intentionally approximate but must be fixed **before** the pilot is run.

## 6.1 Seed policy and statistical rule

Preferred policy:

- run **at least 2 seeds** for all core pilot systems,
- judge go/no-go using **mean performance**, not a single run.

If compute only allows **1 seed** in the first pass, then tighten the decision bars:

- raise the naturalistic-task relative-improvement bar,
- treat marginal wins as non-actionable,
- do not scale based on a noisy single-run bump.

## 6.2 Primary paper-go bars

Proceed to scale-up only if **all** of the following are met:

### Bar 1: naturalistic task gain

On the primary naturalistic task, `C + C-inf` must achieve either:

- **>= 5% relative improvement** on a co-primary metric over baseline across at least **2 seeds**, **or**
- a statistically stable win on **2 of 3** predeclared generation metrics across at least **2 seeds**.

If only **1 seed** is available in the initial pilot, require:

- **>= 8% relative improvement** on a co-primary metric,
- or a visibly strong win on both co-primary metrics.

### Bar 2: synthetic-task gain

On topic-skeleton completion, `C + C-inf` must reduce skeleton error by **>= 20% relative** versus baseline.

If this bar is not met, the planning claim is too weak.

### Bar 3: falsification separation

The matched `C-inf` system must beat the perturbed system by a meaningful margin.

Let:

- matched gain over baseline = `G_main`
- perturbed gain over baseline = `G_pert`

Require:

- `G_pert <= 0.7 * G_main`

In words: at least **30% of the main gain must disappear** under mismatch.

### Bar 4: train-inference alignment

`C + C-inf` must clearly beat `C-only-train`.

Require either:

- **>= 3% relative** advantage on a co-primary metric, or
- consistent advantage across both main naturalistic and synthetic tasks.

### Bar 5: BD gap closure

On the WritingPrompts comparison setting, `C + C-inf` must recover at least **25% of the Block Diffusion gap** over baseline on a co-primary metric.

That is enough for the paper story.

### Bar 6: engineering sanity

Core overhead must remain bounded:

- training throughput slowdown <= **15%**,
- decode-time overhead <= **10%** relative to baseline,
- no major stability failures.

If the method only works with large overhead, the simplicity story weakens too much.

---

## 7. Diagnostics to Implement Early

These should be implemented before the serious pilot finishes.

## 7.1 Must-have diagnostics

1. **Denoising order analysis**
   - Are higher-information tokens / spans committed earlier?

2. **Skeleton span survival**
   - Do early commitments survive later refinement?

3. **Frozen-noise semantic consistency**
   - At fixed denoising checkpoints, are span predictions already globally coherent?

4. **Cross-span topic consistency**
   - Particularly important for the synthetic topic-skeleton task.

## 7.2 Decision rule for diagnostics

At least **2 of the 4** diagnostics should move in the intended direction with clear separation between:

- baseline,
- main,
- at least one falsification condition.

If the final metric improves but diagnostics remain flat, the story becomes much weaker.

---

## 8. A2 Contingency Plan

This must be decided in advance, not after the results arrive.

## 8.1 Path A: clean story

If `C + C-inf` alone meets the main bars:

- keep A2 as a minor strengthening result or appendix ablation;
- the paper headline stays entirely on planning induction via granularity control.

## 8.2 Path B: amplified story

If `C + C-inf` shows the right diagnostics but only moderate end-task gains, and `+A2` materially boosts performance:

- promote A2 into the main paper,
- but keep the narrative as:
  - structured visibility and commitment are the primary mechanism,
  - semantic span supervision amplifies or stabilizes the induced planning behavior.

Avoid reframing the paper as an auxiliary-loss paper.

## 8.3 Stop condition

If neither `C + C-inf` nor `C + C-inf + A2` meets the synthetic and falsification bars, stop scaling and redesign.

---

## 9. Execution Timeline

## Phase 0: Feasibility gate with parallel tracks (3-5 days)

### Goal

Answer one question:

> Is `C + C-inf` operationally stable enough to justify a real pilot?

### Setup

- tiny model,
- short sequence length,
- only baseline vs `C + C-inf`,
- no A2 yet.

### Track A: model / training feasibility

Deliverables:

1. mask-ratio calibration check,
2. stable training curves,
3. working structured decode,
4. throughput overhead estimate.

### Track B: data / evaluation pipeline

Deliverables:

1. outline-to-story prototype,
2. synthetic task prototype generator,
3. metric code for MAUVE plus cross-segment coherence,
4. Block Diffusion codebase inspection memo.

### Explicit exit criteria

Advance only if **all** are met:

1. **mask-ratio calibration:** empirical masking ratio deviates from target `r(t)` by **<= 2% absolute** on average;
2. **training stability:** no divergence / NaN / catastrophic instability in the first pilot-scale training window;
3. **decode correctness:** structured decode produces sensible partial denoising trajectories and no obvious implementation bug;
4. **throughput:** training overhead <= **15%**, decode overhead <= **10%**;
5. **outline prototype:** at least **100** derived outline-story pairs pass a quick manual sanity check;
6. **synthetic prototype:** at least **100** examples show a readable latent skeleton;
7. **BD feasibility memo:** codebase path is known well enough to decide whether Phase 2 BD comparison is realistic.

## Phase 1: Core pilot (7-10 days)

### Systems

- baseline,
- `C-only-train`,
- `C + C-inf`,
- `C + perturbed C-inf`.

Preferred: run **2 seeds**.

### Tasks

- WritingPrompts prompt -> story,
- outline -> story,
- topic-skeleton completion,
- short-text control.

### Deliverables

1. co-primary task metrics,
2. synthetic-task metrics,
3. diagnostics,
4. falsification results,
5. initial `C-inf` sensitivity checks.

## Phase 2: Reference pilot (5-7 days)

### Run only if Phase 1 is positive

Add:

- small Block Diffusion reference on WritingPrompts,
- optional `+A2` if needed.

### Deliverables

1. BD gap-closure table,
2. final pilot decision,
3. paper narrative choice: Path A or Path B.

## Phase 3: Scale-up (only if pilot passes)

### Scale dimensions

- larger model,
- longer sequence length,
- slightly broader evaluation,
- optional light human evaluation.

Do not scale before the pilot bars are met.

---

## 10. Immediate Next Actions

1. build the deterministic outline-extraction prototype and inspect 100 examples,
2. build the topic-skeleton generator and inspect 100 examples,
3. implement baseline vs `C + C-inf` for Phase 0,
4. implement MAUVE plus cross-segment coherence metric code,
5. inspect the Block Diffusion codebase and write a short feasibility note,
6. pre-register the go/no-go bars and seed policy,
7. enter Phase 0.

If the pilot passes, you have a realistic solid-NeurIPS path.
If it does not, you will know **why** it failed:

- no synthetic signal,
- no falsification separation,
- no task-level gain,
- too much decode brittleness,
- or no practical BD reference path.

That is exactly what a good experiment plan should buy you.
