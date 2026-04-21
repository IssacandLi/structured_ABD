# Noise-Dependent Planning for Discrete Diffusion LLMs

## Design Document v2: A Falsifiable Solid-NeurIPS-Oriented Plan

---

## 0. Executive Summary

This v2 revision keeps the strong core of v1 but makes the project substantially more submission-ready.

The central thesis remains:

> **A single-tier discrete diffusion language model can be induced to exhibit coarse-to-fine planning behavior by making its masking and unmasking granularity noise-dependent, thereby approximating key benefits of Block Diffusion without architectural hierarchy.**

However, v2 makes five decisive strategic changes:

1. **Theory is no longer a must-have.** It is optional and only worth including if it supports a genuinely informative experimental asymmetry.
2. **Falsifiability becomes central.** We no longer rely only on correlational diagnostics; we require intervention-style tests that can distinguish genuine planning-like behavior from trivial adaptation to a masking pattern.
3. **Block Diffusion comparison is upgraded.** Block Diffusion can no longer remain merely a conceptual reference. At least one matched-budget empirical comparison is required.
4. **C-inf robustness is now a first-class concern.** Since matched structured unmasking is part of the main method, its sensitivity must be characterized early.
5. **Execution is split into feasibility gate -> pilot -> scale-up.** This reduces the risk that engineering instability or decode-schedule brittleness derails the entire project.

In short, v2 shifts the paper from a clean but still somewhat vulnerable mechanism paper toward a more robust behavioral-science-style paper for diffusion LLMs: one that makes a sharp claim, exposes that claim to falsification, and measures how much of Block Diffusion's benefit is actually recovered.

---

## 1. Revised Core Thesis

### 1.1 Problem Statement

Standard discrete diffusion LLMs (dLLMs) typically use tokenwise random masking across all noise levels. This gives the model no explicit reason to separate:

- **global planning**: deciding the high-level semantic and structural content of a sequence, from
- **local realization**: deciding exact token-level surface form.

At very high noise, the model is therefore trained to reconstruct text from **scattered token remnants** rather than from **coherent local summaries**. This weakens its ability to form, preserve, and exploit high-level plans during denoising.

### 1.2 Main Hypothesis

Noise level already defines a natural coarse-to-fine axis in a dLLM. If masking and unmasking granularity are aligned with that axis, then a standard single-tier dLLM can acquire behavior analogous to hierarchical coarse-to-fine generation:

- at **high noise**, act more like a planner over coherent spans,
- at **low noise**, act more like a token-level infiller.

### 1.3 Headline Claim

The scientific claim is not merely that the masking schedule is smarter. It is:

> **Coarse-to-fine planning behavior can be induced in a standard single-tier dLLM by controlling what semantic granularity is visible and committed at each noise level.**

That is the headline. The paper must not present itself as just another masking-schedule variant.

---

## 2. Positioning Relative to Existing Work

### 2.1 What We Are Not Claiming

We are **not** claiming:

- to have invented the general idea of state-dependent masking,
- to solve planning in full generality,
- or to replace Block Diffusion outright with a minor schedule tweak.

### 2.2 What We Are Claiming

We claim something narrower but sharper:

1. **Block-style planning behavior is not purely architectural.** Some of it can be induced behaviorally within a single-tier model.
2. **Uniform tokenwise masking is a poor planning-induction mechanism.** At high noise it destroys exactly the coherent local evidence needed for global planning.
3. **Noise-dependent structured masking plus matched structured unmasking changes denoising dynamics in a measurable, planning-relevant way.**
4. **The induced behavior is not reducible to a trivial masking-pattern adaptation.** This must be tested empirically by falsification-style interventions.

### 2.3 Relation to Block Diffusion

Block Diffusion achieves coarse-to-fine generation via architectural hierarchy:

- stage 1: block-level plan,
- stage 2: token-level refinement.

Our method aims to recover part of that behavior **without** explicit block latents, a separate planner, or a second generative stage.

The intended paper-level relationship is:

- **Block Diffusion:** architectural decomposition of planning and realization,
- **our method:** behavioral induction of planning-like denoising in a single-tier model.

This makes the two approaches complementary in principle. But in the paper, that claim must be backed by at least one empirical comparison, not just positioning language.

---

## 3. Main Method: Structured Planning Induction in a Single-Tier dLLM

The core method remains deliberately narrow.

### 3.1 Component C: Noise-Dependent Structured Masking

The training corruption process transitions from **span-level masking** at high noise to **token-level masking** at low noise.

#### High-noise regime

At large masking ratios, the model sees a sparse set of coherent visible spans while large contiguous spans are masked. The task becomes:

> infer missing semantic regions from a small number of coherent local windows.

This is planning-like.

#### Low-noise regime

At small masking ratios, the model transitions toward ordinary token-level infilling, recovering lexical and syntactic detail.

#### Formal control variable

Let the masking ratio be \(r(t)\). Define a structuredness variable:

\[
 s(t) = \min\left(1, \frac{r(t)-r_{\text{low}}}{r_{\text{high}}-r_{\text{low}}}\right)^+.
\]

Then:

- \(s(t) \approx 1\): mostly span-level masking,
- \(s(t) \approx 0\): mostly token-level masking.

Span size increases with \(s(t)\).

### 3.2 Component C-inf: Matched Structured Unmasking at Inference

Training-time structured masking alone is insufficient. If inference remains purely tokenwise at all steps, the model faces a granularity mismatch.

Therefore inference should mirror training:

- **high noise:** unmask by span-level confidence or span-prioritized confidence,
- **low noise:** revert to token-level confidence-based unmasking,
- **intermediate noise:** interpolate between span-level and token-level commitment.

In v2, **C-inf is part of the method, not an implementation note**.

### 3.3 Why This Pair Is the Central Object

The pair \((C, C\text{-inf})\) is the minimum intervention required to test the main scientific question:

> can single-tier diffusion denoising be made more planning-like by changing the semantic granularity of context and commitment across noise levels?

This remains the sharpest version of the project.

---

## 4. Optional Strengthening Only: Span-Level Auxiliary Objective

### 4.1 Role of A in v2

The span-level auxiliary loss remains optional. It should only be promoted if the main method clearly benefits from it.

The logic stays simple:

- if **C + C-inf** already works, the paper is cleaner without A,
- if **C + C-inf** is unstable or too weak, A may provide a strengthening mechanism.

### 4.2 Preferred Variant

If A is used, the preferred version is:

- **A2: semantic span embedding prediction**

rather than:

- **A1: bag-of-words KL loss**

A2 better matches the intended claim: improving semantic region-level coherence rather than just lexical set recovery.

### 4.3 Experimental Use

Use the following ladder:

1. establish **C + C-inf**,
2. then test **+A2**,
3. only keep A in the main paper if gains are additive and stable.

---

## 5. What We Explicitly Remove from the Main Paper

### 5.1 Remove B from the Main Story

Noise-dependent attention routing is no longer part of the main method.

Reasons:

1. it adds systems complexity,
2. it muddies the paper's central object,
3. it is unnecessary for testing the core hypothesis,
4. attribution would become harder.

At most, B may appear as an appendix experiment or negative result.

### 5.2 Remove Overclaiming Language

Do **not** say:

- “no additional inference cost”,
- “equivalent to Block Diffusion”,
- “solves global planning”.

Use instead:

- **no architectural hierarchy**,
- **negligible additional inference overhead**,
- **approximates key coarse-to-fine planning behavior**.

---

## 6. What Must Be Proven Empirically

To be a solid NeurIPS submission, the empirical burden has three layers.

### 6.1 Layer I: Performance on planning-sensitive tasks

The method should be tested where planning clearly matters.

### 6.2 Layer II: Behavioral evidence that planning-like denoising emerged

The paper must show that denoising dynamics changed in the intended direction.

### 6.3 Layer III: Falsification-style evidence

The paper must show that the effect is not merely adaptation to a specific masking pattern or decode heuristic.

This third layer is the main upgrade from v1.

---

## 7. Falsification Suite: Making the Core Claim Harder to Dismiss

A tough reviewer may argue that the observed diagnostics are trivial consequences of structured masking rather than evidence of genuine planning-like behavior.

To address this, v2 requires explicit falsification tests.

### 7.1 Minimum intervention matrix

At minimum, run the following four settings in a pilot-scale experiment:

| Train corruption | Inference unmasking | Purpose |
|---|---|---|
| uniform | uniform | baseline |
| structured | uniform | isolate train–test mismatch |
| structured | matched structured (C-inf) | main method |
| structured | wrong-granularity / perturbed structured | falsification |

The fourth row is crucial. A few candidate perturbations:

- **shuffled span grouping**: compute span confidence over artificial spans that do not match training-time grouping,
- **wrong span scale**: train with one span scale but infer with a very different one,
- **forced tokenwise early commitment**: revert to token-level unmasking in the high-noise regime,
- **forced spanwise late commitment**: over-apply span-level commitment in low-noise steps.

### 7.2 What we want to observe

The ideal pattern is:

- `structured + matched structured` works best,
- `structured + uniform` is clearly worse,
- `structured + perturbed structured` is also clearly worse,
- the degradation is aligned with planning diagnostics, not just final score.

That would support the claim that the model has become sensitive to **semantic granularity alignment**, not merely to “more structured noise”.

### 7.3 Stronger optional test

An even stronger test is to compare whether the structured-trained model is disproportionately harmed when forced back to uniform tokenwise inference, relative to the baseline model. This is not by itself definitive, but it adds useful evidence that the learned denoising policy has changed materially.

---

## 8. Task Selection: Lock It Down Early

The v1 categories were directionally right but still too broad. For submission quality, v2 should lock tasks early.

### 8.1 Recommended three-task structure

Use exactly three main task families:

1. **One long-form or document-structured task**
   - preferably *outline/section -> article* or *multi-section completion*, not unconstrained open-ended continuation.

2. **One structured generation task**
   - preferably *docstring -> function/code* or another setting with an explicit skeleton-to-detail dependency.

3. **One synthetic or semi-synthetic planning-sensitive task**
   - designed so that coherent visible spans are genuinely more useful than scattered visible tokens.

This is enough. More than this risks diluting the story.

### 8.2 Control setting

Add one lightweight control:

- short-text generation or standard perplexity-style evaluation.

The purpose is not to win there, but to show that the method is specifically useful when planning matters.

### 8.3 Recommended selection principle

Prefer tasks where all three are true:

- a latent skeleton is meaningful,
- partially denoised states are interpretable,
- success can plausibly depend on early global commitments.

---

## 9. Denoising Diagnostics: Still Central, but No Longer Sufficient Alone

Diagnostics remain central evidence, but in v2 they are interpreted together with falsification tests.

Recommended diagnostics:

1. **Early-step global coherence**
   - does the partially denoised sequence become topically coherent earlier?

2. **Denoising order analysis**
   - are high-information structural/content tokens committed earlier than function words?

3. **Skeleton token/span survival rate**
   - do early high-noise commitments survive later refinement?

4. **Frozen-noise semantic consistency**
   - at intermediate denoising checkpoints, do predicted spans exhibit meaningful cross-span coherence?

5. **Span commitment stability**
   - once a span-level hypothesis forms, how much is it later overwritten?

These should be shown not just for the main method, but also for at least one falsification condition.

---

## 10. Block Diffusion Reference Comparison: No Longer Optional

Since the paper claims to recover part of Block Diffusion's benefit without hierarchy, at least one empirical comparison to Block Diffusion is required.

### 10.1 Minimum requirement

At least one matched or approximately matched setting should report:

- standard dLLM baseline,
- **C + C-inf**,
- a small Block Diffusion reference system.

### 10.2 What to compare

Report, at minimum:

- final task metric,
- one or two planning diagnostics,
- approximate compute, latency, or memory overhead.

### 10.3 What success looks like

The paper does **not** need to beat Block Diffusion outright. A convincing result could be:

- substantial closing of the gap over baseline,
- planning diagnostics that move toward the Block Diffusion pattern,
- much lower architectural complexity.

The key question the table should answer is:

> how much of the hierarchical gain can be recovered behaviorally?

Without this, the “approximates Block Diffusion's key benefits” claim remains too soft.

---

## 11. C-inf Sensitivity Analysis: Must-Have Robustness Check

Because matched structured unmasking is now part of the main method, its robustness must be tested early.

### 11.1 Why this matters

A reviewer may argue that performance comes from schedule overfitting rather than from a stable principle.

### 11.2 Minimal sensitivity grid

At pilot scale, test a small but informative grid over:

1. **span confidence aggregation**
   - mean,
   - top-k mean,
   - max.

2. **commitment style**
   - hard span-level commitment,
   - mixed span-token commitment.

3. **thresholding rule**
   - fixed reveal ratio,
   - adaptive confidence threshold.

### 11.3 Desired conclusion

We do not need full invariance. We need to show that the method has a **broad reasonable basin** and does not depend on a single fragile decode recipe.

---

## 12. Synthetic Benchmark Requirement

A planning paper still needs at least one benchmark where the mechanism has an exact conceptual target.

### 12.1 Why we still need it

Naturalistic tasks alone leave too much room for ambiguous interpretation:

- regularization effect,
- optimization effect,
- data alignment effect.

A synthetic or semi-synthetic task creates a cleaner causal anchor.

### 12.2 Desired property

The ideal task should have:

- a **global latent skeleton**,
- local realization conditioned on that skeleton,
- weak evidence from scattered visible tokens,
- strong evidence from coherent visible spans.

### 12.3 Example directions

1. **template-and-detail generation**
2. **topic-skeleton completion**
3. **program sketch completion**

The benchmark does not need to be fancy. It needs to make the mechanism legible.

---

## 13. Theory / Analysis Layer: Optional Only

Theory is no longer a must-have for this project.

### 13.1 Default position

The default paper version should be able to stand **without** a theorem section.

### 13.2 When theory is worth adding

Only add a small analytical section if all three are true:

1. it is directly tied to one key empirical asymmetry,
2. it is clearly interpretable,
3. it does not consume disproportionate project time.

### 13.3 When theory should be skipped

Skip it if the result is:

- too toy to be informative,
- disconnected from the main empirical story,
- or likely to distract from the behavioral evidence.

For NeurIPS, a sharper empirical package is more valuable here than a weak proposition.

---

## 14. Experimental Matrix for a Solid-NeurIPS Attempt

### 14.1 Core ablations

| Experiment | Structured masking | Structured unmasking | Span aux loss | Purpose |
|---|---:|---:|---:|---|
| Baseline | ✗ | ✗ | ✗ | Standard dLLM |
| C-only-train | ✓ | ✗ | ✗ | show train–test mismatch |
| C + C-inf | ✓ | ✓ | ✗ | main method |
| C + C-inf + A2 | ✓ | ✓ | ✓ | optional strengthening |
| C + perturbed C-inf | ✓ | perturbed | ✗ | falsification |

### 14.2 Main comparisons

Compare against:

1. **strong standard dLLM baseline**,
2. **uniform masking with matched compute and decode budget**,
3. **nearby generalized masking variant**, if available,
4. **a small Block Diffusion reference**.

### 14.3 Main success conditions

A convincing paper should ideally show:

1. **consistent gains on planning-sensitive tasks**,
2. **little or no degradation on non-planning controls**,
3. **clear changes in denoising diagnostics**,
4. **C + C-inf outperforming C-only-train**,
5. **C + perturbed C-inf underperforming matched C-inf**,
6. **at least partial closing of the Block Diffusion gap** in one reference setting.

---

## 15. Risks and How to Manage Them

### Risk 1: Gains are small and look like regularization noise

**Mitigation:**
Use planning-sensitive tasks, emphasize falsification and diagnostics, and include a Block Diffusion reference point.

### Risk 2: The effect disappears without A

**Mitigation:**
Treat A2 as a strengthening term. If it is necessary, the story becomes: structured visibility is primary, semantic span alignment stabilizes the effect.

### Risk 3: Structured training helps, but decode sensitivity is too high

**Mitigation:**
Run the C-inf sensitivity analysis early. If the basin is too narrow, simplify C-inf or reduce claims.

### Risk 4: Reviewers say this is just another masking schedule

**Mitigation:**
The response is the package itself:

- planning-induction framing,
- falsification suite,
- diagnostics,
- Block Diffusion reference comparison,
- synthetic support.

### Risk 5: Engineering feasibility problems delay the project

**Mitigation:**
Introduce a Week 0 feasibility gate before full pilot execution.

### Risk 6: Real tasks are too noisy to show a clean story

**Mitigation:**
Make the synthetic or semi-synthetic task strong enough to carry interpretive weight.

---

## 16. Recommended Paper Narrative

A strong v2 paper narrative should proceed as follows:

### Part I: Observation

Uniform tokenwise masking is poorly aligned with planning in dLLMs. At high noise it destroys coherent local evidence.

### Part II: Hypothesis

If noise level is treated as a semantic granularity axis, then a single-tier dLLM can be induced to plan earlier and realize later.

### Part III: Method

Implement this with:

- structured masking at training time,
- matched structured unmasking at inference time,
- optional semantic span auxiliary loss.

### Part IV: Evidence

Show:

- improved performance on planning-sensitive tasks,
- denoising diagnostics consistent with planning induction,
- falsification tests that rule out trivial schedule adaptation,
- one Block Diffusion reference comparison,
- one synthetic planning-sensitive anchor task.

### Part V: Conclusion

Some benefits of block-style planning are not purely architectural; they can be induced behaviorally in a single-tier diffusion model.

---

## 17. Recommended Scope for the First Submission

### Must-have

1. **C + C-inf fully implemented.**
2. **One long-form/document-structured task.**
3. **One structured generation task.**
4. **One synthetic or semi-synthetic planning-sensitive task.**
5. **Denoising diagnostics.**
6. **Falsification suite.**
7. **C-inf sensitivity analysis.**
8. **One Block Diffusion reference comparison.**
9. **Week 0 feasibility gate before pilot.**

### Nice-to-have

1. **A2 additive gain.**
2. **Human evaluation on long-form coherence.**
3. **A small analytical proposition, only if it is genuinely informative.**

### Not necessary for first submission

1. **B as a main mechanism.**
2. **Heavy mathematical theory.**
3. **Broad reasoning-task claims.**
4. **Large benchmark sprawl.**

---

## 18. Final Assessment

This v2 plan is built around the belief that the project can become a solid NeurIPS submission if and only if it satisfies the following expectations:

1. **A sharp central object:** C + C-inf.
2. **A clear scientific claim:** planning is induced, not merely performance nudged.
3. **Behavioral evidence:** denoising dynamics become more structured.
4. **Falsifiability:** the effect survives tests designed to distinguish it from trivial masking-pattern adaptation.
5. **Task-level evidence where planning matters.**
6. **At least one exact-support setting:** synthetic or semi-synthetic benchmark.
7. **At least one Block Diffusion reference comparison.**
8. **Decode robustness:** the method is not carried by a brittle C-inf recipe.

If these conditions are met, the project has a realistic path to becoming a solid main-conference paper. If not, it is more likely to be perceived as a clever but limited masking heuristic.

---

## 19. Next-Step Execution Plan

### Phase 0: Feasibility gate

Before any serious pilot, run a tiny-scale setup:

- small model,
- short sequence length,
- only **C + C-inf**,
- verify masking calibration,
- verify decode schedule correctness,
- verify training stability,
- estimate throughput overhead.

This phase answers one question:

> is the core mechanism operationally stable enough to justify the real pilot?

### Phase 1: Method locking

Freeze the main method to:

- structured masking,
- structured unmasking,
- optional A2 only if later justified.

### Phase 2: Benchmark locking

Select early:

- one long-form/document-structured task,
- one structured generation task,
- one synthetic planning-sensitive task,
- one lightweight control.

### Phase 3: Diagnostic and falsification pipeline

Implement early, not at the end:

- denoising diagnostics,
- perturbed C-inf conditions,
- C-inf sensitivity grid.

### Phase 4: Pilot study

Run a pilot to answer five yes/no questions:

1. Does **C + C-inf** beat baseline on at least one planning-sensitive task?
2. Do denoising diagnostics move in the expected direction?
3. Does matched **C-inf** clearly beat mismatched or perturbed inference?
4. Is C-inf reasonably robust to small design choices?
5. Does the method recover a meaningful fraction of the Block Diffusion gap in one reference setting?

### Phase 5: Submission decision

Proceed to full-scale paper only if the pilot yields a clearly positive signal on most of the above questions.

As a practical rule:

- if only one signal is positive, stop,
- if two or three are positive, redesign,
- if four or five are positive, scale and write.

