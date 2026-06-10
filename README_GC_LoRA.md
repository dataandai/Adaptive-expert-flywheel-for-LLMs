# GC-LoRA: Gain-Controlled LoRA Expert Pool for Online Adaptation

**A controlled notebook prototype for online adapter adaptation with a frozen LLM and a gain-weighted pool of parallel LoRA experts.**

This repository explores a small but interesting engineering question:

> Can a frozen language model adapt to a changing input stream through a population of lightweight LoRA experts, while limiting harmful adaptation to noisy or low-quality inputs?

The current prototype attaches parallel LoRA expert branches to selected transformer layers of a frozen `Qwen/Qwen2.5-0.5B` backbone. Each expert contributes an adapter update in the same forward pass. A gain-softmax mixture controls how much each expert contributes. A simple garbage-collection loop can remove weak experts and replace them with cloned-and-perturbed copies of stronger experts.

This is a notebook prototype, not a solved continual-learning system.

---

## What this is

This repository is:

- a small online-adaptation experiment,
- a frozen-backbone LoRA prototype,
- a gain-weighted parallel expert pool,
- a lifecycle experiment with expert kill / clone / perturb operations,
- a controlled synthetic-stream notebook for inspecting expert dynamics.

The goal is to make the mechanism visible and easy to modify.

---

## What this is not

This repository is **not**:

- a production continual-learning architecture,
- a self-improving LLM system,
- a proof of stable lifelong learning,
- a replacement for retrieval, fine-tuning, or memory systems,
- a learned MoE router,
- exact per-expert credit assignment,
- evidence that the method generalizes beyond the toy stream.

The current system should be understood as an inspectable prototype for studying online LoRA expert dynamics.

---

## Core idea

A frozen language model provides a stable base.

Small LoRA experts provide plastic adaptation.

A gain controller decides how strongly each expert contributes.

A garbage collector removes persistently weak experts and creates perturbed copies of stronger ones.

```text
input stream
      ↓
frozen Qwen backbone
      ↓
parallel LoRA expert branches
      ↓
gain-weighted mixture
      ↓
self-supervised loss + proxy expert utility
      ↓
EMA gain update
      ↓
kill / clone / perturb garbage collection
      ↓
updated expert population
      ↓
next interaction
```

The backbone stays frozen throughout the experiment. Only the LoRA expert parameters and expert gains are updated.

---

## Why freeze the backbone?

Online learning can be unstable.

If the full model is updated on a noisy or non-stationary stream, it can quickly overwrite useful behavior or collapse numerically.

This prototype avoids that by separating the system into two parts:

```text
frozen backbone  = stable base model
LoRA expert pool = small adaptive layer
```

The backbone preserves the pretrained model.

The expert pool absorbs short-term adaptation.

The garbage collector acts as a crude selection mechanism over experts.

---

## Current implementation

The current notebook uses:

- backbone: `Qwen/Qwen2.5-0.5B`,
- backbone parameters: frozen,
- adaptive modules: custom parallel LoRA experts,
- injection target: selected upper-layer `down_proj` modules,
- routing: gain-softmax mixture,
- training signal: self-supervised next-token loss,
- expert utility: proxy fitness signal,
- fitness smoothing: exponential moving average,
- expert lifecycle: kill / clone / perturb,
- runtime modes:
  - `inference_only`,
  - `learn_from_input`,
  - `quarantine_input`.

The implementation is intentionally compact and notebook-first.

---

## Key design constraint: simultaneous experts

GC-LoRA does **not** evaluate experts one by one.

It does not run:

```text
expert_1 forward
expert_2 forward
expert_3 forward
...
compare after separate runs
```

Instead, all experts operate on the same hidden state during the same forward pass.

For a frozen linear projection:

```math
y_{\mathrm{base}} = Wx
```

each LoRA expert computes:

```math
\Delta y_i = B_i A_i x
```

The adapted output is:

```math
y =
y_{\mathrm{base}}
+
\sum_{i=1}^{K}
\pi_i \Delta y_i
```

where:

```math
\pi_i =
\mathrm{softmax}(\eta g)_i
```

and:

- \(K\) is the number of experts,
- \(g_i\) is the current gain of expert \(i\),
- \(\eta\) is a gain temperature or sharpness parameter,
- \(\pi_i\) is the mixture weight of expert \(i\).

This makes the system closer to a small parallel adapter pool than to offline adapter evaluation.

---

## Expert gains

Each expert has a scalar gain.

The gain controls how much that expert contributes to the adapter mixture.

Higher gain means the expert receives more mixture weight.

Lower gain means the expert is suppressed.

Gains are updated from an exponential moving average of a proxy utility signal:

```math
f_i^{(t)}
=
\beta f_i^{(t-1)}
+
(1-\beta) r_i^{(t)}
```

where:

- \(f_i^{(t)}\) is the smoothed proxy fitness of expert \(i\),
- \(r_i^{(t)}\) is the current proxy utility signal,
- \(\beta\) is the EMA smoothing coefficient.

The gain can then be updated from the smoothed fitness.

---

## Important caveat: proxy fitness is not exact credit assignment

The current notebook does **not** compute exact per-expert counterfactual loss.

Exact credit assignment would require asking:

> What would the loss have been if this expert were removed or isolated?

That would require extra forward passes or a different architecture.

The current prototype instead uses local proxy signals such as gradient alignment or representation-level utility estimates.

That means expert selection is approximate.

A better interpretation is:

```text
fitness = local proxy utility signal
```

not:

```text
fitness = exact causal contribution to loss reduction
```

This is the most important limitation of the current prototype.

---

## Garbage collection

The garbage collector periodically inspects expert fitness.

A simple lifecycle is:

1. rank experts by smoothed proxy fitness,
2. identify weak experts,
3. remove or reset persistently weak experts,
4. choose stronger experts as parents,
5. clone parent LoRA weights,
6. perturb the clone,
7. continue online adaptation.

Conceptually:

```text
weak expert dies
strong expert is copied
copy is slightly perturbed
population continues
```

This is a crude population mechanism, not a full evolutionary algorithm.

The purpose is to prevent the expert pool from becoming stale and to give the system a way to recycle capacity.

---

## Clone and perturb

When a weak expert is replaced, the new expert can be initialized from a stronger parent:

```math
A_{\mathrm{new}}
=
A_{\mathrm{parent}}
+
\epsilon_A
```

```math
B_{\mathrm{new}}
=
B_{\mathrm{parent}}
+
\epsilon_B
```

where \(\epsilon_A\) and \(\epsilon_B\) are small random perturbations.

The goal is to explore small variations around experts that currently appear useful.

This can also fail.

If the same parent dominates repeatedly, the population can collapse into many near-copies of one expert.

That is a known risk and should be measured.

---

## Runtime modes

### `inference_only`

Runs the model without updating LoRA experts or gains.

Use this for evaluation or safe inference.

### `learn_from_input`

Allows online adapter updates from the current input.

Use this only for controlled experiments or trusted streams.

### `quarantine_input`

Processes the input but avoids learning from it.

Use this for noisy, adversarial, low-quality, or unknown-quality inputs.

The quarantine mode is important because an online learner should not adapt to every input it sees.

---

## Synthetic stream

The current notebook uses a controlled synthetic stream with domain-like segments such as:

- Python,
- math,
- technical QA,
- noise.

The stream is designed to test whether the expert pool shows visible adaptation dynamics and whether noisy segments can be survived without numerical collapse.

This is not a benchmark.

It is a toy stream for observing mechanism behavior.

---

## What to measure

A useful run should log:

- training loss over stream position,
- validation loss or held-out loss,
- gain values per expert,
- expert mixture entropy,
- expert birth/death events,
- parent selection counts,
- number of garbage-collection events,
- non-finite loss or NaN events,
- domain-wise performance,
- recovery after noisy segments,
- diversity between expert parameters,
- diversity between expert outputs.

The most important thing is to compare against simpler baselines.

---

## Baselines and ablations

This prototype should be evaluated against:

| Variant | Purpose |
|---|---|
| Frozen backbone only | Checks whether adapters help at all |
| Single online LoRA | Tests whether an expert pool is useful |
| Static LoRA expert pool without GC | Tests whether garbage collection matters |
| Gain mixture without expert replacement | Tests whether gain routing alone is enough |
| Random expert replacement | Tests whether fitness-based GC helps |
| Clone without perturbation | Tests whether perturbation matters |
| Perturbation without clone selection | Tests whether parent choice matters |
| Exact per-expert ablation for small K | Validates the proxy fitness signal |

The system is only interesting if it beats these simpler alternatives under the same stream and compute budget.

---

## Minimal success criteria

A successful toy run should show:

1. no numerical collapse,
2. stable frozen backbone,
3. finite LoRA updates,
4. non-trivial gain differentiation between experts,
5. some useful recovery after noisy segments,
6. lower held-out loss than frozen-backbone-only,
7. better adaptation than a single online LoRA under the same budget,
8. expert diversity that does not collapse immediately into clones of one parent.

Without baselines, gain plots are only interesting diagnostics, not evidence of useful adaptation.

---

## Suggested results table

A minimal report should include:

| Method | Stream loss ↓ | Held-out loss ↓ | NaN steps | GC events | Expert entropy | Parent monopoly rate |
|---|---:|---:|---:|---:|---:|---:|
| Frozen backbone | TBD | TBD | TBD | N/A | N/A | N/A |
| Single online LoRA | TBD | TBD | TBD | N/A | N/A | N/A |
| Expert pool, no GC | TBD | TBD | TBD | 0 | TBD | N/A |
| Expert pool + random GC | TBD | TBD | TBD | TBD | TBD | TBD |
| GC-LoRA | TBD | TBD | TBD | TBD | TBD | TBD |

The key question is not whether the mechanism looks dynamic.

The key question is whether the dynamics improve adaptation relative to simple baselines.

---

## Numerical stability

One practical lesson from the prototype is that dtype matters.

The stable configuration keeps:

```text
frozen backbone: fp16 or model-native low precision
trainable LoRA parameters: fp32
optimizer state: fp32
```

This avoids a common failure mode where small trainable LoRA matrices receive unstable updates in low precision.

Before interpreting online adaptation behavior, first check:

- loss is finite,
- gradients are finite,
- LoRA parameter norms are bounded,
- no NaN or Inf appears in the update path.

The first requirement of an adaptive system is not being adaptive.

It is not exploding.

---

## Failure modes

### Proxy fitness can reward the wrong expert

If fitness is only a local proxy, an expert may be credited for improvement it did not cause.

This is a credit-assignment problem.

### Parent monopoly

One high-gain expert may become the parent of most new experts.

This can reduce population diversity.

### Gain collapse

The softmax over gains may become too sharp.

One expert can dominate the mixture while others stop receiving meaningful learning signal.

### Overfitting to the stream

Online updates may improve immediate stream loss while harming held-out behavior.

### Learning from noise

Without quarantine or input-quality gating, the system may adapt to low-quality or adversarial inputs.

### No hardware benefit

This method adds adapter computation.

It is not a compression or acceleration method.

### Toy-stream overinterpretation

Recovery on a synthetic domain stream does not imply robust continual learning in real deployment.

---

## Interpretation discipline

Use cautious wording.

| Avoid saying | Prefer saying |
|---|---|
| self-improving LLM | frozen LLM with online LoRA adapter updates |
| continual learning solved | toy online adaptation prototype |
| expert fitness | proxy expert utility |
| expert learns a domain | expert gain increased on a domain-like segment |
| evolutionary system | clone / perturb lifecycle mechanism |
| garbage collector finds bad experts | GC removes low-proxy-fitness experts |
| robust adaptation | stable behavior in this controlled run |
| expert specialization | observed gain differentiation |

This keeps the project honest and easier to evaluate.

---

## Relationship to other ideas

GC-LoRA is loosely related to several families of methods:

- LoRA and parameter-efficient fine-tuning,
- mixture-of-experts routing,
- online learning,
- population-based training,
- stability-plasticity tradeoffs,
- continual learning,
- adapter fusion,
- input-quality gating.

These are inspirations, not formal equivalences.

The current notebook does not implement a full MoE router, does not provide regret bounds, does not perform NEAT-style speciation, and does not solve catastrophic forgetting.

---

## Possible extensions

### Exact expert ablation

For small numbers of experts, periodically run extra forward passes with each expert removed.

This would estimate:

```math
\Delta L_i =
L_{\mathrm{without}\ i}
-
L_{\mathrm{with}\ all}
```

and provide a better fitness signal.

### Diversity regularization

Add penalties or constraints to discourage all experts from becoming near-identical.

Possible metrics:

- LoRA parameter cosine similarity,
- output delta cosine similarity,
- parent clone counts,
- expert entropy.

### Better input gating

Use a classifier or heuristic to decide whether an input should be learned from, ignored, or quarantined.

### Domain-held-out evaluation

Evaluate on held-out examples from each synthetic domain after adaptation.

This would separate real learning from short-term stream fitting.

### Realistic streams

Replace the synthetic stream with more realistic data mixtures.

### Learned router

Replace scalar gain-only routing with a token-dependent or sequence-dependent router.

This would move the system closer to an actual adaptive MoE adapter layer.

### CLI experiment runner

Move the notebook into a reproducible runner:

```bash
python run_gc_lora.py \
  --model Qwen/Qwen2.5-0.5B \
  --experts 8 \
  --rank 8 \
  --stream synthetic \
  --gc-interval 50 \
  --output-dir runs/gc_lora_seed1
```

Then compare runs:

```bash
python compare_runs.py \
  --runs runs/frozen runs/single_lora runs/gc_lora_seed1 \
  --metrics stream_loss heldout_loss expert_entropy gc_events
```

The notebook can remain the readable entry point.

A CLI would make seed sweeps and ablations easier.

---

## Suggested repository description

For the GitHub About box:

```text
Controlled notebook prototype for online adaptation with a frozen Qwen LLM and a gain-weighted pool of parallel LoRA experts.
```

Suggested topics:

```text
lora
llm
qwen
online-learning
continual-learning
mixture-of-experts
adapter-tuning
```

---

## Scope statement

GC-LoRA is a small experimental notebook for studying online adapter dynamics.

Its useful contribution is not that it proves a new continual-learning architecture.

Its useful contribution is that it makes a specific mechanism easy to inspect:

> frozen backbone + parallel LoRA experts + gain mixture + proxy fitness + kill/clone/perturb lifecycle.

That mechanism may be useful as a playground for future experiments in online adaptation, expert routing, adapter lifecycle management, and safe learning-from-input policies.

For now, it should be treated as a controlled toy prototype.
