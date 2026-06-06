# GC-LoRA: Adaptive Expert Flywheel for Online Representation Adaptation

> A research proof-of-concept for continuous online adaptation in LLMs using a frozen backbone and a population of parallel LoRA experts with gain-controlled garbage collection.

## Overview

GC-LoRA explores a simple but important question:

> Can a frozen language model acquire useful new representations through a population of small adaptive experts, while suppressing harmful or noisy adaptations?

The project implements an experimental **adaptive expert flywheel** on top of `Qwen/Qwen2.5-0.5B`.

The backbone model remains frozen. Online adaptation happens only through small, parallel LoRA expert branches inserted into selected transformer layers. Each expert receives gradient updates, accumulates a smoothed fitness estimate, competes with other experts through gain-weighted mixture, and may be removed if it becomes unhelpful.

The system therefore behaves less like ordinary fine-tuning and more like a small evolutionary population attached to a frozen LLM.

```text
interaction stream
      ↓
frozen Qwen backbone
      ↓
parallel LoRA expert branches
      ↓
gain-weighted mixture
      ↓
self-supervised loss + local expert fitness
      ↓
EMA gain update
      ↓
kill / clone / perturb garbage collection
      ↓
updated expert population
      ↓
next interaction
```

## Current Status

This repository contains a working proof-of-concept notebook:

```text
GC_LoRA_Adaptive_Expert_Flywheel_Qwen05B_v2_stable.ipynb
```

The current prototype has demonstrated:

- stable execution on a T4-class GPU;
- no NaN collapse after fp32 LoRA stabilization;
- parallel expert competition inside a single forward pass;
- online adaptation across controlled domain streams;
- expert birth/death events through garbage collection;
- recovery after noisy data segments;
- observable expert gain dynamics.

This is **not** a production continual-learning system. It is an experimental research notebook intended to make the mechanism inspectable.

## Motivation

Modern LLMs are powerful but mostly static after deployment. New information is usually added through:

- full retraining;
- offline fine-tuning;
- parameter-efficient fine-tuning;
- retrieval-augmented generation;
- manually curated memory.

These approaches either require offline optimization or avoid changing the model parameters.

GC-LoRA investigates a different route:

> keep the backbone stable, but let a small population of adaptation modules evolve online.

This is motivated by the stability–plasticity dilemma:

> A learning system must remain stable enough to preserve useful knowledge, but plastic enough to incorporate new information.

GC-LoRA addresses this by separating:

```text
frozen backbone      = slow stable system
LoRA expert pool     = fast adaptive system
garbage collector    = selection / pruning mechanism
```

## Key Design Constraint

GC-LoRA does **not** evaluate experts sequentially.

It does not run:

```text
expert_1 forward
expert_2 forward
expert_3 forward
...
compare afterwards
```

Instead, all experts operate on the same hidden state in the same forward pass:

```text
hidden state h
      ↓
LoRA expert 1 computes Δh_1
LoRA expert 2 computes Δh_2
LoRA expert 3 computes Δh_3
...
      ↓
h' = h + Σ_i softmax(η gain_i) Δh_i
```

This is essential. The goal is simultaneous expert competition, not offline adapter evaluation.

## Architecture

The current notebook uses:

- backbone: `Qwen/Qwen2.5-0.5B`;
- backbone parameters: frozen;
- adaptive modules: custom parallel LoRA expert branches;
- injection target: upper-layer `down_proj` modules;
- routing: gain-softmax, no learned router;
- fitness tracking: EMA-smoothed expert reward;
- garbage collection: kill / clone / perturb;
- runtime modes:
  - `inference_only`;
  - `learn_from_input`;
  - `quarantine_input`.

### Parallel LoRA Layer

For a frozen linear projection:

```text
y_base = W x
```

each LoRA expert computes:

```text
Δy_i = B_i A_i x
```

and the wrapped layer returns:

```text
y = y_base + Σ_i w_i Δy_i
```

where:

```text
w_i = softmax(η gain_i)
```

The experts are stored as separate low-rank parameter sets inside one parallel layer.

## Literature Mapping

GC-LoRA combines several existing ideas, but does not claim to be identical to any one of them.

| Component | Related literature / concept | Role |
|---|---|---|
| Frozen backbone | Complementary Learning Systems | Stable slow representation system |
| LoRA experts | Low-Rank Adaptation | Parameter-efficient adaptation |
| Parallel expert branches | Mixture of Experts | Multiple competing transformations |
| Gain weighting | Hedge / multiplicative weights | Online weighting of competing hypotheses |
| EMA fitness | Adam first moment | Smooth noisy fitness signals |
| Survival threshold | Adaptive Resonance Theory vigilance | Stability–plasticity control |
| Clone / perturb | Population Based Training, NEAT | Exploit useful experts and explore variants |
| Garbage collection | Pruning / selection dynamics | Remove harmful or redundant experts |
| Flywheel loop | Data flywheel / MAPE-style loops | Continuous adaptation cycle |

The working hypothesis is that expert-level selection can provide a practical substrate for online representation adaptation.

## Notebook Structure

The notebook is deliberately organized as a research document, not just a script.

Main sections:

1. Research goal and motivation
2. Literature mapping
3. Configuration
4. Environment setup
5. Load frozen Qwen2.5-0.5B
6. Define expert state
7. Define `ParallelLoRALinear`
8. Patch selected Qwen layers
9. Optimizer
10. Controlled data stream
11. Validation splits
12. Parallel-compatible fitness signals
13. Garbage collector
14. Training step
15. Controlled online training loop
16. Gain plots
17. Validation plots
18. Expert lifecycle events
19. Runtime flywheel API
20. Interactive examples
21. Baseline plan
22. Limitations
23. Next experiments

Each major code section is preceded by a markdown explanation describing what it does and why it exists.

## Installation

A clean environment should include:

```bash
pip install "transformers>=4.45.0" "accelerate>=0.34.0" "datasets>=2.20.0" \
            "matplotlib>=3.7.0" "pandas>=2.0.0" "sentencepiece" "einops"
```

The notebook is designed for a CUDA GPU. A T4-class GPU is sufficient for the current settings.

## Running the Notebook

1. Open:

```text
GC_LoRA_Adaptive_Expert_Flywheel_Qwen05B_v2_stable.ipynb
```

2. Run the cells from top to bottom.

3. Confirm that patching worked:

```text
Patched parallel LoRA layers: 4
Trainable dtypes: ['torch.float32']
```

4. Run the controlled online training loop.

5. Inspect:

- expert gain dynamics;
- validation loss by domain;
- GC lifecycle events;
- skipped non-finite steps;
- final expert weights.

A healthy run should show:

```text
Skipped non-finite steps: 0
nonfinite_params=0
```

and at least some gain separation between experts.

## Important Configuration Values

The core configuration is stored in one notebook cell.

Representative defaults:

```python
CONFIG = {
    "model_name": "Qwen/Qwen2.5-0.5B",
    "freeze_backbone": True,

    "torch_dtype": "float16",
    "lora_param_dtype": "float32",

    "target_module_names": ["down_proj"],
    "num_patched_layers": 4,

    "num_experts": 8,
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,

    "initial_gain": 1.0,
    "gain_ema_beta": 0.90,
    "hedge_eta": 0.30,

    "gc_interval_batches": 100,
    "min_lifetime_gc_cycles": 2,
    "death_quantile": 0.20,
    "clone_source_quantile": 0.20,
    "mutation_sigma": 0.005,

    "learning_rate": 5e-5,
    "max_grad_norm": 0.5,
}
```

### Numeric Stability Note

The first prototype became unstable because trainable LoRA parameters were converted to fp16 and updated by AdamW.

The stable version keeps:

```text
frozen backbone: fp16
trainable LoRA experts: fp32
```

This is important for T4 stability.

## Data Stream

The notebook uses a controlled synthetic stream with four domains:

| Domain | Purpose |
|---|---|
| `python` | Code-like adaptation |
| `math` | Symbolic / structured examples |
| `technical_qa` | Conceptual technical language |
| `noise` | Corrupted / harmful stream |

The noise domain is intentional. It tests whether the adaptive system can avoid being permanently damaged by bad updates.

## Validation

Validation is separated by domain:

```text
python
math
technical_qa
noise
```

The goal is not just to reduce current training loss. The goal is to observe:

- adaptation to the current stream;
- interference with previous domains;
- recovery after noise;
- whether noise is rejected rather than learned.

## Interpreting Results

### Good signs

A promising run shows:

- no NaN or non-finite parameter corruption;
- training loss decreases inside domains;
- validation loss improves for at least some useful domains;
- expert gains separate over time;
- garbage collection occurs occasionally;
- noisy data does not permanently collapse useful domains.

### Warning signs

Potential problems:

- all gains stay identical;
- one expert monopolizes the population;
- every GC event clones from the same parent;
- GC happens too often;
- GC never happens;
- technical validation loss does not improve;
- noise validation improves too much, suggesting the system may be learning noise.

## Example Observations

A representative stable run showed:

```text
Skipped non-finite steps: 0
GC events: 8
```

and domain-related GC events such as:

```text
step 80   math          expert 5 ← expert 7
step 160  technical_qa  expert 3 ← expert 2
step 240  noise         expert 0 ← expert 7
step 320  python        expert 1 ← expert 4
step 400  math          expert 7 ← expert 3
step 480  technical_qa  expert 2 ← expert 4
```

This suggests that the population dynamics are sensitive to domain changes.

## Runtime Flywheel API

The notebook defines:

```python
adaptive_flywheel_step(user_text, mode)
```

with three modes:

### `inference_only`

Runs generation without learning.

### `learn_from_input`

Treats the input as trusted online learning data and updates the expert population.

### `quarantine_input`

Stores the input without learning from it.

This gating is important. An adaptive system should not learn from every interaction.

## Limitations

GC-LoRA is currently a proof-of-concept.

Known limitations:

1. **No exact per-expert counterfactual CE loss**

   Exact per-expert loss would require separate forward passes or a more invasive architecture with expert-specific logits. The current implementation uses local gradient-alignment and representation-level proxy signals.

2. **Synthetic data**

   The current data stream is intentionally small and controlled. It is not a benchmark.

3. **No learned router**

   The current system uses gain-softmax rather than input-conditioned learned routing. This avoids router drift but limits expressivity.

4. **Weak technical QA generalization**

   The technical QA domain currently has too few examples to support strong held-out generalization.

5. **Parent monopoly risk**

   A strong expert can become parent too often. Future versions should add parent diversity constraints.

6. **No claim of solved continual learning**

   A successful run demonstrates a mechanism worth studying. It does not prove robust open-ended continual learning.

## Roadmap

Recommended next steps:

### v3: Better GC Dynamics

Add:

```python
"max_deaths_per_gc": 2
"max_clones_per_parent_per_gc": 1
"parent_selection": "gain_weighted_without_replacement"
```

This should reduce parent monopoly and preserve expert diversity.

### v4: Better Data

Expand each domain to 20–50 examples, especially `technical_qa`.

Split validation into:

```text
seen_style
heldout_concept
interference
noise_resilience
```

### v5: Baselines

Implement:

| Baseline | Purpose |
|---|---|
| Frozen Qwen only | Base model behavior |
| Single online LoRA | Catastrophic drift baseline |
| Static parallel LoRA without GC | Tests whether GC matters |
| Random GC | Tests whether gain-based selection matters |
| Full GC-LoRA | Proposed method |

### v6: Context-Conditioned Expert Selection

Keep slow EMA gain, but add fast input-conditioned modulation:

```text
effective_gain_i = slow_gain_i + fast_context_score_i
```

This would connect the system more directly to fast weights and dynamic routing.

### v7: Checkpointing

Save and restore:

- LoRA expert parameters;
- expert gains;
- expert ages;
- lifecycle events;
- quarantine buffer.

## Repository Layout

Suggested structure:

```text
.
├── README.md
├── notebooks/
│   └── GC_LoRA_Adaptive_Expert_Flywheel_Qwen05B_v2_stable.ipynb
├── results/
│   ├── example_logs.csv
│   ├── example_eval.csv
│   └── gc_events.csv
├── docs/
│   └── literature_notes.md
└── LICENSE
```

## Citation / Attribution

If you use this repository, please cite the conceptual components appropriately:

- LoRA for parameter-efficient adaptation;
- Mixture of Experts for expert-branch architectures;
- Hedge / multiplicative weights for online expert weighting;
- Adam for EMA-style moment estimates;
- Adaptive Resonance Theory for vigilance / stability–plasticity framing;
- Population Based Training and NEAT for clone / perturb population dynamics.

## Disclaimer

This project is experimental research code.

It is not intended for deployment in high-stakes systems.  
It does not provide a guarantee of safe or reliable continual learning.  
It is a controlled prototype for studying adaptive expert population dynamics in LLMs.
