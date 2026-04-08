# Architecture

## Core constraints

`pptrain` is intentionally small. The core owns only the upstream stage:

1. create synthetic token sequences
2. train a causal LM on them
3. export a transfer bundle for downstream language pretraining

The downstream stack remains the user's own stack.

## Main interfaces

### `Mechanism`

Responsible only for synthetic data generation and tokenization.

- expose a `TokenizerSpec`
- build train and eval datasets
- export config and metadata

### `TokenSequenceMechanism`

Optional helper for mechanisms that only need to sample token sequences and light metadata.

- centralizes the usual train/eval split handling
- keeps Dyck-like or procedural mechanisms from duplicating dataset boilerplate
- avoids leaking sequence-specific shortcuts into mechanisms like NCA that need custom masking or tokenization

### `ModelAdapter`

Responsible only for bridging a model family into `pptrain`.

- create a model for upstream pre-pre-training
- load a downstream model later

### `TransferPolicy`

Responsible only for moving useful weights across stages.

`v0.1` ships one safe default:

- copy matching parameters
- skip mismatched shapes
- re-initialize input/output embeddings on the target model

### `EvalTask`

Responsible only for one evaluation contract.

The eval layer is intentionally thin so benchmark-specific assumptions do not leak into the core training path.

## Why NCA first

NCA is a good first mechanism because it exercises the actual abstractions we need:

- synthetic non-linguistic data generation
- mechanism-specific tokenization
- mechanism-level complexity controls
- transfer into a text model with a different vocabulary

## Extension path

New mechanisms should only need:

1. a config dataclass
2. a mechanism class implementing the base interface, or subclassing `TokenSequenceMechanism`
3. registration in the mechanism registry with a short user-facing description

That keeps later additions such as Dyck languages, procedural tasks, or artificial-language generators local to their own modules.
