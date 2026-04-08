from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pptrain.core.config import RunConfig


CLAIM_TRANSFER_SIGNAL = "transfer_signal"
CLAIM_CONVERGENCE_GAIN = "convergence_gain"
CLAIM_COMPUTE_MATCHED_GAIN = "compute_matched_gain"
CLAIM_REASONING_TRANSFER = "reasoning_transfer"
CLAIM_ALGORITHMIC_TRANSFER = "algorithmic_transfer"
CLAIM_SYNTHETIC_ORDERING = "synthetic_ordering"
CLAIM_NEAR_REAL_BASELINE = "near_real_baseline"

CLAIM_COLUMNS = (
    CLAIM_TRANSFER_SIGNAL,
    CLAIM_CONVERGENCE_GAIN,
    CLAIM_COMPUTE_MATCHED_GAIN,
    CLAIM_REASONING_TRANSFER,
    CLAIM_ALGORITHMIC_TRANSFER,
    CLAIM_SYNTHETIC_ORDERING,
    CLAIM_NEAR_REAL_BASELINE,
)


@dataclass(frozen=True, slots=True)
class TextDatasetSpec:
    source: str
    formatter: str
    dataset_name: str | None = None
    dataset_config_name: str | None = None
    subset: str | None = None
    warmup_split: str | None = None
    train_split: str | None = None
    eval_split: str | None = None
    text_field: str = "text"
    inline_warmup_texts: tuple[str, ...] = ()
    inline_train_texts: tuple[str, ...] = ()
    inline_eval_texts: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class NeedleProbeConfig:
    num_examples: int
    haystack_size: int
    max_new_tokens: int = 8


@dataclass(frozen=True, slots=True)
class ArithmeticProbeConfig:
    num_examples: int
    max_addend: int
    max_new_tokens: int = 16


@dataclass(frozen=True, slots=True)
class GSM8KEvalConfig:
    split: str = "test[:32]"
    max_new_tokens: int = 128
    fewshot_examples: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class MechanismStudySpec:
    mechanism_name: str
    primary_preset: str
    dataset_key: str
    claim_categories: tuple[str, ...]
    paper_source: str
    paper_note: str
    sequence_count_override: int
    eval_sequence_count_override: int
    max_length_override: int
    comparison_presets: dict[str, str] = field(default_factory=dict)
    compare_against_natural_warmup: bool = False
    run_reasoning_probe: bool = False
    run_algorithmic_probe: bool = False


@dataclass(frozen=True, slots=True)
class ReplicationProfile:
    name: str
    description: str
    model_name_or_path: str
    context_length: int
    seed_values: tuple[int, ...]
    config_overrides: dict[str, Any]
    synthetic_run_config: RunConfig
    downstream_run_config: RunConfig
    natural_warmup_run_config: RunConfig
    datasets: dict[str, TextDatasetSpec]
    studies: tuple[MechanismStudySpec, ...]
    needle_probe: NeedleProbeConfig | None = None
    arithmetic_probe: ArithmeticProbeConfig | None = None
    gsm8k_eval: GSM8KEvalConfig | None = None


def build_replication_profile(
    profile_name: str,
    *,
    output_dir: str,
    test_mode: bool,
    model_name_or_path: str | None = None,
    context_length: int | None = None,
) -> ReplicationProfile:
    if test_mode or profile_name == "smoke":
        return _build_smoke_profile(output_dir=output_dir, model_name_or_path=model_name_or_path)
    if profile_name != "paper_proxy_2048":
        raise KeyError(f"Unknown replication profile '{profile_name}'.")
    return _build_paper_proxy_profile(
        output_dir=output_dir,
        model_name_or_path=model_name_or_path,
        context_length=context_length,
    )


def _build_smoke_profile(
    *,
    output_dir: str,
    model_name_or_path: str | None = None,
) -> ReplicationProfile:
    general_train = (
        "A small language model can learn from synthetic data before natural text.",
        "Pre pre training should help transfer when the abstraction matches the task.",
        "Long context behavior matters when the downstream task needs retrieval.",
        "Synthetic curricula can be useful even when they do not look like language.",
    )
    math_train = (
        "Question: If Alice has 3 apples and buys 4 more, how many apples does she have? Answer: 7",
        "Question: If a box has 5 red balls and 2 blue balls, how many balls are in the box? Answer: 7",
        "Question: If one bag holds 6 marbles and another holds 3 marbles, how many marbles are there? Answer: 9",
    )
    summary_train = (
        "Article: Synthetic tasks can improve transfer for summarization.\nTL;DR: synthetic tasks help summarization transfer.",
        "Article: Nonsense corpora can still teach compression and selection.\nTL;DR: nonsense data can teach summarization skills.",
        "Article: Structured pretraining can be cheap and useful.\nTL;DR: structured pretraining may be efficient.",
    )
    datasets = {
        "general_text": TextDatasetSpec(
            source="inline",
            formatter="plain_text",
            inline_warmup_texts=general_train[:2],
            inline_train_texts=general_train,
            inline_eval_texts=general_train[1:],
        ),
        "math_text": TextDatasetSpec(
            source="inline",
            formatter="plain_text",
            inline_warmup_texts=math_train[:2],
            inline_train_texts=math_train,
            inline_eval_texts=math_train,
        ),
        "summary_text": TextDatasetSpec(
            source="inline",
            formatter="plain_text",
            inline_warmup_texts=summary_train[:2],
            inline_train_texts=summary_train,
            inline_eval_texts=summary_train[1:],
        ),
    }
    studies = (
        MechanismStudySpec(
            mechanism_name="nca",
            primary_preset="smoke",
            dataset_key="general_text",
            claim_categories=(
                CLAIM_TRANSFER_SIGNAL,
                CLAIM_CONVERGENCE_GAIN,
                CLAIM_COMPUTE_MATCHED_GAIN,
                CLAIM_REASONING_TRANSFER,
            ),
            paper_source="Lee et al. 2026",
            paper_note="Proxy for downstream LM gain, convergence, compute-matched NL baseline, and reasoning transfer.",
            sequence_count_override=24,
            eval_sequence_count_override=8,
            max_length_override=128,
            compare_against_natural_warmup=True,
            run_reasoning_probe=True,
        ),
        MechanismStudySpec(
            mechanism_name="lime",
            primary_preset="smoke",
            dataset_key="math_text",
            claim_categories=(
                CLAIM_TRANSFER_SIGNAL,
                CLAIM_CONVERGENCE_GAIN,
                CLAIM_COMPUTE_MATCHED_GAIN,
                CLAIM_REASONING_TRANSFER,
            ),
            paper_source="Wu et al. 2021",
            paper_note="Proxy for mathematical reasoning transfer.",
            sequence_count_override=32,
            eval_sequence_count_override=8,
            max_length_override=96,
            compare_against_natural_warmup=True,
            run_reasoning_probe=True,
        ),
        MechanismStudySpec(
            mechanism_name="simpler_tasks",
            primary_preset="smoke",
            dataset_key="general_text",
            claim_categories=(
                CLAIM_TRANSFER_SIGNAL,
                CLAIM_CONVERGENCE_GAIN,
                CLAIM_COMPUTE_MATCHED_GAIN,
            ),
            paper_source="Wu et al. 2022",
            paper_note="Proxy for transfer signal under simple synthetic warm-up.",
            sequence_count_override=32,
            eval_sequence_count_override=8,
            max_length_override=96,
            compare_against_natural_warmup=True,
        ),
        MechanismStudySpec(
            mechanism_name="procedural",
            primary_preset="smoke",
            dataset_key="general_text",
            claim_categories=(
                CLAIM_TRANSFER_SIGNAL,
                CLAIM_CONVERGENCE_GAIN,
                CLAIM_COMPUTE_MATCHED_GAIN,
                CLAIM_ALGORITHMIC_TRANSFER,
            ),
            paper_source="Jiang et al. 2026",
            paper_note="Proxy for semantic transfer and algorithmic long-context gains.",
            sequence_count_override=32,
            eval_sequence_count_override=8,
            max_length_override=96,
            compare_against_natural_warmup=True,
            run_algorithmic_probe=True,
        ),
        MechanismStudySpec(
            mechanism_name="dyck",
            primary_preset="smoke",
            dataset_key="general_text",
            claim_categories=(
                CLAIM_TRANSFER_SIGNAL,
                CLAIM_CONVERGENCE_GAIN,
                CLAIM_COMPUTE_MATCHED_GAIN,
                CLAIM_ALGORITHMIC_TRANSFER,
            ),
            paper_source="Jiang et al. 2026",
            paper_note="Proxy for Needle-in-a-Haystack style gains.",
            sequence_count_override=32,
            eval_sequence_count_override=8,
            max_length_override=128,
            compare_against_natural_warmup=True,
            run_algorithmic_probe=True,
        ),
        MechanismStudySpec(
            mechanism_name="summarization",
            primary_preset="smoke",
            dataset_key="summary_text",
            claim_categories=(
                CLAIM_TRANSFER_SIGNAL,
                CLAIM_CONVERGENCE_GAIN,
                CLAIM_COMPUTE_MATCHED_GAIN,
                CLAIM_SYNTHETIC_ORDERING,
                CLAIM_NEAR_REAL_BASELINE,
            ),
            paper_source="Krishna et al. 2021 / Nath et al. 2021",
            paper_note="Proxy for OurTasks vs STEP ordering and closeness to natural-text warm-up.",
            sequence_count_override=32,
            eval_sequence_count_override=8,
            max_length_override=128,
            comparison_presets={"step": "smoke"},
            compare_against_natural_warmup=True,
        ),
    )
    return ReplicationProfile(
        name="smoke",
        description="Tiny local replication smoke run for pipeline validation.",
        model_name_or_path=model_name_or_path or "sshleifer/tiny-gpt2",
        context_length=128,
        seed_values=(41,),
        config_overrides={"n_positions": 128},
        synthetic_run_config=RunConfig(
            output_dir=f"{output_dir}/synthetic",
            max_steps=4,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            logging_steps=1,
            save_steps=4,
            eval_steps=2,
        ),
        downstream_run_config=RunConfig(
            output_dir=f"{output_dir}/downstream",
            max_steps=4,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            logging_steps=1,
            save_steps=4,
            eval_steps=2,
        ),
        natural_warmup_run_config=RunConfig(
            output_dir=f"{output_dir}/warmup",
            max_steps=2,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            logging_steps=1,
            save_steps=2,
            eval_steps=2,
        ),
        datasets=datasets,
        studies=studies,
        needle_probe=NeedleProbeConfig(num_examples=4, haystack_size=8, max_new_tokens=6),
        arithmetic_probe=ArithmeticProbeConfig(num_examples=4, max_addend=9, max_new_tokens=8),
    )


def _build_paper_proxy_profile(
    *,
    output_dir: str,
    model_name_or_path: str | None,
    context_length: int | None,
) -> ReplicationProfile:
    resolved_context = context_length or 2048
    datasets = {
        "general_text": TextDatasetSpec(
            source="hf",
            formatter="plain_text",
            dataset_name="wikitext",
            dataset_config_name="wikitext-2-raw-v1",
            warmup_split="train[:256]",
            train_split="train[256:1024]",
            eval_split="validation[:128]",
        ),
        "math_text": TextDatasetSpec(
            source="hf",
            formatter="gsm8k_qa",
            dataset_name="openai/gsm8k",
            subset="main",
            warmup_split="train[:128]",
            train_split="train[128:512]",
            eval_split="test[:64]",
        ),
        "summary_text": TextDatasetSpec(
            source="hf",
            formatter="cnn_dm_tldr",
            dataset_name="cnn_dailymail",
            dataset_config_name="3.0.0",
            warmup_split="train[:256]",
            train_split="train[256:1024]",
            eval_split="validation[:128]",
        ),
    }
    studies = (
        MechanismStudySpec(
            mechanism_name="nca",
            primary_preset="paper_web_text",
            dataset_key="general_text",
            claim_categories=(
                CLAIM_TRANSFER_SIGNAL,
                CLAIM_CONVERGENCE_GAIN,
                CLAIM_COMPUTE_MATCHED_GAIN,
                CLAIM_REASONING_TRANSFER,
            ),
            paper_source="Lee et al. 2026",
            paper_note="Proxy for LM gain, faster convergence, compute-matched natural baseline, and GSM8K transfer.",
            sequence_count_override=1024,
            eval_sequence_count_override=128,
            max_length_override=resolved_context,
            compare_against_natural_warmup=True,
            run_reasoning_probe=True,
        ),
        MechanismStudySpec(
            mechanism_name="lime",
            primary_preset="paper_benchmark_100k",
            dataset_key="math_text",
            claim_categories=(
                CLAIM_TRANSFER_SIGNAL,
                CLAIM_CONVERGENCE_GAIN,
                CLAIM_COMPUTE_MATCHED_GAIN,
                CLAIM_REASONING_TRANSFER,
            ),
            paper_source="Wu et al. 2021",
            paper_note="Proxy for math reasoning gains using GSM8K.",
            sequence_count_override=1024,
            eval_sequence_count_override=128,
            max_length_override=512,
            compare_against_natural_warmup=True,
            run_reasoning_probe=True,
        ),
        MechanismStudySpec(
            mechanism_name="simpler_tasks",
            primary_preset="paper_unary_core_100k",
            dataset_key="general_text",
            claim_categories=(
                CLAIM_TRANSFER_SIGNAL,
                CLAIM_CONVERGENCE_GAIN,
                CLAIM_COMPUTE_MATCHED_GAIN,
            ),
            paper_source="Wu et al. 2022",
            paper_note="Proxy for transfer signal under simpler synthetic warm-up.",
            sequence_count_override=1024,
            eval_sequence_count_override=128,
            max_length_override=512,
            compare_against_natural_warmup=True,
        ),
        MechanismStudySpec(
            mechanism_name="procedural",
            primary_preset="paper_set_len64",
            dataset_key="general_text",
            claim_categories=(
                CLAIM_TRANSFER_SIGNAL,
                CLAIM_CONVERGENCE_GAIN,
                CLAIM_COMPUTE_MATCHED_GAIN,
                CLAIM_ALGORITHMIC_TRANSFER,
            ),
            paper_source="Jiang et al. 2026",
            paper_note="Proxy for semantic transfer and long-context algorithmic transfer.",
            sequence_count_override=1024,
            eval_sequence_count_override=128,
            max_length_override=512,
            compare_against_natural_warmup=True,
            run_algorithmic_probe=True,
        ),
        MechanismStudySpec(
            mechanism_name="dyck",
            primary_preset="paper_k64",
            dataset_key="general_text",
            claim_categories=(
                CLAIM_TRANSFER_SIGNAL,
                CLAIM_CONVERGENCE_GAIN,
                CLAIM_COMPUTE_MATCHED_GAIN,
                CLAIM_ALGORITHMIC_TRANSFER,
            ),
            paper_source="Jiang et al. 2026",
            paper_note="Proxy for Dyck-to-Needle long-context transfer.",
            sequence_count_override=1024,
            eval_sequence_count_override=128,
            max_length_override=resolved_context,
            compare_against_natural_warmup=True,
            run_algorithmic_probe=True,
        ),
        MechanismStudySpec(
            mechanism_name="summarization",
            primary_preset="paper_ourtasks_subset_100k",
            dataset_key="summary_text",
            claim_categories=(
                CLAIM_TRANSFER_SIGNAL,
                CLAIM_CONVERGENCE_GAIN,
                CLAIM_COMPUTE_MATCHED_GAIN,
                CLAIM_SYNTHETIC_ORDERING,
                CLAIM_NEAR_REAL_BASELINE,
            ),
            paper_source="Krishna et al. 2021 / Nath et al. 2021",
            paper_note="Proxy for OurTasks vs STEP ordering and closeness to natural-text warm-up on article-summary format.",
            sequence_count_override=1024,
            eval_sequence_count_override=128,
            max_length_override=512,
            comparison_presets={"step": "paper_step_tasks_100k"},
            compare_against_natural_warmup=True,
        ),
    )
    return ReplicationProfile(
        name="paper_proxy_2048",
        description="Paper-aligned proxy campaign with 2k context and public datasets.",
        model_name_or_path=model_name_or_path or "EleutherAI/pythia-160m-deduped",
        context_length=resolved_context,
        seed_values=(11, 23, 37),
        config_overrides={"max_position_embeddings": resolved_context},
        synthetic_run_config=RunConfig(
            output_dir=f"{output_dir}/synthetic",
            max_steps=60,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            logging_steps=10,
            save_steps=60,
            eval_steps=20,
            gradient_accumulation_steps=4,
        ),
        downstream_run_config=RunConfig(
            output_dir=f"{output_dir}/downstream",
            max_steps=80,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            logging_steps=10,
            save_steps=80,
            eval_steps=20,
            gradient_accumulation_steps=4,
        ),
        natural_warmup_run_config=RunConfig(
            output_dir=f"{output_dir}/warmup",
            max_steps=40,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            logging_steps=10,
            save_steps=40,
            eval_steps=20,
            gradient_accumulation_steps=4,
        ),
        datasets=datasets,
        studies=studies,
        needle_probe=NeedleProbeConfig(num_examples=32, haystack_size=128, max_new_tokens=8),
        gsm8k_eval=GSM8KEvalConfig(split="test[:32]", max_new_tokens=96),
    )
