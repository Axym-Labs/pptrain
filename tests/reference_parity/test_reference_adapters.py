from __future__ import annotations

import json
import sys
from pathlib import Path

from pptrain.core.registry import create_task
from pptrain.reference_parity_exporters import (
    TASK_REFERENCE_ADAPTERS,
    build_lime_reference_fixture_from_files,
    build_nca_reference_fixture_from_export_json,
    build_nca_reference_fixture_from_repo,
    build_nca_reference_fixture_from_rows,
    build_procedural_reference_fixture_from_repo,
    build_procedural_reference_fixture_from_rows,
    build_summarization_reference_fixture_from_jsonl,
    build_summarization_reference_fixture_from_repo,
    parse_lime_reference_example,
    parse_procedural_reference_example,
    parse_summarization_reference_example,
)


def test_reference_task_adapters_cover_core_open_source_tasks() -> None:
    expected = {"lime", "summarization", "procedural", "nca"}
    assert expected.issubset(TASK_REFERENCE_ADAPTERS)
    assert TASK_REFERENCE_ADAPTERS["lime"].comparison_target == "normalized_examples"
    assert TASK_REFERENCE_ADAPTERS["summarization"].comparison_target == "normalized_examples"
    assert TASK_REFERENCE_ADAPTERS["procedural"].comparison_target == "normalized_examples"
    assert TASK_REFERENCE_ADAPTERS["nca"].comparison_target == "dataset_bundle"


def test_parse_lime_reference_example_normalizes_induction_rows() -> None:
    example = parse_lime_reference_example(
        src_line="<UPPER> A B <LOWER> x y z <MATH> + - <space> x y z + <space> A : [ x ] , B : [ y z ]",
        tgt_line="A B +",
    )

    assert example == {
        "mode": "induct",
        "upper_vocab": ["A", "B"],
        "lower_vocab": ["x", "y", "z"],
        "math_vocab": ["+", "-"],
        "pattern": ["A", "B", "+"],
        "result": ["x", "y", "z", "+"],
        "substitution_pairs": [
            ["A", ["x"]],
            ["B", ["y", "z"]],
        ],
    }


def test_build_lime_reference_fixture_from_files(tmp_path) -> None:
    train_src = tmp_path / "train.src"
    train_tgt = tmp_path / "train.tgt"
    valid_src = tmp_path / "valid.src"
    valid_tgt = tmp_path / "valid.tgt"

    train_src.write_text(
        "<UPPER> A B <LOWER> x y z <MATH> + - <space> x y z + <space> A : [ x ] , B : [ y z ]\n",
        encoding="utf-8",
    )
    train_tgt.write_text("A B +\n", encoding="utf-8")
    valid_src.write_text(
        "<UPPER> A B <LOWER> x y z <MATH> + - <space> A B + <space> A : [ x ] , B : [ y z ]\n",
        encoding="utf-8",
    )
    valid_tgt.write_text("x y z +\n", encoding="utf-8")

    task = create_task("lime", {"preset": "smoke", "sequence_count": 1, "eval_sequence_count": 1, "modes": ("induct",)})
    fixture = build_lime_reference_fixture_from_files(
        task,
        train_src_path=train_src,
        train_tgt_path=train_tgt,
        eval_src_path=valid_src,
        eval_tgt_path=valid_tgt,
        preset_name="paper_fixture",
    )

    assert fixture.task_name == "lime"
    assert fixture.comparison_target == "normalized_examples"
    assert fixture.train.examples == [
        {
            "mode": "induct",
            "upper_vocab": ["A", "B"],
            "lower_vocab": ["x", "y", "z"],
            "math_vocab": ["+", "-"],
            "pattern": ["A", "B", "+"],
            "result": ["x", "y", "z", "+"],
            "substitution_pairs": [
                ["A", ["x"]],
                ["B", ["y", "z"]],
            ],
        }
    ]
    assert fixture.eval is not None
    assert fixture.eval.examples == [
        {
            "mode": "deduct",
            "upper_vocab": ["A", "B"],
            "lower_vocab": ["x", "y", "z"],
            "math_vocab": ["+", "-"],
            "pattern": ["A", "B", "+"],
            "result": ["x", "y", "z", "+"],
            "substitution_pairs": [
                ["A", ["x"]],
                ["B", ["y", "z"]],
            ],
        }
    ]


def test_parse_procedural_reference_example_normalizes_union_rows() -> None:
    example = parse_procedural_reference_example(
        task_name="union",
        input_ids=[5, 6, 100, 6, 4, 100, 5, 6, 4, 101],
        separator_token_id=100,
        pad_token_id=101,
    )

    assert example == {
        "task": "union",
        "inputs": [[0, 1], [1, 2]],
        "target": [0, 1, 2],
    }


def test_parse_procedural_reference_example_normalizes_delete_rows() -> None:
    example = parse_procedural_reference_example(
        task_name="delete",
        input_ids=[7, 3, 7, 9, 100, 7, 100, 3, 7, 9, 101],
        separator_token_id=100,
        pad_token_id=101,
    )

    assert example == {
        "task": "delete",
        "inputs": [[0, 1, 0, 2], [0]],
        "target": [1, 0, 2],
    }


def test_build_procedural_reference_fixture_from_rows() -> None:
    task = create_task("procedural", {"preset": "paper_union_len16"})
    fixture = build_procedural_reference_fixture_from_rows(
        task,
        reference_task_name="union",
        train_input_rows=[[5, 6, 100, 6, 4, 100, 5, 6, 4, 101]],
        eval_input_rows=[[8, 9, 100, 9, 10, 100, 8, 9, 10, 101]],
        separator_token_id=100,
        pad_token_id=101,
        preset_name="paper_union_len16",
    )

    assert fixture.task_name == "procedural"
    assert fixture.comparison_target == "normalized_examples"
    assert fixture.train.examples == [
        {
            "task": "union",
            "inputs": [[0, 1], [1, 2]],
            "target": [0, 1, 2],
        }
    ]


def test_build_procedural_reference_fixture_from_repo(tmp_path) -> None:
    repo_root = tmp_path / "fake_repo"
    module_dir = repo_root / "procedural_data"
    module_dir.mkdir(parents=True)
    (module_dir / "reverse.py").write_text(
        """
class ReverseDataset:
    def __init__(self, seq_len, vocab_size, num_samples, seed=None):
        self.samples = [[5, 6, vocab_size, 6, 5, vocab_size + 1] for _ in range(num_samples)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {"input_ids": self.samples[idx]}
""".strip(),
        encoding="utf-8",
    )

    task = create_task("procedural", {"preset": "paper_reverse_len16"})
    fixture = build_procedural_reference_fixture_from_repo(
        task,
        repo_root=repo_root,
        reference_task_name="reverse",
        seq_len=2,
        vocab_size=100,
        train_sequence_count=1,
        eval_sequence_count=1,
        seed=7,
        preset_name="paper_reverse_len16",
    )

    assert fixture.task_name == "procedural"
    assert fixture.comparison_target == "normalized_examples"
    assert fixture.train.examples == [{"task": "reverse", "inputs": [[0, 1]], "target": [1, 0]}]
    assert fixture.eval is not None
    assert fixture.eval.examples == [{"task": "reverse", "inputs": [[0, 1]], "target": [1, 0]}]


def test_parse_summarization_reference_example_normalizes_bulleted_rows() -> None:
    example = parse_summarization_reference_example(
        task_name="copy_bulleted",
        article_lines=["bullet alpha beta .", "gamma delta ."],
        summary_lines=["alpha beta ."],
    )

    assert example == {
        "task": "copy_bulleted",
        "input_document": [["bullet", 0, 1], [2, 3]],
        "target_document": [[0, 1]],
    }


def test_parse_summarization_reference_example_canonicalizes_keywords() -> None:
    example = parse_summarization_reference_example(
        task_name="copy_keyword_multiple_sorted",
        article_lines=["alpha __d4__keyword_7__ .", "__d4__keyword_2__ beta ."],
        summary_lines=["__d4__keyword_2__ beta .", "alpha __d4__keyword_7__ ."],
    )

    assert example == {
        "task": "copy_keyword_multiple_sorted",
        "input_document": [[0, "keyword:1"], ["keyword:0", 1]],
        "target_document": [["keyword:0", 1], [0, "keyword:1"]],
    }


def test_build_summarization_reference_fixture_from_jsonl(tmp_path) -> None:
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    train_path.write_text(
        '{"article_lines":["bullet alpha beta .","gamma delta ."],"summary_lines":["alpha beta ."]}\n',
        encoding="utf-8",
    )
    eval_path.write_text(
        '{"article_lines":["bullet red blue ."],"summary_lines":["red blue ."]}\n',
        encoding="utf-8",
    )

    task = create_task("summarization", {"preset": "paper_copy_bulleted_100k"})
    fixture = build_summarization_reference_fixture_from_jsonl(
        task,
        reference_task_name="copy_bulleted",
        train_jsonl_path=train_path,
        eval_jsonl_path=eval_path,
        preset_name="paper_copy_bulleted_100k",
    )

    assert fixture.task_name == "summarization"
    assert fixture.comparison_target == "normalized_examples"
    assert fixture.train.examples == [
        {
            "task": "copy_bulleted",
            "input_document": [["bullet", 0, 1], [2, 3]],
            "target_document": [[0, 1]],
        }
    ]
    assert fixture.eval is not None
    assert fixture.eval.examples == [
        {
            "task": "copy_bulleted",
            "input_document": [["bullet", 0, 1]],
            "target_document": [[0, 1]],
        }
    ]


def test_build_summarization_reference_fixture_from_repo(tmp_path) -> None:
    dataset_root = tmp_path / "dataset_root" / "pretraining_datasets" / "copy_bulleted_paper"
    dataset_root.mkdir(parents=True)
    (dataset_root / "train.jsonl").write_text(
        '{"article_lines":["bullet alpha beta .","gamma delta ."],"summary_lines":["alpha beta ."]}\n',
        encoding="utf-8",
    )
    (dataset_root / "val.jsonl").write_text(
        '{"article_lines":["bullet red blue ."],"summary_lines":["red blue ."]}\n',
        encoding="utf-8",
    )

    task = create_task("summarization", {"preset": "paper_copy_bulleted_100k"})
    fixture = build_summarization_reference_fixture_from_repo(
        task,
        repo_root=tmp_path,
        dataset_name="copy_bulleted_paper",
        reference_task_name="copy_bulleted",
        preset_name="paper_copy_bulleted_100k",
    )

    assert fixture.task_name == "summarization"
    assert fixture.comparison_target == "normalized_examples"
    assert fixture.metadata["dataset_name"] == "copy_bulleted_paper"
    assert fixture.train.examples == [
        {
            "task": "copy_bulleted",
            "input_document": [["bullet", 0, 1], [2, 3]],
            "target_document": [[0, 1]],
        }
    ]
    assert fixture.eval is not None
    assert fixture.eval.examples == [
        {
            "task": "copy_bulleted",
            "input_document": [["bullet", 0, 1]],
            "target_document": [[0, 1]],
        }
    ]


def test_build_nca_reference_fixture_from_rows() -> None:
    task = create_task("nca", {"preset": "smoke"})
    fixture = build_nca_reference_fixture_from_rows(
        task,
        train_sequences=[[36, 1, 2, 37, 36, 3, 4]],
        train_labels=[[-100, -100, -100, 36, 3, 4, 37]],
        eval_sequences=[[36, 5, 6, 37, 36, 7, 8]],
        eval_labels=[[-100, -100, -100, 36, 7, 8, 37]],
        preset_name="smoke",
    )

    assert fixture.task_name == "nca"
    assert fixture.comparison_target == "dataset_bundle"
    assert fixture.train.sequences == [[36, 1, 2, 37, 36, 3, 4]]
    assert fixture.train.labels == [[-100, -100, -100, 36, 3, 4, 37]]
    assert fixture.eval is not None
    assert fixture.eval.sequences == [[36, 5, 6, 37, 36, 7, 8]]
    assert fixture.eval.labels == [[-100, -100, -100, 36, 7, 8, 37]]


def test_build_nca_reference_fixture_from_export_json(tmp_path) -> None:
    export_path = tmp_path / "nca_export.json"
    export_path.write_text(
        '{"train":{"sequences":[[36,1,2,37,36,3,4]],"labels":[[-100,-100,-100,36,3,4,37]]},"eval":{"sequences":[[36,5,6,37,36,7,8]],"labels":[[-100,-100,-100,36,7,8,37]]},"metadata":{"source":"paper"}}',
        encoding="utf-8",
    )

    task = create_task("nca", {"preset": "smoke"})
    fixture = build_nca_reference_fixture_from_export_json(
        task,
        export_json_path=export_path,
        preset_name="smoke",
    )

    assert fixture.task_name == "nca"
    assert fixture.comparison_target == "dataset_bundle"
    assert fixture.metadata["source"] == "paper"
    assert fixture.train.sequences == [[36, 1, 2, 37, 36, 3, 4]]
    assert fixture.eval is not None
    assert fixture.eval.labels == [[-100, -100, -100, 36, 7, 8, 37]]


def test_build_nca_reference_fixture_from_repo(tmp_path) -> None:
    task = create_task(
        "nca",
        {
            "preset": "smoke",
            "sequence_count": 1,
            "eval_sequence_count": 1,
            "rule_count": 1,
            "eval_rule_count": 1,
            "complexity_min": 0.0,
            "complexity_max": 1.0,
        },
    )
    bundle = task.build_datasets(seed=5)
    repo_root = _build_fake_nca_reference_repo(tmp_path / "fake_nca_repo", bundle)

    fixture = build_nca_reference_fixture_from_repo(
        task,
        repo_root=repo_root,
        preset_name="smoke",
        seed=5,
        python_executable=sys.executable,
    )

    assert fixture.task_name == "nca"
    assert fixture.comparison_target == "dataset_bundle"
    assert fixture.train.sequences == bundle.train_dataset.sequences
    assert fixture.train.labels == bundle.train_dataset.labels
    assert fixture.eval is not None
    assert fixture.eval.sequences == bundle.eval_dataset.sequences
    assert fixture.eval.labels == bundle.eval_dataset.labels


def test_build_nca_reference_fixture_from_rows_validates_row_counts() -> None:
    task = create_task("nca", {"preset": "smoke"})

    try:
        build_nca_reference_fixture_from_rows(
            task,
            train_sequences=[[36, 1, 2]],
            train_labels=[[-100, -100, -100], [-100, 1, 2]],
        )
    except ValueError as exc:
        assert "same number of rows" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected row-count validation to fail for NCA reference rows.")


def _build_fake_nca_reference_repo(repo_root: Path, bundle) -> Path:
    repo_root.mkdir(parents=True)
    (repo_root / "utils").mkdir()
    payload = {
        "train": {
            "raw_sequences": [
                _recover_raw_nca_sequence(input_ids, labels)
                for input_ids, labels in zip(bundle.train_dataset.sequences, bundle.train_dataset.labels)
            ]
        },
        "eval": {
            "raw_sequences": [
                _recover_raw_nca_sequence(input_ids, labels)
                for input_ids, labels in zip(bundle.eval_dataset.sequences, bundle.eval_dataset.labels)
            ]
        },
    }
    (repo_root / "fixture_payload.json").write_text(json.dumps(payload), encoding="utf-8")
    (repo_root / "utils" / "__init__.py").write_text("", encoding="utf-8")
    (repo_root / "utils" / "nca.py").write_text(
        "\n".join(
            [
                "def generate_rules_batch(seed, num_rules, tokenizer=None, **kwargs):",
                "    return list(range(num_rules))",
                "",
                "def generate_nca_dataset(seed, num_sims, grid=12, d_state=10, n_groups=1, identity_bias=0.0, temperature=0.0, num_examples=1, num_rules=1, dT=1, start_step=0, rule_seeds=None):",
                "    split = 'train' if (rule_seeds is None or int(rule_seeds[0]) == 0) else 'eval'",
                "    return {'split': split, 'count': num_sims}",
            ]
        ),
        encoding="utf-8",
    )
    (repo_root / "utils" / "tokenizers.py").write_text(
        "\n".join(
            [
                "import json",
                "from pathlib import Path",
                "",
                "_PAYLOAD = json.loads((Path(__file__).resolve().parents[1] / 'fixture_payload.json').read_text(encoding='utf-8'))",
                "",
                "class NCA_Tokenizer:",
                "    def __init__(self, patch, num_colors=10):",
                "        self.patch = patch",
                "        self.num_colors = num_colors",
                "",
                "    def encode_task(self, sims):",
                "        split = sims['split']",
                "        rows = _PAYLOAD[split]['raw_sequences']",
                "        return rows, rows",
            ]
        ),
        encoding="utf-8",
    )
    return repo_root


def _recover_raw_nca_sequence(input_ids: list[int], labels: list[int]) -> list[int]:
    if labels and int(labels[-1]) != -100:
        tail_token = int(labels[-1])
    else:
        tail_token = int(input_ids[-1])
    return [*map(int, input_ids), tail_token]
