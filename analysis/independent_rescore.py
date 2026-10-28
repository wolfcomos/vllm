"""CPU-only rescore of raw predictions against the pinned GSM8K data archive."""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tarfile


def answer(text):
    # Preserve the repository's final-integer scoring convention.
    numbers = re.findall(r"\d+", text.replace(",", ""))
    if not numbers:
        return -9999999
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return -9999999


def digest(value):
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


def audit(arm_dir, notes=None):
    if notes is None:
        notes = Path(__file__).resolve().parent.parent
    manifest = json.loads((notes / "gsm8k/dataset-manifest.json").read_text())
    datasets = {}
    with tarfile.open(notes / "pinned-gsm8k-dataset.tar") as archive:
        for name, expected in manifest["files"].items():
            raw = archive.extractfile(name).read()
            assert hashlib.sha256(raw).hexdigest() == expected
            datasets[name] = [json.loads(line) for line in raw.splitlines()]
    train, test = datasets["train.jsonl"], datasets["test.jsonl"]
    assert len(train) == 7473 and len(test) == 1319
    prefix = "".join(f"Question: {row['question']}\nAnswer: {row['answer']}\n\n" for row in train[:5])
    prompts = [prefix + f"Question: {row['question']}\nAnswer:" for row in test]
    labels = [answer(row["answer"]) for row in test]
    assert digest(prompts) == manifest["prompts_sha256"]
    assert digest(labels) == manifest["labels_sha256"]
    result = json.loads((arm_dir / "result.json").read_text())
    raw = (arm_dir / "predictions.jsonl").read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    assert result["status"] == "complete" and len(rows) == 1319
    assert [row["id"] for row in rows] == list(range(1319))
    assert result["prompts_sha256"] == digest(prompts)
    assert result["labels_sha256"] == digest(labels)
    correct, invalid, truncated, total_tokens = 0, 0, 0, 0
    wrong_ids = []
    for row, prompt, label in zip(rows, prompts, labels, strict=True):
        prediction = answer(row["text"])
        assert row["prompt"] == prompt and row["label"] == label
        assert row["prediction"] == prediction and row["correct"] == (prediction == label)
        assert row["prompt_token_ids"] and 0 < len(row["token_ids"]) <= 1024
        assert row["finish_reason"] in ("stop", "length")
        correct += prediction == label
        invalid += prediction == -9999999
        truncated += row["finish_reason"] == "length"
        total_tokens += len(row["token_ids"])
        if prediction != label:
            wrong_ids.append(row["id"])
    assert correct / 1319 == result["metrics"]["accuracy"]
    return {
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": result["mode"], "source_commit": result["source_commit"],
        "questions": 1319, "correct": correct, "accuracy": correct / 1319,
        "invalid": invalid, "truncated": truncated, "output_tokens": total_tokens,
        "wrong_ids": wrong_ids, "prompts_rebuilt_from_pinned_archive": True,
        "labels_rebuilt_from_pinned_archive": True,
        "predictions_sha256": hashlib.sha256(raw).hexdigest(),
        "output_token_sequences_sha256": digest([row["token_ids"] for row in rows]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm_dir", type=Path)
    parser.add_argument("--notes", type=Path)
    args = parser.parse_args()
    report = audit(args.arm_dir, args.notes)
    (args.arm_dir / "independent-rescore.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "wrong_ids"}, indent=2))
