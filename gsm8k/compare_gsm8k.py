"""Recompute paired GSM8K results and enforce the declared regression gates."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re


def answer(text):
    values = re.findall(r"\d+", text.replace(",", ""))
    if not values:
        return -9999999
    try:
        return ast.literal_eval(values[-1])
    except SyntaxError:
        return -9999999


def read(root):
    result = json.loads((root / "result.json").read_text())
    rows = [json.loads(s) for s in (root / "predictions.jsonl").read_text().splitlines()]
    assert result["status"] == "complete" and len(rows) == 1319
    assert [r["id"] for r in rows] == list(range(1319))
    for row in rows:
        assert answer(row["text"]) == row["prediction"]
        assert row["correct"] == (row["prediction"] == row["label"])
    correct = sum(r["correct"] for r in rows)
    invalid = sum(r["prediction"] == -9999999 for r in rows)
    truncated = sum(r["finish_reason"] == "length" for r in rows)
    assert correct / len(rows) == result["metrics"]["accuracy"]
    return result, rows, {"correct": correct, "questions": len(rows), "accuracy": correct / len(rows), "invalid": invalid, "truncated": truncated}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    data = {m: read(args.root / m) for m in ["baseline", "bf16", "fp16"]}
    base_result, base_rows, base_metrics = data["baseline"]
    summary = {"criterion": base_result["criterion"], "baseline": base_metrics, "comparisons": {}}
    passed = base_metrics["accuracy"] >= 0.80 and base_metrics["invalid"] / 1319 <= 0.01 and base_metrics["truncated"] / 1319 <= 0.01
    for mode in ["bf16", "fp16"]:
        result, rows, metrics = data[mode]
        assert result["sampling"] == base_result["sampling"]
        assert result["dataset"] == base_result["dataset"]
        assert result["chat_template_kwargs"] == base_result["chat_template_kwargs"]
        assert result["prompts_sha256"] == base_result["prompts_sha256"]
        assert result["labels_sha256"] == base_result["labels_sha256"]
        assert result['packages'] == base_result['packages']
        settings = json.loads(json.dumps(result['settings']))
        settings['attention_config'].pop('minimax_m3_msa_decode_fp16_dot', None)
        assert settings.pop('worker_extension_cls') == 'gsm8k_cache_seed.FlashInferCacheSeed'
        assert settings == base_result['settings']
        lost = gained = same_tokens = 0
        discordant = []
        for old, new in zip(base_rows, rows, strict=True):
            assert (old["id"], old["prompt"], old["prompt_token_ids"], old["label"]) == (new["id"], new["prompt"], new["prompt_token_ids"], new["label"])
            lost += old["correct"] and not new["correct"]
            gained += new["correct"] and not old["correct"]
            same_tokens += old["token_ids"] == new["token_ids"]
            if old["correct"] != new["correct"]:
                discordant.append({"id": new["id"], "baseline_correct": old["correct"], "candidate_correct": new["correct"]})
        n = lost + gained
        p_value = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(lost, gained) + 1)) / (2 ** n)) if n else 1.0
        drop = (base_metrics["correct"] - metrics["correct"]) / 1319
        gate = drop <= 0.005 and metrics["invalid"] / 1319 <= 0.01 and metrics["truncated"] / 1319 <= 0.01
        passed &= gate
        summary["comparisons"][mode] = {**metrics, "accuracy_delta_percentage_points": -100 * drop,
            "lost_correct": lost, "gained_correct": gained, "same_output_token_sequences": same_tokens,
            "mcnemar_exact_two_sided_p": p_value, "regression_gate_passed": gate,
            "discordant_examples": discordant,
            "predictions_sha256": hashlib.sha256((args.root / mode / "predictions.jsonl").read_bytes()).hexdigest()}
    summary["passed"] = passed
    (args.root / "comparison.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
