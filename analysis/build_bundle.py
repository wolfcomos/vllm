"""Build a bounded public evidence projection; never pushes or publishes it."""

import argparse
import ast
import csv
from datetime import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

ARMS = ("baseline", "default-a", "fp16-a", "fp16-b", "default-b", "cutlass")
QUALITY_ARMS = ("baseline", "bf16", "fp16")
METRICS = ("seconds", "mean_tpot_seconds", "mean_ttft_seconds", "common_tail_seconds", "common_tail_tpot_seconds")
E2E_PROTOCOL = "gb200-current-main-e2e-v4"
GSM8K_PROTOCOL = "gb200-current-main-gsm8k-v4"
MODEL = "nvidia/MiniMax-M3-NVFP4"
REVISION = "901464083161bf8612a29ff7ad29914cd4ab4a85"
HEX = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
SIMPLE = re.compile(r"[A-Za-z0-9_.+:-]+")
VERSION = re.compile(r"[A-Za-z0-9_.+-]+")
SOURCE_PATH = re.compile(r"(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.py")
STATUS = {"starting", "running", "complete", "failed", "smoke_passed"}
PACKAGE_NAMES = {"vllm", "torch", "triton", "nvidia-cutlass-dsl", "flashinfer-python", "transformers", "tokenizers", "quack-kernels", "depyf"}
HARNESSES = {
    "e2e": ("e2e_support.py", "run_e2e.py", "compare_e2e.py", "make_manifest.py", "run_sequence.sh", "PROTOCOL.md"),
    "gsm8k": ("run_gsm8k.py", "gsm8k_cache_seed.py", "compare_gsm8k.py", "check_cache_log.py", "run_all.sh", "PROTOCOL.md", "dataset-manifest.json"),
}
HISTORY_V3_MANIFEST_SHA256 = "c18b5381ab64f6742d98d39a8a3bda49f4dc17066ec4daca5885b30c77786a47"
HISTORY_V3_FILES = (
    "LICENSE", "PROTOCOL.md", "README.md", "SHA256SUMS.json", "frozen-comparison.json",
    "harness/compare_e2e.py", "harness/orchestrate.py", "harness/run_e2e.py",
    "observations.json", "posthoc-tail.json", "provenance.json", "raw-evidence-sha256.json", "redaction-audit.json",
)


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def value_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def checksum(value):
    assert isinstance(value, str) and HEX.fullmatch(value), "Invalid source/digest field"
    return value


def number(value):
    assert isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    return value


def flag(value):
    assert isinstance(value, bool)
    return value


def scalar(value):
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return number(value)
    assert isinstance(value, str) and SIMPLE.fullmatch(value), "Unexpected scalar string"
    return value


def simple_tree(value):
    """For a fixed schema's configuration fields; never pass arbitrary results."""
    if isinstance(value, list):
        return [simple_tree(item) for item in value]
    if isinstance(value, dict):
        assert all(SIMPLE.fullmatch(key) for key in value)
        return {key: simple_tree(item) for key, item in value.items()}
    return scalar(value)


def version(value):
    assert isinstance(value, str) and VERSION.fullmatch(value), "Unexpected runtime version"
    return value


def hashes(mapping):
    assert isinstance(mapping, dict)
    assert all(SOURCE_PATH.fullmatch(name) or SIMPLE.fullmatch(name) for name in mapping)
    return {name: checksum(value) for name, value in mapping.items()}


def runtime(value):
    assert set(value["packages"]).issubset(PACKAGE_NAMES), "Unexpected package metadata requires explicit projection review"
    projected = {"torch": version(value["torch"]), "cuda": version(value["cuda"]),
                 "packages": {name: version(v) for name, v in value["packages"].items()}}
    assert all(SIMPLE.fullmatch(name) for name in projected["packages"])
    if "native_binaries" in value:
        projected["native_binary_sha256"] = hashes(value["native_binaries"])
    return projected


def settings(value):
    assert value["revision"] == value["tokenizer_revision"] == REVISION
    assert value["model"] == MODEL or Path(value["model"]).name == "MiniMax-M3-NVFP4"
    keys = ("language_model_only", "revision", "tokenizer_revision", "tensor_parallel_size", "distributed_executor_backend",
            "dtype", "kv_cache_dtype", "block_size", "seed", "enforce_eager", "max_model_len", "max_num_seqs",
            "max_num_batched_tokens", "kv_cache_memory_bytes", "enable_chunked_prefill", "enable_prefix_caching",
            "generation_config", "disable_log_stats", "cudagraph_metrics", "compilation_config", "kernel_config")
    result = {key: simple_tree(value[key]) for key in keys if key in value}
    result["model"] = MODEL
    attn = value["attention_config"]
    result["attention_config"] = {key: scalar(attn[key]) for key in (
        "indexer_kv_dtype", "minimax_m3_msa_decode_backend", "minimax_m3_msa_decode_fp16_dot") if key in attn}
    return result


def environment(value):
    return {key: scalar(value[key]) for key in ("OMP_NUM_THREADS", "VLLM_BATCH_INVARIANT", "VLLM_USE_V2_MODEL_RUNNER",
            "VLLM_ENABLE_V1_MULTIPROCESSING", "VLLM_WORKER_MULTIPROC_METHOD") if key in value}


def cache_records(records):
    return [{"rank": number(row["rank"]), "sha256": checksum(row["sha256"]),
             "configuration_count": number(row["configuration_count"]),
             "metadata_sha256": hashlib.sha256(json.dumps(row["metadata"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
            for row in records]


def timestamp(value):
    assert isinstance(value, str)
    datetime.fromisoformat(value)
    return value


def trial(row):
    result = {key: number(row[key]) for key in ("batch", "repeat", "seconds", "output_tokens", "output_tokens_per_second",
              "mean_tpot_seconds", "mean_ttft_seconds", "common_tail_seconds", "common_tail_tpot_seconds", "tail_token_intervals") if key in row}
    assert row["stage"] in ("warmup", "measured")
    result["stage"] = row["stage"]
    for key in ("started_at_utc", "finished_at_utc"):
        if key in row:
            result[key] = timestamp(row[key])
    for key in ("output_token_ids_sha256", "input_before_sha256", "input_after_sha256", "returned_prompts_sha256", "normalized_sampling_sha256"):
        if key in row:
            result[key] = checksum(row[key])
    native = row.get("request_metrics", [])
    if native:
        origin = min(m["queued_ts_monotonic"] for m in native)
        wall_origin = min(m["arrival_time_wall"] for m in native)
        result["request_observations"] = []
        for metric in native:
            record = {key: number(metric[key]) for key in ("num_generation_tokens", "num_preemptions", "ttft_seconds", "queue_seconds",
                      "prefill_seconds", "decode_seconds", "mean_tpot_seconds") if key in metric}
            for key in ("queued_ts_monotonic", "scheduled_ts_monotonic", "first_token_ts_monotonic", "last_token_ts_monotonic"):
                record[key.replace("_ts_monotonic", "_offset_seconds")] = number(metric[key] - origin)
            record["arrival_offset_seconds"] = number(metric["arrival_time_wall"] - wall_origin)
            result["request_observations"].append(record)
        first = max(m["first_token_ts_monotonic"] for m in native)
        last = max(m["last_token_ts_monotonic"] for m in native)
        # Anonymous array positions are not the runtime request identifiers.
        result["tail_witness_positions"] = [i for i, m in enumerate(native)
                                            if m["first_token_ts_monotonic"] == first and m["last_token_ts_monotonic"] == last]
        if "common_tail_seconds" in row:
            assert row["common_tail_seconds"] == last - first and result["tail_witness_positions"]
    return result


def summaries(value):
    result = {}
    for arm, by_batch in value.items():
        assert arm in ARMS
        result[arm] = {}
        for batch, by_metric in by_batch.items():
            assert batch in ("8", "40")
            result[arm][batch] = {}
            for metric, row in by_metric.items():
                assert metric in METRICS
                result[arm][batch][metric] = {key: [number(v) for v in item] if key == "values" else number(item)
                                            for key, item in row.items() if key in ("values", "median", "min", "max", "relative_mad")}
    return result


def e2e_comparison(raw):
    assert raw["protocol"] == E2E_PROTOCOL and raw["status"] == "complete"
    result = {"status": "complete", "protocol": E2E_PROTOCOL, "integrity_passed": flag(raw["integrity_passed"]),
              "result_sha256": hashes(raw["result_sha256"]), "statistics": summaries(raw["statistics"]),
              "ratios": [], "decisions": {}, "cutlass_matches_triton_baseline": simple_tree(raw["cutlass_matches_triton_baseline"]),
              "cutlass_output_hashes": hashes(raw["cutlass_output_hashes"])}
    for row in raw["ratios"]:
        assert row["numerator"] in ARMS and row["denominator"] in ARMS and row["metric"] in METRICS
        result["ratios"].append({key: scalar(row[key]) for key in ("batch", "metric", "numerator", "denominator", "ratio", "percent_change")})
    for batch, decision in raw["decisions"].items():
        assert batch in ("8", "40")
        projected = {key: scalar(decision[key]) for key in ("maximum_control_tail_drift", "review_evidence_gate_passed",
                     "repeatability_passed", "previous_3pct_mean_tpot_rule_passed")}
        for group, keys in (("primary_comparisons", ("fp16", "default", "tail_benefit", "required_benefit_greater_than", "e2e_ratio", "passed")),
                            ("previous_3pct_mean_tpot_comparisons", ("fp16", "default", "mean_tpot_ratio", "passed"))):
            projected[group] = [{key: scalar(row[key]) for key in keys} for row in decision[group]]
        projected["repeatability"] = []
        for row in decision["repeatability"]:
            assert re.fullmatch(r"(baseline|default-[ab]|fp16-[ab]|cutlass)/(seconds|common_tail_seconds)/MAD", row["name"])
            projected["repeatability"].append({"name": row["name"], "value": number(row["value"]), "passed": flag(row["passed"])})
        result["decisions"][batch] = projected
    return result


def quality_comparison(raw):
    assert raw["protocol"] == GSM8K_PROTOCOL
    metric_keys = ("correct", "questions", "accuracy", "invalid", "truncated")
    result = {"protocol": GSM8K_PROTOCOL, "passed": flag(raw["passed"]), "criterion": simple_tree(raw["criterion"]),
              "baseline": {key: number(raw["baseline"][key]) for key in metric_keys}, "comparisons": {},
              "source_commits": hashes(raw["source_commits"]), "result_sha256": hashes(raw["result_sha256"])}
    for arm in ("bf16", "fp16"):
        row = raw["comparisons"][arm]
        result["comparisons"][arm] = {key: number(row[key]) for key in (*metric_keys, "accuracy_delta_percentage_points",
            "lost_correct", "gained_correct", "same_output_token_sequences", "mcnemar_exact_two_sided_p")}
        result["comparisons"][arm].update(regression_gate_passed=flag(row["regression_gate_passed"]), predictions_sha256=checksum(row["predictions_sha256"]))
    return result


def privacy_scan(name, data):
    text = data.decode("utf-8")
    forbidden = (r"/Users/", r"/home/[^\s\"']+", r"/raid/", r"/mnt/[^\s\"']+", r"gitlab-master", r"\.nvidia\.com(?::|/)",
                 r"remote-ssh", r"GPU-[0-9a-fA-F-]{16,}", r"\b(?:\d{1,3}\.){3}\d{1,3}\b", r"\b\d{8}-\d{4}-\d{4}-\d{4}-\d{12}\b")
    for pattern in forbidden:
        assert not re.search(pattern, text), f"Private/infrastructure pattern in public file {name}"
    if name.endswith(".json"):
        def walk(value):
            if isinstance(value, dict):
                assert not set(value).intersection({"request_id", "request_ids", "queued_request_ids", "gpu_uuid", "hostname", "host", "job_id",
                                                   "source", "root", "path", "actual_file", "prompt", "text", "token_ids", "prompt_token_ids", "traceback"}), name
                for item in value.values():
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)
        walk(json.loads(text))


class Builder:
    def __init__(self, notes, repository, allow_incomplete, e2e_telemetry=None):
        self.notes = notes
        self.raw = notes / "remote-results"
        self.repository = repository
        self.allow_incomplete = allow_incomplete
        self.bundle = {}
        self.availability = {}
        self.evidence_hashes = {}
        self.telemetry_paths = {"gsm8k": self.raw / "results/gpu-telemetry.csv",
                                "e2e": e2e_telemetry or self.raw / "ready/gpu-telemetry.csv"}
        assert self.telemetry_paths["gsm8k"].resolve() != self.telemetry_paths["e2e"].resolve(), "Telemetry scopes require distinct inputs"
        self.harness_roots = {
            "e2e": self.raw / "ready/harness" if (self.raw / "ready/harness/run_e2e.py").exists() else notes / "harness",
            "gsm8k": self.raw / "gsm8k" if (self.raw / "gsm8k/run_gsm8k.py").exists() else notes / "gsm8k",
        }

    def record(self, label, path):
        self.evidence_hashes[label] = sha(path)

    def get(self, label, path, terminal=True):
        if not path.exists():
            self.availability[label] = {"state": "unavailable", "reason": "No collected artifact"}
            if not self.allow_incomplete:
                raise RuntimeError(f"Missing required evidence: {label}")
            return None
        self.record(label, path)
        raw = load(path)
        if "status" in raw:
            assert raw["status"] in STATUS, (label, "Unknown status")
            if terminal and raw["status"] != "complete":
                self.availability[label] = {"state": "incomplete", "source_status": raw["status"]}
                if not self.allow_incomplete:
                    raise RuntimeError(f"Incomplete required evidence: {label}")
            else:
                self.availability[label] = {"state": "available", "source_status": raw["status"]}
        else:
            self.availability[label] = {"state": "available"}
        return raw

    def put(self, name, value):
        self.bundle[name] = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()

    def verify_harness(self, group, expected):
        for name, expected_sha in expected.items():
            assert name in HARNESSES[group], (group, "Unexpected executed harness file")
            assert sha(self.harness_roots[group] / name) == expected_sha, (group, name, "Executed harness differs")

    def static(self):
        pins = load(self.notes / "source-pins.json")
        result = {}
        for kind in ("baseline", "candidate"):
            row = pins[kind]
            result[kind] = {"commit": checksum(row["commit"]), "source_hashes": hashes(row["source_hashes"])}
            for relative, expected in result[kind]["source_hashes"].items():
                data = subprocess.check_output(["git", "-C", str(self.repository), "show", f"{row['commit']}:vllm/{relative}"])
                assert hashlib.sha256(data).hexdigest() == expected
                if relative.endswith("/ops/sparse_attn.py"):
                    self.bundle[f"sources/{kind}/sparse_attn.py"] = data
        self.put("source-pins.json", result)
        self.pins = result
        for group, names in HARNESSES.items():
            for name in names:
                self.bundle[f"{group}/{name}"] = (self.harness_roots[group] / name).read_bytes()
        self.bundle["operator-validation/run_tests.py"] = (self.notes / "launch/run_tests.py").read_bytes()
        self.bundle["LICENSE"] = subprocess.check_output(["git", "-C", str(self.repository), "show", f"{pins['baseline']['commit']}:LICENSE"])
        return result

    def tests(self):
        status_path = self.raw / "tests-ready/status.json"
        status = self.get("tests/status", status_path, terminal=False)
        report = {"scope": "Full original attention suites, without test filtering", "arms": {}}
        for arm in ("baseline", "candidate"):
            path = self.raw / f"tests-ready/{arm}.xml"
            if not path.exists():
                self.availability[f"tests/{arm}"] = {"state": "unavailable", "reason": "No collected JUnit XML"}
                if not self.allow_incomplete:
                    raise RuntimeError(f"Missing test XML for {arm}")
                continue
            self.record(f"tests/{arm}.xml", path)
            cases, counts = [], {"passed": 0, "failed": 0, "error": 0, "skipped": 0}
            for item in ET.parse(path).getroot().iter("testcase"):
                result = next((tag for tag in ("error", "failure", "skipped") if item.find(tag) is not None), "passed")
                result = "failed" if result == "failure" else result
                counts[result] += 1
                # Names derive from repository tests; diagnostics/tracebacks are deliberately excluded.
                classname, name = item.get("classname", ""), item.get("name", "")
                if result == "error" and not classname.startswith("tests.kernels."):
                    # Pytest collection errors can have empty classnames or path labels.
                    case = {"label_scope": "Anonymous collection error; original label retained privately"}
                else:
                    assert classname.startswith("tests.kernels.")
                    assert re.fullmatch(r"[A-Za-z0-9_.:\[\](), +='-]+", name), "Unexpected test label"
                    case = {"classname": classname, "name": name}
                case.update(status=result, seconds=number(float(item.get("time", "0"))))
                cases.append(case)
            assert cases, "Empty JUnit evidence"
            row = {"counts": counts, "case_count": len(cases), "cases": cases}
            matching = [run for run in status["runs"] if run["arm"] == arm] if status else []
            assert len(matching) <= 1
            if matching and "exit_code" in matching[0]:
                row["exit_code"] = number(matching[0]["exit_code"])
                self.availability[f"tests/{arm}"] = {"state": "available"}
            else:
                self.availability[f"tests/{arm}"] = {"state": "incomplete", "reason": "No final test process exit code"}
                if not self.allow_incomplete:
                    raise RuntimeError(f"Test process has no final exit code: {arm}")
            report["arms"][arm] = row
        self.put("operator-validation/results.json", report)

    def quality(self):
        report = {"protocol": GSM8K_PROTOCOL, "arms": {}}
        private_answers = {}
        for arm in QUALITY_ARMS:
            raw = self.get(f"gsm8k/{arm}", self.raw / f"gsm8k/{arm}/result.json")
            if raw is None:
                continue
            assert raw["protocol"] == GSM8K_PROTOCOL and raw["mode"] == arm
            assert raw["source_commit"] == self.pins["baseline" if arm == "baseline" else "candidate"]["commit"]
            self.verify_harness("gsm8k", raw["harness_hashes"])
            projected = {"status": raw["status"], "source_commit": checksum(raw["source_commit"]), "settings": settings(raw["settings"]),
                         "runtime": runtime(raw), "criterion": simple_tree(raw["criterion"]), "sampling": simple_tree(raw["sampling"]),
                         "harness_hashes": hashes(raw["harness_hashes"]), "source_hashes": hashes(raw["source_hashes"]),
                         "api_hashes": hashes(raw["api_hashes"])}
            projected["runtime_flags"] = environment(raw.get("environment", {}))
            for key in ("startup_seconds", "smoke_seconds"):
                if key in raw:
                    projected[key] = number(raw[key])
            if "metrics" in raw:
                projected["metrics"] = {key: number(raw["metrics"][key]) for key in (
                    "accuracy", "invalid_rate", "truncation_rate", "total_output_tokens", "num_questions", "num_shots", "max_tokens", "latency") if key in raw["metrics"]}
                projected["elapsed_scope"] = "Instrumented quality evaluation; not a performance claim"
            for key in ("prompts_sha256", "labels_sha256", "predictions_sha256", "repository_eval_sha256", "dataset_manifest_sha256"):
                if key in raw:
                    projected[key] = checksum(raw[key])
            for key in ("autotune_cache_after_startup", "autotune_cache_after_generation"):
                if key in raw:
                    projected[key] = cache_records(raw[key])
            if "full_dispatch" in raw:
                projected["actual_decode_dispatch"] = []
                for worker in raw["full_dispatch"]:
                    row = {key: scalar(worker[key]) for key in ("rank", "launches", "split_sensitive_shape_launches", "expected_policy", "max_total_q")}
                    row["variants"] = []
                    for key, count in worker["variants"].items():
                        meta = json.loads(key)
                        allowed = ("grid", "q_shape", "q_dtype", "kv_dtype", "total_q", "num_kv_heads", "topk", "USE_FP16_DOT", "USE_FP8", "KV_SCALE_MODE", "NUM_TOPK_CHUNKS")
                        row["variants"].append({"metadata": {field: simple_tree(meta[field]) for field in allowed}, "launches": number(count)})
                    projected["actual_decode_dispatch"].append(row)
            if "shutdown_completed" in raw:
                projected["shutdown_completed"] = flag(raw["shutdown_completed"])
            if raw["status"] == "complete":
                answers_path = self.raw / f"gsm8k/{arm}/predictions.jsonl"
                assert answers_path.exists(), "Complete GSM8K result lacks raw scoring evidence"
                self.record(f"gsm8k/{arm}/predictions", answers_path)
                assert sha(answers_path) == raw["predictions_sha256"]
                rows = [json.loads(line) for line in answers_path.read_text().splitlines()]
                assert len(rows) == 1319 and [row["id"] for row in rows] == list(range(1319))
                for row in rows:
                    values = re.findall(r"\d+", row["text"].replace(",", ""))
                    try:
                        prediction = ast.literal_eval(values[-1]) if values else -9999999
                    except SyntaxError:
                        prediction = -9999999
                    assert prediction == row["prediction"] and row["correct"] == (prediction == row["label"])
                assert sum(row["correct"] for row in rows) / 1319 == raw["metrics"]["accuracy"]
                assert sum(row["prediction"] == -9999999 for row in rows) / 1319 == raw["metrics"]["invalid_rate"]
                assert sum(row["finish_reason"] == "length" for row in rows) / 1319 == raw["metrics"]["truncation_rate"]
                projected["raw_answers_rescored"] = True
                private_answers[arm] = rows
            report["arms"][arm] = projected
        self.put("gsm8k/results.json", report)
        raw = self.get("gsm8k/comparison", self.raw / "gsm8k/comparison.json")
        if raw is not None:
            for arm, expected in raw["result_sha256"].items():
                assert self.evidence_hashes[f"gsm8k/{arm}"] == expected
            assert set(private_answers) == set(QUALITY_ARMS), "Comparison lacks full raw scoring evidence"
            base = private_answers["baseline"]
            for arm in ("bf16", "fp16"):
                candidate = private_answers[arm]
                paired = list(zip(base, candidate, strict=True))
                assert all((a["id"], a["label"], a["prompt_token_ids"]) == (b["id"], b["label"], b["prompt_token_ids"]) for a, b in paired)
                check = raw["comparisons"][arm]
                assert sum(a["token_ids"] == b["token_ids"] for a, b in paired) == check["same_output_token_sequences"]
                assert sum(a["correct"] and not b["correct"] for a, b in paired) == check["lost_correct"]
                assert sum(not a["correct"] and b["correct"] for a, b in paired) == check["gained_correct"]
            self.put("gsm8k/comparison.json", quality_comparison(raw))

    def e2e(self):
        report = {"protocol": E2E_PROTOCOL, "arms": {},
                  "request_timing_projection": "All request timing observations retained in source array order; monotonic origins shifted to earliest queued timestamp per trial, arrival origins shifted independently. Runtime request IDs removed.",
                  "backend_reference_scope": "The arm named cutlass uses current-main default backend and native Triton fallback. Static eligibility and policy snapshots do not prove actual per-step kernel counts."}
        reference_hashes = {}
        for arm in ARMS:
            raw = self.get(f"e2e/{arm}", self.raw / f"ready/results/{arm}/result.json")
            if raw is None:
                continue
            assert raw["protocol"] == E2E_PROTOCOL and raw["mode"] == arm
            assert raw["source_commit"] == self.pins["baseline" if arm in ("baseline", "cutlass") else "candidate"]["commit"]
            self.verify_harness("e2e", raw["harness_hashes"])
            projected = {"status": raw["status"], "source_commit": checksum(raw["source_commit"]), "settings": settings(raw["settings"]),
                         "sampling": simple_tree(raw["sampling"]), "harness_hashes": hashes(raw["harness_hashes"]),
                         "batches": [number(v) for v in raw["batches"]], "input_len": number(raw["input_len"]),
                         "output_len": number(raw["output_len"]), "warmup_count": number(raw["warmup_count"]), "repeats": number(raw["repeats"]),
                         "trials": [trial(row) for row in raw.get("trials", [])], "warmups": [trial(row) for row in raw.get("warmups", [])]}
            projected["runtime_flags"] = environment(raw.get("environment", {}))
            for key in ("source_hashes", "api_hashes", "references"):
                if key in raw:
                    projected[key] = hashes(raw[key])
            for key in ("manifest_sha256", "prompts_sha256", "model_revision"):
                projected[key] = checksum(raw[key])
            if "runtime" in raw:
                projected["runtime"] = runtime(raw["runtime"])
            for key in ("cache_after_startup", "cache_after_generation"):
                if key in raw:
                    projected[key] = cache_records(raw[key])
            if "shutdown_completed" in raw:
                projected["shutdown_completed"] = flag(raw["shutdown_completed"])
            if "workers" in raw:
                projected["workers"] = []
                for worker in raw["workers"]:
                    assert "GB200" in worker["gpu"]
                    row = {"rank": number(worker["rank"]), "gpu_model": "NVIDIA GB200", "num_sms": number(worker["num_sms"]),
                           "capability": simple_tree(worker["capability"]), "resolved_graph_mode": scalar(worker["resolved_graph_mode"]),
                           "captured_entries": simple_tree(worker["captured_entries"]), "dispatch": simple_tree(worker["dispatch"])}
                    row["implementations"] = [{key: simple_tree(impl[key]) for key in ("class", "fp16_dot", "use_cutlass_decode", "kv_cache_dtype", "k_scale_numel", "v_scale_numel") if key in impl}
                                              for impl in worker["implementations"]]
                    if "native_backend_policies" in worker:
                        row["native_backend_policies"] = [{key: simple_tree(policy[key]) for key in ("geometry", "min_batch_override", "shapes")}
                                                           for policy in worker["native_backend_policies"]]
                    projected["workers"].append(row)
            reference_hashes[arm] = {}
            observed_hashes, integrity_failures = {}, []
            rows = raw.get("warmups", []) + raw.get("trials", [])
            for row in rows:
                assert re.fullmatch(r"tokens-(warmup|measured)-b(8|40)-r\d+\.json", row["tokens_file"])
                token_path = self.raw / f"ready/results/{arm}" / row["tokens_file"]
                tokens = load(token_path)
                assert len(tokens) == row["batch"] and all(len(seq) == raw["output_len"] for seq in tokens)
                actual = value_hash(tokens)
                assert actual == row["output_token_ids_sha256"]
                if row["stage"] == "measured":
                    key = str(row["batch"])
                    reference_hashes[arm].setdefault(key, actual)
                    observed_hashes.setdefault(key, []).append(actual)
                    if actual != reference_hashes[arm][key]:
                        integrity_failures.append({"check": "measured_within_arm_outputs", "batch": number(row["batch"]),
                                                   "repeat": number(row["repeat"]), "reference_sha256": reference_hashes[arm][key],
                                                   "observed_sha256": actual})
            projected["verified_trial_output_count"] = len(rows)
            if rows:
                projected["raw_trial_output_hashes_verified"] = True
            projected["measured_output_consistency"] = {
                "scope": "Collected measured trials only; this does not establish full protocol integrity",
                "state": "failed" if integrity_failures else "consistent_observations" if observed_hashes else "unavailable",
                "batches": {key: {"trial_count": len(values), "unique_output_hashes": sorted(set(values))}
                            for key, values in observed_hashes.items()},
            }
            if integrity_failures:
                projected["integrity_failures"] = integrity_failures
                self.availability[f"e2e/{arm}"].update(state="incomplete", observed_integrity_passed=False)
                if not self.allow_incomplete:
                    raise RuntimeError(f"Measured within-arm outputs vary: {arm}")
            report["arms"][arm] = projected
        self.put("e2e/results.json", report)
        raw = self.get("e2e/comparison", self.raw / "ready/comparison.json")
        if raw is not None:
            for arm, expected in raw["result_sha256"].items():
                assert self.evidence_hashes[f"e2e/{arm}"] == expected
            assert all(row["status"] == "complete" and not row.get("integrity_failures") for row in report["arms"].values()), "Comparison contradicts incomplete or failed evidence"
            assert reference_hashes["baseline"] == reference_hashes["default-a"] == reference_hashes["default-b"]
            assert reference_hashes["fp16-a"] == reference_hashes["fp16-b"]
            self.put("e2e/comparison.json", e2e_comparison(raw))

    def history(self):
        root = self.notes / "publication/historical-v3"
        if not root.exists():
            self.availability["history/v3"] = {"state": "unavailable", "reason": "Audited historical supplement not collected"}
            if not self.allow_incomplete:
                raise RuntimeError("Missing required evidence: history/v3")
            return
        entries = list(root.rglob("*"))
        assert not any(path.is_symlink() for path in entries), "Historical supplement contains symlinks"
        actual = {str(path.relative_to(root)) for path in entries if path.is_file()}
        assert actual == set(HISTORY_V3_FILES), "Historical supplement file list differs from audit"
        manifest_path = root / "SHA256SUMS.json"
        assert sha(manifest_path) == HISTORY_V3_MANIFEST_SHA256, "Historical manifest differs from independent audit"
        manifest = load(manifest_path)
        assert set(manifest) == set(HISTORY_V3_FILES) - {"SHA256SUMS.json"}
        for name, expected in manifest.items():
            assert sha(root / name) == checksum(expected), "Historical artifact checksum mismatch"
        for name in HISTORY_V3_FILES:
            self.bundle[f"history/v3/{name}"] = (root / name).read_bytes()
        self.record("history/v3/manifest", manifest_path)
        self.availability["history/v3"] = {"state": "available", "files": len(HISTORY_V3_FILES),
                                           "original_performance_passed": False}

    def telemetry(self):
        columns = ("timestamp", "index", "pstate", "temperature.gpu", "power.draw [W]", "power.limit [W]",
                   "clocks.current.sm [MHz]", "clocks.current.memory [MHz]", "utilization.gpu [%]", "utilization.memory [%]", "memory.used [MiB]")
        bitmask_aliases = ("clocks_event_reasons.active", "clocks_throttle_reasons.active")
        for scope, path in self.telemetry_paths.items():
            label = f"telemetry/{scope}"
            if not path.exists():
                self.availability[label] = {"state": "unavailable", "reason": "No collected sampler CSV", "scope": scope}
                continue
            self.record(label, path)
            output = io.StringIO()
            count = 0
            with path.open() as stream:
                reader = csv.DictReader(stream, skipinitialspace=True)
                assert set(columns).issubset(reader.fieldnames or []), "Incomplete telemetry columns"
                bitmasks = tuple(name for name in reader.fieldnames if name in bitmask_aliases)
                assert len(bitmasks) == len(set(bitmasks)), "Duplicate throttle bitmask columns"
                numeric_masks = {name: 0 for name in bitmasks}
                selected = columns + bitmasks
                writer = csv.DictWriter(output, fieldnames=selected)
                writer.writeheader()
                for row in reader:
                    datetime.strptime(row["timestamp"], "%Y/%m/%d %H:%M:%S.%f")
                    assert row["index"] in ("0", "1", "2", "3") and re.fullmatch(r"P\d+", row["pstate"])
                    for key in columns[3:]:
                        number(float(row[key].split()[0]))
                    for bitmask in bitmasks:
                        value = row[bitmask].strip()
                        if re.fullmatch(r"0x[0-9a-fA-F]{1,16}", value):
                            numeric_masks[bitmask] += 1
                        else:
                            assert value in ("N/A", "[N/A]", "Not Supported", "[Not Supported]", "NotSupported", "[NotSupported]"), "Invalid throttle bitmask"
                        row[bitmask] = value
                    writer.writerow({key: row[key] for key in selected})
                    count += 1
            self.bundle[f"telemetry/{scope}-gpu-samples.csv"] = output.getvalue().encode()
            self.availability[label] = {"state": "available" if count else "incomplete", "scope": scope, "sample_rows": count,
                                       "throttle_bitmask_column_present": bool(bitmasks), "throttle_bitmask_columns": list(bitmasks),
                                       "numeric_throttle_bitmask_rows": numeric_masks,
                                       "timestamp_scope": "NVSMI naive timestamp retained; no timezone conversion inferred by this projector"}

    def finish(self, pins):
        raw_runtime = self.get("runtime", self.raw / "results/runtime-provenance.json", terminal=False)
        if raw_runtime is not None:
            self.put("runtime-summary.json", runtime(raw_runtime))
        self.tests()
        self.quality()
        self.e2e()
        self.telemetry()
        self.history()
        self.put("availability.json", self.availability)
        self.put("raw-evidence-sha256.json", self.evidence_hashes)
        incomplete = any(row["state"] != "available" for key, row in self.availability.items() if not key.startswith("telemetry/"))
        status = "Incomplete: unavailable or non-complete records are explicitly listed in availability.json." if incomplete else "All required result groups are available. Consult their actual gates; completeness does not mean they passed."
        template = Path(__file__).with_name("README.template.md").read_text()
        template = template.replace("@BASELINE@", pins["baseline"]["commit"]).replace("@CANDIDATE@", pins["candidate"]["commit"]).replace("@AVAILABILITY@", status)
        history = "[Historical v3 evidence](history/v3/README.md) retains all 120 observations and its nine failed performance gates. Its exploratory tail diagnostic remains explicitly post hoc."
        if self.availability["history/v3"]["state"] != "available":
            history = "Historical v3 supplement unavailable; see availability.json. No historical result is inferred."
        template = template.replace("@HISTORY@", history)
        self.bundle["README.md"] = template.encode()
        self.bundle["analysis/build_bundle.py"] = Path(__file__).read_bytes()
        self.bundle["analysis/README.template.md"] = Path(__file__).with_name("README.template.md").read_bytes()
        self.bundle["analysis/prepare_inputs.py"] = Path(__file__).with_name("prepare_inputs.py").read_bytes()
        # The public builder contains its own detector expressions, not private data.
        for name, data in self.bundle.items():
            if name != "analysis/build_bundle.py":
                privacy_scan(name, data)
        assert sum(map(len, self.bundle.values())) <= 15 * 1024**2, "Bundle exceeds 15 MiB bound"
        assert all(len(data) <= 8 * 1024**2 for data in self.bundle.values()), "One artifact exceeds 8 MiB bound"
        self.put("SHA256SUMS.json", {name: hashlib.sha256(data).hexdigest() for name, data in sorted(self.bundle.items())})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notes", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--e2e-telemetry", type=Path, help="Explicit E2E sampler CSV; default is remote-results/ready/gpu-telemetry.csv. The separate results/gpu-telemetry.csv is GSM8K only.")
    parser.add_argument("--allow-incomplete", action="store_true", help="Keep available observations and emit explicit omissions; never fabricate missing results")
    args = parser.parse_args()
    assert not args.output.exists(), "Use a fresh output directory"
    builder = Builder(args.notes, args.source_repo, args.allow_incomplete, args.e2e_telemetry)
    pins = builder.static()
    builder.finish(pins)
    args.output.mkdir(parents=True)
    for relative, data in builder.bundle.items():
        target = args.output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    print(json.dumps({"files": len(builder.bundle), "bytes": sum(map(len, builder.bundle.values())),
                      "manifest_sha256": sha(args.output / "SHA256SUMS.json"),
                      "availability": builder.availability}, indent=2))


if __name__ == "__main__":
    main()
