"""CPU-only synthetic-file and gate-boundary checks for the v2 comparator."""

import copy
import json
from pathlib import Path
import tempfile

import compare_admission_controlled as check


def write(path, value):
    path.write_text(json.dumps(value))


def main():
    root = Path(__file__).resolve().parent
    notes = root.parent
    baseline_path = notes / "graph-perf/results/baseline/result.json"
    prompts_path = notes / "graph-perf/prompts.json"
    pins_path = notes / "staging/source-pins.json"
    baseline, prompts, pins = map(check.load, (baseline_path, prompts_path, pins_path))
    checks = []
    with tempfile.TemporaryDirectory(prefix="m3-comparator-cpu-") as temporary:
        tmp = Path(temporary)
        paths, fixtures = [], []
        for index, name in enumerate(check.ARMS):
            directory = tmp / name
            directory.mkdir()
            result = copy.deepcopy(baseline)
            result.update(protocol=check.PROTOCOL, mode=name, shutdown_completed=True,
                          warmup_count=3, repeats=7, logging_handler_installed=False,
                          protocol_sha256=check.sha(root / "PROTOCOL.md"),
                          baseline_v1_result_sha256=check.sha(baseline_path),
                          source_pins_sha256=check.sha(pins_path),
                          admission_api_hashes={"synthetic-api": "0" * 64},
                          compilation_cache_environment={}, interval="synthetic-fixed-interval",
                          started_at_utc=f"2026-10-28T22:{index * 2:02d}:00+00:00",
                          finished_at_utc=f"2026-10-28T22:{index * 2 + 1:02d}:00+00:00",
                          trials=[], warmups=[], references={})
            result["harness_hashes"].update({
                "run_admission_controlled.py": check.sha(root / "run_admission_controlled.py"),
                "PROTOCOL.md": result["protocol_sha256"]})
            pin = pins["baseline" if name == "baseline" else "candidate"]
            result["source_commit"], result["source_hashes"] = pin["commit"], pin["source_hashes"]
            if name.startswith("fp16"):
                result["settings"]["attention_config"]["minimax_m3_msa_decode_fp16_dot"] = True
            result["tuner_after_generation"] = []
            for worker in result["workers"]:
                worker["source_sha256"] = pin["source_hashes"]["models/minimax_m3/common/ops/sparse_attn.py"]
                for key in check.ACTIVITY:
                    worker["flashinfer_statistics"][key] = {
                        op: 0 for op in worker["flashinfer_statistics"][key]}
                result["tuner_after_generation"].append({"rank": worker["rank"],
                    "statistics": copy.deepcopy(worker["flashinfer_statistics"])})
            result["cache_after_startup"] = copy.deepcopy(result["cache_after_generation"])
            next_id = 0
            for batch in (8, 40):
                reference = [[i + int(name.startswith("fp16"))] * 256 for i in range(batch)]
                write(directory / f"reference-b{batch}.json", reference)
                result["references"][str(batch)] = check.digest(reference)
                for stage, count in (("warmup", 3), ("measured", 7)):
                    for repeat in range(count):
                        tokens = reference if stage == "measured" else [
                            [i + repeat + 100] * 256 for i in range(batch)]
                        ids = [str(next_id + i) for i in range(batch)]
                        next_id += batch
                        row = dict(batch=batch, repeat=repeat, stage=stage, seed=42,
                            output_tokens=batch * 256, output_token_ids_sha256=check.digest(tokens),
                            first_token_ids_sha256=check.digest([r[0] for r in tokens]),
                            normalized_sampling_sha256=check.digest({**baseline["sampling"], "stop": []}),
                            inputs_unchanged=True, sampling_unchanged=True,
                            request_ids=ids, queued_request_ids=[i + "-abcdef12" for i in ids])
                        for key in ("input_before_sha256", "input_after_sha256", "returned_prompts_sha256"):
                            row[key] = check.digest(prompts["prompts"][:batch])
                        if stage == "measured":
                            # B8 intentionally regresses; B40 passes the prescribed benefit gate.
                            row["seconds"] = (1.10 if batch == 8 else .95) if name.startswith("fp16") else 1.0
                        result["trials" if stage == "measured" else "warmups"].append(row)
                        stem = "tokens" if stage == "measured" else "tokens-warmup"
                        write(directory / f"{stem}-b{batch}-r{repeat}.json", tokens)
            (directory / "run.log").write_text("Synthetic cache log\n")
            log_check = copy.deepcopy(check.load(notes / "graph-perf/results/default/cache-log-check.json"))
            log_check.update(mode=name, log_sha256=check.sha(directory / "run.log"))
            write(directory / "cache-log-check.json", log_check)
            paths.append(directory / "result.json")
            fixtures.append(result)

        def persist():
            for index, result in enumerate(fixtures):
                if index:
                    result["witness_result_sha256"] = check.sha(paths[0])
                    result["previous_result_sha256"] = check.sha(paths[index - 1])
                write(paths[index], result)

        def compare():
            return check.compare(paths, prompts_path, baseline_path, pins_path)

        persist()
        report = compare()
        assert report["passed"] and report["status"] == "complete"
        assert len(report["performance_gates"]) == 18
        assert report["workloads"]["fp16-a"]["8"]["median_seconds"] == 1.10
        checks.append("Five-arm synthetic file flow passes; warmups vary and B8 regresses without invalidating B40-specific gate")
        original = copy.deepcopy(fixtures)

        def reject(label, mutate):
            fixtures[:] = copy.deepcopy(original)
            mutate()
            persist()
            try:
                compare()
            except (AssertionError, KeyError, ValueError, TypeError):
                checks.append(label)
            else:
                raise AssertionError(f"Comparator accepted {label}")

        reject("Missing measured repeat rejects", lambda: fixtures[0]["trials"].pop())
        reject("NaN timing rejects", lambda: fixtures[0]["trials"][0].update(seconds=float("nan")))
        reject("Runtime mismatch rejects", lambda: fixtures[2].update(torch="changed"))
        reject("Diagnostic phase cannot supply timings", lambda: fixtures[2].update(phase="diagnostic"))
        reject("Input mismatch rejects", lambda: fixtures[0]["trials"][0].update(input_after_sha256="0" * 64))
        reject("Sampling mutation rejects", lambda: fixtures[0]["trials"][0].update(sampling_unchanged=False))
        reject("Request order mismatch rejects", lambda: fixtures[0]["trials"][0]["request_ids"].reverse())
        reject("Startup tuning rejects", lambda: fixtures[0]["workers"][0]["flashinfer_statistics"]["tuned_op_successful_configs"].update(mxfp8_gemm=1))
        reject("Statistics replacement rejects", lambda: fixtures[0]["tuner_after_generation"][0]["statistics"].update(statistics_object_id=0))
        reject("Changed tactic hash rejects", lambda: fixtures[0]["cache_after_generation"][0].update(sha256="0" * 64))
        reject("Source mismatch rejects", lambda: fixtures[1].update(source_commit=check.COMMITS["baseline"]))
        reject("Protocol hash mismatch rejects", lambda: fixtures[0].update(protocol_sha256="0" * 64))
        reject("Raw token hash mismatch rejects", lambda: fixtures[0]["trials"][0].update(output_token_ids_sha256="0" * 64))
        fixtures[:] = copy.deepcopy(original)
        persist()
        token_path = paths[0].parent / "tokens-b8-r0.json"
        tokens = check.load(token_path)
        tokens[0][0] += 1
        write(token_path, tokens)
        fixtures[0]["trials"][0].update(output_token_ids_sha256=check.digest(tokens),
            first_token_ids_sha256=check.digest([r[0] for r in tokens]))
        persist()
        try:
            compare()
        except AssertionError as error:
            assert "Within-arm output variation" in str(error)
            checks.append("Within-arm token difference rejects even with internally consistent hashes")
        else:
            raise AssertionError("Token variation accepted")

        # Give an arm a different but completely self-consistent measured output.
        write(token_path, check.load(paths[0].parent / "reference-b8.json"))
        for index, expected in ((1, "Default token parity failed"),
                                (3, "FP16 repeat token parity failed")):
            fixtures[:] = copy.deepcopy(original)
            directory = paths[index].parent
            old_reference = check.load(directory / "reference-b8.json")
            changed = copy.deepcopy(old_reference)
            changed[0][0] += 1
            write(directory / "reference-b8.json", changed)
            fixtures[index]["references"]["8"] = check.digest(changed)
            for row in fixtures[index]["trials"]:
                if row["batch"] == 8:
                    write(directory / f"tokens-b8-r{row['repeat']}.json", changed)
                    row.update(output_token_ids_sha256=check.digest(changed),
                               first_token_ids_sha256=check.digest([r[0] for r in changed]))
            persist()
            try:
                compare()
            except AssertionError as error:
                assert expected in str(error), str(error)
                checks.append(expected + " with internally consistent raw output hashes")
            else:
                raise AssertionError("Cross-arm variation accepted")
            write(directory / "reference-b8.json", old_reference)
            for repeat in range(7):
                write(directory / f"tokens-b8-r{repeat}.json", old_reference)

    summaries = report["workloads"]
    for value, expected in ((.02, True), (.020001, False)):
        changed = copy.deepcopy(summaries)
        changed["baseline"]["8"]["relative_mad"] = value
        assert check.performance_gates(changed)[0]["passed"] is expected
    checks.append("Relative MAD accepts exact 2 percent boundary, rejects above it")
    for ratio, expected in ((1.02, True), (.98, True), (1.020001, False), (.979999, False)):
        changed = copy.deepcopy(summaries)
        changed["default-b"]["40"]["median_seconds"] = ratio
        gate = next(g for g in check.performance_gates(changed) if g["gate"] == "default_anchor_drift" and g["batch"] == 40)
        assert gate["passed"] is expected
    checks.append("Anchor drift uses absolute two-sided 2 percent threshold")
    for ratio, expected in ((.97, True), (.970001, False)):
        changed = copy.deepcopy(summaries)
        changed["fp16-b"]["40"]["median_seconds"] = ratio
        selected = [g for g in check.performance_gates(changed) if g["gate"] == "B40_at_least_3_percent_lower_latency" and g["fp16"] == "fp16-b"]
        assert all(g["passed"] is expected for g in selected)
    checks.append("Both FP16 engines must meet exact 3 percent B40 threshold against both controls")
    for seconds in ([1.0] * 6, [0.0] * 7, [float("inf")] * 7):
        try:
            check.summarize(seconds)
        except AssertionError:
            pass
        else:
            raise AssertionError("Invalid timing accepted")
    checks.append("Wrong sample count, zero and infinite timing reject")
    result = {"status": "passed", "scope": "CPU-only synthetic fixtures; no GPU or real model execution",
              "comparator_sha256": check.sha(root / "compare_admission_controlled.py"),
              "checker_sha256": check.sha(Path(__file__)), "checks": checks}
    write(root / "comparator-cpu-checks.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
