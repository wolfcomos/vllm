"""CPU-only protocol/lifecycle checks with fake vLLM and torch modules."""

import argparse
import contextlib
import copy
import hashlib
import io
import json
import logging
import logging.config
from pathlib import Path
import sys
import tempfile
import types

import run_admission_controlled as driver


def main():
    root = Path(__file__).resolve().parent
    notes = root.parent
    baseline_path = notes / "graph-perf/results/baseline/result.json"
    baseline = driver.load(baseline_path)
    pin_path = notes / "staging/source-pins.json"
    pins = driver.load(pin_path)
    original_digest = driver.file_digest
    state = {}

    def file_digest(path):
        if str(path).startswith("/cpu-fake-vllm/"):
            name = str(path).removeprefix("/cpu-fake-vllm/")
            pin = pins["baseline" if state["arm"] == "baseline" else "candidate"]
            return pin["source_hashes"].get(name, hashlib.sha256(name.encode()).hexdigest())
        return original_digest(path)

    driver.file_digest = file_digest
    driver.importlib.metadata.version = lambda name: baseline["packages"][name]
    driver.cache_manifest = lambda output: copy.deepcopy(baseline["cache_after_generation"])

    def clock():
        state["events"].append("clock")
        state["clock"] += 0.95 if state["arm"].startswith("fp16") else 1.0
        return state["clock"]

    driver.time.perf_counter = clock

    class Params:
        def __init__(self, **kwargs):
            self.__dict__.update(copy.deepcopy(kwargs))
            self.n = 1
            if self.stop is None:
                self.stop = []

    class LLM:
        def __init__(self, **settings):
            state["created"] += 1
            self.settings = settings
            self.calls = 0
            self.paused = False
            self.pending = None
            self.workers = copy.deepcopy(baseline["workers"])
            for worker in self.workers:
                worker["source_sha256"] = file_digest("/cpu-fake-vllm/" + driver.SOURCE_FILES[1])
                for field in driver.validate_tuner_startup.__globals__["TUNING_ACTIVITY_FIELDS"]:
                    worker["flashinfer_statistics"][field] = {
                        key: 0 for key in worker["flashinfer_statistics"][field]
                    }
            if settings["cudagraph_metrics"]:
                # Reproduce logging initialization clearing an early child handler.
                logging.config.dictConfig({"version": 1, "disable_existing_loggers": False,
                                           "loggers": {"vllm": {"handlers": [], "level": "INFO"}}})
            self.llm_engine = types.SimpleNamespace(do_log_stats=lambda: None,
                engine_core=types.SimpleNamespace(shutdown=self.shutdown))

        def collective_rpc(self, fn, timeout):
            if fn.__name__ == "inspect_worker":
                return copy.deepcopy(self.workers)
            return [{"rank": w["rank"], "statistics": copy.deepcopy(w["flashinfer_statistics"])}
                    for w in self.workers]

        def sleep(self, level, mode):
            assert level == 0 and mode == "wait" and self.pending is None
            self.paused = True
            state["events"].append("sleep")

        def enqueue(self, prompts, sampling_params, use_tqdm):
            assert self.paused and self.pending is None
            state["events"].append("enqueue")
            state["references"].append((prompts, sampling_params))
            self.calls += 1
            ids = [f"{1000 * self.calls + i}-abcdef12" for i in range(len(prompts))]
            sampling_params.output_kind = "FINAL_ONLY"
            self.pending = (prompts, sampling_params, ids)
            return ids

        def wake_up(self, tags):
            assert tags == ["scheduling"] and self.paused and self.pending
            self.paused = False
            state["events"].append("wake")
            return state["problem"] != "wake"

        def wait_for_completion(self, use_tqdm):
            assert not self.paused and self.pending
            state["events"].append("wait")
            prompts, params, ids = self.pending
            self.pending = None
            batch = len(prompts)
            per_batch = 4 if self.settings["cudagraph_metrics"] else 10
            call = (self.calls - 1) % per_batch
            value = 1 if state["arm"].startswith("fp16") else 0
            # Deliberately varying warmups must not trigger the measured parity gate.
            if call < 3:
                value += 10 + call
            if state["problem"] == "repeat" and call == 4:
                value += 1
            if state["problem"] == "cross" and call >= 3:
                value += 1
            if state["problem"] == "input":
                prompts[0]["prompt_token_ids"][0] += 1
            if state["problem"] == "sampling":
                params.stop.append("mutated")
            outputs = [types.SimpleNamespace(request_id=rid.partition("-")[0],
                prompt_token_ids=p["prompt_token_ids"], finished=True,
                outputs=[types.SimpleNamespace(token_ids=[value + i] * 256)])
                for i, (rid, p) in enumerate(zip(ids, prompts))]
            if state["problem"] == "order":
                outputs.reverse()
            if self.settings["cudagraph_metrics"]:
                count = 0 if state["problem"] == "metrics" else 200
                logger = logging.getLogger("vllm.v1.metrics.loggers")
                logger.debug(f"| {batch} | {batch} | 0 | FULL | {count} |")
            return outputs

        def shutdown(self, timeout):
            state["shutdown"] += 1

    vllm = types.ModuleType("vllm")
    vllm.__file__ = "/cpu-fake-vllm/__init__.py"
    vllm.LLM, vllm.SamplingParams = LLM, Params
    inputs = types.ModuleType("vllm.inputs")
    inputs.TokensPrompt = dict
    torch = types.ModuleType("torch")
    torch.__version__ = baseline["torch"]
    torch.version = types.SimpleNamespace(cuda=baseline["cuda"])
    sys.modules.update({"vllm": vllm, "vllm.inputs": inputs, "torch": torch})
    checks = []
    with tempfile.TemporaryDirectory(prefix="m3-admission-cpu-") as directory:
        tmp = Path(directory)

        def execute(name, arm, problem="none", witness=None, previous=None,
                    phase="timing", timing=None, comparison=None):
            state.update(arm=arm, problem=problem, created=0, shutdown=0,
                         events=[], references=[], clock=0.0)
            args = argparse.Namespace(arm=arm, phase=phase, baseline_v1_result=baseline_path,
                prompts=notes / "graph-perf/prompts.json", source_pins=pin_path,
                cache_dir=tmp / f"cache-{name}", output=tmp / name,
                witness_result=witness, previous_result=previous,
                timing_result=timing, timing_comparison=comparison)
            failed = False
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    driver.run(args)
            except AssertionError:
                failed = True
            assert failed == (problem != "none"), (name, failed)
            result = driver.load(args.output / "result.json")
            assert state["shutdown"] == 1 and result["shutdown_completed"] is True
            assert result["status"] == ("failed" if failed else "complete")
            if not failed:
                assert len(result["warmups"]) == 6
                assert len(result["trials"]) == (2 if phase == "diagnostic" else 14)
                refs = state["references"]
                assert len({id(params) for _, params in refs}) == len(refs)
                assert len({id(p["prompt_token_ids"]) for prompts, _ in refs for p in prompts}) == sum(len(p) for p, _ in refs)
                events = state["events"]
                unit = ["sleep", "enqueue", "wake", "wait"]
                timed = ["sleep", "clock", "enqueue", "wake", "wait", "clock"]
                assert events == (unit * 3 + (unit if phase == "diagnostic" else timed * 7)) * 2
            checks.append({"case": name, "expected_failure": failed,
                           "completed_trials": len(result["trials"]), "shutdown": True})
            return args.output / "result.json"

        results = {}
        previous = None
        for arm in driver.ARMS:
            results[arm] = execute(arm, arm, witness=results.get("baseline"), previous=previous)
            previous = results[arm]
        execute("baseline-repeat-failure", "baseline", "repeat")
        execute("default-cross-failure", "default-a", "cross", results["baseline"], results["baseline"])
        execute("fp16b-cross-failure", "fp16-b", "cross", results["baseline"], results["fp16-a"])
        for problem in ("input", "sampling", "order", "wake"):
            execute(problem, "baseline", problem)
        comparison = tmp / "comparison.json"
        comparison.write_text(json.dumps({"status": "complete", "passed": True,
            "protocol": driver.PROTOCOL,
            "result_sha256": {arm: original_digest(path) for arm, path in results.items()}}))
        execute("diagnostic", "fp16-a", witness=results["baseline"], phase="diagnostic",
                timing=results["fp16-a"], comparison=comparison)
        execute("diagnostic-metrics-failure", "fp16-a", "metrics", witness=results["baseline"],
                phase="diagnostic", timing=results["fp16-a"], comparison=comparison)

    report = {"status": "passed", "driver_sha256": original_digest(root / "run_admission_controlled.py"),
              "checker_sha256": original_digest(__file__), "checks": checks,
              "scope": "CPU-only fake vLLM/torch state machine, normalized stop semantics, "
                       "fresh object identity, timed boundaries, exact-output gates, logger initialization and cleanup. "
                       "No real vLLM, torch, CUDA or GPU execution."}
    (root / "cpu-static-checks.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
