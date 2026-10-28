"""Run pinned operator regression controls and complete repository suites."""

import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


ROOT = Path("/work/attention-tests")
PYTHON = "/work/.venv/bin/python"
SOURCES = {"baseline": Path("/work/baseline"), "candidate": Path("/work/source"), "old": Path("/work/old")}
SUITES = [
    "tests/kernels/attention/test_minimax_m3.py",
    "tests/kernels/test_minimax_m3_sparse_attn_fp8_scale.py",
    "tests/kernels/attention/test_minimax_m3_fp8_triton_indexer.py",
    "tests/kernels/attention/test_minimax_m3_msa_cutlass_sparse_decode.py",
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def environment(source):
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="0", PYTHONPATH=f"{source}:/work", OMP_NUM_THREADS="4",
               VLLM_CACHE_ROOT="/work/cache/vllm", TRITON_CACHE_DIR="/work/cache/triton")
    for key in ("VLLM_MINIMAX_M3_SPARSE_DECODE_FP16_DOT", "VLLM_BATCH_INVARIANT"):
        env.pop(key, None)
    return env


def run_pytest(source, label, suite, extra=()):
    output = ROOT / label
    output.mkdir(parents=True, exist_ok=True)
    name = Path(suite).stem
    xml = output / (name + ".xml")
    cmd = [PYTHON, "-m", "pytest", "-q", "-ra", "--durations=20", f"--junitxml={xml}", *extra, str(suite)]
    start = time.time()
    record = {"source": str(source), "command": cmd, "CUDA_VISIBLE_DEVICES": "0",
              "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    print(json.dumps({"starting": label, "suite": str(suite), "time": record["started_at"]}), flush=True)
    with (output / (name + ".log")).open("w") as log:
        try:
            result = subprocess.run(cmd, cwd=source, env=environment(source), stdout=log,
                                    stderr=subprocess.STDOUT, timeout=1500)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = 124
    (output / (name + ".exitcode")).write_text(str(code) + "\n")
    record.update(exitcode=code, seconds=time.time() - start)
    if xml.exists():
        suites = ET.parse(xml).getroot().findall("testsuite")
        record["counts"] = {key: sum(int(s.attrib[key]) for s in suites)
                            for key in ("tests", "errors", "failures", "skipped")}
        record["failures"] = [{"name": case.attrib["name"], "text": failure.text}
                              for s in suites for case in s.findall("testcase")
                              for failure in case.findall("failure")]
        record["cases"] = [{"name": case.attrib["name"],
                             "skipped": [skip.attrib for skip in case.findall("skipped")]}
                            for s in suites for case in s.findall("testcase")]
    write(output / (name + ".invocation.json"), record)
    print(json.dumps({"finished": label, "suite": str(suite), "exitcode": code,
                      "counts": record.get("counts"), "seconds": record["seconds"]}), flush=True)
    return record


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    assert not (ROOT / "summary.json").exists(), "Do not overwrite a prior validation run"
    pins = json.loads(Path("/work/source-pins.json").read_text())
    provenance = {}
    for arm, source in SOURCES.items():
        actual = {name: sha(source / "vllm" / name) for name in pins[arm]["source_hashes"]}
        assert actual == pins[arm]["source_hashes"], (arm, actual)
        code = """import hashlib,json,torch,vllm
from pathlib import Path
from vllm.models.minimax_m3.common.ops import sparse_attn
print(json.dumps({'vllm':vllm.__file__,'operator':sparse_attn.__file__,'operator_sha256':hashlib.sha256(Path(sparse_attn.__file__).read_bytes()).hexdigest(),'gpu':str(torch.cuda.get_device_properties(0)),'device_count':torch.cuda.device_count(),'torch':torch.__version__,'cuda':torch.version.cuda}))
"""
        smoke = subprocess.run([PYTHON, "-c", code], cwd=source, env=environment(source),
                               capture_output=True, text=True, check=True, timeout=120)
        (ROOT / f"{arm}-import.log").write_text(smoke.stdout + smoke.stderr)
        live = json.loads(smoke.stdout.splitlines()[-1])
        assert live["device_count"] == 1
        assert live["operator_sha256"] == actual["models/minimax_m3/common/ops/sparse_attn.py"]
        provenance[arm] = {**pins[arm], "live_import": live,
                           "test_hashes": {name: sha(source / name) for name in SUITES}}
    write(ROOT / "source-provenance.json", provenance)
    controls = ROOT / "controls"
    controls.mkdir(exist_ok=True)
    test = controls / "test_final_split_regression.py"
    shutil.copyfile(SOURCES["candidate"] / SUITES[1], test)
    assert sha(test) == provenance["candidate"]["test_hashes"][SUITES[1]]
    selection = ("-c", "/dev/null", "--import-mode=importlib", "-k", "test_minimax_m3_sparse_decode_fp8_split_scales")
    negative = run_pytest(SOURCES["old"], "negative-old", test, selection)
    positive = run_pytest(SOURCES["candidate"], "positive-final", test, selection)
    negative_passed = (negative["exitcode"] == 1 and negative.get("counts", {}).get("errors") == 0
                       and negative.get("counts", {}).get("skipped") == 0
                       and negative.get("counts", {}).get("tests") == 6
                       and negative.get("failures")
                       and all("Tensor-likes are not equal" in (failure["text"] or "")
                               for failure in negative["failures"]))
    summary = {"negative_control_expected_failure_observed": bool(negative_passed),
               "negative_control": negative, "positive_control": positive, "suites": {}}
    passed = bool(negative_passed) and positive["exitcode"] == 0
    write(ROOT / "progress.json", summary)
    for arm in ("baseline", "candidate"):
        summary["suites"][arm] = []
        for suite in SUITES:
            result = run_pytest(SOURCES[arm], arm, suite)
            summary["suites"][arm].append(result)
            passed &= result["exitcode"] == 0
            write(ROOT / "progress.json", summary)
    summary["original_case_preservation"] = []
    for before, after in zip(summary["suites"]["baseline"], summary["suites"]["candidate"], strict=True):
        old_names = {row["name"] for row in before.get("cases", [])}
        new_names = {row["name"] for row in after.get("cases", [])}
        removed = sorted(old_names - new_names)
        summary["original_case_preservation"].append({"suite": before["command"][-1],
                                                      "removed": removed, "added": sorted(new_names-old_names)})
        passed &= not removed
    summary["passed"] = passed
    summary["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    write(ROOT / "summary.json", summary)
    print(json.dumps({"complete": True, "passed": passed, "negative_control": bool(negative_passed)}), flush=True)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
