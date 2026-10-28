"""Three fixed production-head fallback cases; external validation only."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy

os.environ["VLLM_BATCH_INVARIANT"] = "0"

import torch
from vllm.models.minimax_m3.common.ops import sparse_attn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "candidate"], required=True)
    args = parser.parse_args()
    source = Path("/work/baseline" if args.mode == "baseline" else "/work/source")
    pins = json.loads(Path("/work/source-pins.json").read_text())[args.mode]
    assert hashlib.sha256(Path(sparse_attn.__file__).read_bytes()).hexdigest() == pins["source_hashes"]["models/minimax_m3/common/ops/sparse_attn.py"]
    fixture = runpy.run_path(str(source / "tests/kernels/test_minimax_m3_sparse_attn_fp8_scale.py"))
    torch.manual_seed(2)
    q = torch.randn(40, 16, 128, device="cuda", dtype=torch.bfloat16) * 0.1
    kv, dequant = fixture["_make_kv_cache"](16, seed=3)
    ks, vs = (torch.tensor(value, device="cuda", dtype=torch.float32) for value in (0.3, 0.7))
    dequant[..., :128] = (kv[..., :128].float() * ks).to(q.dtype)
    dequant[..., 128:] = (kv[..., 128:].float() * vs).to(q.dtype)
    blocks = torch.arange(16, device="cuda", dtype=torch.int32)
    topk = blocks.expand(1, 40, -1).contiguous()
    block_table = blocks.expand(40, -1).contiguous()
    seq_lens = torch.full((40,), 2048, device="cuda", dtype=torch.int32)
    seq_lens[0] -= 64
    outputs = {}
    cases = [
        ("scalar_fp8_default", kv, ks, vs, False),
        ("per_token_fp8_fallback", kv, ks.expand(1, 2048).contiguous(), vs.expand(1, 2048).contiguous(), True),
        ("bf16_cache_fallback", dequant, None, None, True),
    ]
    for name, cache, k_scale, v_scale, request_fp16 in cases:
        output = torch.empty_like(q)
        options = {"fp16_dot": request_fp16} if args.mode == "candidate" else {}
        sparse_attn.minimax_m3_sparse_attn_decode(
            q, cache, topk, block_table, seq_lens, 1, 128**-0.5, output,
            decode_query_len=1, k_scale=k_scale, v_scale=v_scale, **options,
        )
        outputs[name] = output.cpu()
        assert torch.isfinite(outputs[name]).all()
    root = Path("/work/attention-tests/direct-baseline")
    root.mkdir(exist_ok=True)
    torch.save(outputs, root / f"{args.mode}.pt")
    report = {"mode": args.mode, "commit": pins["commit"], "q_shape": list(q.shape),
              "gpu": str(torch.cuda.get_device_properties(0)),
              "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "output_sha256": {name: hashlib.sha256(out.view(torch.uint8).numpy().tobytes()).hexdigest()
                                for name, out in outputs.items()}}
    if args.mode == "candidate":
        baseline = torch.load(root / "baseline.pt", map_location="cpu", weights_only=True)
        assert json.loads((root / "baseline.json").read_text())["gpu"] == report["gpu"]
        report["bitwise_equal"] = {name: torch.equal(output, baseline[name]) for name, output in outputs.items()}
    (root / f"{args.mode}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if args.mode == "candidate":
        assert all(report["bitwise_equal"].values()), report


if __name__ == "__main__":
    main()
