"""Reproduce the fixed synthetic input locally; public bundle contains no arrays."""

import argparse
import hashlib
import json
from pathlib import Path
import random

EXPECTED = "0b75efe9de7157be8e82b5d482cfb0e8b445951b91198c3f5e852db15fa8e13a"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rng = random.Random(42)
    prompts = [[rng.randrange(10000) for _ in range(2048)] for _ in range(40)]
    checksum = hashlib.sha256(json.dumps(prompts, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert checksum == EXPECTED, "Synthetic input recipe does not reproduce the frozen hash"
    assert not args.output.exists(), "Use a fresh input path"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"seed": 42, "input_len": 2048, "prompts_sha256": checksum, "prompts": prompts}, indent=2) + "\n")
    print(checksum)


if __name__ == "__main__":
    main()
