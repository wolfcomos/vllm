# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`--drop-checkpoint-page-cache` reaches the worker, which evicts the files the
model was loaded from (pulled weights take precedence over the model name)."""

from types import SimpleNamespace

import pytest

from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.worker.gpu_worker import Worker


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ([], False),
        (["--drop-checkpoint-page-cache"], True),
        (["--no-drop-checkpoint-page-cache"], False),
    ],
)
def test_cli_sets_load_config(options, expected):
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = EngineArgs.from_cli_args(parser.parse_args(options))
    assert args.create_load_config().drop_checkpoint_page_cache is expected


@pytest.mark.parametrize(
    ("model_weights", "expected"),
    [("", "org/model"), ("/pulled/weights", "/pulled/weights")],
)
def test_worker_drops_the_loaded_checkpoint(monkeypatch, model_weights, expected):
    calls = []
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.weight_utils.drop_checkpoint_cache",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    worker = SimpleNamespace(
        model_config=SimpleNamespace(
            model="org/model", model_weights=model_weights, revision="rev"
        ),
        load_config=SimpleNamespace(download_dir="/cache"),
    )
    Worker._drop_checkpoint_page_cache(worker)
    assert calls == [((expected,), {"revision": "rev", "cache_dir": "/cache"})]
