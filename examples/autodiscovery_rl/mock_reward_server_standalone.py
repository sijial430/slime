#!/usr/bin/env python3
"""Dependency-free mock AutoDiscovery reward server (stdlib only).

Speaks the exact HTTP contract that slime's
``slime.rollout.rm_hub.autodiscovery.autodiscovery_rm`` expects, so it can run
inside the slime training image (which does NOT have the ``autodiscovery``
package) as a co-located sidecar for pipeline smoke tests. The reward is a
deterministic pseudo-surprise derived from the hypothesis text — it validates
the rollout -> reward -> update loop and wandb logging, but teaches the policy
nothing meaningful. Swap for ``python -m autodiscovery.slime_reward`` (real
datasets on weka) for actual training.

Mirrors the logic of
asta-autodiscovery/packages/autodiscovery/scripts/slime/mock_reward_server.py
(same sha256 reward, request_id dedup, dataset_id routing), minus any deps.

Contract:
    GET  /health           -> {"status": "ok", "datasets": [<dataset_id>, ...]}
    POST /reward           request  {"hypothesis", "dataset_id", "request_id"?}
                           response {"reward": float, "success": bool, ...}

Usage:
    python mock_reward_server_standalone.py --registry data/registry.json \
        --host 0.0.0.0 --port 8137
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_SURPRISAL_WIDTH = 0.2
_lock = threading.Lock()
_n_calls = 0
_dedup: OrderedDict[str, dict] = OrderedDict()
_DEDUP_MAX = 8192
_DATASETS: set[str] = set()


def score(hypothesis: str) -> dict:
    """Deterministic pseudo-surprise reward from the hypothesis text."""
    global _n_calls
    with _lock:
        _n_calls += 1
        call_index = _n_calls
    digest = int(hashlib.sha256(hypothesis.encode()).hexdigest(), 16)
    prior = (digest % 1000) / 1000.0
    posterior = ((digest // 1000) % 1000) / 1000.0
    belief_change = abs(posterior - prior)
    return {
        "reward": float(belief_change / _SURPRISAL_WIDTH),
        "success": True,
        "surprising": belief_change > _SURPRISAL_WIDTH,
        "belief_change": belief_change,
        "kl_divergence": belief_change * 2.0,
        "prior_mean": prior,
        "posterior_mean": posterior,
        "hypothesis": hypothesis,
        "error": None,
        "mock": True,
        "call_index": call_index,
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            self._send(200, {"status": "ok", "datasets": sorted(_DATASETS)})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        if self.path != "/reward":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            hypothesis = payload["hypothesis"]
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            self._send(400, {"error": f"bad request: {e}"})
            return

        dataset_id = payload.get("dataset_id")
        if _DATASETS and dataset_id is not None and dataset_id not in _DATASETS:
            self._send(400, {"error": f"unknown dataset_id {dataset_id!r}"})
            return

        request_id = payload.get("request_id")
        if request_id is not None:
            with _lock:
                if request_id in _dedup:
                    self._send(200, _dedup[request_id])
                    return

        result = score(hypothesis)
        result["dataset_id"] = dataset_id
        if request_id is not None:
            with _lock:
                _dedup[request_id] = result
                while len(_dedup) > _DEDUP_MAX:
                    _dedup.popitem(last=False)
        self._send(200, result)

    def log_message(self, fmt, *args):  # keep stdout readable in job logs
        return


def main() -> None:
    ap = argparse.ArgumentParser(description="Dependency-free mock reward server.")
    ap.add_argument("--registry", default=None, help="registry.json; its keys are the valid dataset_ids.")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8137)
    args = ap.parse_args()

    if args.registry:
        with open(args.registry) as f:
            _DATASETS.update(json.load(f).keys())

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[mock-reward] listening on http://{args.host}:{args.port} "
          f"({len(_DATASETS)} dataset_ids)", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
