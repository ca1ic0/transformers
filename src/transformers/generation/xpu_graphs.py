# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
XPU graph capture for the decode loop in `.generate()`.

Intel XPU (`torch.xpu`) is the CUDA-Graphs-equivalent runtime for Intel GPUs, exposed as
`torch.xpu.graphs` (`torch.xpu.XPUGraph`, `torch.xpu.graph`, `torch.xpu.graph_pool_handle`).
On XPU, `torch.compile(mode="reduce-overhead")` silently skips graph capture: an internal
CUDA-centric gate emits *"skipping cudagraphs due to multiple devices: device(type='xpu')"* and
falls back to plain inductor. The only way to get CUDA-Graphs-style whole-decode-step capture on
XPU today is **manual** capture via `torch.xpu.graphs`.

This module implements that manual path as a drop-in replacement for the per-step
`model_forward(...)` call in the sampling loop. The wrapper:

1. Flattens the per-step `(args, kwargs)` into a pytree and keys a cache of `XPUGraph`s by
   `(tensor shapes/dtypes, cache position)`. Host-read scalars (positions, mask sizes) are baked
   into the captured kernels, so one graph per decode position is required.
2. On a key miss, snapshots host-side cache counters, runs the wrapped callable **once** on a side
   stream (warmup: triggers lazy inits / plan caches), restores the counters, and records the graph
   with `torch.xpu.graph(...)`.
3. On a key hit, copies the live inputs into the captured static buffers, calls `graph.replay()`,
   then advances the host-side cache counters itself (the in-graph counter `add_` is a host op on a
   CPU tensor and is therefore **not** recorded, so replays must advance it explicitly).
4. Returns the captured outputs with tensor leaves cloned and `Cache` object leaves **substituted**
   with the current call's cache objects. The captured output holds capture-time state; the
   generate loop must receive the *current* cache to keep decoding correctly.

The engine is generic; model-specific capture-hostility fixes (host-side tensor reads, warning
branches that sync) are the responsibility of the model's `forward` or a thin adapter.
"""

from __future__ import annotations

import functools
from typing import Any, Callable

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

from ..utils import logging


logger = logging.get_logger(__name__)


def _is_cache_obj(x: Any) -> bool:
    """Heuristic: a Cache object has `layers` and `get_seq_length`."""
    return hasattr(x, "layers") and hasattr(x, "get_seq_length")


def _cache_position(cache: Any) -> int:
    """Truthful cache position, robust to `is_initialized` gating.

    A cache whose prefill was served by a graph *replay* is never lazy-initialized (no python model
    code runs), so `cache.get_seq_length()` may gate on `is_initialized` and return 0. Read the
    per-layer counter tensor directly; the wrapper's counter-advance keeps it truthful on replays.
    """
    try:
        for layer in getattr(cache, "layers", []):
            cl = getattr(layer, "cumulative_length", None)
            if torch.is_tensor(cl):
                return int(cl)
        return int(cache.get_seq_length())
    except Exception:
        return -1


def _iter_layers(cache: Any):
    return getattr(cache, "layers", [])


# -- pytree keying -----------------------------------------------------------------------------


def _cache_identity(cache: Any):
    """Identity of a cache object for graph keying: a per-object token + backing-store pointers.

    A recorded graph replays the *capture-time* keys/values tensor addresses (`update()` records
    its writes), so a replay is only valid for the exact same backing store. A fresh cache object —
    e.g. CSM's per-frame depth caches — must therefore NEVER hit a graph captured for another
    object, even at the same position: tokens make different objects miss, and the store pointers
    additionally catch in-place-identity-preserving swaps (offload/prefetch, `reset()` re-allocs).
    """
    tok = getattr(cache, "_xpu_graph_token", None)
    if tok is None:
        tok = object()
        try:
            cache._xpu_graph_token = tok
        except Exception:
            tok = ("id", id(cache))
    stores = []
    for layer in _iter_layers(cache):
        k = getattr(layer, "keys", None)
        v = getattr(layer, "values", None)
        stores.append((k.data_ptr() if torch.is_tensor(k) else -1, v.data_ptr() if torch.is_tensor(v) else -1))
    return (tok, tuple(stores))


def _leaf_key(x: Any):
    if torch.is_tensor(x):
        return ("t", tuple(x.shape), str(x.dtype))
    if _is_cache_obj(x):
        # Key on (position, cache identity): a replay is only valid for the exact backing store the
        # graph was recorded with, and only at the baked-in position. Input tensor shapes elsewhere
        # in the key disambiguate e.g. prefill vs step at the same position.
        return ("cache", _cache_position(x), _cache_identity(x))
    return ("v", repr(x))


class XPUGraphCacheLayerMixin:
    """Optional mixin for a `Cache` layer to make it XPU-graph capturable.

    The default `StaticLayer` keeps `cumulative_length` as a GPU tensor after lazy init. Two
    capture-hostile consumers read it as a *bound*:
      * `torch.arange(past_seen_tokens, ...)` — a GPU tensor bound forces a D2H sync during capture
        (L0 error 44, `UR_RESULT_ERROR_UNSUPPORTED_FEATURE`).
      * `masking_utils` `q_offset.to(device)` — a CPU tensor bound is an H2D copy, also L0 error 44.

    Keeping the counter on CPU and returning a plain python `int` from `get_seq_length()` makes both
    paths host-reads (capture-safe). The in-place `add_` then runs on the CPU tensor (a host op that
    is *not* recorded), so the wrapper advances the counter itself on every replay.
    """

    def xpu_graph_enable(self) -> None:
        cl = getattr(self, "cumulative_length", None)
        if torch.is_tensor(cl) and cl.device.type != "cpu":
            self.cumulative_length = cl.cpu()

    def get_seq_length(self) -> int:  # noqa: D102
        # Deliberately NOT gated on `is_initialized`: a replay-served cache is never lazy-initialized
        # but its counter is kept truthful by the wrapper, so this must still report the position.
        cl = getattr(self, "cumulative_length", None)
        return int(cl) if torch.is_tensor(cl) else 0


class XPUGraphCapturedForward:
    """Wraps a model's `forward` with XPU-graph capture + replay for the decode step.

    Drop-in for the per-step `model_forward(...)` call in the sampling loop. Signatures are
    preserved (`functools.update_wrapper`) so `generate`'s kwargs validation keeps working.

    The wrapper is stateful: it owns a dict of `XPUGraph`s keyed by the flattened per-step inputs,
    a shared memory pool, and the host-side counter snapshot/restore machinery. It is intended to be
    created once per `generate()` call (or reused across calls via the model).

    Args:
        fn (`Callable`):
            The bound `forward` to wrap.
        name (`str`):
            Human-readable name for logging (e.g. the sub-model name).
        pool (`tuple`, *optional*):
            A `torch.xpu.graph_pool_handle()` to share captured-memory across graphs. Pass the same
            handle to every wrapper that should share a pool.
    """

    def __init__(self, fn: Callable, name: str, pool: Any = None):
        self.fn = fn
        self.name = name
        self.pool = pool
        self.graphs: dict = {}
        # Torch-XPU forbids capturing a graph replay inside another capture. When one wrapped
        # sub-model (e.g. CSM's depth decoder) is captured, it may internally invoke another
        # already-captured callable; such inner calls must run eagerly so only the outer graph
        # records real device work.
        self._capturing = False
        self._in_capture_warmup = False
        self.n_capture = 0
        self.n_replay = 0
        self.broken = False
        functools.update_wrapper(self, fn)

    # -- host-side cache counter bookkeeping ---------------------------------------------

    @staticmethod
    def _snapshot_counters(caches: list) -> list:
        """Snapshot counters as `(layer, int_value)`; lazy init *replaces* the tensor during warmup,
        so snapshotting the tensor object would restore into an orphaned tensor."""
        snaps = []
        for c in caches:
            for layer in _iter_layers(c):
                cl = getattr(layer, "cumulative_length", None)
                if torch.is_tensor(cl):
                    snaps.append((layer, int(cl)))
        return snaps

    @staticmethod
    def _restore_counters(snaps: list) -> None:
        for layer, v in snaps:
            cl = getattr(layer, "cumulative_length", None)
            if torch.is_tensor(cl):
                cl.fill_(v)

    # -- capture / replay ------------------------------------------------------------------

    def _capture(self, flat: list, spec: Any, caches: list) -> dict:
        static_flat = [l.clone() if torch.is_tensor(l) else l for l in flat]
        static_args, static_kwargs = tree_unflatten(static_flat, spec)

        # NOTE: no cache pre-initialization here. A cache whose prefill was served by a graph
        # *replay* is never lazy-initialized, so the first real forward — which may be this capture
        # — can hit `lazy_initialization` inside the record region. That is safe: `StaticLayer` is
        # capture-aware once `xpu_graph_enable()` has run (see `cache_utils.StaticLayer`), keeping
        # its counter on CPU instead of uploading it (an H2D copy that would fail with L0 error 44).
        # The enable loop below runs before the record, and keys/values allocation during capture is
        # pool-owned and stays alive via this cache's references.

        # Enable CPU-counter semantics for every cache layer (no-op if already CPU / non-static).
        for c in caches:
            for layer in _iter_layers(c):
                enable = getattr(layer, "xpu_graph_enable", None)
                if callable(enable):
                    enable()

        snaps = self._snapshot_counters(caches)
        s = torch.xpu.Stream()
        s.wait_stream(torch.xpu.current_stream())
        # Exactly ONE warmup, executed through the *raw* callable. Re-entering `self` here caused
        # unbounded recursion when the wrapped forward internally re-invoked the same wrapper; the
        # warmup only needs to trigger lazy inits / plan caches, which the raw fn does fine.
        if self._in_capture_warmup:
            raise RuntimeError(f"XPUGraph: re-entrant capture for {self.name}; aborting to avoid recursion.")
        self._in_capture_warmup = True
        try:
            with torch.xpu.stream(s):
                self.fn(*static_args, **static_kwargs)
        finally:
            self._in_capture_warmup = False
        torch.xpu.current_stream().wait_stream(s)
        self._restore_counters(snaps)
        for c in caches:
            for layer in _iter_layers(c):
                enable = getattr(layer, "xpu_graph_enable", None)
                if callable(enable):
                    enable()  # lazy init may have re-uploaded counters during warmup

        g = torch.xpu.XPUGraph()
        ctx = torch.xpu.graph(g, pool=self.pool) if self.pool is not None else torch.xpu.graph(g)
        # During the record phase, suppress this wrapper entirely: any re-entrant call runs the raw
        # fn, so torch-xpu never sees a graph replay inside the capture. Without this, recording
        # `self(...)` would hit an inner already-captured graph's replay and fail with
        # "Cannot prepare for replay during capturing stage".
        prev = self._capturing
        self._capturing = True
        try:
            with ctx:
                static_out = self(*static_args, **static_kwargs)
        finally:
            self._capturing = prev
        out_leaves, out_spec = tree_flatten(static_out)

        # Counter deltas observed at capture time (post-record minus pre-warmup snapshot). The
        # in-graph `add_` is a host op on the CPU counter: it runs during the record pass but is NOT
        # recorded, so replays must advance the counter by exactly this delta — never by a shape
        # heuristic (the query length is not reliably recoverable from the flattened inputs).
        cap_snaps = self._snapshot_counters(caches)
        counter_delta = {id(layer): v2 - v1 for (layer, v1), (_, v2) in zip(snaps, cap_snaps)}

        self.n_capture += 1
        logger.info(
            "XPUGraph: captured graph #%d for %s (pos=%s)",
            self.n_capture,
            self.name,
            _cache_position(caches[0]) if caches else "-",
        )
        return {
            "graph": g,
            "static_flat": static_flat,
            "out_leaves": out_leaves,
            "out_spec": out_spec,
            "counter_delta": counter_delta,
        }

    def __call__(self, *args, **kwargs):
        if self.broken:
            return self.fn(*args, **kwargs)

        flat, spec = tree_flatten((args, kwargs))
        caches = [l for l in flat if _is_cache_obj(l)]
        key = tuple(_leaf_key(l) for l in flat)
        ent = self.graphs.get(key)

        try:
            # Hard guard: never replay while ANY capture is recording on the current stream (torch-xpu
            # forbids it). This catches re-entrant cases the flag-based check misses.
            if torch.xpu.is_current_stream_capturing():
                return self.fn(*args, **kwargs)
            # If we're inside another wrapper's capture *record* phase, we must not replay (torch-xpu
            # forbids a graph replay inside a capture). The warmup phase re-enters `self` so nested
            # captured callables replay there; the record phase uses the raw `fn` and never re-enters.
            if self._capturing and not self._in_capture_warmup:
                return self.fn(*args, **kwargs)
            advancing = ent is not None
            if ent is None:
                self._capturing = True
                try:
                    ent = self._capture(flat, spec, caches)
                finally:
                    self._capturing = False
                self.graphs[key] = ent
            else:
                for st, lv in zip(ent["static_flat"], flat):
                    if torch.is_tensor(st):
                        st.copy_(lv, non_blocking=True)
            ent["graph"].replay()
            self.n_replay += 1
            if advancing:
                # The in-graph counter add_ is a host op on the CPU counter: it ran during capture
                # but is NOT recorded, so replays advance it by the capture-time delta.
                for c in caches:
                    for layer in _iter_layers(c):
                        cl = getattr(layer, "cumulative_length", None)
                        d = ent["counter_delta"].get(id(layer))
                        if torch.is_tensor(cl) and d:
                            cl.add_(d)
            # Substitute cache leaves with the CURRENT call's cache objects: the captured output
            # holds the capture-time cache (stale state, wrong counters); the generate loop must
            # receive the current cache to keep decoding correctly.
            in_caches = [l for l in flat if _is_cache_obj(l)]
            cache_i = 0
            out = []
            for l in ent["out_leaves"]:
                if torch.is_tensor(l):
                    out.append(l.clone())
                elif _is_cache_obj(l) and cache_i < len(in_caches):
                    out.append(in_caches[cache_i])
                    cache_i += 1
                else:
                    out.append(l)
            return tree_unflatten(out, ent["out_spec"])
        except Exception as e:
            logger.warning(
                "XPUGraph: capture/replay failed for %s: %r; falling back to eager for this call.",
                self.name,
                e,
            )
            self.broken = True
            return self.fn(*args, **kwargs)


def wrap_forward_for_xpu_graph(fn: Callable, name: str, pool: Any = None) -> XPUGraphCapturedForward:
    """Convenience constructor: wrap `fn` (a bound `forward`) with XPU-graph capture."""
    return XPUGraphCapturedForward(fn, name, pool)
