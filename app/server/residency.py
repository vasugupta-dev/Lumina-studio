"""Model residency manager for Lumina Studio — honest, stdlib only.

What this module owns
---------------------
`serve.CHAT` holds the single resident chat model; media jobs (video/image
work) run in the same process and need the RAM the model may be holding.  This
module answers three questions with measurements instead of promises:

1. How much RAM is really free, and how much does this process hold?
   `measure()` reads /proc/self/statm (+ /proc/meminfo) first, falls back to
   `resource.getrusage(RUSAGE_SELF).ru_maxrss` (a lifetime PEAK, named as
   such), and reports which source it used.  Setting
   LUMINA_RESIDENCY_MEASURE=rusage forces the fallback.
2. Is a chat model resident, and how big are its ARRAYS?  `resident()`.
3. May a job that declares `need_mb` proceed, and should the model be unloaded
   first?  `ensure_headroom()`, plus `finish()` after the job.

Policy pairing (this is the contract media.py follows)
------------------------------------------------------
    r = residency.ensure_headroom(need_mb, "reload_after")
    try:
        ... run the job ...
    finally:
        residency.finish(r.get("policy_used"), r.get("unloaded_model"))

- "off"          never touch the model (the declared need is unverified; the
                 dict says so in need_source).
- "unload"       unload the model only when free RAM does not already cover
                 the declared need; the model stays unloaded afterwards.
- "reload_after" the same unload, and `finish()` reloads the model after the
                 job.  `finish()` is a no-op unless policy_used is
                 "reload_after" AND a model was actually unloaded, so a
                 "reload_after" job that never needed an unload causes no
                 churn, and a refusal that happened BEFORE any unload never
                 triggers a reload of a model that is still resident.

Measured realities this module does not pretend away
----------------------------------------------------
- Small-model RSS does not drop on del+gc (allocator arena retention).  The
  `freed_mb` reported here is a MEASUREMENT of the RSS delta, not a promise:
  it may be ~0.0 even though the model's arrays were released.
- A closed fp16 allocation of >= 128 MB DOES return to the OS on this device
  (tests/test_residency.py prints the numbers; that is the justification for
  whole-model unload of real models).
- PARTIAL residency (keep K layers in RAM, stream the rest from disk) is NOT
  implementable with the current loader: ChatModel materializes every tensor
  eagerly (chat.py:118-143) and both adapters read the whole file
  (adapters/safetensors.py:87-91, adapters/lumina.py:95-99).  With no mmap
  accessor there is nothing to stream, so it is DEFERRED — hence
  `resident()["partial_residency"]["supported"] is False`, with the reason.

Locking note: `serve.api_unload()`/`serve.api_load()` take `serve._CHAT_LOCK`
themselves, so this module never wraps them in that lock (it is a plain
`threading.Lock`, not reentrant — wrapping would deadlock).  Ordering is
always _ENSURE_LOCK -> serve._CHAT_LOCK.

`import serve` happens lazily INSIDE functions only: this module imports
standalone and never touches the server at import time.
"""
from __future__ import annotations

import gc
import os
import re
import threading

POLICIES = ("off", "unload", "reload_after")

# one headroom decision at a time: two media jobs must not unload/reload the
# model around each other.
_ENSURE_LOCK = threading.Lock()

_STATM = "/proc/self/statm"
_MEMINFO = "/proc/meminfo"
_MB = 1048576.0
_NEED_SOURCE = ("declared estimate (another process's true demand cannot be "
                "measured)")
_PARTIAL_RESIDENCY = {
    "supported": False,
    "reason": ("loader materializes all tensors eagerly; no mmap accessor "
               "(deferred)"),
}


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------
def _r3(x):
    return None if x is None else round(float(x), 3)


def _pagesize() -> int:
    try:
        return int(os.sysconf("SC_PAGE_SIZE"))
    except Exception:
        return 4096


def _meminfo_mb():
    """(available_mb, total_mb) from /proc/meminfo; (None, None) if unreadable."""
    avail = total = None
    try:
        with open(_MEMINFO) as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) / 1024.0
                elif line.startswith("MemTotal:"):
                    total = int(line.split()[1]) / 1024.0
                if avail is not None and total is not None:
                    break
    except Exception:
        return None, None
    return avail, total


def _statm_rss_mb():
    """Current RSS from /proc/self/statm (field 2 = resident pages)."""
    try:
        with open(_STATM) as f:
            fields = f.read().split()
        if len(fields) < 7:
            return None
        return int(fields[1]) * _pagesize() / _MB
    except Exception:
        return None


def _rusage_peak_mb():
    """ru_maxrss: the process LIFETIME peak RSS (kB on Linux)."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return None


def measure() -> dict:
    """RAM snapshot. Never raises; unreadable numbers come back as None.

    source == "proc_self_statm" -> rss_mb is the CURRENT resident set.
    source == "ru_maxrss"       -> rss_mb is the lifetime PEAK (no statm).
    source == "none"            -> nothing readable, all RSS fields None.
    """
    forced = os.environ.get("LUMINA_RESIDENCY_MEASURE", "").strip().lower() == "rusage"
    available_mb, total_mb = _meminfo_mb()
    peak = _rusage_peak_mb()
    current = None if forced else _statm_rss_mb()
    if forced:
        current, source = peak, ("ru_maxrss" if peak is not None else "none")
    elif current is not None:
        source = "proc_self_statm"
    else:
        current, source = peak, ("ru_maxrss" if peak is not None else "none")
    if current is not None:
        # The kernel's peak counter lags the live RSS on this device (measured:
        # ru_maxrss 38.0 MB while /proc/self/statm read 38.6 MB), so report a
        # PEAK FLOOR instead of a number that contradicts the current reading.
        peak = current if peak is None else max(peak, current)
    return {"rss_mb": _r3(current), "peak_rss_mb": _r3(peak),
            "available_mb": _r3(available_mb), "total_mb": _r3(total_mb),
            "page_size": _pagesize(), "source": source}


# ---------------------------------------------------------------------------
# policy (persisted in serve.CFG["media"]["policy"])
# ---------------------------------------------------------------------------
def _stored_policy() -> str:
    """Policy currently persisted in serve.CFG["media"]["policy"] ("off" if unset)."""
    import serve
    media = serve.CFG.get("media")
    p = media.get("policy") if isinstance(media, dict) else None
    return p if p is not None else "off"


def policy() -> dict:
    return {"policy": _stored_policy(), "policies": list(POLICIES)}


def set_policy(name: str) -> dict:
    if name not in POLICIES:
        return {"error": "policy must be one of off|unload|reload_after"}
    import serve
    media = serve.CFG.get("media")
    if not isinstance(media, dict):
        media = {}
        serve.CFG["media"] = media
    media["policy"] = name
    serve._save_cfg()
    return {"policy": name, "saved": True}


# ---------------------------------------------------------------------------
# residency state
# ---------------------------------------------------------------------------
def _model_arrays(model):
    """Every ndarray the loaded model actually holds: fp16 originals + cache."""
    for arr in (getattr(model, "t", None) or {}).values():
        yield arr
    for arr in (getattr(model, "_wcache", None) or {}).values():
        yield arr


def _weights_mb(model) -> float | None:
    if model is None:
        return None
    try:
        total = sum(int(getattr(a, "nbytes", 0)) for a in _model_arrays(model))
        return round(total / 1e6, 4)
    except Exception:
        return None


def _params_by_layer(model) -> dict | None:
    """Parameter count per transformer layer, from tensor names. Optional."""
    if model is None:
        return None
    try:
        counts: dict[int, int] = {}
        for name, arr in (getattr(model, "t", None) or {}).items():
            m = re.search(r"layers\.(\d+)\.", str(name))
            if m:
                li = int(m.group(1))
                counts[li] = counts.get(li, 0) + int(getattr(arr, "size", 0))
        return dict(sorted(counts.items()))
    except Exception:
        return None


def resident() -> dict:
    """Is a model resident, how big are its arrays, and how much RAM is free.

    weights_mb is decimal MB of ndarray bytes (sum .nbytes / 1e6): what the
    model HOLDS, not what RSS shows; rss_mb/peak_rss_mb are MiB from the
    kernel.  They are different measures on purpose — both are reported.
    """
    import serve
    m = measure()
    model = serve.CHAT.get("model")
    loaded = model is not None
    return {"loaded": loaded,
            "model_id": serve.CHAT.get("model_id"),
            "weights_mb": _weights_mb(model) if loaded else None,
            "params_by_layer": _params_by_layer(model) if loaded else None,
            "rss_mb": m["rss_mb"], "peak_rss_mb": m["peak_rss_mb"],
            "available_mb": m["available_mb"],
            "policy": policy()["policy"],
            "measure_source": m["source"],
            "partial_residency": dict(_PARTIAL_RESIDENCY)}


def _released_arrays_mb() -> float:
    """MB of ndarray bytes the resident model holds (call BEFORE unload)."""
    try:
        import serve
        return _weights_mb(serve.CHAT.get("model")) or 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# headroom
# ---------------------------------------------------------------------------
def _result(need, m, policy_used, error=None, reason="media job", **kw) -> dict:
    """Every outcome carries the same honest skeleton."""
    d = {"requested_mb": need, "need_source": _NEED_SOURCE,
         "policy_used": policy_used,
         "available_mb_before": m["available_mb"],
         "available_mb_after": m["available_mb"],
         "freed_mb": 0.0, "array_mb": 0.0, "unloaded_model": None,
         "rss_before_mb": m["rss_mb"], "rss_after_mb": m["rss_mb"],
         "malloc_trim_released_mb": 0.0, "measure_source": m["source"],
         "reloaded": False, "reload_error": None,
         "reason": reason, "error": error}
    d.update(kw)
    return d


def _reload_locked(model_id: str | None) -> dict:
    """Reload through serve.api_load. Caller must hold _ENSURE_LOCK."""
    if not model_id:
        return {"error": "no model id to reload"}
    try:
        import serve
        r = serve.api_load(model_id)
    except Exception as e:  # never raise out of a headroom decision
        return {"error": "reload failed: " + str(e)[:200]}
    return r if isinstance(r, dict) else {"error": "reload returned non-dict"}


def reload_model(model_id: str | None = None) -> dict:
    """Explicit reload (the caller supplies the id it wants back)."""
    with _ENSURE_LOCK:
        return _reload_locked(model_id)


def _malloc_trim_mb(rss_before_trim):
    """Try libc malloc_trim(0); return the RSS delta of just this step."""
    try:
        import ctypes
        libc = ctypes.CDLL(None)
        libc.malloc_trim(0)
        after = measure()["rss_mb"]
        if rss_before_trim is None or after is None:
            return 0.0
        return round(max(0.0, rss_before_trim - after), 3)
    except Exception:
        return 0.0


def ensure_headroom(need_mb: int | float, policy: str | None = None,
                    reason: str = "media job", force: bool = False) -> dict:
    """Decide whether a job declaring `need_mb` may start.

    force=True runs the unload path even when free RAM already covers the
    declared need (diagnostics/tests only — media.py should not need it).
    A refusal that can be decided before touching the model never unloads it.
    """
    try:
        need = float(need_mb)
    except Exception:
        return {"error": "need_mb must be a number"}
    if need != need or need < 0:
        return {"error": "need_mb must be a non-negative number"}
    # NB: the parameter is named `policy` (contract), so the module-level
    # policy() reader must be reached through _stored_policy(), not policy().
    if policy is None:
        try:
            eff = _stored_policy()
        except Exception as e:  # serve unavailable: honest error, never a raise
            return {"error": "residency policy unreadable: " + str(e)[:150]}
    else:
        eff = policy
    if eff not in POLICIES:
        return _result(need, measure(), "invalid",
                       error="policy must be one of off|unload|reload_after",
                       policy_requested=eff, reason=reason)
    m_before = measure()
    avail_before = m_before["available_mb"]
    if eff == "off":
        return _result(need, m_before, "off", reason=reason)
    if not force and avail_before is not None and avail_before >= need:
        return _result(need, m_before, "not_needed", reason=reason,
                       note="available RAM already covers the declared need")
    with _ENSURE_LOCK:
        import serve
        model_id = serve.CHAT.get("model_id")
        array_mb = _released_arrays_mb()
        m_now = measure()
        avail_now = m_now["available_mb"]
        # Feasibility first: unloading a small model cannot free 50 GB, so
        # refuse WITHOUT churning a resident model the job may still need.
        if avail_now is not None and avail_now + array_mb < need:
            return _result(
                need, m_now, "insufficient", reason=reason, array_mb=array_mb,
                error=(f"insufficient RAM: need {need:g}MB, have {avail_now:g}MB "
                       f"and unloading {model_id} could release at most about "
                       f"{array_mb:.1f}MB of arrays ({_NEED_SOURCE}); refusing "
                       "to start the job"),
                note="refused before unloading: model left resident")
        if model_id is None:
            if avail_now is not None and avail_now < need:
                return _result(
                    need, m_now, "insufficient", reason=reason,
                    error=(f"insufficient RAM: need {need:g}MB, have "
                           f"{avail_now:g}MB and no chat model is loaded "
                           "(nothing to unload); refusing to start the job"))
            return _result(need, m_now, "not_needed", reason=reason,
                           note="no chat model is loaded; nothing to unload")
        rss_before = measure()["rss_mb"]
        serve.api_unload()  # takes serve._CHAT_LOCK itself (never re-wrap it)
        gc.collect()
        trim_before = measure()["rss_mb"]
        trim_mb = _malloc_trim_mb(trim_before)
        m_after = measure()
        rss_after = m_after["rss_mb"]
        avail_after = m_after["available_mb"]
        freed = (round(rss_before - rss_after, 3)
                 if (rss_before is not None and rss_after is not None) else None)
        common = dict(array_mb=array_mb, unloaded_model=model_id,
                      available_mb_after=avail_after, freed_mb=freed,
                      rss_before_mb=rss_before, rss_after_mb=rss_after,
                      malloc_trim_released_mb=trim_mb,
                      measure_source=m_after["source"], reason=reason,
                      forced=bool(force))
        notes = []
        if force:
            notes.append("unload forced although available RAM covered the need")
        if freed is not None and freed <= 0.5:
            notes.append("RSS dropped by only %.1f MB at unload: small "
                         "allocations stay in allocator arenas; multi-MB fp16 "
                         "blocks are the ones that return to the OS" % freed)
        # Post-unload reality check (the declared need still may not fit).
        if avail_after is not None and avail_after < need:
            out = _result(need, m_now, "insufficient",
                          error=(f"insufficient RAM: need {need:g}MB, have "
                                 f"{avail_after:g}MB after unloading {model_id}; "
                                 "refusing to start the job"),
                          note="; ".join(notes) or None, **common)
            rr = _reload_locked(model_id) if eff == "reload_after" else None
            if rr is not None:
                out["reloaded"] = "error" not in rr
                out["reload_error"] = rr.get("error")
            return out
        if eff == "reload_after":
            notes.append("model stays unloaded until residency.finish("
                         "'reload_after', %r) runs after the job" % model_id)
        return _result(need, m_now, eff, note="; ".join(notes) or None, **common)


def finish(policy_used: str, model_id: str | None) -> dict:
    """Call AFTER a media job. Reloads only when the job unloaded the model."""
    if policy_used != "reload_after" or not model_id:
        return {"reloaded": False, "reload_error": None, "model_id": model_id,
                "note": "nothing to reload (policy %r)" % policy_used}
    r = reload_model(model_id)
    if "error" in r:
        return {"reloaded": False, "reload_error": r["error"], "model_id": model_id}
    return {"reloaded": True, "reload_error": None, "model_id": model_id}
