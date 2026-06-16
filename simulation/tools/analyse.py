"""Headless analysis primitives for processed buddies artifacts.

Used from short scripts (not the Qt viewer) to ask scriptable questions
about a sim's output and produce dense text -- markdown tables and
one-line summaries -- not images. Sister to ``buddies.viewer`` which is
the interactive UI; this module exists so an analyst (human or LLM) can
load an artifact, slice it, compute spectra/statistics, refit models
out-of-band, and tabulate results without ever rendering a plot.

Typical use::

    from tools import inspect
    art = inspect.load("model_link")
    for name, shot in art.shots.items():
        rx = inspect.channel_values(shot, "RX phys (V)")
        print(name, inspect.describe_str(rx))
"""

import os

import numpy as np

from buddies import channel_model, store

ARTIFACT_ROOT = "output"


# ---- Loading & channel lookup ----------------------------------------

def load(sim_name, out_name="default"):
    """Open ``output/<sim_name>/processed/<out_name>`` as a Store."""
    return store.open_store(
        os.path.join(ARTIFACT_ROOT, sim_name, "processed", out_name)
    )


def channel(shot, name):
    """Return the channel with that exact name; KeyError if absent."""
    for ch in shot.channels:
        if ch.name == name:
            return ch
    raise KeyError(
        f"shot {shot.name!r} has no channel {name!r}; "
        f"available: {[c.name for c in shot.channels]}"
    )


def channel_values(shot, name):
    """Channel values as a float64 numpy array."""
    return np.asarray(channel(shot, name).values, dtype=np.float64)


def channel_names(shot):
    """All channel names in this shot, in declaration order."""
    return [c.name for c in shot.channels]


# ---- Geometric / timing helpers --------------------------------------

def prop_delay(tx_pos, rx_pos, c):
    """Straight-line propagation delay between two positions at speed c."""
    return float(np.hypot(rx_pos[0] - tx_pos[0], rx_pos[1] - tx_pos[1]) / c)


def _tx_rx_channels(shot):
    """Pull the TX and RX channels by name-prefix convention."""
    tx = next((c for c in shot.channels if c.name.startswith("TX")), None)
    rx = next((c for c in shot.channels if c.name.startswith("RX")), None)
    return tx, rx


def bit_windows(shot, *, bit_dur=None, n_bits=None, prop_delay_s=None,
                c=None, second_half_only=True):
    """Per-bit slicer windows as a list of (start, end) sample indices.

    Defaults are pulled from ``shot.extras`` (``bit_dur``, ``sent_bits``)
    and the TX/RX channel positions; pass overrides for ad-hoc work.
    ``second_half_only=True`` mirrors the decoder's per-bit integration
    window (which skips the rising edge to avoid ISI from the prior bit).
    """
    if bit_dur is None:
        bit_dur = shot.extras.get("bit_dur")
    if bit_dur is None:
        raise ValueError("no bit_dur in extras; pass one explicitly")
    if n_bits is None:
        sent = shot.extras.get("sent_bits") or ()
        n_bits = len(sent)
    if n_bits == 0:
        raise ValueError("no sent_bits in extras; pass n_bits explicitly")
    dt = shot.channels[0].dt
    if prop_delay_s is None:
        if c is None:
            raise ValueError("pass c (sound speed) or prop_delay_s explicitly")
        tx, rx = _tx_rx_channels(shot)
        if tx is None or rx is None:
            raise ValueError("can't locate TX/RX channels by name prefix")
        prop_delay_s = prop_delay(tx.pos, rx.pos, c)
    spb = int(round(bit_dur / dt))
    delay = int(round(prop_delay_s / dt))
    windows = []
    for i in range(n_bits):
        start = delay + i * spb + (spb // 2 if second_half_only else 0)
        end = delay + (i + 1) * spb
        windows.append((start, end))
    return windows


def active_region(shot, *, bit_dur=None, n_bits=None,
                  prop_delay_s=None, c=None):
    """(start, end) covering the message's active samples.

    Same defaulting rules as ``bit_windows``. Useful for cropping a
    channel array down to "just the part where the message is" before
    spectra or NRMSE.
    """
    if bit_dur is None:
        bit_dur = shot.extras.get("bit_dur")
    if n_bits is None:
        n_bits = len(shot.extras.get("sent_bits") or ())
    dt = shot.channels[0].dt
    if prop_delay_s is None and c is not None:
        tx, rx = _tx_rx_channels(shot)
        if tx is not None and rx is not None:
            prop_delay_s = prop_delay(tx.pos, rx.pos, c)
    if prop_delay_s is None:
        prop_delay_s = 0.0
    start = int(round(prop_delay_s / dt))
    end = start + n_bits * int(round(bit_dur / dt))
    return start, end


# ---- Statistical primitives -----------------------------------------

def describe(x, percentiles=(5, 50, 95)):
    """Summary dict: ``n``, ``mean``, ``std``, ``min``, requested
    percentile keys (e.g. ``p5``), and ``max``."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return {"n": 0}
    pcts = np.percentile(x, list(percentiles))
    out = {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(x.min()),
    }
    for p, v in zip(percentiles, pcts):
        out[f"p{int(p)}"] = float(v)
    out["max"] = float(x.max())
    return out


def describe_str(x, percentiles=(5, 50, 95)):
    """One-line text rendering of ``describe()``."""
    s = describe(x, percentiles=percentiles)
    if s["n"] == 0:
        return "n=0"
    parts = [f"n={s['n']}", f"mean={s['mean']:+.4g}",
             f"std={s['std']:.4g}", f"min={s['min']:+.4g}"]
    for p in percentiles:
        parts.append(f"p{int(p)}={s[f'p{int(p)}']:+.4g}")
    parts.append(f"max={s['max']:+.4g}")
    return "  ".join(parts)


def spectrum(values, dt):
    """One-sided rFFT magnitude. Returns ``(freqs_Hz, magnitudes)``."""
    x = np.asarray(values, dtype=np.float64)
    return np.fft.rfftfreq(len(x), d=dt), np.abs(np.fft.rfft(x))


def peaks(freqs, mags, k=5, min_separation_hz=None):
    """Top-k local maxima of ``mags`` as ``[(freq, mag), ...]``, descending.

    A point qualifies as a local maximum when strictly greater than both
    neighbours. ``min_separation_hz`` (optional) suppresses peaks closer
    than that to any already-selected peak -- useful when an FFT bin
    cluster around one feature would otherwise dominate the top-k."""
    freqs = np.asarray(freqs, dtype=np.float64)
    mags = np.asarray(mags, dtype=np.float64)
    if len(mags) < 3:
        return []
    is_peak = np.zeros(len(mags), dtype=bool)
    is_peak[1:-1] = (mags[1:-1] > mags[:-2]) & (mags[1:-1] > mags[2:])
    idx = np.where(is_peak)[0]
    if len(idx) == 0:
        return []
    order = np.argsort(mags[idx])[::-1]
    selected, used = [], []
    for i in order:
        f = float(freqs[idx[i]])
        if min_separation_hz is not None and any(
            abs(f - uf) < min_separation_hz for uf in used
        ):
            continue
        selected.append((f, float(mags[idx[i]])))
        used.append(f)
        if len(selected) >= k:
            break
    return selected


# ---- NRMSE convenience -----------------------------------------------

def nrmse(y_true, y_pred):
    """Pass-through to ``channel_model.nrmse`` so callers don't have to
    pull both modules in."""
    return channel_model.nrmse(y_true, y_pred)


def nrmse_per_window(y_true, y_pred, windows):
    """NRMSE on each window slice. Returns list parallel to ``windows``."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return [nrmse(y_true[s:e], y_pred[s:e]) for s, e in windows]


# ---- Out-of-band model refit -----------------------------------------

def refit_fir(v_tx, v_rx, n_taps):
    """Fresh FIR fit on the given pair. Returns the model with ``.h``
    accessible for inspection; doesn't touch any artifact on disk."""
    fir = channel_model.FIRModel(n_taps=n_taps)
    fir.fit(v_tx, v_rx)
    return fir


def taps_diff(h_a, h_b, k=5):
    """Compare two impulse responses tap-by-tap (index-aligned).

    Returns ``{"rms_diff": float, "top": [(lag, h_a, h_b, abs_diff), ...]}``.
    The shorter array is zero-padded so they have the same length; the
    top-k entries are the lags with the largest absolute difference."""
    h_a = np.asarray(h_a, dtype=np.float64)
    h_b = np.asarray(h_b, dtype=np.float64)
    n = max(len(h_a), len(h_b))
    pad_a = np.pad(h_a, (0, n - len(h_a)))
    pad_b = np.pad(h_b, (0, n - len(h_b)))
    diff = pad_b - pad_a
    rms = float(np.sqrt(np.mean(diff ** 2)))
    abs_diff = np.abs(diff)
    top_idx = np.argsort(abs_diff)[::-1][:k]
    top = [
        (int(i), float(pad_a[i]), float(pad_b[i]), float(abs_diff[i]))
        for i in top_idx
    ]
    return {"rms_diff": rms, "top": top}


# ---- Cross-shot comparison -------------------------------------------

def compare(shots, fn):
    """Apply ``fn(shot)`` to each shot, return list of rows for ``table()``.

    If ``fn`` returns a dict, the row is ``{"name": shot.name, **result}``.
    If it returns a scalar, the row is ``[shot.name, value]``.
    If a tuple/list, the row is ``[shot.name, *result]``."""
    rows = []
    for shot in shots:
        result = fn(shot)
        if isinstance(result, dict):
            rows.append({"name": shot.name, **result})
        elif isinstance(result, (list, tuple)):
            rows.append([shot.name, *result])
        else:
            rows.append([shot.name, result])
    return rows


# ---- Markdown table output -------------------------------------------

def table(rows, headers=None, fmt=None):
    """Render ``rows`` as a GitHub-flavored markdown table.

    ``rows`` is a list of dicts or a list of lists/tuples. If dicts and
    ``headers`` is omitted, columns are the union of keys in
    first-row order. ``fmt`` is an optional dict
    ``{column_name: fmt_string}`` controlling per-column float formatting
    (default is ``{:.4g}`` for floats)."""
    if not rows:
        return ""

    if isinstance(rows[0], dict):
        if headers is None:
            headers = list(rows[0].keys())
            for r in rows[1:]:
                for k in r:
                    if k not in headers:
                        headers.append(k)
        data = [[r.get(h, "") for h in headers] for r in rows]
    else:
        if headers is None:
            headers = [f"c{i}" for i in range(len(rows[0]))]
        data = [list(r) for r in rows]

    fmt = fmt or {}

    def cellstr(value, header):
        f = fmt.get(header)
        if f is not None and isinstance(
            value, (int, float, np.floating, np.integer)
        ):
            return f.format(value)
        if isinstance(value, (float, np.floating)):
            return f"{value:.4g}"
        return str(value)

    cells = [
        [cellstr(v, h) for v, h in zip(row, headers)] for row in data
    ]
    widths = [
        max(len(h), *(len(row[i]) for row in cells))
        for i, h in enumerate(headers)
    ]

    header_line = "| " + " | ".join(
        h.ljust(w) for h, w in zip(headers, widths)
    ) + " |"
    sep_line = "| " + " | ".join("-" * w for w in widths) + " |"
    body = [
        "| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |"
        for row in cells
    ]
    return "\n".join([header_line, sep_line, *body])
