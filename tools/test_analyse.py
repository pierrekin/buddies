"""Smoke + correctness tests for ``tools.analyse``.

Run::

    uv run python tools/test_analyse.py

Each test prints PASS/FAIL plus a short note; non-zero exit on any
failure. Some tests are pure (synthetic inputs, known outputs); others
exercise the primitives end-to-end against the ``model_link`` artifact
on disk so we catch interface drift between the analysis layer and the
sims that feed it."""

import math
import sys
import traceback

import numpy as np

import analyse as inspect


FAIL_COUNT = 0
PASS_COUNT = 0


def check(name, ok, note=""):
    global FAIL_COUNT, PASS_COUNT
    if ok:
        PASS_COUNT += 1
        print(f"  PASS  {name}" + (f"  ({note})" if note else ""))
    else:
        FAIL_COUNT += 1
        print(f"  FAIL  {name}" + (f"  ({note})" if note else ""))


def section(title):
    print(f"\n== {title} ==")


def case(label, fn):
    try:
        fn()
    except Exception as e:
        global FAIL_COUNT
        FAIL_COUNT += 1
        print(f"  FAIL  {label}  (uncaught: {e!r})")
        traceback.print_exc()


# ---- Synthetic / pure-function tests --------------------------------

def test_describe():
    s = inspect.describe([1.0, 2.0, 3.0])
    check("describe.mean", abs(s["mean"] - 2.0) < 1e-12, f"got {s['mean']}")
    check("describe.std", abs(s["std"] - math.sqrt(2 / 3)) < 1e-12, f"got {s['std']}")
    check("describe.min", s["min"] == 1.0)
    check("describe.max", s["max"] == 3.0)
    check("describe.n", s["n"] == 3)
    txt = inspect.describe_str([1.0, 2.0, 3.0])
    check("describe_str.has_mean", "mean=" in txt, txt)
    check("describe.empty", inspect.describe([]) == {"n": 0})


def test_spectrum_and_peaks():
    # Pure 5 kHz cosine sampled at 200 kHz for 0.05 s -- the rFFT bin
    # at 5 kHz should dominate; with N = 10000 the resolution is 20 Hz
    # so 5000 lands exactly on bin 250.
    fs = 200_000.0
    dt = 1.0 / fs
    t = np.arange(0, 0.05, dt)
    f0 = 5_000.0
    x = np.cos(2 * np.pi * f0 * t)
    freqs, mag = inspect.spectrum(x, dt)
    peak_freq = freqs[int(np.argmax(mag))]
    check("spectrum.peak_5kHz", abs(peak_freq - f0) < 50, f"got {peak_freq}")

    # Now build a hand-crafted three-bump magnitude array, confirm peaks
    # finds the right three local maxima in descending magnitude.
    freqs2 = np.linspace(0, 1000, 11)         # 0, 100, ..., 1000
    mags2 = np.array([0, 5, 0, 3, 0, 7, 0, 1, 0, 4, 0], dtype=float)
    got = inspect.peaks(freqs2, mags2, k=3)
    expected = [(500.0, 7.0), (100.0, 5.0), (900.0, 4.0)]
    check("peaks.top3", got == expected, f"got {got}")

    got_k1 = inspect.peaks(freqs2, mags2, k=1)
    check("peaks.k=1", got_k1 == [expected[0]], f"got {got_k1}")

    # min_separation_hz should drop the 100 Hz peak when separation
    # exceeds gap to 500 Hz.
    got_sep = inspect.peaks(freqs2, mags2, k=3, min_separation_hz=500)
    check(
        "peaks.min_separation",
        got_sep == [(500.0, 7.0)],
        f"got {got_sep}",
    )

    # Empty / no-peaks edge cases.
    check("peaks.short_input", inspect.peaks([0, 1], [0, 1]) == [])
    flat = np.zeros(50)
    check("peaks.flat_input", inspect.peaks(np.arange(50), flat) == [])


def test_nrmse():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    check("nrmse.identity", abs(inspect.nrmse(a, a)) < 1e-12)
    check("nrmse.zero_pred", abs(inspect.nrmse(a, np.zeros_like(a)) - 1.0) < 1e-12)

    # nrmse_per_window: windows over an exact-match should all give 0.
    windows = [(0, 2), (2, 4)]
    per = inspect.nrmse_per_window(a, a, windows)
    check("nrmse_per_window.identity", all(abs(v) < 1e-12 for v in per), str(per))


def test_taps_diff():
    h_a = np.array([1.0, 2.0, 3.0])
    h_b = np.array([1.0, 5.0, 3.0])
    res = inspect.taps_diff(h_a, h_b, k=2)
    # rms_diff = sqrt(((0)**2 + 3**2 + 0**2)/3) = sqrt(3) = 1.732
    check(
        "taps_diff.rms",
        abs(res["rms_diff"] - math.sqrt(3)) < 1e-12,
        f"got {res['rms_diff']}",
    )
    check(
        "taps_diff.top[0]",
        res["top"][0] == (1, 2.0, 5.0, 3.0),
        f"got {res['top']}",
    )
    # Zero-pad shorter array: h_a length 2, h_b length 3 -> pad h_a to length 3.
    short = inspect.taps_diff(np.array([1.0, 0.0]), np.array([1.0, 0.0, 4.0]))
    check(
        "taps_diff.pad",
        short["top"][0][0] == 2 and short["top"][0][3] == 4.0,
        f"got {short['top'][0]}",
    )


def test_table():
    rows = [
        {"name": "a", "val": 1.5},
        {"name": "b", "val": 2.0},
    ]
    out = inspect.table(rows, fmt={"val": "{:.2f}"})
    check("table.has_header", "name" in out and "val" in out, out.splitlines()[0])
    check("table.has_separator", "---" in out, out)
    check("table.formatted_value", "1.50" in out, out)
    # List-of-list with explicit headers.
    out2 = inspect.table([[1, 2], [3, 4]], headers=["a", "b"])
    check("table.list_rows", "| 1 " in out2 and "| 4 " in out2, out2)
    check("table.empty", inspect.table([]) == "")


def test_compare():
    # Compare on synthetic mini-shots: just dicts with a name attribute.
    class _MiniShot:
        def __init__(self, name, value):
            self.name = name
            self.value = value
    shots = [_MiniShot("a", 10), _MiniShot("b", 20)]
    rows = inspect.compare(shots, lambda s: {"value": s.value, "double": s.value * 2})
    check(
        "compare.dict_fn",
        rows == [
            {"name": "a", "value": 10, "double": 20},
            {"name": "b", "value": 20, "double": 40},
        ],
        str(rows),
    )
    rows2 = inspect.compare(shots, lambda s: s.value + 1)
    check(
        "compare.scalar_fn",
        rows2 == [["a", 11], ["b", 21]],
        str(rows2),
    )


def test_prop_delay():
    # 3-4-5 triangle: distance 5, c=10 -> 0.5 s.
    d = inspect.prop_delay((0.0, 0.0), (3.0, 4.0), c=10.0)
    check("prop_delay.345", abs(d - 0.5) < 1e-12, f"got {d}")


# ---- Artifact-backed tests ------------------------------------------

def test_artifact_load():
    art = inspect.load("model_link")
    check("load.has_shots", len(art.shots) > 0, f"n={len(art.shots)}")
    check("load.has_dt", "dt" in art.meta, str(art.meta.keys()))
    return art


def test_channel_lookup(art):
    shot = art.shots["link_1000us"]
    ch = inspect.channel(shot, "TX (V)")
    check("channel.found", ch.name == "TX (V)")
    try:
        inspect.channel(shot, "nope nope nope")
    except KeyError:
        check("channel.missing_raises", True)
    else:
        check("channel.missing_raises", False, "did not raise")
    values = inspect.channel_values(shot, "TX (V)")
    check("channel_values.array", isinstance(values, np.ndarray))
    check("channel_values.len", len(values) > 0, f"n={len(values)}")
    names = inspect.channel_names(shot)
    check("channel_names.list", "TX (V)" in names, str(names))


def test_bit_windows(art):
    shot = art.shots["link_1000us"]
    c = art.meta["c"]
    wins = inspect.bit_windows(shot, c=c)
    n_bits = len(shot.extras["sent_bits"])
    check("bit_windows.count", len(wins) == n_bits, f"got {len(wins)} expected {n_bits}")
    check(
        "bit_windows.monotonic",
        all(wins[i][1] <= wins[i + 1][1] for i in range(len(wins) - 1)),
    )
    # Every window's end must be within the channel.
    rx = inspect.channel_values(shot, "RX phys (V)")
    check(
        "bit_windows.in_bounds",
        all(end <= len(rx) for _, end in wins),
        f"max end={max(e for _, e in wins)} vs len={len(rx)}",
    )
    # Second-half windows are half the full window.
    half = inspect.bit_windows(shot, c=c, second_half_only=True)
    full = inspect.bit_windows(shot, c=c, second_half_only=False)
    half_len = half[0][1] - half[0][0]
    full_len = full[0][1] - full[0][0]
    check(
        "bit_windows.second_half_size",
        abs(2 * half_len - full_len) <= 1,
        f"half={half_len}, full={full_len}",
    )


def test_active_region(art):
    shot = art.shots["link_1000us"]
    c = art.meta["c"]
    start, end = inspect.active_region(shot, c=c)
    rx = inspect.channel_values(shot, "RX phys (V)")
    check("active_region.start", 0 <= start < end, f"start={start} end={end}")
    check("active_region.end", end <= len(rx), f"end={end} len={len(rx)}")


def test_refit_fir(art):
    # Refit on the link_0500us pair (TX, RX phys) and confirm the fitted
    # model produces output of the right shape and a finite-energy h.
    shot = art.shots["link_0500us"]
    v_tx = inspect.channel_values(shot, "TX (V)")
    v_rx = inspect.channel_values(shot, "RX phys (V)")
    fir = inspect.refit_fir(v_tx, v_rx, n_taps=256)
    check("refit_fir.has_h", fir.h is not None and len(fir.h) == 256)
    check("refit_fir.finite", np.all(np.isfinite(fir.h)))
    # The refit should at least beat predicting zero (NRMSE < 1) on the
    # same pair it trained on.
    pred = fir.predict(v_tx)[: len(v_rx)]
    nr = inspect.nrmse(v_rx, pred)
    check("refit_fir.fit_beats_zero", nr < 1.0, f"NRMSE={nr}")


def test_spectrum_on_real_signal(art):
    # The OOK 1 ms link's TX is a square at 15 kHz; the spectrum's top
    # peak (above DC) should land in the 14.5 -- 15.5 kHz band.
    shot = art.shots["link_1000us"]
    v_tx = inspect.channel_values(shot, "TX (V)")
    dt = shot.channels[0].dt
    freqs, mag = inspect.spectrum(v_tx, dt)
    # Drop the DC bin so the peak finder isn't dominated by zero-mean
    # rounding.
    nonzero = freqs > 100
    peak_freq = freqs[nonzero][int(np.argmax(mag[nonzero]))]
    check(
        "spectrum.real_signal_15kHz",
        14_500 <= peak_freq <= 15_500,
        f"got {peak_freq} Hz",
    )


def test_table_with_compare(art):
    # End-to-end: compare returns rows, table renders them.
    link_shots = [s for n, s in art.shots.items() if n.startswith("link_")]
    rows = inspect.compare(
        link_shots,
        lambda s: {"bit_dur_us": float(s.extras["bit_dur"]) * 1e6,
                   "ber_phys": float(s.extras["ber_phys"])},
    )
    rendered = inspect.table(
        rows, fmt={"bit_dur_us": "{:.0f}", "ber_phys": "{:.4f}"}
    )
    check(
        "compare+table.has_rows",
        len(rendered.splitlines()) == 2 + len(link_shots),
        f"lines={len(rendered.splitlines())} shots={len(link_shots)}",
    )


# ---- Driver ----------------------------------------------------------

def main():
    section("synthetic primitives")
    case("describe", test_describe)
    case("spectrum + peaks", test_spectrum_and_peaks)
    case("nrmse", test_nrmse)
    case("taps_diff", test_taps_diff)
    case("table", test_table)
    case("compare", test_compare)
    case("prop_delay", test_prop_delay)

    section("artifact-backed (requires processed model_link)")
    try:
        art = test_artifact_load()
    except Exception as e:
        global FAIL_COUNT
        FAIL_COUNT += 1
        print(f"  FAIL  load model_link  ({e!r})")
        print("  skipping artifact-backed tests")
        art = None

    if art is not None:
        case("channel lookup", lambda: test_channel_lookup(art))
        case("bit_windows", lambda: test_bit_windows(art))
        case("active_region", lambda: test_active_region(art))
        case("refit_fir", lambda: test_refit_fir(art))
        case("spectrum on TX", lambda: test_spectrum_on_real_signal(art))
        case("compare + table", lambda: test_table_with_compare(art))

    print(f"\n{PASS_COUNT} passed, {FAIL_COUNT} failed")
    sys.exit(0 if FAIL_COUNT == 0 else 1)


if __name__ == "__main__":
    main()
