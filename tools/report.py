"""Headless report on a model_link artifact.

Example use of ``tools.analyse``. Prints four sections to stdout, all as
markdown tables:

  1. Overview        -- one row per shot (role, bit_dur, BERs, NRMSE)
  2. Residual stats  -- describe() of each shot's phys-vs-model residual
  3. Residual peaks  -- top-3 spectral peaks of each residual
  4. Tap-diff vs train -- refit a FIR per validate shot, compare its
                          impulse response against the trained FIR's

Run::

    uv run python tools/report.py [sim_name]   # default: model_link

The report is the *only* output. No PNGs, no Qt. If you want to add a
new question, write another section function and call it from main()."""

import sys

import numpy as np

import analyse


SIM_DEFAULT = "model_link"
FIR_REFIT_TAPS = 1024  # match what the train shot used


# ---- Section helpers -------------------------------------------------

def section(title):
    print()
    print(f"## {title}")
    print()


def overview(art):
    section("overview")
    rows = []
    for name, shot in art.shots.items():
        ex = shot.extras
        rows.append({
            "shot": name,
            "role": ex.get("role", ""),
            "bit_dur_us": (float(ex["bit_dur"]) * 1e6) if ex.get("bit_dur") else "",
            "offset_mm": (float(ex["rx_offset_m"]) * 1000) if "rx_offset_m" in ex else "",
            "ber_phys": ex.get("ber_phys", ""),
            "ber_model": ex.get("ber_model", ""),
            "agreement": ex.get("agreement", ""),
            "waveform_nrmse": ex.get("waveform_nrmse", ""),
        })
    print(analyse.table(rows, fmt={
        "bit_dur_us": "{:.0f}",
        "offset_mm": "{:.0f}",
        "ber_phys": "{:.3f}",
        "ber_model": "{:.3f}",
        "agreement": "{:.3f}",
        "waveform_nrmse": "{:.4f}",
    }))


def residual_stats(art):
    section("residual describe() (RX phys - RX model, full trace)")
    for name, shot in art.shots.items():
        try:
            residual = analyse.channel_values(shot, "residual phys-model (V)")
        except KeyError:
            continue
        print(f"  {name:>30}: {analyse.describe_str(residual)}")


def residual_peaks(art):
    section("residual top-3 spectral peaks (Hz, magnitude)")
    rows = []
    for name, shot in art.shots.items():
        try:
            residual = analyse.channel_values(shot, "residual phys-model (V)")
        except KeyError:
            continue
        dt = shot.channels[0].dt
        freqs, mag = analyse.spectrum(residual, dt)
        # Drop DC bin and group nearby peaks within 1 kHz.
        nonzero = freqs > 100
        top = analyse.peaks(freqs[nonzero], mag[nonzero], k=3,
                            min_separation_hz=1000.0)
        row = {"shot": name}
        for i, (f, m) in enumerate(top):
            row[f"f{i+1}_Hz"] = f
            row[f"m{i+1}"] = m
        rows.append(row)
    print(analyse.table(rows, fmt={
        "f1_Hz": "{:.0f}", "f2_Hz": "{:.0f}", "f3_Hz": "{:.0f}",
        "m1": "{:.3g}", "m2": "{:.3g}", "m3": "{:.3g}",
    }))


def tapdiff_per_validate(art):
    """Refit a FIR on each validate shot's (TX, RX phys) pair and diff
    its impulse response against the chirp-trained FIR. Where the taps
    differ tells us which lags carry the geometry-specific multipath
    the original FIR couldn't see."""
    section(
        f"refit FIR per validate shot vs train (N={FIR_REFIT_TAPS}, "
        "top-3 tap differences)"
    )
    train_shot = art.shots.get("chirp_train")
    if train_shot is None or "fir_h" not in train_shot.extras:
        print("(no chirp_train.fir_h; skipping)")
        return
    h_train = np.asarray(train_shot.extras["fir_h"])

    rows = []
    for name, shot in art.shots.items():
        if shot.extras.get("role") != "validate":
            continue
        v_tx = analyse.channel_values(shot, "TX (V)")
        v_rx = analyse.channel_values(shot, "RX phys offset (V)")
        fir = analyse.refit_fir(v_tx, v_rx, n_taps=FIR_REFIT_TAPS)
        diff = analyse.taps_diff(h_train, fir.h, k=3)
        dt = shot.channels[0].dt
        row = {"shot": name, "rms_tap_diff": diff["rms_diff"]}
        for i, (lag, ha, hb, ad) in enumerate(diff["top"]):
            row[f"lag{i+1}_us"] = lag * dt * 1e6
            row[f"abs_diff_{i+1}"] = ad
        rows.append(row)
    print(analyse.table(rows, fmt={
        "rms_tap_diff": "{:.4g}",
        "lag1_us": "{:.0f}", "lag2_us": "{:.0f}", "lag3_us": "{:.0f}",
        "abs_diff_1": "{:.3g}", "abs_diff_2": "{:.3g}", "abs_diff_3": "{:.3g}",
    }))


# ---- Driver ----------------------------------------------------------

def main():
    sim = sys.argv[1] if len(sys.argv) > 1 else SIM_DEFAULT
    art = analyse.load(sim)
    print(f"# {sim} -- analysis report")
    print()
    print(f"shots: {list(art.shots.keys())}")
    overview(art)
    residual_stats(art)
    residual_peaks(art)
    tapdiff_per_validate(art)


if __name__ == "__main__":
    main()
