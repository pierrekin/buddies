//! Chirp generation + matched-filter detection.
//!
//! Parameters here must stay in sync with `firmware/host/channels.py`'s
//! chirp constants. Mismatched parameters break the matched filter.

use core::f32::consts::PI;

pub const SAMPLE_RATE_HZ: f32 = 200_000.0;
pub const CHIRP_LEN: usize = 500;
pub const CHIRP_F_LO_HZ: f32 = 30_000.0;
pub const CHIRP_F_HI_HZ: f32 = 50_000.0;

/// Fill `out` with an LFM chirp sweeping from `CHIRP_F_LO_HZ` to
/// `CHIRP_F_HI_HZ` over `CHIRP_LEN / SAMPLE_RATE_HZ` seconds.
pub fn generate(out: &mut [f32; CHIRP_LEN]) {
    let duration_s = CHIRP_LEN as f32 / SAMPLE_RATE_HZ;
    let k = (CHIRP_F_HI_HZ - CHIRP_F_LO_HZ) / duration_s;
    for i in 0..CHIRP_LEN {
        let t = i as f32 / SAMPLE_RATE_HZ;
        let phase = 2.0 * PI * (CHIRP_F_LO_HZ * t + 0.5 * k * t * t);
        out[i] = libm::sinf(phase);
    }
}

/// Cross-correlate `rx` against `chirp`, writing `rx.len() - CHIRP_LEN + 1`
/// values into `corr`. corr[lag] = sum_i chirp[i] * rx[lag + i].
pub fn cross_correlate(rx: &[f32], chirp: &[f32; CHIRP_LEN], corr: &mut [f32]) {
    let n_lags = rx.len() - CHIRP_LEN + 1;
    for lag in 0..n_lags {
        let mut acc = 0.0f32;
        for i in 0..CHIRP_LEN {
            acc += chirp[i] * rx[lag + i];
        }
        corr[lag] = acc;
    }
}

/// (peak lag, peak value).
pub fn argmax(corr: &[f32]) -> (usize, f32) {
    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, &v) in corr.iter().enumerate() {
        if v > best.1 {
            best = (i, v);
        }
    }
    best
}

/// Parabolic sub-sample interpolation around `peak` in `corr`. Returns
/// the refined lag as f32.
pub fn parabolic_interp(corr: &[f32], peak: usize) -> f32 {
    if peak == 0 || peak >= corr.len() - 1 {
        return peak as f32;
    }
    let y0 = corr[peak - 1];
    let y1 = corr[peak];
    let y2 = corr[peak + 1];
    let denom = y0 - 2.0 * y1 + y2;
    if libm::fabsf(denom) < 1e-12 {
        return peak as f32;
    }
    peak as f32 + 0.5 * (y0 - y2) / denom
}
