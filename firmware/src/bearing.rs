//! TDOA bearing + range estimation from per-channel matched-filter lags.
//!
//! Receiver array geometry must stay in sync with `firmware/host/channels.py`.

use crate::chirp::SAMPLE_RATE_HZ;

pub const N_RX: usize = 4;
pub const SOUND_SPEED_M_PER_S: f32 = 1500.0;

/// Receiver positions in body frame (metres). +y is forward, +x is right.
pub const RX_POSITIONS_M: [(f32, f32); N_RX] = [
    (-0.03, 0.03),   // front-left
    (0.03, 0.03),    // front-right
    (0.03, -0.03),   // rear-right
    (-0.03, -0.03),  // rear-left
];

/// Body-relative compass bearing (degrees, 0 = forward, clockwise).
pub fn estimate_bearing_deg(lags_samples: &[f32; N_RX]) -> f32 {
    let mut ata = [[0.0f32; 2]; 2];
    let mut atb = [0.0f32; 2];
    let inv_fs = 1.0 / SAMPLE_RATE_HZ;
    for i in 1..N_RX {
        let dx = RX_POSITIONS_M[0].0 - RX_POSITIONS_M[i].0;
        let dy = RX_POSITIONS_M[0].1 - RX_POSITIONS_M[i].1;
        let dt = (lags_samples[i] - lags_samples[0]) * inv_fs;
        let rhs = SOUND_SPEED_M_PER_S * dt;
        ata[0][0] += dx * dx;
        ata[0][1] += dx * dy;
        ata[1][0] += dx * dy;
        ata[1][1] += dy * dy;
        atb[0] += dx * rhs;
        atb[1] += dy * rhs;
    }
    let det = ata[0][0] * ata[1][1] - ata[0][1] * ata[1][0];
    if libm::fabsf(det) < 1e-12 {
        return 0.0;
    }
    let inv_det = 1.0 / det;
    let ux = (ata[1][1] * atb[0] - ata[0][1] * atb[1]) * inv_det;
    let uy = (-ata[1][0] * atb[0] + ata[0][0] * atb[1]) * inv_det;
    libm::atan2f(ux, uy) * 180.0 / core::f32::consts::PI
}

pub fn estimate_range_m(lag0_samples: f32) -> f32 {
    lag0_samples / SAMPLE_RATE_HZ * SOUND_SPEED_M_PER_S
}
