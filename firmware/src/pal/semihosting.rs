use cortex_m_semihosting::{hprint, hprintln};

use crate::peers::{Peer, PeerLocator};

use super::{Adc, Rgb, RgbStrip};

/// Output format: `strip: #ff0000 #ff7f00 ...`.
pub struct SemihostingStrip;

impl SemihostingStrip {
    pub const fn new() -> Self {
        Self
    }
}

impl RgbStrip for SemihostingStrip {
    fn write(&mut self, pixels: &[Rgb]) {
        hprint!("strip:");
        for p in pixels {
            hprint!(" #{:02x}{:02x}{:02x}", p.r, p.g, p.b);
        }
        hprintln!();
    }
}

impl PeerLocator for SemihostingStrip {
    fn scan(&mut self) -> Option<Peer> {
        Some(Peer {
            id: 1,
            bearing_deg: 45.0,
            range_m: 2.0,
        })
    }
}

impl Adc for SemihostingStrip {
    fn read_block(&mut self, n_samples: usize, n_channels: usize, out: &mut [f32]) {
        for v in &mut out[..n_samples * n_channels] {
            *v = 0.0;
        }
    }
}
