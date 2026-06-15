use cortex_m_semihosting::{hprint, hprintln};

use super::{Rgb, RgbStrip};

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
