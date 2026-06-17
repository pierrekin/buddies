use cortex_m_semihosting::{hprint, hprintln};

use super::{Adc, Frame, RgbStrip, ROWS, TapInput};

/// Output format: `strip: #ff0000 #ff7f00 ...`.
pub struct SemihostingStrip;

impl SemihostingStrip {
    pub const fn new() -> Self {
        Self
    }
}

impl RgbStrip for SemihostingStrip {
    fn write(&mut self, frame: &Frame) {
        hprint!("strip:");
        for row in 0..ROWS {
            for p in frame.row(row) {
                hprint!(" #{:02x}{:02x}{:02x}", p.r, p.g, p.b);
            }
        }
        hprintln!();
    }
}

impl Adc for SemihostingStrip {
    fn read_block(&mut self, n_samples: usize, n_channels: usize, out: &mut [f32]) {
        for v in &mut out[..n_samples * n_channels] {
            *v = 0.0;
        }
    }
}

impl TapInput for SemihostingStrip {
    fn poll_taps(&mut self) -> u8 {
        0
    }
}

impl core::fmt::Write for SemihostingStrip {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        hprint!("{}", s);
        Ok(())
    }
}
