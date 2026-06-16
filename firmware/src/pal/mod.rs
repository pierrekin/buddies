//! Platform abstraction layer.

#[cfg(feature = "pal-semihosting")]
pub mod semihosting;

#[cfg(feature = "pal-socket")]
pub mod socket;
#[cfg(feature = "pal-socket")]
mod uart;

#[cfg(feature = "pal-semihosting")]
pub type ActiveStrip = semihosting::SemihostingStrip;

#[cfg(feature = "pal-socket")]
pub type ActiveStrip = socket::SocketStrip;

#[derive(Copy, Clone, Debug)]
pub struct Rgb {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Rgb {
    pub const OFF: Rgb = Rgb::new(0, 0, 0);

    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }
}

/// RGB strip. Each call to `write` fully specifies the pixel state.
pub trait RgbStrip {
    fn write(&mut self, pixels: &[Rgb]);
}

/// Multi-channel ADC. `out` is filled as a sequential layout:
/// `[ch0_sample0..ch0_sampleN-1, ch1_sample0..ch1_sampleN-1, ...]`.
pub trait Adc {
    fn read_block(&mut self, n_samples: usize, n_channels: usize, out: &mut [f32]);
}
