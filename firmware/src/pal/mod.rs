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

/// LED frame types live in the UI crate so the rendering logic can stay
/// host-testable; the PAL just serialises whatever the UI produces.
pub use buddies_ui::{Frame, ROWS};

/// RGB side panel. Each call to `write` fully specifies the frame.
pub trait RgbStrip {
    fn write(&mut self, frame: &Frame);
}

/// Multi-channel ADC. `out` is filled as a sequential layout:
/// `[ch0_sample0..ch0_sampleN-1, ch1_sample0..ch1_sampleN-1, ...]`.
pub trait Adc {
    fn read_block(&mut self, n_samples: usize, n_channels: usize, out: &mut [f32]);
}

/// Tap input for the gesture UI.
///
/// Returns the number of taps seen since the last poll; the gesture state
/// machine debounces those into single/double/long bursts. Today the host
/// mock supplies taps over the link; an IMU driver will detect them on-device
/// behind this same trait.
pub trait TapInput {
    fn poll_taps(&mut self) -> u8;
}
