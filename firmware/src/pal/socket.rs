use core::fmt::Write;

use cortex_m_semihosting::hprintln;

use super::uart::Uart0;
use super::{Rgb, RgbStrip};

/// Sync byte the host sends once it's connected and ready. Prevents
/// losing the first frames if the listener attaches late.
const SYNC_BYTE: u8 = b'g';

pub struct SocketStrip {
    uart: Uart0,
}

impl SocketStrip {
    pub fn new() -> Self {
        let mut uart = Uart0::init();
        hprintln!("socket pal: waiting for host sync byte on uart0");
        loop {
            if uart.read_byte_blocking() == SYNC_BYTE {
                break;
            }
        }
        hprintln!("socket pal: synced");
        Self { uart }
    }
}

impl RgbStrip for SocketStrip {
    fn write(&mut self, pixels: &[Rgb]) {
        let mut w = ByteSink { uart: &mut self.uart };
        let _ = write!(w, "strip");
        for p in pixels {
            let _ = write!(w, " {} {} {}", p.r, p.g, p.b);
        }
        let _ = writeln!(w);
    }
}

struct ByteSink<'a> {
    uart: &'a mut Uart0,
}

impl core::fmt::Write for ByteSink<'_> {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        for b in s.bytes() {
            self.uart.write_byte(b);
        }
        Ok(())
    }
}
