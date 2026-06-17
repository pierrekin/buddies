use core::fmt::Write;

use cortex_m_semihosting::hprintln;

use super::uart::Uart0;
use super::{Adc, Frame, RgbStrip, ROWS, TapInput};

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
    fn write(&mut self, frame: &Frame) {
        let mut w = ByteSink { uart: &mut self.uart };
        let _ = write!(w, "strip");
        for row in 0..ROWS {
            for p in frame.row(row) {
                let _ = write!(w, " {} {} {}", p.r, p.g, p.b);
            }
        }
        let _ = writeln!(w);
    }
}

impl TapInput for SocketStrip {
    fn poll_taps(&mut self) -> u8 {
        for &b in b"taps\n" {
            self.uart.write_byte(b);
        }
        let mut buf = [0u8; 32];
        let n = read_line(&mut self.uart, &mut buf);
        parse_taps_response(&buf[..n])
    }
}

impl Adc for SocketStrip {
    fn read_block(&mut self, n_samples: usize, n_channels: usize, out: &mut [f32]) {
        {
            let mut w = ByteSink { uart: &mut self.uart };
            let _ = write!(w, "rx {} {}\n", n_samples, n_channels);
        }
        let total = n_samples * n_channels;
        for slot in &mut out[..total] {
            let mut b = [0u8; 4];
            for byte in &mut b {
                *byte = self.uart.read_byte_blocking();
            }
            *slot = f32::from_le_bytes(b);
        }
    }
}

fn read_line(uart: &mut Uart0, buf: &mut [u8]) -> usize {
    let mut len = 0;
    loop {
        let b = uart.read_byte_blocking();
        if b == b'\n' {
            return len;
        }
        if len < buf.len() {
            buf[len] = b;
            len += 1;
        }
    }
}

fn parse_taps_response(line: &[u8]) -> u8 {
    let Ok(s) = core::str::from_utf8(line) else {
        return 0;
    };
    let mut parts = s.split_whitespace();
    if parts.next() != Some("taps") {
        return 0;
    }
    parts.next().and_then(|v| v.parse::<u8>().ok()).unwrap_or(0)
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

impl core::fmt::Write for SocketStrip {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        for b in s.bytes() {
            self.uart.write_byte(b);
        }
        Ok(())
    }
}
