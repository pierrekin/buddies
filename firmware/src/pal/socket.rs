use core::fmt::Write;

use cortex_m_semihosting::hprintln;

use crate::peers::{Peer, PeerLocator};

use super::uart::Uart0;
use super::{Adc, Rgb, RgbStrip};

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

impl PeerLocator for SocketStrip {
    fn scan(&mut self) -> Option<Peer> {
        for &b in b"scan\n" {
            self.uart.write_byte(b);
        }
        let mut buf = [0u8; 96];
        let n = read_line(&mut self.uart, &mut buf);
        parse_peer_response(&buf[..n])
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

fn parse_peer_response(line: &[u8]) -> Option<Peer> {
    let s = core::str::from_utf8(line).ok()?;
    let mut parts = s.split_whitespace();
    if parts.next()? != "peer" {
        return None;
    }
    let first = parts.next()?;
    if first == "none" {
        return None;
    }
    let id = first.parse::<u8>().ok()?;
    let bearing_deg = parts.next()?.parse::<f32>().ok()?;
    let range_m = parts.next()?.parse::<f32>().ok()?;
    Some(Peer { id, bearing_deg, range_m })
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
