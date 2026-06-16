#![no_std]
#![no_main]

mod pal;
mod peers;

use cortex_m_rt::entry;
use embassy_executor::{Executor, Spawner};
use panic_semihosting as _;
use static_cell::StaticCell;

use crate::pal::{ActiveStrip, Rgb, RgbStrip};
use crate::peers::{Peer, PeerLocator};

const N_PIXELS: usize = 8;
const LIT: Rgb = Rgb::new(0, 255, 0);

#[embassy_executor::task]
async fn app(mut backend: ActiveStrip) {
    let mut buf = [Rgb::OFF; N_PIXELS];
    let mut iter: usize = 0;
    loop {
        let peer = backend.scan();
        render(&mut buf, peer);
        backend.write(&buf);
        cortex_m::asm::delay(5_000_000);
        iter = iter.wrapping_add(1);

        #[cfg(feature = "pal-semihosting")]
        if iter >= 5 {
            cortex_m_semihosting::debug::exit(cortex_m_semihosting::debug::EXIT_SUCCESS);
        }
    }
}

fn render(buf: &mut [Rgb; N_PIXELS], peer: Option<Peer>) {
    for p in buf.iter_mut() {
        *p = Rgb::OFF;
    }
    let Some(peer) = peer else { return };

    let center = bearing_to_led(peer.bearing_deg) as i32;
    let half = (range_to_width(peer.range_m) / 2) as i32;
    for offset in -half..=half {
        let idx = center + offset;
        if (0..N_PIXELS as i32).contains(&idx) {
            buf[idx as usize] = LIT;
        }
    }
}

fn bearing_to_led(bearing_deg: f32) -> usize {
    let b = normalize_180(bearing_deg);
    if b < -90.0 {
        return 0;
    }
    if b > 90.0 {
        return 7;
    }
    let pos = 1.0 + ((b + 90.0) / 180.0) * 5.0;
    (pos + 0.5) as usize
}

fn range_to_width(range_m: f32) -> usize {
    if range_m < 1.0 { 3 } else { 1 }
}

fn normalize_180(deg: f32) -> f32 {
    let mut x = deg % 360.0;
    if x > 180.0 {
        x -= 360.0;
    }
    if x < -180.0 {
        x += 360.0;
    }
    x
}

static EXECUTOR: StaticCell<Executor> = StaticCell::new();

#[entry]
fn main() -> ! {
    let backend = ActiveStrip::new();
    let executor = EXECUTOR.init(Executor::new());
    executor.run(|spawner: Spawner| {
        spawner.spawn(app(backend)).unwrap();
    })
}
