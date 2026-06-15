#![no_std]
#![no_main]

mod pal;

use cortex_m_rt::entry;
use embassy_executor::{Executor, Spawner};
use panic_semihosting as _;
use static_cell::StaticCell;

use crate::pal::{ActiveStrip, Rgb, RgbStrip};

const N_PIXELS: usize = 8;

const PALETTE: [Rgb; 8] = [
    Rgb::new(255, 0, 0),     // red
    Rgb::new(255, 127, 0),   // orange
    Rgb::new(255, 255, 0),   // yellow
    Rgb::new(0, 255, 0),     // green
    Rgb::new(0, 255, 255),   // cyan
    Rgb::new(0, 0, 255),     // blue
    Rgb::new(127, 0, 255),   // violet
    Rgb::new(255, 255, 255), // white
];

#[embassy_executor::task]
async fn strip_animator(mut strip: ActiveStrip) {
    let mut buf = [Rgb::OFF; N_PIXELS];
    let mut tick: usize = 0;
    loop {
        for i in 0..N_PIXELS {
            buf[i] = PALETTE[(i + tick) % PALETTE.len()];
        }
        strip.write(&buf);
        cortex_m::asm::delay(2_000_000);
        tick = tick.wrapping_add(1);

        #[cfg(feature = "pal-semihosting")]
        if tick >= 2 * PALETTE.len() {
            cortex_m_semihosting::debug::exit(cortex_m_semihosting::debug::EXIT_SUCCESS);
        }
    }
}

static EXECUTOR: StaticCell<Executor> = StaticCell::new();

#[entry]
fn main() -> ! {
    let strip = ActiveStrip::new();
    let executor = EXECUTOR.init(Executor::new());
    executor.run(|spawner: Spawner| {
        spawner.spawn(strip_animator(strip)).unwrap();
    })
}
