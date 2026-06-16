#![no_std]
#![no_main]

mod bearing;
mod chirp;
mod pal;
mod peers;

use core::fmt::Write;

use cortex_m_rt::entry;
use embassy_executor::{Executor, Spawner};
use panic_semihosting as _;
use static_cell::StaticCell;

use crate::bearing::N_RX;
use crate::chirp::CHIRP_LEN;
use crate::pal::{ActiveStrip, Adc, Rgb, RgbStrip};
use crate::peers::Peer;

const N_PIXELS: usize = 8;
const LIT: Rgb = Rgb::new(0, 255, 0);

const ADC_BLOCK: usize = 1024;
const ADC_CHANNELS: usize = N_RX;
const CORR_LEN: usize = ADC_BLOCK - CHIRP_LEN + 1;

const DETECT_THRESHOLD: f32 = 50.0;

#[embassy_executor::task]
async fn app(mut backend: ActiveStrip) {
    let mut buf = [Rgb::OFF; N_PIXELS];
    let mut adc_buf = [0f32; ADC_BLOCK * ADC_CHANNELS];
    let mut chirp_buf = [0f32; CHIRP_LEN];
    let mut corr_buf = [0f32; CORR_LEN];
    chirp::generate(&mut chirp_buf);

    let mut iter: usize = 0;
    loop {
        backend.read_block(ADC_BLOCK, ADC_CHANNELS, &mut adc_buf);

        let mut lags = [0.0f32; N_RX];
        let mut peak_sum = 0.0f32;
        for ch in 0..N_RX {
            let ch_data = &adc_buf[ch * ADC_BLOCK..(ch + 1) * ADC_BLOCK];
            chirp::cross_correlate(ch_data, &chirp_buf, &mut corr_buf);
            let (peak_idx, peak_val) = chirp::argmax(&corr_buf);
            lags[ch] = chirp::parabolic_interp(&corr_buf, peak_idx);
            peak_sum += peak_val;
        }
        let peak_avg = peak_sum / N_RX as f32;

        let bearing_deg = bearing::estimate_bearing_deg(&lags);
        let range_m = bearing::estimate_range_m(lags[0]);

        let peer = if peak_avg > DETECT_THRESHOLD {
            Some(Peer {
                id: 0,
                bearing_deg,
                range_m,
            })
        } else {
            None
        };
        render(&mut buf, peer);
        backend.write(&buf);

        let _ = writeln!(
            backend,
            "bearing {} {} {}",
            bearing_deg, range_m, peak_avg
        );

        cortex_m::asm::delay(5_000_000);
        iter = iter.wrapping_add(1);

        #[cfg(feature = "pal-semihosting")]
        if iter >= 3 {
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
