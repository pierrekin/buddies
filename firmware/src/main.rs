#![no_std]
#![no_main]

mod bearing;
mod chirp;
mod pal;
mod time_driver;

use core::fmt::Write;

use cortex_m_rt::entry;
use embassy_executor::{Executor, Spawner};
use embassy_time::{Duration, Instant, Ticker};
use panic_semihosting as _;
use static_cell::StaticCell;

use buddies_ui::{self as ui, Detection, Event, OledFrame, TapDebouncer, Ui};

use crate::bearing::N_RX;
use crate::chirp::CHIRP_LEN;
use crate::pal::{ActiveStrip, Adc, Heading, OledOut, RgbStrip, TapInput};

const ADC_BLOCK: usize = 1024;
const ADC_CHANNELS: usize = N_RX;
const CORR_LEN: usize = ADC_BLOCK - CHIRP_LEN + 1;

const DETECT_THRESHOLD: f32 = 50.0;

const LOOP_HZ: u64 = 10;

// Mocked identity and battery the OLED reports; BLE will supply real diver
// names and a fuel-gauge reading later.
const OWN_NAME: &str = "marie";
const BUDDY_NAME: &str = "james";
const BATTERY_PCT: u8 = 82;

/// The OLED backbuffer. `blank()` is const, so this is zero-initialised in .bss
/// with no construction on the stack. Single-core, single-task access only.
static mut OLED_FB: OledFrame = OledFrame::blank();

#[embassy_executor::task]
async fn app(mut backend: ActiveStrip) {
    let mut adc_buf = [0f32; ADC_BLOCK * ADC_CHANNELS];
    let mut chirp_buf = [0f32; CHIRP_LEN];
    let mut corr_buf = [0f32; CORR_LEN];
    chirp::generate(&mut chirp_buf);

    // The whole device is a state machine; this task is just its driver. Each
    // loop it gathers events (taps, the tick, the latest detection) and lets
    // the pure `ui` logic decide the next mode and the frame to show.
    let mut state = Ui::boot();
    let mut taps = TapDebouncer::new();

    let mut ticker = Ticker::every(Duration::from_hz(LOOP_HZ));
    loop {
        let now = Instant::now().as_millis();

        if let Some(burst) = taps.update(backend.poll_taps(), now) {
            state = ui::step(state, Event::Tap(burst), now);
        }
        state = ui::step(state, Event::Tick, now);

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

        let det = Detection {
            present: peak_avg > DETECT_THRESHOLD,
            bearing_deg,
        };
        let frame = ui::render(&state, &det, now);
        backend.write(&frame);

        // Draw into a single static backbuffer. At 256x64 the framebuffer is
        // 32 KB, too large for the task stack, so it lives in .bss and is reused
        // and cleared each frame.
        {
            let hud = ui::Hud {
                own_name: OWN_NAME,
                battery_pct: BATTERY_PCT,
                heading_deg: backend.poll_heading(),
                buddy_name: BUDDY_NAME,
            };
            let fb = unsafe { &mut *core::ptr::addr_of_mut!(OLED_FB) };
            fb.clear();
            let _ = ui::draw_oled(fb, &hud, &det);
            backend.show_oled(fb);
        }

        let _ = writeln!(backend, "bearing {} {} {}", bearing_deg, range_m, peak_avg);

        ticker.next().await;
    }
}

static EXECUTOR: StaticCell<Executor> = StaticCell::new();

#[entry]
fn main() -> ! {
    let cp = cortex_m::Peripherals::take().unwrap();
    time_driver::init(cp.SYST);

    let backend = ActiveStrip::new();
    let executor = EXECUTOR.init(Executor::new());
    executor.run(|spawner: Spawner| {
        spawner.spawn(app(backend)).unwrap();
    })
}
