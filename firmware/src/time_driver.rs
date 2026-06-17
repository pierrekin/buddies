//! `embassy-time` driver backed by the Cortex-M SysTick.
//!
//! SysTick free-runs at `TICK_HZ`. Each interrupt bumps a 64-bit tick count
//! (the monotonic clock embassy reads) and wakes any timers that have come
//! due. The cadence is exact in wall-clock terms because QEMU slaves SysTick
//! to host time; only `SYSTICK_CLK_HZ` has to match the clock the target
//! actually feeds SysTick.

use core::cell::{Cell, RefCell};
use core::task::Waker;

use cortex_m::peripheral::SYST;
use cortex_m::peripheral::syst::SystClkSource;
use cortex_m_rt::exception;
use critical_section::Mutex;
use embassy_time_driver::Driver;
use embassy_time_queue_utils::Queue;

/// Clock feeding SysTick on the current target.
///
/// QEMU `mps2-an500` runs the Cortex-M7 from a fixed 25 MHz FPGA clock
/// (measured: 83,886,080 counts in 3.37 s). Real H7 silicon differs; this
/// constant moves to the hardware PAL when that lands.
const SYSTICK_CLK_HZ: u32 = 25_000_000;

/// Tick rate. Must match the `tick-hz-*` feature on `embassy-time`.
const TICK_HZ: u32 = 1_000;

struct SystickDriver {
    ticks: Mutex<Cell<u64>>,
    queue: Mutex<RefCell<Queue>>,
}

embassy_time_driver::time_driver_impl!(
    static DRIVER: SystickDriver = SystickDriver {
        ticks: Mutex::new(Cell::new(0)),
        queue: Mutex::new(RefCell::new(Queue::new())),
    }
);

impl Driver for SystickDriver {
    fn now(&self) -> u64 {
        critical_section::with(|cs| self.ticks.borrow(cs).get())
    }

    fn schedule_wake(&self, at: u64, waker: &Waker) {
        critical_section::with(|cs| {
            self.queue.borrow(cs).borrow_mut().schedule_wake(at, waker);
        });
    }
}

/// Start the tick. Takes `SYST` so nothing else reprograms it.
pub fn init(mut syst: SYST) {
    let reload = SYSTICK_CLK_HZ / TICK_HZ - 1;
    syst.set_clock_source(SystClkSource::Core);
    syst.set_reload(reload);
    syst.clear_current();
    syst.enable_counter();
    syst.enable_interrupt();
    core::mem::forget(syst);
}

#[exception]
fn SysTick() {
    critical_section::with(|cs| {
        let cell = DRIVER.ticks.borrow(cs);
        let now = cell.get() + 1;
        cell.set(now);
        // Free-running tick: every millisecond, release whatever's due.
        DRIVER.queue.borrow(cs).borrow_mut().next_expiration(now);
    });
}
