//! Operating-mode state machine.

use crate::frame::{COLS, Frame, ROWS, Rgb};

const GREEN: Rgb = Rgb::new(0, 220, 0);
const WHITE: Rgb = Rgb::new(235, 235, 235);
const BLUE: Rgb = Rgb::new(70, 150, 255);
const CENTER: usize = COLS / 2;

const MAX_DIVER: u8 = 5;

/// How long a one-tap reveal stays up before snapping back to plain tracking.
const REVEAL_MS: u64 = 3_000;
/// Idle after the last tap before a tracking change commits.
const TRACK_COMMIT_MS: u64 = 10_000;
/// Idle before an identity change commits.
const IDENTITY_COMMIT_MS: u64 = 10_000;
/// Blink half-period for "being set" screens.
const BLINK_MS: u64 = 450;
/// A tap burst of at least this many opens identity (the rare, guarded path).
const IDENTITY_BURST: u8 = 4;

/// Committed device state plus the current interaction mode.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Ui {
    pub identity: u8,
    pub target: u8,
    pub mode: Mode,
}

/// Each variant carries exactly the data that mode needs. A `commit_at`/`until`
/// deadline is a timestamp in the same `u64` ms clock the caller passes to
/// `step`.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Mode {
    Boot,
    SettingIdentity { candidate: u8, commit_at: u64 },
    Tracking,
    RevealTracking { until: u64 },
    SettingTracking { candidate: u8, commit_at: u64 },
}

/// Inputs to the state machine. `step` also takes `now` so deadlines are exact.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Event {
    /// Fires every loop; drives deadline-based transitions.
    Tick,
    /// A debounced burst of `n` taps.
    Tap(u8),
}

/// Latest acoustic detection, used only for rendering the live bearing.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Detection {
    pub present: bool,
    pub bearing_deg: f32,
}

impl Ui {
    pub const fn boot() -> Self {
        Self {
            identity: 1,
            target: 2,
            mode: Mode::Boot,
        }
    }
}

/// Pure transition. No I/O, exhaustive over `(mode, event)`.
pub fn step(ui: Ui, event: Event, now: u64) -> Ui {
    match event {
        Event::Tap(n) => on_tap(ui, n, now),
        Event::Tick => on_tick(ui, now),
    }
}

fn on_tap(ui: Ui, n: u8, now: u64) -> Ui {
    let mode = match ui.mode {
        // Boot's first interaction just drops into identity setting.
        Mode::Boot => set_identity(ui.identity, now),

        // From a resting view, a tap burst chooses an action by size.
        Mode::Tracking | Mode::RevealTracking { .. } => match n {
            1 => Mode::RevealTracking { until: now + REVEAL_MS },
            _ if n >= IDENTITY_BURST => set_identity(ui.identity, now),
            _ => set_tracking(ui.target, now),
        },

        // Inside a set mode any tap just steps the candidate and re-arms the
        // commit timer; the burst size no longer matters.
        Mode::SettingTracking { candidate, .. } => {
            set_tracking(cycle_skip(candidate, ui.identity), now)
        }
        Mode::SettingIdentity { candidate, .. } => set_identity(cycle(candidate), now),
    };
    Ui { mode, ..ui }
}

fn on_tick(ui: Ui, now: u64) -> Ui {
    match ui.mode {
        Mode::Boot => Ui {
            mode: set_identity(ui.identity, now),
            ..ui
        },
        Mode::RevealTracking { until } if now >= until => Ui {
            mode: Mode::Tracking,
            ..ui
        },
        Mode::SettingTracking { candidate, commit_at } if now >= commit_at => Ui {
            target: candidate,
            mode: Mode::Tracking,
            ..ui
        },
        Mode::SettingIdentity { candidate, commit_at } if now >= commit_at => Ui {
            identity: candidate,
            mode: Mode::Tracking,
            ..ui
        },
        _ => ui,
    }
}

fn set_identity(candidate: u8, now: u64) -> Mode {
    Mode::SettingIdentity { candidate, commit_at: now + IDENTITY_COMMIT_MS }
}

fn set_tracking(candidate: u8, now: u64) -> Mode {
    Mode::SettingTracking { candidate, commit_at: now + TRACK_COMMIT_MS }
}

fn cycle(n: u8) -> u8 {
    if n >= MAX_DIVER { 1 } else { n + 1 }
}

/// Cycle, skipping `skip` (you never track yourself).
fn cycle_skip(n: u8, skip: u8) -> u8 {
    let next = cycle(n);
    if next == skip { cycle(next) } else { next }
}

/// Render the current state to a frame. `now` only sets the blink phase, so
/// the output is a pure function of its inputs.
pub fn render(ui: &Ui, det: &Detection, now: u64) -> Frame {
    let mut f = Frame::blank();
    let blink_on = (now / BLINK_MS) % 2 == 0;

    match ui.mode {
        Mode::Boot => {}
        Mode::Tracking => paint_bearing(&mut f, det, 0, ROWS),
        Mode::RevealTracking { .. } => {
            // Bearing drops to the lower rows so tracking pips can ride the top.
            paint_bearing(&mut f, det, 1, ROWS);
            paint_pips(&mut f, 0, ui.target, BLUE);
        }
        Mode::SettingTracking { candidate, .. } => {
            if blink_on {
                paint_pips(&mut f, 0, candidate, BLUE);
                f.set(1, CENTER, BLUE);
                f.set(2, CENTER, BLUE);
            }
        }
        Mode::SettingIdentity { candidate, .. } => {
            if blink_on {
                paint_pips(&mut f, 2, candidate, WHITE);
                f.set(0, CENTER, WHITE);
                f.set(1, CENTER, WHITE);
            }
        }
    }
    f
}

fn paint_bearing(f: &mut Frame, det: &Detection, row_lo: usize, row_hi: usize) {
    if !det.present {
        return;
    }
    let col = bearing_to_col(det.bearing_deg);
    for row in row_lo..row_hi {
        f.set(row, col, GREEN);
    }
}

fn paint_pips(f: &mut Frame, row: usize, n: u8, color: Rgb) {
    let n = n as usize;
    let spacing: isize = 2;
    let start = CENTER as isize - (n as isize - 1) * spacing / 2;
    for i in 0..n {
        let col = start + i as isize * spacing;
        if (0..COLS as isize).contains(&col) {
            f.set(row, col as usize, color);
        }
    }
}

/// Map a body-relative bearing (deg, 0 = ahead) onto a column. The frontal
/// ±90° arc spreads across the inner columns; the two ends flag a peer beyond
/// it. Mirrors the firmware's old `bearing_to_led`.
fn bearing_to_col(bearing_deg: f32) -> usize {
    let b = normalize_180(bearing_deg);
    if b < -90.0 {
        return 0;
    }
    if b > 90.0 {
        return COLS - 1;
    }
    let span = (COLS - 3) as f32;
    let pos = 1.0 + ((b + 90.0) / 180.0) * span;
    (pos + 0.5) as usize
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

#[cfg(test)]
mod tests {
    use super::*;

    const NONE: Detection = Detection { present: false, bearing_deg: 0.0 };

    #[test]
    fn boot_tick_enters_identity_setting() {
        let ui = step(Ui::boot(), Event::Tick, 0);
        assert_eq!(ui.mode, Mode::SettingIdentity { candidate: 1, commit_at: IDENTITY_COMMIT_MS });
    }

    #[test]
    fn identity_taps_cycle_then_commit() {
        let mut ui = step(Ui::boot(), Event::Tick, 0); // SettingIdentity candidate 1
        ui = step(ui, Event::Tap(2), 100); // any tap -> +1
        assert!(matches!(ui.mode, Mode::SettingIdentity { candidate: 2, .. }));
        ui = step(ui, Event::Tap(1), 200);
        assert!(matches!(ui.mode, Mode::SettingIdentity { candidate: 3, .. }));
        // Idle past the commit window commits identity and returns to tracking.
        ui = step(ui, Event::Tick, 200 + IDENTITY_COMMIT_MS);
        assert_eq!(ui.identity, 3);
        assert_eq!(ui.mode, Mode::Tracking);
    }

    #[test]
    fn identity_cycles_wrap() {
        let mut ui = Ui { identity: 5, target: 2, mode: Mode::SettingIdentity { candidate: 5, commit_at: 0 } };
        ui = step(ui, Event::Tap(1), 10);
        assert!(matches!(ui.mode, Mode::SettingIdentity { candidate: 1, .. }));
    }

    #[test]
    fn one_tap_reveals_then_expires() {
        let base = Ui { identity: 1, target: 2, mode: Mode::Tracking };
        let ui = step(base, Event::Tap(1), 1_000);
        assert_eq!(ui.mode, Mode::RevealTracking { until: 1_000 + REVEAL_MS });
        // Still up before the deadline.
        let ui2 = step(ui, Event::Tick, 1_000 + REVEAL_MS - 1);
        assert!(matches!(ui2.mode, Mode::RevealTracking { .. }));
        // Snaps back after it.
        let ui3 = step(ui, Event::Tick, 1_000 + REVEAL_MS);
        assert_eq!(ui3.mode, Mode::Tracking);
    }

    #[test]
    fn double_tap_sets_tracking_and_skips_self() {
        let base = Ui { identity: 3, target: 2, mode: Mode::Tracking };
        let mut ui = step(base, Event::Tap(2), 0);
        assert!(matches!(ui.mode, Mode::SettingTracking { candidate: 2, .. }));
        // Stepping lands on 3 == identity, so it skips to 4.
        ui = step(ui, Event::Tap(2), 100);
        assert!(matches!(ui.mode, Mode::SettingTracking { candidate: 4, .. }));
        ui = step(ui, Event::Tick, 100 + TRACK_COMMIT_MS);
        assert_eq!(ui.target, 4);
        assert_eq!(ui.mode, Mode::Tracking);
    }

    #[test]
    fn big_burst_opens_identity_from_tracking() {
        let base = Ui { identity: 1, target: 2, mode: Mode::Tracking };
        let ui = step(base, Event::Tap(IDENTITY_BURST), 0);
        assert!(matches!(ui.mode, Mode::SettingIdentity { candidate: 1, .. }));
    }

    #[test]
    fn tracking_renders_green_at_bearing() {
        let ui = Ui { identity: 1, target: 2, mode: Mode::Tracking };
        let det = Detection { present: true, bearing_deg: 0.0 };
        let f = render(&ui, &det, 0);
        // Dead ahead -> a full-height stem at one column.
        let stem: usize = (0..COLS).filter(|&c| f.px[0][c] == GREEN).count();
        assert_eq!(stem, 1);
        assert_eq!(f.px[0], f.px[2]); // all three rows lit identically
    }

    #[test]
    fn setting_blink_blanks_on_off_phase() {
        let ui = Ui { identity: 1, target: 2, mode: Mode::SettingIdentity { candidate: 2, commit_at: 9_999 } };
        let on = render(&ui, &NONE, 0); // blink on
        let off = render(&ui, &NONE, BLINK_MS); // blink off
        assert_ne!(on, Frame::blank());
        assert_eq!(off, Frame::blank());
    }
}
