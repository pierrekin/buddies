//! Device UI logic: the operating-mode state machine and LED rendering.
//!
//! Pure, dependency-free, and host-testable. Time is a plain monotonic `u64`
//! millisecond count supplied by the caller, so this crate never touches
//! embassy-time or any hardware, and `cargo test` runs it on the host.
//!
//! Shape of the design:
//!   - `Ui` holds the committed state (identity, target) plus the current
//!     interaction `Mode`. Each `Mode` variant carries exactly its own data,
//!     so illegal states (e.g. tracking with no target) cannot be built.
//!   - `step(ui, event, now)` is a pure transition: one exhaustive match, no
//!     I/O. `render(ui, detection, now)` turns state into a `Frame`.
//!   - `TapDebouncer` turns raw per-poll tap counts into `Event::Tap(n)`
//!     bursts. It is the only place that knows about the classify window.
#![cfg_attr(not(test), no_std)]

mod frame;
mod mode;
mod tap;

pub use frame::{COLS, Frame, ROWS, Rgb};
pub use mode::{Detection, Event, Mode, Ui, render, step};
pub use tap::TapDebouncer;
