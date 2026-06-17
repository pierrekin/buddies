//! Tap debouncer: raw per-poll tap counts in, debounced bursts out.
//!
//! Taps arrive a few per poll (from the host mock today, an IMU later). A burst
//! is "done" once the panel has been quiet for `QUIET_MS`; only then do we know
//! whether it was a single, double, or long burst. That quiet window is the
//! irreducible classify latency every gesture pays.

/// Quiet time after the last tap before a burst is reported.
const QUIET_MS: u64 = 350;

#[derive(Copy, Clone, Debug)]
pub struct TapDebouncer {
    count: u8,
    last_tap_ms: u64,
    active: bool,
}

impl TapDebouncer {
    pub const fn new() -> Self {
        Self {
            count: 0,
            last_tap_ms: 0,
            active: false,
        }
    }

    /// Feed the taps seen this poll and the current time. Returns `Some(n)`
    /// once a burst completes (`QUIET_MS` after its last tap).
    pub fn update(&mut self, raw_taps: u8, now: u64) -> Option<u8> {
        if raw_taps > 0 {
            self.count = self.count.saturating_add(raw_taps);
            self.last_tap_ms = now;
            self.active = true;
            return None;
        }
        if self.active && now.saturating_sub(self.last_tap_ms) >= QUIET_MS {
            let burst = self.count;
            self.count = 0;
            self.active = false;
            return Some(burst);
        }
        None
    }
}

impl Default for TapDebouncer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_tap_reports_after_quiet() {
        let mut d = TapDebouncer::new();
        assert_eq!(d.update(1, 0), None); // tap
        assert_eq!(d.update(0, 100), None); // still within the window
        assert_eq!(d.update(0, QUIET_MS), Some(1)); // quiet elapsed
    }

    #[test]
    fn taps_accumulate_into_one_burst() {
        let mut d = TapDebouncer::new();
        assert_eq!(d.update(1, 0), None);
        assert_eq!(d.update(1, 50), None); // second tap extends the window
        assert_eq!(d.update(0, 50 + QUIET_MS - 1), None);
        assert_eq!(d.update(0, 50 + QUIET_MS), Some(2));
    }

    #[test]
    fn multiple_taps_in_one_poll_count() {
        let mut d = TapDebouncer::new();
        assert_eq!(d.update(3, 0), None);
        assert_eq!(d.update(0, QUIET_MS), Some(3));
    }

    #[test]
    fn idle_reports_nothing() {
        let mut d = TapDebouncer::new();
        assert_eq!(d.update(0, 1_000), None);
    }
}
