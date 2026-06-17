//! The LED side panel: an `ROWS` x `COLS` matrix of RGB pixels.

pub const COLS: usize = 16;
pub const ROWS: usize = 3;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Rgb {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Rgb {
    pub const OFF: Rgb = Rgb::new(0, 0, 0);

    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    pub const fn is_off(self) -> bool {
        self.r == 0 && self.g == 0 && self.b == 0
    }
}

/// A full panel frame. Row 0 is the top lightpipe, row 1 the middle, row 2 the
/// bottom; rows are spatial (bearing rides the stem, identity the bottom,
/// tracking the top).
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Frame {
    pub px: [[Rgb; COLS]; ROWS],
}

impl Frame {
    pub const fn blank() -> Self {
        Self {
            px: [[Rgb::OFF; COLS]; ROWS],
        }
    }

    pub fn set(&mut self, row: usize, col: usize, rgb: Rgb) {
        if row < ROWS && col < COLS {
            self.px[row][col] = rgb;
        }
    }

    pub fn row(&self, row: usize) -> &[Rgb; COLS] {
        &self.px[row]
    }
}

impl Default for Frame {
    fn default() -> Self {
        Self::blank()
    }
}
