//! The OLED screen: the single-buddy dive compass, rendered with embedded-graphics.
//!
//! This is the device's primary display. The drawing is generic over any
//! `DrawTarget<Color = Rgb565>`, so the exact same code paints:
//!   - the real panel on-device (a controller driver is a `DrawTarget`),
//!   - the host simulator/PNG path (`OledFrame` below is a `DrawTarget`),
//!   - the Qt harness (the PAL streams `OledFrame` over the link).
//!
//! The layout: a status bar (own name + battery), an absolute compass tape that
//! scrolls under a fixed centre reference, and one buddy marker placed at its
//! bearing and coloured by which side it sits on (port = red, centred = white,
//! starboard = green) with a matching edge glow.
//!
//! The geometry is authored once in a 256x64 "design space" and mapped onto the
//! real panel by `Layout`: a uniform scale plus centring. A wider panel (the
//! RM67162's 536x240) keeps the same proportions with true-black letterbox bars,
//! and fonts step up a tier so they stay crisp rather than upscaled-chunky.
//!
//! `OledFrame` owns the pixels so the firmware can serialise them and the host
//! can encode a PNG, both byte-for-byte identical to what the panel would show.

use core::fmt::Write as _;

use embedded_graphics::{
    draw_target::DrawTarget,
    geometry::{OriginDimensions, Size},
    mono_font::{
        ascii::{FONT_4X6, FONT_6X9, FONT_6X10, FONT_6X13_BOLD, FONT_9X15, FONT_10X20},
        MonoTextStyle,
    },
    pixelcolor::{IntoStorage, Rgb565},
    prelude::*,
    primitives::{Line, PrimitiveStyle, Rectangle, Triangle},
    text::{Alignment, Text},
    Pixel,
};
use heapless::String;

use crate::mode::normalize_180;
use crate::Detection;

/// Panel size, selected by feature. The default is the bring-up panel; enable
/// `panel-rm67162` for the 536x240 QSPI AMOLED dev module.
#[cfg(feature = "panel-rm67162")]
pub const OLED_W: usize = 536;
#[cfg(feature = "panel-rm67162")]
pub const OLED_H: usize = 240;
#[cfg(not(feature = "panel-rm67162"))]
pub const OLED_W: usize = 256;
#[cfg(not(feature = "panel-rm67162"))]
pub const OLED_H: usize = 64;

// The fixed canvas the layout is authored in; every coordinate below is in these
// units and mapped to the panel through `Layout`.
const DESIGN_W: f32 = 256.0;
const DESIGN_H: f32 = 64.0;
const MARGIN_X: f32 = 12.0;
const TAPE_Y: f32 = 44.0;
const VIS_HALF_DEG: f32 = 75.0;
/// A buddy within this many degrees of dead-ahead reads as "centred".
const CENTER_BAND_DEG: f32 = 7.0;
/// Pixels per degree along the tape, in design space.
const PX_PER_DEG: f32 = (DESIGN_W / 2.0 - MARGIN_X) / VIS_HALF_DEG;

// Palette, lifted from the design's hex values and packed to RGB565.
const C_TEXT: Rgb565 = rgb(0x9f, 0xb0, 0xbf); // names, battery
const C_TAPE: Rgb565 = rgb(0x6a, 0x7d, 0x90); // compass baseline
const C_TICK_MAJOR: Rgb565 = rgb(0x5a, 0x6b, 0x7e);
const C_TICK_MINOR: Rgb565 = rgb(0x3a, 0x46, 0x54);
const C_REF: Rgb565 = rgb(0x8f, 0xa3, 0xb6); // centre reference triangle
const C_DIM: Rgb565 = rgb(0x4a, 0x52, 0x5a); // "no contact"

// Buddy colour by side: a solid marker colour plus a lighter accent for its name.
const C_PORT: Rgb565 = rgb(0xff, 0x5a, 0x52);
const C_PORT_HI: Rgb565 = rgb(0xff, 0x9c, 0x96);
const C_CENTER: Rgb565 = rgb(0xe8, 0xeb, 0xef);
const C_STARBOARD: Rgb565 = rgb(0x46, 0xe0, 0x8a);
const C_STARBOARD_HI: Rgb565 = rgb(0x8a, 0xf0, 0xbb);

const fn rgb(r: u8, g: u8, b: u8) -> Rgb565 {
    Rgb565::new(r >> 3, g >> 2, b >> 3)
}

/// Maps design-space coordinates onto the real panel: a uniform scale that keeps
/// the 256x64 proportions, centred so a taller/wider panel letterboxes in black.
struct Layout {
    scale: f32,
    ox: f32,
    oy: f32,
    /// Step fonts up a tier once the panel is large enough to warrant it.
    big: bool,
}

impl Layout {
    fn new() -> Self {
        let scale = (OLED_W as f32 / DESIGN_W).min(OLED_H as f32 / DESIGN_H);
        Self {
            scale,
            ox: (OLED_W as f32 - DESIGN_W * scale) / 2.0,
            oy: (OLED_H as f32 - DESIGN_H * scale) / 2.0,
            big: scale >= 1.6,
        }
    }

    fn x(&self, dx: f32) -> i32 {
        (self.ox + dx * self.scale + 0.5) as i32
    }
    fn y(&self, dy: f32) -> i32 {
        (self.oy + dy * self.scale + 0.5) as i32
    }
    fn pt(&self, dx: f32, dy: f32) -> Point {
        Point::new(self.x(dx), self.y(dy))
    }
    /// Scale a design-space length to device pixels, never below 1.
    fn len(&self, d: f32) -> u32 {
        ((d * self.scale + 0.5) as i32).max(1) as u32
    }
    fn stroke(&self) -> u32 {
        self.len(1.0)
    }
    /// Character advance of the name font at the current tier (for centring).
    fn name_char_w(&self) -> i32 {
        if self.big { 10 } else { 6 }
    }
}

/// Everything the screen needs that does not come from the acoustic detection:
/// the diver's own identity/battery and current heading. In the single-buddy
/// build these are mocked by the firmware; BLE will supply them later.
pub struct Hud<'a> {
    pub own_name: &'a str,
    pub battery_pct: u8,
    pub heading_deg: f32,
    pub buddy_name: &'a str,
}

/// An owned RGB565 framebuffer that is itself an embedded-graphics target.
#[derive(Clone)]
pub struct OledFrame {
    /// Row-major RGB565, `px[y * OLED_W + x]`.
    pub px: [u16; OLED_W * OLED_H],
}

impl OledFrame {
    pub const fn blank() -> Self {
        Self {
            px: [0; OLED_W * OLED_H],
        }
    }

    pub fn pixel(&self, x: usize, y: usize) -> u16 {
        self.px[y * OLED_W + x]
    }

    /// Zero every pixel in place, so a single buffer can be reused across frames
    /// without building a fresh frame on the stack.
    pub fn clear(&mut self) {
        self.px.fill(0);
    }
}

impl Default for OledFrame {
    fn default() -> Self {
        Self::blank()
    }
}

impl OriginDimensions for OledFrame {
    fn size(&self) -> Size {
        Size::new(OLED_W as u32, OLED_H as u32)
    }
}

impl DrawTarget for OledFrame {
    type Color = Rgb565;
    type Error = core::convert::Infallible;

    fn draw_iter<I>(&mut self, pixels: I) -> Result<(), Self::Error>
    where
        I: IntoIterator<Item = Pixel<Self::Color>>,
    {
        for Pixel(p, color) in pixels {
            if p.x >= 0 && p.y >= 0 && (p.x as usize) < OLED_W && (p.y as usize) < OLED_H {
                self.px[p.y as usize * OLED_W + p.x as usize] = color.into_storage();
            }
        }
        Ok(())
    }
}

/// Expand a packed RGB565 value to 8-bit-per-channel RGB, the form PNG encoders
/// and Qt's `Format_RGB888` want. Bit-replication keeps full-scale full-scale.
pub const fn rgb565_to_rgb888(v: u16) -> [u8; 3] {
    let r5 = ((v >> 11) & 0x1f) as u8;
    let g6 = ((v >> 5) & 0x3f) as u8;
    let b5 = (v & 0x1f) as u8;
    [
        (r5 << 3) | (r5 >> 2),
        (g6 << 2) | (g6 >> 4),
        (b5 << 3) | (b5 >> 2),
    ]
}

/// Render the compass HUD into any RGB565 draw target. Pure function of its
/// inputs: same `(hud, det)` always paints the same frame.
pub fn draw_oled<D>(target: &mut D, hud: &Hud, det: &Detection) -> Result<(), D::Error>
where
    D: DrawTarget<Color = Rgb565>,
{
    let l = Layout::new();
    let band = det.present.then(|| side_band(det.bearing_deg));

    // Glow first, so the tape and marker sit on top of it.
    if let Some(b) = band {
        draw_side_glow(target, &l, b)?;
    }
    draw_compass_tape(target, &l, hud.heading_deg)?;
    draw_center_reference(target, &l)?;

    match band {
        Some(b) => draw_buddy(target, &l, det.bearing_deg, hud.buddy_name, b)?,
        None => draw_label(target, &l, "no contact", l.pt(DESIGN_W / 2.0, TAPE_Y - 20.0), C_DIM)?,
    }

    // Status bar last so the name and battery always read clearly over the glow.
    draw_status_bar(target, &l, hud)?;
    Ok(())
}

/// Convenience for the host (tests, PNG export) and the PAL: render into an
/// owned framebuffer. Drawing into `OledFrame` is infallible.
pub fn render_oled(hud: &Hud, det: &Detection) -> OledFrame {
    let mut f = OledFrame::blank();
    let _ = draw_oled(&mut f, hud, det);
    f
}

/// Which side of dead-ahead the buddy sits on, with the marker/accent colours.
#[derive(Copy, Clone)]
struct Band {
    marker: Rgb565,
    accent: Rgb565,
    /// +1 = starboard (right), -1 = port (left), 0 = centred.
    side: i8,
}

fn side_band(bearing_deg: f32) -> Band {
    let b = normalize_180(bearing_deg);
    if b.abs() <= CENTER_BAND_DEG {
        Band { marker: C_CENTER, accent: C_CENTER, side: 0 }
    } else if b < 0.0 {
        Band { marker: C_PORT, accent: C_PORT_HI, side: -1 }
    } else {
        Band { marker: C_STARBOARD, accent: C_STARBOARD_HI, side: 1 }
    }
}

fn draw_status_bar<D>(target: &mut D, l: &Layout, hud: &Hud) -> Result<(), D::Error>
where
    D: DrawTarget<Color = Rgb565>,
{
    draw_name(target, l, hud.own_name, l.pt(MARGIN_X, 12.0), Alignment::Left, C_TEXT)?;

    // Battery icon at the far right: a 24x12 body, a proportional fill, and a nub.
    let pct = hud.battery_pct.min(100) as u32;
    let body_x = DESIGN_W - MARGIN_X - 2.0 - 24.0;
    Rectangle::new(l.pt(body_x, 2.0), Size::new(l.len(24.0), l.len(12.0)))
        .into_styled(PrimitiveStyle::with_stroke(C_TEXT, l.stroke()))
        .draw(target)?;
    let fill_w = 20 * pct / 100;
    if fill_w > 0 {
        Rectangle::new(l.pt(body_x + 2.0, 4.0), Size::new(l.len(fill_w as f32), l.len(8.0)))
            .into_styled(PrimitiveStyle::with_fill(C_TEXT))
            .draw(target)?;
    }
    Rectangle::new(l.pt(DESIGN_W - MARGIN_X, 5.0), Size::new(l.len(2.0), l.len(6.0)))
        .into_styled(PrimitiveStyle::with_fill(C_TEXT))
        .draw(target)?;

    let mut pct_label: String<5> = String::new();
    let _ = write!(pct_label, "{}%", pct);
    draw_pct(target, l, &pct_label, l.pt(body_x - 4.0, 11.0), C_TEXT)?;
    Ok(())
}

fn draw_compass_tape<D>(target: &mut D, l: &Layout, heading_deg: f32) -> Result<(), D::Error>
where
    D: DrawTarget<Color = Rgb565>,
{
    // Baseline bar across the inner width.
    Rectangle::new(
        l.pt(MARGIN_X, TAPE_Y - 1.0),
        Size::new(l.len(DESIGN_W - 2.0 * MARGIN_X), l.len(2.0)),
    )
    .into_styled(PrimitiveStyle::with_fill(C_TAPE))
    .draw(target)?;

    // Ticks sit at absolute multiples of 15deg; majors (multiples of 30deg) carry
    // a label. Walking absolute degrees keeps the cardinals fixed to the world as
    // the diver's heading scrolls the tape past the centre reference.
    let lo = heading_deg - VIS_HALF_DEG;
    let hi = heading_deg + VIS_HALF_DEG;
    let mut k = (lo / 15.0) as i32;
    while (k as f32) * 15.0 < lo {
        k += 1;
    }
    let mut a = k * 15;
    while (a as f32) <= hi {
        let dx = DESIGN_W / 2.0 + (a as f32 - heading_deg) * PX_PER_DEG;
        if dx >= MARGIN_X && dx <= DESIGN_W - MARGIN_X {
            if a.rem_euclid(30) == 0 {
                Line::new(l.pt(dx, TAPE_Y - 9.0), l.pt(dx, TAPE_Y - 2.0))
                    .into_styled(PrimitiveStyle::with_stroke(C_TICK_MAJOR, l.stroke()))
                    .draw(target)?;
                let mut label: String<4> = String::new();
                let _ = write!(label, "{}", cardinal(a));
                draw_label(target, l, &label, l.pt(dx, TAPE_Y + 10.0), C_TICK_MAJOR)?;
            } else {
                Line::new(l.pt(dx, TAPE_Y - 5.0), l.pt(dx, TAPE_Y - 2.0))
                    .into_styled(PrimitiveStyle::with_stroke(C_TICK_MINOR, l.stroke()))
                    .draw(target)?;
            }
        }
        a += 15;
    }
    Ok(())
}

/// A downward triangle hanging over the tape centre: the diver's own heading.
fn draw_center_reference<D>(target: &mut D, l: &Layout) -> Result<(), D::Error>
where
    D: DrawTarget<Color = Rgb565>,
{
    let cx = DESIGN_W / 2.0;
    Triangle::new(
        l.pt(cx - 5.0, TAPE_Y - 18.0),
        l.pt(cx + 5.0, TAPE_Y - 18.0),
        l.pt(cx, TAPE_Y - 11.0),
    )
    .into_styled(PrimitiveStyle::with_fill(C_REF))
    .draw(target)?;
    Ok(())
}

fn draw_buddy<D>(
    target: &mut D,
    l: &Layout,
    bearing_deg: f32,
    name: &str,
    band: Band,
) -> Result<(), D::Error>
where
    D: DrawTarget<Color = Rgb565>,
{
    // Place the marker by bearing, clamped to the visible window so an off-screen
    // buddy pins to the edge it lies beyond rather than vanishing.
    let b = normalize_180(bearing_deg).clamp(-VIS_HALF_DEG, VIS_HALF_DEG);
    let dx = (DESIGN_W / 2.0 + b * PX_PER_DEG).clamp(MARGIN_X + 2.0, DESIGN_W - MARGIN_X - 2.0);

    // A vertical bar straddling the tape baseline.
    Rectangle::new(l.pt(dx - 2.0, TAPE_Y - 12.0), Size::new(l.len(4.0), l.len(20.0)))
        .into_styled(PrimitiveStyle::with_fill(band.marker))
        .draw(target)?;

    // Name above the marker, centred on it but kept fully on-screen.
    let half = (name.len() as i32 * l.name_char_w()) / 2;
    let nx = l.x(dx).clamp(l.x(MARGIN_X) + half, l.x(DESIGN_W - MARGIN_X) - half);
    draw_name(target, l, name, Point::new(nx, l.y(TAPE_Y - 18.0)), Alignment::Center, band.accent)?;
    Ok(())
}

/// A soft colour wash on the edge the buddy lies toward. Drawn in device space as
/// columns whose brightness ramps down inward (no true alpha blend on the panel).
fn draw_side_glow<D>(target: &mut D, l: &Layout, band: Band) -> Result<(), D::Error>
where
    D: DrawTarget<Color = Rgb565>,
{
    if band.side == 0 {
        return Ok(());
    }
    let glow_w = l.len(44.0) as i32;
    let (r0, g0, b0) = base_rgb(band.side);
    let y0 = l.y(18.0);
    let y1 = OLED_H as i32;
    let left = l.x(MARGIN_X);
    let right = l.x(DESIGN_W - MARGIN_X);
    for i in 0..glow_w {
        // Brightest at the edge, fading to nothing inward; capped low so the wash
        // stays a hint rather than a block.
        let f = 0.30 * (1.0 - i as f32 / glow_w as f32);
        let col = rgb((r0 as f32 * f) as u8, (g0 as f32 * f) as u8, (b0 as f32 * f) as u8);
        let x = if band.side > 0 { right - i } else { left + i };
        Line::new(Point::new(x, y0), Point::new(x, y1))
            .into_styled(PrimitiveStyle::with_stroke(col, 1))
            .draw(target)?;
    }
    Ok(())
}

/// 8-bit base colour for the side glow (starboard green / port red).
fn base_rgb(side: i8) -> (u8, u8, u8) {
    if side > 0 {
        (0x46, 0xe0, 0x8a)
    } else {
        (0xff, 0x5a, 0x52)
    }
}

// Text helpers pick a font tier from the layout so glyphs stay crisp at any panel
// size. The role-specific fonts mirror the design's relative type sizes.

fn draw_name<D>(target: &mut D, l: &Layout, s: &str, pt: Point, align: Alignment, color: Rgb565) -> Result<(), D::Error>
where
    D: DrawTarget<Color = Rgb565>,
{
    if l.big {
        Text::with_alignment(s, pt, MonoTextStyle::new(&FONT_10X20, color), align).draw(target)?;
    } else {
        Text::with_alignment(s, pt, MonoTextStyle::new(&FONT_6X13_BOLD, color), align).draw(target)?;
    }
    Ok(())
}

fn draw_pct<D>(target: &mut D, l: &Layout, s: &str, pt: Point, color: Rgb565) -> Result<(), D::Error>
where
    D: DrawTarget<Color = Rgb565>,
{
    if l.big {
        Text::with_alignment(s, pt, MonoTextStyle::new(&FONT_9X15, color), Alignment::Right).draw(target)?;
    } else {
        Text::with_alignment(s, pt, MonoTextStyle::new(&FONT_6X9, color), Alignment::Right).draw(target)?;
    }
    Ok(())
}

fn draw_label<D>(target: &mut D, l: &Layout, s: &str, pt: Point, color: Rgb565) -> Result<(), D::Error>
where
    D: DrawTarget<Color = Rgb565>,
{
    if l.big {
        Text::with_alignment(s, pt, MonoTextStyle::new(&FONT_6X10, color), Alignment::Center).draw(target)?;
    } else {
        Text::with_alignment(s, pt, MonoTextStyle::new(&FONT_4X6, color), Alignment::Center).draw(target)?;
    }
    Ok(())
}

/// Cardinal letter for the quadrant points, else a zero-padded heading in degrees.
fn cardinal(deg: i32) -> Cardinal {
    match deg.rem_euclid(360) {
        0 => Cardinal::Letter("N"),
        90 => Cardinal::Letter("E"),
        180 => Cardinal::Letter("S"),
        270 => Cardinal::Letter("W"),
        d => Cardinal::Degrees(d as u16),
    }
}

enum Cardinal {
    Letter(&'static str),
    Degrees(u16),
}

impl core::fmt::Display for Cardinal {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Cardinal::Letter(s) => f.write_str(s),
            Cardinal::Degrees(d) => write!(f, "{:03}", d),
        }
    }
}
