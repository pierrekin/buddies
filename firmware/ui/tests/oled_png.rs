//! Golden-image harness: render each OLED compass state and write a PNG.
//!
//! Run with `cargo test -p buddies-ui --target <host>` (from `firmware/`). Files
//! land in `firmware/ui/render/`. This is the host path: no QEMU, no hardware,
//! just the same `render_oled` the firmware ships, encoded to PNG. Diffing these
//! PNGs is how a screen change gets reviewed and regression-tested.

use std::fs;
use std::path::PathBuf;

use buddies_ui::{rgb565_to_rgb888, render_oled, Detection, Hud, OLED_H, OLED_W};

fn render_dir() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("render");
    fs::create_dir_all(&dir).expect("create render dir");
    dir
}

fn save(name: &str, hud: &Hud, det: &Detection) -> usize {
    let frame = render_oled(hud, det);

    let mut rgb = Vec::with_capacity(OLED_W * OLED_H * 3);
    let mut lit = 0;
    for &px in frame.px.iter() {
        if px != 0 {
            lit += 1;
        }
        rgb.extend_from_slice(&rgb565_to_rgb888(px));
    }

    let path = render_dir().join(format!("{name}.png"));
    let file = fs::File::create(&path).expect("create png");
    let mut enc = png::Encoder::new(file, OLED_W as u32, OLED_H as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()
        .expect("png header")
        .write_image_data(&rgb)
        .expect("png data");
    println!("wrote {} ({lit} lit px)", path.display());
    lit
}

fn hud(own: &'static str, battery_pct: u8, heading_deg: f32, buddy: &'static str) -> Hud<'static> {
    Hud { own_name: own, battery_pct, heading_deg, buddy_name: buddy }
}

const NONE: Detection = Detection { present: false, bearing_deg: 0.0 };
fn contact(bearing_deg: f32) -> Detection {
    Detection { present: true, bearing_deg }
}

#[test]
fn render_oled_gallery() {
    // One PNG per situation the single-buddy screen can show.
    let ahead = save("01_buddy_ahead", &hud("marie", 82, 45.0, "james"), &contact(0.0));
    save("02_buddy_port", &hud("marie", 82, 45.0, "james"), &contact(-40.0));
    save("03_buddy_starboard", &hud("marie", 82, 45.0, "james"), &contact(55.0));
    save("04_buddy_off_edge", &hud("marie", 60, 45.0, "james"), &contact(120.0));
    let none = save("05_no_contact", &hud("marie", 17, 312.0, "james"), &NONE);

    // The compass tape plus status bar draw on every screen, so none is blank;
    // a present buddy adds its marker and glow on top.
    assert!(ahead > 0, "buddy-ahead screen should not be blank");
    assert!(none > 0, "no-contact screen still draws the tape and status bar");
    assert!(ahead > none, "a detected buddy lights more pixels than no contact");
}

#[test]
fn battery_fill_scales_with_charge() {
    // A fuller battery lights more pixels in the status-bar icon.
    let full = render_oled(&hud("marie", 100, 0.0, "james"), &NONE);
    let low = render_oled(&hud("marie", 5, 0.0, "james"), &NONE);
    let count = |f: &buddies_ui::OledFrame| f.px.iter().filter(|&&p| p != 0).count();
    assert!(count(&full) > count(&low), "fuller battery should light more pixels");
}
