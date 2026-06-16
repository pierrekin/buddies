//! Peer locator: "where is my buddy?"

pub struct Peer {
    #[allow(dead_code)]
    pub id: u8,
    pub bearing_deg: f32,
    pub range_m: f32,
}

#[allow(dead_code)]
pub trait PeerLocator {
    fn scan(&mut self) -> Option<Peer>;
}
