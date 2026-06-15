//! Driver for the ARM CMSDK UART exposed by QEMU's MPS2 boards.
//!
//! Layout (MPS2-AN500): UART0 base = 0x4000_4000.
//!   DATA     (0x00) byte FIFO (RW)
//!   STATE    (0x04) bit0 TXBF, bit1 RXBF
//!   CTRL     (0x08) bit0 TX_EN, bit1 RX_EN
//!   BAUDDIV  (0x10) divider (>=16)

const UART0_BASE: usize = 0x4000_4000;
const OFF_DATA: usize = 0x00;
const OFF_STATE: usize = 0x04;
const OFF_CTRL: usize = 0x08;
const OFF_BAUDDIV: usize = 0x10;

const STATE_TXBF: u32 = 1 << 0;
const STATE_RXBF: u32 = 1 << 1;
const CTRL_TX_EN: u32 = 1 << 0;
const CTRL_RX_EN: u32 = 1 << 1;

pub struct Uart0;

impl Uart0 {
    pub fn init() -> Self {
        unsafe {
            write_reg(OFF_BAUDDIV, 16);
            write_reg(OFF_CTRL, CTRL_TX_EN | CTRL_RX_EN);
        }
        Self
    }

    pub fn write_byte(&mut self, byte: u8) {
        unsafe {
            while read_reg(OFF_STATE) & STATE_TXBF != 0 {}
            write_reg(OFF_DATA, byte as u32);
        }
    }

    pub fn read_byte_blocking(&mut self) -> u8 {
        unsafe {
            while read_reg(OFF_STATE) & STATE_RXBF == 0 {}
            read_reg(OFF_DATA) as u8
        }
    }
}

unsafe fn read_reg(offset: usize) -> u32 {
    unsafe { core::ptr::read_volatile((UART0_BASE + offset) as *const u32) }
}

unsafe fn write_reg(offset: usize, value: u32) {
    unsafe { core::ptr::write_volatile((UART0_BASE + offset) as *mut u32, value); }
}
