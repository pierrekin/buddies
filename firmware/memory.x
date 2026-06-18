/* Memory layout for the QEMU mps2-an500 (Cortex-M7) dev target.
 *
 * The MPS2 board maps several MB of SSRAM at 0x20000000, so this is sized for
 * headroom (high-resolution framebuffers fit comfortably), not the 64K it used
 * to guess at. The production STM32H747/H745 has its own map (about 1 MB of
 * internal SRAM plus FMC/SDRAM) and takes a separate linker script on-target.
 */
MEMORY
{
  FLASH : ORIGIN = 0x00000000, LENGTH = 1M
  RAM   : ORIGIN = 0x20000000, LENGTH = 8M
}
