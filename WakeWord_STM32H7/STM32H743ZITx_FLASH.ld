/*
 * STM32H743ZITx_FLASH.ld – Linker script for STM32H743ZI (2 MB Flash, 1 MB RAM)
 *
 * Memory map (Nucleo-H743ZI2):
 *   FLASH  : 0x08000000  2 MB    (program + read-only data + AI weights)
 *   DTCMRAM: 0x20000000  128 KB  (stack, time-critical data)
 *   RAM_D1 : 0x24000000  512 KB  (AXI SRAM – main data, DMA buffers)
 *   RAM_D2 : 0x30000000  288 KB  (DMA-capable SRAM – I2S DMA buffer)
 *   RAM_D3 : 0x38000000  64 KB   (backup SRAM)
 *   ITCMRAM: 0x00000000  64 KB   (instruction TCM – time-critical code)
 */

ENTRY(Reset_Handler)

_Min_Heap_Size  = 0x8000;   /* 32 KB heap  */
_Min_Stack_Size = 0x4000;   /* 16 KB stack */

MEMORY
{
    FLASH    (rx)  : ORIGIN = 0x08000000, LENGTH = 2048K
    DTCMRAM  (xrw) : ORIGIN = 0x20000000, LENGTH = 128K
    RAM_D1   (xrw) : ORIGIN = 0x24000000, LENGTH = 512K
    RAM_D2   (xrw) : ORIGIN = 0x30000000, LENGTH = 288K
    RAM_D3   (xrw) : ORIGIN = 0x38000000, LENGTH = 64K
    ITCMRAM  (xrw) : ORIGIN = 0x00000000, LENGTH = 64K
}

SECTIONS
{
    /* ── Interrupt vector table ──────────────────────────────────────────── */
    .isr_vector :
    {
        . = ALIGN(4);
        KEEP(*(.isr_vector))
        . = ALIGN(4);
    } >FLASH

    /* ── Program code ────────────────────────────────────────────────────── */
    .text :
    {
        . = ALIGN(4);
        *(.text)
        *(.text*)
        *(.glue_7)
        *(.glue_7t)
        *(.eh_frame)
        KEEP(*(.init))
        KEEP(*(.fini))
        . = ALIGN(4);
        _etext = .;
    } >FLASH

    /* ── Read-only data (constants, AI weight tables) ─────────────────────── */
    .rodata :
    {
        . = ALIGN(4);
        *(.rodata)
        *(.rodata*)
        . = ALIGN(4);
    } >FLASH

    /* ARM exception unwind tables */
    .ARM.extab :
    {
        *(.ARM.extab* .gnu.linkonce.armextab.*)
    } >FLASH

    .ARM :
    {
        __exidx_start = .;
        *(.ARM.exidx*)
        __exidx_end = .;
    } >FLASH

    .preinit_array :
    {
        PROVIDE_HIDDEN(__preinit_array_start = .);
        KEEP(*(.preinit_array*))
        PROVIDE_HIDDEN(__preinit_array_end = .);
    } >FLASH

    .init_array :
    {
        PROVIDE_HIDDEN(__init_array_start = .);
        KEEP(*(SORT(.init_array.*)))
        KEEP(*(.init_array*))
        PROVIDE_HIDDEN(__init_array_end = .);
    } >FLASH

    .fini_array :
    {
        PROVIDE_HIDDEN(__fini_array_start = .);
        KEEP(*(SORT(.fini_array.*)))
        KEEP(*(.fini_array*))
        PROVIDE_HIDDEN(__fini_array_end = .);
    } >FLASH

    /* ── Initialized data (copied from FLASH to RAM_D1 at startup) ─────────── */
    _sidata = LOADADDR(.data);

    .data :
    {
        . = ALIGN(4);
        _sdata = .;
        *(.data)
        *(.data*)
        *(.RamFunc)
        *(.RamFunc*)
        . = ALIGN(4);
        _edata = .;
    } >RAM_D1 AT >FLASH

    /* ── Zero-initialized data (BSS) ──────────────────────────────────────── */
    .bss :
    {
        . = ALIGN(4);
        _sbss = .;
        __bss_start__ = _sbss;
        *(.bss)
        *(.bss*)
        *(COMMON)
        . = ALIGN(4);
        _ebss = .;
        __bss_end__ = _ebss;
    } >RAM_D1

    /* ── I2S DMA buffer in D2 SRAM (accessible by DMA1/DMA2) ─────────────── */
    .dma_buffers (NOLOAD) :
    {
        . = ALIGN(4);
        *(.dma_buffers)
        *(.dma_buffers*)
        . = ALIGN(4);
    } >RAM_D2

    /* ── User heap and stack ──────────────────────────────────────────────── */
    ._user_heap_stack :
    {
        . = ALIGN(8);
        PROVIDE(end = .);
        PROVIDE(_end = .);
        . = . + _Min_Heap_Size;
        . = . + _Min_Stack_Size;
        . = ALIGN(8);
    } >RAM_D1

    /* ── Discard C++ exception metadata (not used in bare-metal C) ────────── */
    /DISCARD/ :
    {
        libc.a(*)
        libm.a(*)
        libgcc.a(*)
    }

    .ARM.attributes 0 : { *(.ARM.attributes) }
}
