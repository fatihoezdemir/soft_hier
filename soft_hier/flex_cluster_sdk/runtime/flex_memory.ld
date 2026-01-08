
/* Copyright 2020 ETH Zurich and University of Bologna. */
/* Solderpad Hardware License, Version 0.51, see LICENSE for details. */
/* SPDX-License-Identifier: SHL-0.51 */

/*
 * Configurable Linker Script for SoftHier with HBM Edge Placement
 *
 * This linker script is automatically generated from the architecture configuration.
 *
 * HBM Memory Layout:
 *   - WEST:  0xc0000000  (Length: 0x20000000)
 *   - NORTH: 0xe0000000 (Length: 0x1000000)
 *   - EAST:  0xe1000000  (Length: 0x1000000)
 *   - SOUTH: 0xe2000000 (Length: 0x1e000000)
 *
 * Usage - Place data on specific HBM edges:
 *   __attribute__((section(".hbm_west")))  float matrix_a[1024];
 *   __attribute__((section(".hbm_north"))) float matrix_b[1024];
 *   __attribute__((section(".hbm_east")))  float matrix_c[1024];
 *   __attribute__((section(".hbm_south"))) float result[1024];
 *
 * Available symbols for runtime allocation:
 *   __hbm_west_heap_start, __hbm_north_heap_start, __hbm_east_heap_start, __hbm_south_heap_start
 *   __hbm_west_start, __hbm_north_start, __hbm_east_start, __hbm_south_start
 */

OUTPUT_ARCH( "riscv" )
ENTRY(_start)

PHDRS
{
  l3   PT_LOAD;
  hbm  PT_LOAD;
}

/* Memory section should be provided in a separate, platform-specific */
/* file. It should define at least the L1 and L3 memory blocks. */
MEMORY
{
    L1  (rwxa) : ORIGIN = 0, LENGTH = 0x00100000
    L3  (rwxa) : ORIGIN = 0x80000000, LENGTH = 0x80000000


    HBM_WEST (rwxa) : ORIGIN = 0xc0000000, LENGTH = 0x20000000
    HBM_NORTH (rwxa) : ORIGIN = 0xe0000000, LENGTH = 0x1000000
    HBM_EAST (rwxa) : ORIGIN = 0xe1000000, LENGTH = 0x1000000
    HBM_SOUTH (rwxa) : ORIGIN = 0xe2000000, LENGTH = 0x1e000000

}

SECTIONS
{
  .l1 (NOLOAD) :
  {
    __l1_data_start = .;
    *(.l1_prio)
    *(.l1)
    . = ALIGN(4);
    __l1_heap_start = .;
    __l1_end = .;
  } >L1 

  /* Program code goes into L3 */
  .text :
  {
    . = ALIGN(4);
    *(.init)
    *(.text.init)
    *(.text.startup)
    *(.text)
    *(.text*)
    *(.text)
    . = ALIGN(4);
    _etext = .;
  } >L3 :l3

  /* By default, constant data goes into L3, right after code section */
  .rodata :
  {
    . = ALIGN(4);
    *(.rodata)
    *(.rodata*)
    . = ALIGN(4);
  } >L3 :l3

  /* HTIF section for FESVR */
  .htif         : { } >L3 :l3

  /* Thread Local Storage sections */
  .tdata    :
  {
    __tdata_start = .;
    *(.tdata .tdata.* .gnu.linkonce.td.*)
    __tdata_end = .;
  } >L3 :l3
  .tbss :
  {
    __tbss_start = .;
    *(.tbss .tbss.* .gnu.linkonce.tb.*)
    *(.tcommon)
    __tbss_end = .;
    __tbss_end2 = .;
  } >L3 :l3
  /* add a section after .tbss to put the __tbss_end symbol into for
     the LLD linker */
  /* .tbssend : { __tbss_end2 = .; } */

  /* Cluster Local Storage sections */
  .cdata    :
  {
    __cdata_start = .;
    *(.cdata .cdata.*)
    __cdata_end = .;
  } >L3 :l3
  .cbss :
  {
    __cbss_start = .;
    KEEP(*(.cbss .cbss.*))
    __cbss_end = .;
  } >L3 :l3

  /* small data section that can be addressed through the global pointer */
  .sdata          :
  {
    __SDATA_BEGIN__ = .;
    __global_pointer$ = . + 0x7f0;
    *(.srodata.cst16) *(.srodata.cst8) *(.srodata.cst4) *(.srodata.cst2) *(.srodata .srodata.*)
    *(.sdata .sdata.* .gnu.linkonce.s.*)
  } >L3 :l3

  /* Initialized data sections goes into L3 */
  _sidata = LOADADDR(.data);
  .data :
  {
    __DATA_BEGIN__ = .;
    *(.data .data.* .gnu.linkonce.d.*)
    SORT(CONSTRUCTORS)
    . = ALIGN(4);
  } >L3 AT>L3 :l3
  _edata = .; PROVIDE (edata = .);

  /* small bss section */
  . = .;
  __bss_start = .;
  .sbss           :
  {
    *(.dynsbss)
    *(.sbss .sbss.* .gnu.linkonce.sb.*)
    *(.scommon)
  } >L3 :l3

  /* Uninitialized data section */
  .bss            :
  {
   *(.dynbss)
   *(.bss .bss.* .gnu.linkonce.b.*)
   *(COMMON)
   /* Align here to ensure that the .bss section occupies space up to
      _end.  Align after .bss to ensure correct alignment even if the
      .bss section disappears because there are no input sections. */
   . = ALIGN(. != 0 ? 32 / 8 : 1);
  } >L3 :l3
  . = ALIGN(32 / 8);
  . = SEGMENT_START("ldata-segment", .);
  . = ALIGN(32 / 8);
  __BSS_END__ = .;
  __bss_end = .;
  _end = .; PROVIDE (end = .);


  /* Uninitialized data section in L3 */
  .dram :
  {
    *(.dram)
    _edram = .;
  } >L3 :l3

  /* HBM memory sections - for edge-specific data placement */
  /* Users can place data on specific HBM edges using __attribute__((section(".hbm_west"))) */
  .hbm_west :
  {
    *(.hbm_prio)
    . = ALIGN(1024);
    *(.hbm_west)
    *(.hbm_west*)
    . = ALIGN(1024);
    __hbm_heap_start = .;
  } >HBM_WEST :hbm

  .hbm_north :
  {
    *(.hbm_north)
    *(.hbm_north*)
    . = ALIGN(1024);
  } >HBM_NORTH :hbm

  .hbm_east :
  {
    *(.hbm_east)
    *(.hbm_east*)
    . = ALIGN(1024);
  } >HBM_EAST :hbm

  .hbm_south :
  {
    *(.hbm_south)
    *(.hbm_south*)
    . = ALIGN(1024);
  } >HBM_SOUTH :hbm

  /* HBM edge base addresses for runtime use */
  PROVIDE(__hbm_west_base = 0xc0000000);
  PROVIDE(__hbm_north_base = 0xe0000000);
  PROVIDE(__hbm_east_base = 0xe1000000);
  PROVIDE(__hbm_south_base = 0xe2000000);

  /* HBM edge lengths */
  PROVIDE(__hbm_west_length = 0x20000000);
  PROVIDE(__hbm_north_length = 0x1000000);
  PROVIDE(__hbm_east_length = 0x1000000);
  PROVIDE(__hbm_south_length = 0x1e000000);
}
