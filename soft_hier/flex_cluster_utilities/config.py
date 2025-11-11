import re
import ast
import argparse
from pathlib import Path
parser = argparse.ArgumentParser(description="Generate C and S header files from a SoftHier configuration file.")
parser.add_argument("input_file", nargs="?", default="soft_hier/flex_cluster/flex_cluster_arch.py", help="Path to the input Python file")
args = parser.parse_args()
input_file = args.input_file
import ast, math
# Read the input Python file
C_header_file = 'soft_hier/flex_cluster_sdk/runtime/include/flex_cluster_arch.h'
S_header_file = 'soft_hier/flex_cluster_sdk/runtime/include/flex_cluster_arch.inc'
Linker_script_file = 'soft_hier/flex_cluster_sdk/runtime/flex_memory.ld'
# Initialize a dictionary to store the class attributes and their values
attributes = {}
def to_int(x):
    if isinstance(x, int): return x
    s = str(x).strip().rstrip(',')
    # try literal (handles 0x..., decimals)
    try: return int(ast.literal_eval(s)) if s.startswith(('0x','0X')) else int(s, 0)
    except: return int(s, 0)

# Read the input file and extract the class attributes
with open(input_file, 'r') as file:
    lines = file.readlines()
    for line in lines:
        match = re.match(r'\s*self\.(\w+)\s*=\s*(.+)', line)
        if match:
            attr_name = match.group(1)
            attr_value = match.group(2)
            attributes[attr_name] = attr_value

# Write the output C header file
with open(C_header_file, 'w') as file:
    file.write('#ifndef FLEXCLUSTERARCH_H\n')
    file.write('#define FLEXCLUSTERARCH_H\n\n')
    num_core_per_cluster = 0
    
    for attr_name, attr_value in attributes.items():
        # Convert attribute name to uppercase and prefix with 'ARCH_'
        define_name = f'ARCH_{attr_name.upper()}'
        if define_name == 'ARCH_NUM_CORE_PER_CLUSTER':
            num_core_per_cluster = int(attr_value)
            pass
        if define_name == 'ARCH_SPATZ_ATTACED_CORE_LIST':
            core_list = ast.literal_eval(attr_value)
            file.write(f'#define ARCH_SPATZ_ATTACED_CORES {len(core_list)}\n')
            attach_list = []
            sid_list = []
            sid = 0
            for x in range(num_core_per_cluster):
                if x in core_list:
                    attach_list.append(1)
                    sid_list.append(sid)
                    sid = sid + 1
                else:
                    attach_list.append(0)
                    sid_list.append(0)
                    pass
                pass
            attach_list_str = str(attach_list).replace("[", "{").replace("]", "}")
            file.write(f'#define ARCH_SPATZ_ATTACED_CHECK_LIST {attach_list_str}\n')
            sid_list_str = str(sid_list).replace("[", "{").replace("]", "}")
            file.write(f'#define ARCH_SPATZ_ATTACED_SID_LIST {sid_list_str}\n')
            attr_value = attr_value.replace("[", "{").replace("]", "}")
            pass
        file.write(f'#define {define_name} {attr_value}\n')
    
    file.write('\n#endif // FLEXCLUSTERARCH_H\n')

print(f'Header file "{C_header_file}" generated successfully.')

# Write the output S header file
with open(S_header_file, 'w') as file:
    file.write('#ifndef FLEXCLUSTERARCH_H\n')
    file.write('#define FLEXCLUSTERARCH_H\n\n')
    
    for attr_name, attr_value in attributes.items():
        # Convert attribute name to uppercase and prefix with 'ARCH_'
        define_name = f'ARCH_{attr_name.upper()}'
        if define_name == 'ARCH_HBM_CHAN_PLACEMENT' or define_name == 'ARCH_SPATZ_ATTACED_CORE_LIST' or define_name == 'ARCH_HBM_TYPE':
            continue
            pass
        file.write(f'.set {define_name}, {attr_value}\n')
    
    file.write('\n#endif // FLEXCLUSTERARCH_H\n')
print(f'Header file "{S_header_file}" generated successfully.')

nx  = to_int(attributes['num_cluster_x'])
ny  = to_int(attributes['num_cluster_y'])
l1_origin = to_int(attributes.get('cluster_tcdm_base', '0x00000010'))
ali = to_int(attributes.get('hbm_node_aliase', '1'))
HBM_BASE = to_int(attributes['hbm_start_base'])
NODE = to_int(attributes['hbm_node_addr_space'])
hbm_chan_placement = attributes.get('hbm_chan_placement', '[0,0,0,0]')

def cdiv(a,b): return (a + b - 1)//b

# Parse HBM channel placement (format: [west, north, east, south])
try:
    chan_counts = ast.literal_eval(hbm_chan_placement)
    west_chans, north_chans, east_chans, south_chans = chan_counts
except (ValueError, SyntaxError, TypeError):
    west_chans, north_chans, east_chans, south_chans = 0, 0, 0, 0


# From HBM_BASE (0xc0000000) 
AVAILABLE_HBM_SPACE = 0x100000000 - HBM_BASE  # This is 0x40000000 (1GB) for base 0xc0000000

# Count total active edges (those with channels)
active_edges = sum([1 for c in [west_chans, north_chans, east_chans, south_chans] if c > 0])
# Allocate space based on channel counts
total_chans = west_chans + north_chans + east_chans + south_chans

if total_chans > 0:
    # llocation based on channel counts
    # Reserve minimum 16MB for edges without channels to avoid linker issues
    MIN_EDGE_SIZE = 0x01000000  # 16MB minimum

    WEST_BASE = HBM_BASE
    WEST_LEN = max((AVAILABLE_HBM_SPACE * west_chans) // total_chans, MIN_EDGE_SIZE if west_chans == 0 else 0)

    NORTH_BASE = WEST_BASE + (WEST_LEN if WEST_LEN > 0 else MIN_EDGE_SIZE)
    NORTH_LEN = max((AVAILABLE_HBM_SPACE * north_chans) // total_chans, MIN_EDGE_SIZE if north_chans == 0 else 0)

    EAST_BASE = NORTH_BASE + (NORTH_LEN if NORTH_LEN > 0 else MIN_EDGE_SIZE)
    EAST_LEN = max((AVAILABLE_HBM_SPACE * east_chans) // total_chans, MIN_EDGE_SIZE if east_chans == 0 else 0)

    SOUTH_BASE = EAST_BASE + (EAST_LEN if EAST_LEN > 0 else MIN_EDGE_SIZE)
    # For south, use remaining space to avoid overflow, or min size
    if south_chans > 0:
        SOUTH_LEN = min((0x100000000 - SOUTH_BASE), (AVAILABLE_HBM_SPACE * south_chans) // total_chans)
    else:
        SOUTH_LEN = MIN_EDGE_SIZE
else:
    # Equal division if no channel info - divide into 4 parts??
    space_per_edge = AVAILABLE_HBM_SPACE // 4

    WEST_BASE = HBM_BASE
    WEST_LEN = space_per_edge

    NORTH_BASE = WEST_BASE + WEST_LEN
    NORTH_LEN = space_per_edge

    EAST_BASE = NORTH_BASE + NORTH_LEN
    EAST_LEN = space_per_edge

    SOUTH_BASE = EAST_BASE + EAST_LEN
    SOUTH_LEN = space_per_edge
ld=f"""
/* Copyright 2020 ETH Zurich and University of Bologna. */
/* Solderpad Hardware License, Version 0.51, see LICENSE for details. */
/* SPDX-License-Identifier: SHL-0.51 */

/*
 * Configurable Linker Script for SoftHier with HBM Edge Placement
 *
 * This linker script is automatically generated from the architecture configuration.
 *
 * HBM Memory Layout:
 *   - WEST:  {hex(WEST_BASE)}  (Length: {hex(WEST_LEN)})
 *   - NORTH: {hex(NORTH_BASE)} (Length: {hex(NORTH_LEN)})
 *   - EAST:  {hex(EAST_BASE)}  (Length: {hex(EAST_LEN)})
 *   - SOUTH: {hex(SOUTH_BASE)} (Length: {hex(SOUTH_LEN)})
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
{{
  l3   PT_LOAD;
  hbm  PT_LOAD;
}}

/* Memory section should be provided in a separate, platform-specific */
/* file. It should define at least the L1 and L3 memory blocks. */
MEMORY
{{
    L1  (rwxa) : ORIGIN = {l1_origin}, LENGTH = 0x00100000
    L3  (rwxa) : ORIGIN = 0x80000000, LENGTH = 0x80000000


    HBM_WEST (rwxa) : ORIGIN = {hex(WEST_BASE)}, LENGTH = {hex(WEST_LEN)}
    HBM_NORTH (rwxa) : ORIGIN = {hex(NORTH_BASE)}, LENGTH = {hex(NORTH_LEN)}
    HBM_EAST (rwxa) : ORIGIN = {hex(EAST_BASE)}, LENGTH = {hex(EAST_LEN)}
    HBM_SOUTH (rwxa) : ORIGIN = {hex(SOUTH_BASE)}, LENGTH = {hex(SOUTH_LEN)}

}}

SECTIONS
{{
  .l1 (NOLOAD) :
  {{
    __l1_data_start = .;
    *(.l1_prio)
    *(.l1)
    . = ALIGN(4);
    __l1_heap_start = .;
    __l1_end = .;
  }} >L1 

  /* Program code goes into L3 */
  .text :
  {{
    . = ALIGN(4);
    *(.init)
    *(.text.init)
    *(.text.startup)
    *(.text)
    *(.text*)
    *(.text)
    . = ALIGN(4);
    _etext = .;
  }} >L3 :l3

  /* By default, constant data goes into L3, right after code section */
  .rodata :
  {{
    . = ALIGN(4);
    *(.rodata)
    *(.rodata*)
    . = ALIGN(4);
  }} >L3 :l3

  /* HTIF section for FESVR */
  .htif         : {{ }} >L3 :l3

  /* Thread Local Storage sections */
  .tdata    :
  {{
    __tdata_start = .;
    *(.tdata .tdata.* .gnu.linkonce.td.*)
    __tdata_end = .;
  }} >L3 :l3
  .tbss :
  {{
    __tbss_start = .;
    *(.tbss .tbss.* .gnu.linkonce.tb.*)
    *(.tcommon)
    __tbss_end = .;
    __tbss_end2 = .;
  }} >L3 :l3
  /* add a section after .tbss to put the __tbss_end symbol into for
     the LLD linker */
  /* .tbssend : {{ __tbss_end2 = .; }} */

  /* Cluster Local Storage sections */
  .cdata    :
  {{
    __cdata_start = .;
    *(.cdata .cdata.*)
    __cdata_end = .;
  }} >L3 :l3
  .cbss :
  {{
    __cbss_start = .;
    KEEP(*(.cbss .cbss.*))
    __cbss_end = .;
  }} >L3 :l3

  /* small data section that can be addressed through the global pointer */
  .sdata          :
  {{
    __SDATA_BEGIN__ = .;
    __global_pointer$ = . + 0x7f0;
    *(.srodata.cst16) *(.srodata.cst8) *(.srodata.cst4) *(.srodata.cst2) *(.srodata .srodata.*)
    *(.sdata .sdata.* .gnu.linkonce.s.*)
  }} >L3 :l3

  /* Initialized data sections goes into L3 */
  _sidata = LOADADDR(.data);
  .data :
  {{
    __DATA_BEGIN__ = .;
    *(.data .data.* .gnu.linkonce.d.*)
    SORT(CONSTRUCTORS)
    . = ALIGN(4);
  }} >L3 AT>L3 :l3
  _edata = .; PROVIDE (edata = .);

  /* small bss section */
  . = .;
  __bss_start = .;
  .sbss           :
  {{
    *(.dynsbss)
    *(.sbss .sbss.* .gnu.linkonce.sb.*)
    *(.scommon)
  }} >L3 :l3

  /* Uninitialized data section */
  .bss            :
  {{
   *(.dynbss)
   *(.bss .bss.* .gnu.linkonce.b.*)
   *(COMMON)
   /* Align here to ensure that the .bss section occupies space up to
      _end.  Align after .bss to ensure correct alignment even if the
      .bss section disappears because there are no input sections. */
   . = ALIGN(. != 0 ? 32 / 8 : 1);
  }} >L3 :l3
  . = ALIGN(32 / 8);
  . = SEGMENT_START("ldata-segment", .);
  . = ALIGN(32 / 8);
  __BSS_END__ = .;
  __bss_end = .;
  _end = .; PROVIDE (end = .);


  /* Uninitialized data section in L3 */
  .dram :
  {{
    *(.dram)
    _edram = .;
  }} >L3 :l3

  /* HBM memory sections - for edge-specific data placement */
  /* Users can place data on specific HBM edges using __attribute__((section(".hbm_west"))) */
  .hbm_west :
  {{
    *(.hbm_prio)
    . = ALIGN(1024);
    *(.hbm_west)
    *(.hbm_west*)
    . = ALIGN(1024);
    __hbm_heap_start = .;
  }} >HBM_WEST :hbm

  .hbm_north :
  {{
    *(.hbm_north)
    *(.hbm_north*)
    . = ALIGN(1024);
  }} >HBM_NORTH :hbm

  .hbm_east :
  {{
    *(.hbm_east)
    *(.hbm_east*)
    . = ALIGN(1024);
  }} >HBM_EAST :hbm

  .hbm_south :
  {{
    *(.hbm_south)
    *(.hbm_south*)
    . = ALIGN(1024);
  }} >HBM_SOUTH :hbm

  /* HBM edge base addresses for runtime use */
  PROVIDE(__hbm_west_base = {hex(WEST_BASE)});
  PROVIDE(__hbm_north_base = {hex(NORTH_BASE)});
  PROVIDE(__hbm_east_base = {hex(EAST_BASE)});
  PROVIDE(__hbm_south_base = {hex(SOUTH_BASE)});

  /* HBM edge lengths */
  PROVIDE(__hbm_west_length = {hex(WEST_LEN)});
  PROVIDE(__hbm_north_length = {hex(NORTH_LEN)});
  PROVIDE(__hbm_east_length = {hex(EAST_LEN)});
  PROVIDE(__hbm_south_length = {hex(SOUTH_LEN)});
}}
"""

# Write the linker script file
with open(Linker_script_file, 'w') as file:
    file.write(ld)

print(f'Linker script file "{Linker_script_file}" generated successfully.')