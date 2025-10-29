#!/usr/bin/env python3
"""
Parse performance counter output from make dqsw
Correlates cycle counts with operation labels
"""

import re
import subprocess
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

def run_benchmark():
    """Run make dqsw and capture output"""
    result = subprocess.run(['make', 'dqsw'],
                          capture_output=True,
                          text=True,
                          cwd='/home/fo/Desktop/master/soft_hier/examples/SoftHier/assembled/DQ')
    return result.stdout + result.stderr

def parse_perf_output(output):
    """Parse performance counters and match with labels"""
    lines = output.split('\n')

    # Storage for results
    perf_data = []
    label_stack = []

    for i, line in enumerate(lines):
        # Match performance counter
        perf_match = re.search(r'\[Performance Counter\]: Execution period is (\d+) ns', line)

        if perf_match:
            cycles_ns = int(perf_match.group(1))

            # Check if [SYNC] appears on same line or previous line
            label = "unlabeled"
            if '[SYNC]' in line:
                label = "[SYNC]"
                # Subtract printf overhead (234 ns) from sync measurements
                cycles_ns = max(0, cycles_ns - 234)
            elif i - 1 >= 0 and '[SYNC]' in lines[i - 1]:
                label = "[SYNC]"
                # Subtract printf overhead (234 ns) from sync measurements
                cycles_ns = max(0, cycles_ns - 234)
            else:
                # Look backwards for the most recent label
                for j in range(i-1, max(0, i-10), -1):
                    prev_line = lines[j].strip()
                    # Match various label patterns
                    if re.search(r'\[(DMA|REDMULE|SPATZ|DEBUG|TIMER|SYNC)\]', prev_line):
                        label = prev_line
                        break
                    elif re.search(r'>> [AB]-', prev_line):
                        label = prev_line
                        break
                    elif re.search(r'>>> B-TILE', prev_line):
                        label = prev_line
                        break

            perf_data.append((label, cycles_ns))

    return perf_data

def aggregate_stats(perf_data):
    """Aggregate statistics by operation type"""
    stats = defaultdict(list)

    for label, cycles in perf_data:
        # Categorize by operation type
        if '[SYNC]' in label:
            stats['Sync_overhead'].append(cycles)
        elif '[DMA]' in label:
            if 'Load A' in label:
                stats['DMA_Load_A'].append(cycles)
            elif 'Load B' in label or 'indices' in label:
                stats['DMA_Load_B_indices'].append(cycles)
            elif 'Storing' in label or 'store' in label:
                stats['DMA_Store'].append(cycles)
            else:
                stats['DMA_Other'].append(cycles)
        elif '[REDMULE]' in label:
            stats['Compute_REDMULE'].append(cycles)
        elif '[SPATZ]' in label or '[spatz]' in label:
            if 'Dequantizing' in label:
                stats['Dequant_SPATZ'].append(cycles)
            else:
                stats['Compute_SPATZ'].append(cycles)
        elif '>> A-' in label:
            stats['Tile_A_iteration'].append(cycles)
        elif '>>> B-TILE' in label:
            stats['Tile_B_iteration'].append(cycles)
        else:
            stats['Other'].append(cycles)

    return stats

def print_summary(stats):
    """Print statistical summary"""
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)

    total_cycles = 0

    for op_type in sorted(stats.keys()):
        cycles_list = stats[op_type]
        count = len(cycles_list)
        total = sum(cycles_list)
        avg = total / count if count > 0 else 0
        min_c = min(cycles_list) if cycles_list else 0
        max_c = max(cycles_list) if cycles_list else 0

        total_cycles += total

        print(f"\n{op_type}:")
        print(f"  Count: {count}")
        print(f"  Total: {total:,} ns  ({total/1000:.2f} us)")
        print(f"  Avg:   {avg:.1f} ns")
        print(f"  Range: [{min_c} - {max_c}] ns")

    print(f"\n" + "-"*80)
    print(f"TOTAL MEASURED: {total_cycles:,} ns ({total_cycles/1000:.2f} us)")
    print("="*80)

def print_detailed(perf_data):
    """Print detailed line-by-line breakdown"""
    print("\n" + "="*80)
    print("DETAILED BREAKDOWN")
    print("="*80)

    for i, (label, cycles) in enumerate(perf_data):
        print(f"{i:3d}. [{cycles:6d} ns] {label}")

def plot_gantt_chart(perf_data):
    """Create Gantt chart showing parallel execution on DMA and SPATZ cores"""
    fig, ax = plt.subplots(figsize=(20, 10))

    # Color mapping for operation types
    colors = {
        'DMA_Load_A': '#3498db',      # Blue - Activation
        'DMA_Load_B': '#1abc9c',      # Teal - Weight indices
        'DMA_Store': '#2ecc71',       # Green
        'REDMULE': '#e74c3c',         # Red
        'SPATZ': '#f39c12',           # Orange
        'Dequant': '#9b59b6',         # Purple
        'Sync': '#95a5a6',            # Gray
        'Other': '#34495e'            # Dark gray
    }

    # Parse based on actual code execution pattern
    # Between syncs: DMA and SPATZ run in parallel (take max time)
    # After sync: both start at same time

    dma_timeline = []
    spatz_timeline = []
    sync_events = []

    current_time = 0  # Global synchronized time
    pending_dma = []  # DMA ops waiting to execute in parallel
    pending_spatz = []  # SPATZ ops waiting to execute in parallel

    for label, cycles in perf_data:
        display_label = label[:60]

        # Detect sync points: explicit [SYNC] or when operation type changes between cores
        is_sync = '[SYNC]' in label

        # If we have pending ops from both cores, assume they ran in parallel until now
        if not is_sync and pending_dma and pending_spatz:
            # Flush as parallel execution
            dma_dur = sum(op['duration'] for op in pending_dma)
            spatz_dur = sum(op['duration'] for op in pending_spatz)
            parallel_dur = max(dma_dur, spatz_dur)

            # Add DMA ops
            dma_t = current_time
            for op in pending_dma:
                dma_timeline.append({**op, 'start': dma_t})
                dma_t += op['duration']

            # Add SPATZ ops starting at same time (parallel)
            spatz_t = current_time
            for op in pending_spatz:
                spatz_timeline.append({**op, 'start': spatz_t})
                spatz_t += op['duration']

            current_time += parallel_dur
            pending_dma = []
            pending_spatz = []

        if is_sync:
            # Explicit sync barrier
            sync_events.append({'time': current_time, 'duration': cycles})
            current_time += cycles
            continue

        # Categorize operation
        if '[DMA]' in label:
            if 'Load A' in label:
                category = 'DMA_Load_A'
                display_label = "Load A1" if 'A1' in label else "Load A0" if 'A0' in label else "Load A"
            elif 'Load B' in label or 'indices' in label or 'Prefetch' in label:
                category = 'DMA_Load_B'
                display_label = "Load B1 idx" if 'B1' in label else "Load B0 idx" if 'B0' in label else "Load B idx"
            elif 'Storing' in label or 'store' in label:
                category = 'DMA_Store'
                if 'A0 x B0' in label:
                    display_label = "Store A0×B0"
                elif 'A0 x B1' in label:
                    display_label = "Store A0×B1"
                elif 'A1 x B0' in label:
                    display_label = "Store A1×B0"
                elif 'A1 x B1' in label:
                    display_label = "Store A1×B1"
                else:
                    display_label = "Store C"
            else:
                category = 'Other'

            pending_dma.append({
                'duration': cycles,
                'label': display_label,
                'category': category,
                'color': colors.get(category, colors['Other']),
                'lane': 0
            })

        elif '[REDMULE]' in label or '[SPATZ]' in label or '[spatz]' in label:
            if '[REDMULE]' in label:
                category = 'REDMULE'
                display_label = "REDMULE"
            elif 'Dequant' in label:
                category = 'Dequant'
                display_label = "Dequant"
            else:
                category = 'SPATZ'
                display_label = "SPATZ"

            pending_spatz.append({
                'duration': cycles,
                'label': display_label,
                'category': category,
                'color': colors.get(category, colors['Other']),
                'lane': 1
            })

    # Flush any remaining pending operations at the end
    if pending_dma or pending_spatz:
        dma_dur = sum(op['duration'] for op in pending_dma)
        spatz_dur = sum(op['duration'] for op in pending_spatz)
        parallel_dur = max(dma_dur, spatz_dur) if (pending_dma and pending_spatz) else (dma_dur + spatz_dur)

        dma_t = current_time
        for op in pending_dma:
            dma_timeline.append({**op, 'start': dma_t})
            dma_t += op['duration']

        spatz_t = current_time
        for op in pending_spatz:
            spatz_timeline.append({**op, 'start': spatz_t})
            spatz_t += op['duration']

        current_time += parallel_dur

    # Combine timelines
    all_items = dma_timeline + spatz_timeline
    max_time = current_time

    # Plot bars for each core lane
    for item in all_items:
        y_pos = item['lane']
        ax.barh(y_pos, item['duration'], left=item['start'],
                color=item['color'], edgecolor='black', linewidth=0.8, height=0.7)

        # Add label text
        if item['duration'] > max_time * 0.015:
            ax.text(item['start'] + item['duration']/2, y_pos,
                   item['label'],
                   ha='center', va='center', fontsize=8, color='white', weight='bold')

    # Plot sync barriers as vertical lines
    for sync in sync_events:
        ax.axvline(x=sync['time'], color='#95a5a6', linestyle='--', linewidth=2, alpha=0.7)
        ax.text(sync['time'], 1.8, f"SYNC\n{sync['duration']}ns",
                ha='center', fontsize=7, color='#95a5a6', weight='bold')

    # Formatting
    ax.set_xlabel('Time (ns)', fontsize=14, weight='bold')
    ax.set_ylabel('Core', fontsize=14, weight='bold')
    ax.set_title('Parallel Execution Timeline (DMA Core vs SPATZ Core)', fontsize=16, weight='bold')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['DMA Core', 'SPATZ Core'], fontsize=12)
    ax.set_ylim(-0.5, 2.2)
    ax.set_xlim(0, max_time)
    ax.grid(axis='x', alpha=0.3)

    # Legend
    legend_items = [
        mpatches.Patch(color=colors['DMA_Load_A'], label='Load Activation'),
        mpatches.Patch(color=colors['DMA_Load_B'], label='Load B Indices'),
        mpatches.Patch(color=colors['DMA_Store'], label='Store Result'),
        mpatches.Patch(color=colors['Dequant'], label='Dequantize'),
        mpatches.Patch(color=colors['REDMULE'], label='REDMULE Compute'),
    ]
    ax.legend(handles=legend_items, loc='upper right', fontsize=10, ncol=2)

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'plots/gantt_parallel_{timestamp}.png'
    plt.savefig(filename, dpi=150)
    print(f"\nParallel Gantt chart saved to: {filename}")
    plt.close()

def plot_pie_chart(stats):
    """Create pie chart showing time distribution"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Prepare data
    categories = []
    totals = []
    colors_list = []

    color_map = {
        'DMA_Load': '#3498db',
        'DMA_Store': '#2ecc71',
        'Compute_REDMULE': '#e74c3c',
        'Dequant_SPATZ': '#9b59b6',
        'Sync_overhead': '#95a5a6',
    }

    for cat in sorted(stats.keys()):
        total = sum(stats[cat])
        if total > 0:
            categories.append(cat)
            totals.append(total)
            colors_list.append(color_map.get(cat, '#34495e'))

    # Plot
    wedges, texts, autotexts = ax.pie(totals, labels=categories, autopct='%1.1f%%',
                                       colors=colors_list, startangle=90)

    # Format percentage text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(10)
        autotext.set_weight('bold')

    ax.set_title('Time Distribution by Operation Type', fontsize=14, weight='bold')

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'plots/pie_chart_{timestamp}.png'
    plt.savefig(filename, dpi=150)
    print(f"Pie chart saved to: {filename}")
    plt.close()

def plot_bar_chart(stats):
    """Create bar chart comparing operation times"""
    fig, ax = plt.subplots(figsize=(14, 8))

    categories = []
    averages = []
    totals = []

    for cat in sorted(stats.keys()):
        if stats[cat]:
            categories.append(cat.replace('_', '\n'))
            averages.append(sum(stats[cat]) / len(stats[cat]))
            totals.append(sum(stats[cat]))

    x = range(len(categories))
    width = 0.35

    bars1 = ax.bar([i - width/2 for i in x], averages, width, label='Average (ns)', color='#3498db')
    bars2 = ax.bar([i + width/2 for i in x], totals, width, label='Total (ns)', color='#e74c3c')

    ax.set_xlabel('Operation Type', fontsize=12)
    ax.set_ylabel('Time (ns)', fontsize=12)
    ax.set_title('Performance Comparison', fontsize=14, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'plots/bar_chart_{timestamp}.png'
    plt.savefig(filename, dpi=150)
    print(f"Bar chart saved to: {filename}")
    plt.close()

if __name__ == '__main__':
    print("Running make dqsw...")
    output = run_benchmark()

    print("\nParsing performance counters...")
    perf_data = parse_perf_output(output)

    print(f"Found {len(perf_data)} performance measurements")

    # Aggregate statistics
    stats = aggregate_stats(perf_data)

    # Print summary
    print_summary(stats)

    # Optionally print detailed breakdown
    if '--detailed' in sys.argv:
        print_detailed(perf_data)
    else:
        print("\nRun with --detailed flag for line-by-line breakdown")

    # Generate visualizations
    if '--plot' in sys.argv or '--all' in sys.argv:
        print("\nGenerating visualizations...")
        try:
            plot_gantt_chart(perf_data)
            plot_pie_chart(stats)
            plot_bar_chart(stats)
            print("\nAll visualizations generated successfully!")
        except Exception as e:
            print(f"Error generating plots: {e}")
            print("Make sure matplotlib is installed: pip install matplotlib")
    elif '--no-plot' not in sys.argv:
        print("\nRun with --plot flag to generate Gantt chart, pie chart, and bar chart visualizations")
