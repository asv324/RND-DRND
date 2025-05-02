#!/usr/bin/env python3
"""
Simple TensorBoard Log Averager

This script finds and processes TensorBoard event files in a directory structure,
and generates averaged plots for each metric.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def find_event_files(parent_dir):
    """Find all TensorBoard event files in a directory structure."""
    print(f"Searching for TensorBoard event files in: {parent_dir}")
    
    # List all subdirectories
    run_dirs = []
    
    if not os.path.exists(parent_dir):
        print(f"Error: Directory {parent_dir} does not exist")
        return []
    
    # Check if parent_dir itself has event files
    event_files = glob.glob(os.path.join(parent_dir, "events.out.tfevents.*"))
    if event_files:
        print(f"Found event files directly in {parent_dir}")
        run_dirs.append(parent_dir)
    else:
        # Look for subdirectories
        for item in os.listdir(parent_dir):
            item_path = os.path.join(parent_dir, item)
            if os.path.isdir(item_path):
                # Check if this directory has event files
                event_files = glob.glob(os.path.join(item_path, "events.out.tfevents.*"))
                if event_files:
                    print(f"Found event files in: {item_path}")
                    run_dirs.append(item_path)
                else:
                    # Look in common TensorBoard subdirectories
                    for tb_dir in ["tensorboard", "tblogs", "logs", "tb", "tfevents"]:
                        tb_path = os.path.join(item_path, tb_dir)
                        if os.path.exists(tb_path) and os.path.isdir(tb_path):
                            event_files = glob.glob(os.path.join(tb_path, "events.out.tfevents.*"))
                            if event_files:
                                print(f"Found event files in: {tb_path}")
                                run_dirs.append(tb_path)
    
    print(f"Found {len(run_dirs)} directories with event files")
    return run_dirs

def get_scalar_data(event_file_dir, selected_tags=None):
    """Extract scalar data from TensorBoard event files."""
    try:
        event_acc = EventAccumulator(event_file_dir)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        
        if not tags:
            print(f"No scalar tags found in {event_file_dir}")
            return {}
            
        print(f"Available tags in {os.path.basename(event_file_dir)}: {tags}")
        
        # Filter tags if specified
        if selected_tags:
            tags = [tag for tag in tags if tag in selected_tags]
        
        tag_data = {}
        for tag in tags:
            event_data = event_acc.Scalars(tag)
            tag_data[tag] = pd.DataFrame([
                {"step": event.step, "value": event.value, "wall_time": event.wall_time}
                for event in event_data
            ])
            print(f"  Extracted {len(tag_data[tag])} points for '{tag}'")
        
        return tag_data
    except Exception as e:
        print(f"Error processing {event_file_dir}: {e}")
        import traceback
        traceback.print_exc()
        return {}

def process_and_plot(parent_dir, output_dir="./averaged_plots", selected_tags=None):
    """Process TensorBoard logs and create averaged plots."""
    # Find all event file directories
    run_dirs = find_event_files(parent_dir)
    
    if not run_dirs:
        print("No run directories found with event files.")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Collect data from all runs
    all_run_data = []
    for run_dir in run_dirs:
        run_data = get_scalar_data(run_dir, selected_tags)
        if run_data:
            all_run_data.append(run_data)
    
    if not all_run_data:
        print("No data could be extracted from any run.")
        return
    
    # Get all unique tags across all runs
    all_tags = set()
    for run_data in all_run_data:
        all_tags.update(run_data.keys())
    
    print(f"Tags to be processed: {all_tags}")
    
    # Process each tag
    for tag in all_tags:
        print(f"Processing tag: {tag}")
        # Collect data for this tag from all runs
        tag_frames = [run_data[tag] for run_data in all_run_data if tag in run_data]
        
        if not tag_frames:
            print(f"  No data available for {tag}")
            continue
        
        if len(tag_frames) == 1:
            print(f"  Only one run has data for {tag}, no averaging needed")
            
        # Find common step range
        min_step = max(df["step"].min() for df in tag_frames)
        max_step = min(df["step"].max() for df in tag_frames)
        
        if min_step > max_step:
            print(f"  No overlapping steps for {tag}, skipping")
            continue
        
        # Create common step axis (use 500 points or less if range is small)
        num_points = min(500, int(max_step - min_step) + 1)
        common_steps = np.linspace(min_step, max_step, num_points)
        
        # Interpolate all runs to common step axis
        interpolated_values = []
        for df in tag_frames:
            if len(df) >= 2:  # Need at least 2 points for interpolation
                interpolated = np.interp(
                    common_steps,
                    df["step"].values,
                    df["value"].values,
                    left=np.nan,
                    right=np.nan
                )
                interpolated_values.append(interpolated)
        
        if not interpolated_values:
            print(f"  Not enough data points for interpolation in {tag}")
            continue
        
        # Calculate mean and standard deviation
        values_array = np.array(interpolated_values)
        mean_values = np.nanmean(values_array, axis=0)
        std_values = np.nanstd(values_array, axis=0) if len(interpolated_values) > 1 else np.zeros_like(mean_values)
        
        # Create and save plot
        plt.figure(figsize=(10, 6))
        plt.plot(common_steps, mean_values, label="Mean", linewidth=2)
        
        if len(interpolated_values) > 1:
            plt.fill_between(
                common_steps,
                mean_values - std_values,
                mean_values + std_values,
                alpha=0.3,
                label="±1 std"
            )
        
        # Clean up tag name for display
        display_tag = tag.split("/")[-1] if "/" in tag else tag
        
        plt.title(f"{display_tag} (Averaged over {len(interpolated_values)} runs)")
        plt.xlabel("Step")
        plt.ylabel(display_tag)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        
        # Save plot
        clean_tag = tag.replace("/", "_").replace(" ", "_")
        plot_path = os.path.join(output_dir, f"{clean_tag}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"  Created plot: {plot_path}")
        
        # Also save the data as CSV
        result_df = pd.DataFrame({
            "step": common_steps,
            "mean": mean_values,
            "std": std_values if len(interpolated_values) > 1 else np.zeros_like(mean_values),
            "runs": np.sum(~np.isnan(values_array), axis=0)
        })
        
        csv_path = os.path.join(output_dir, f"{clean_tag}.csv")
        result_df.to_csv(csv_path, index=False)
        print(f"  Saved data to: {csv_path}")
    
    print("Processing complete!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Average TensorBoard logs across runs")
    parser.add_argument("--dir", type=str, required=True, 
                        help="Parent directory containing run folders")
    parser.add_argument("--out", type=str, default="./averaged_plots",
                        help="Output directory for plots (default: ./averaged_plots)")
    parser.add_argument("--tags", type=str, default=None,
                        help="Comma-separated list of tags to process (default: all tags)")
    
    args = parser.parse_args()
    
    selected_tags = None
    if args.tags:
        selected_tags = [tag.strip() for tag in args.tags.split(",")]
        
    process_and_plot(args.dir, args.out, selected_tags)