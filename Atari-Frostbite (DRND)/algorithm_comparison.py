#!/usr/bin/env python3
"""
Algorithm Comparison Tool

This script compares TensorBoard metrics between two different algorithm folders
(e.g., DRND vs RND) and generates comparison plots with custom labels.
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from collections import defaultdict

def find_event_dirs(parent_dir):
    """Find all directories containing TensorBoard event files."""
    print(f"Searching for TensorBoard event files in: {parent_dir}")
    
    # Check if directory exists
    if not os.path.exists(parent_dir):
        print(f"Error: Directory {parent_dir} does not exist")
        return []
    
    run_dirs = []
    
    # Check if parent_dir itself has event files
    event_files = glob.glob(os.path.join(parent_dir, "events.out.tfevents.*"))
    if event_files:
        print(f"Found event files directly in {parent_dir}")
        run_dirs.append(parent_dir)
        return run_dirs
    
    # Look for subdirectories with event files
    for item in os.listdir(parent_dir):
        item_path = os.path.join(parent_dir, item)
        if os.path.isdir(item_path):
            # Check if this directory has event files
            event_files = glob.glob(os.path.join(item_path, "events.out.tfevents.*"))
            if event_files:
                print(f"Found event files in: {item_path}")
                run_dirs.append(item_path)
                continue
                
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

def process_algorithm_data(algo_dir, algo_name):
    """
    Process all runs for a given algorithm directory
    
    Args:
        algo_dir: Directory containing runs for this algorithm
        algo_name: Name of the algorithm for display purposes
        
    Returns:
        Dictionary mapping tag -> (mean_values, std_values, steps)
    """
    print(f"\nProcessing {algo_name} data from: {algo_dir}")
    
    # Find all run directories
    run_dirs = find_event_dirs(algo_dir)
    if not run_dirs:
        print(f"No run directories found for {algo_name}")
        return {}
    
    # Collect tag data from all runs
    all_run_data = []
    for run_dir in run_dirs:
        try:
            # Load event file
            event_acc = EventAccumulator(run_dir)
            event_acc.Reload()
            tags = event_acc.Tags()["scalars"]
            
            if not tags:
                print(f"No scalar tags found in {run_dir}")
                continue
                
            # Process each tag
            run_data = {}
            for tag in tags:
                event_data = event_acc.Scalars(tag)
                if event_data:
                    df = pd.DataFrame([
                        {"step": event.step, "value": event.value}
                        for event in event_data
                    ])
                    run_data[tag] = df
            
            if run_data:
                all_run_data.append(run_data)
                print(f"Extracted data from {run_dir} with {len(run_data)} tags")
            
        except Exception as e:
            print(f"Error processing {run_dir}: {e}")
            import traceback
            traceback.print_exc()
    
    if not all_run_data:
        print(f"No data could be extracted for {algo_name}")
        return {}
    
    # Get all unique tags
    all_tags = set()
    for run_data in all_run_data:
        all_tags.update(run_data.keys())
    
    # For each tag, compute the mean and std across runs
    result = {}
    for tag in all_tags:
        # Collect data frames for this tag from all runs
        tag_frames = [run_data[tag] for run_data in all_run_data if tag in run_data]
        if not tag_frames:
            continue
            
        # Number of runs for this tag
        num_runs = len(tag_frames)
        print(f"Processing {tag} with {num_runs} runs")
        
        if num_runs == 1:
            # Only one run, no need for interpolation
            steps = tag_frames[0]["step"].values
            values = tag_frames[0]["value"].values
            result[tag] = (values, np.zeros_like(values), steps)
            continue
        
        # Find common step range
        min_step = max(df["step"].min() for df in tag_frames)
        max_step = min(df["step"].max() for df in tag_frames)
        
        if min_step > max_step:
            print(f"  No overlapping steps for {tag}, skipping")
            continue
        
        # Create common step axis with 500 points
        num_points = min(500, int(max_step - min_step) + 1)
        common_steps = np.linspace(min_step, max_step, num_points)
        
        # Interpolate each run to common steps
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
            continue
        
        # Calculate mean and std
        values_array = np.array(interpolated_values)
        mean_values = np.nanmean(values_array, axis=0)
        std_values = np.nanstd(values_array, axis=0) if len(interpolated_values) > 1 else np.zeros_like(mean_values)
        
        result[tag] = (mean_values, std_values, common_steps)
    
    return result

def create_comparison_plots(drnd_data, rnd_data, output_dir, custom_labels=None):
    """
    Create comparison plots for each tag that exists in both datasets
    
    Args:
        drnd_data: Data for DRND algorithm (from process_algorithm_data)
        rnd_data: Data for RND algorithm (from process_algorithm_data)
        output_dir: Directory to save plots and CSVs
        custom_labels: Dictionary with custom axis labels and titles
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get common tags
    drnd_tags = set(drnd_data.keys())
    rnd_tags = set(rnd_data.keys())
    common_tags = drnd_tags.intersection(rnd_tags)
    
    print(f"\nCreating comparison plots for {len(common_tags)} common metrics")
    
    # If no custom labels provided, initialize empty dict
    if custom_labels is None:
        custom_labels = {}
    
    for tag in common_tags:
        print(f"Creating comparison plot for: {tag}")
        
        # Get data for both algorithms
        drnd_mean, drnd_std, drnd_steps = drnd_data[tag]
        rnd_mean, rnd_std, rnd_steps = rnd_data[tag]
        
        # Create plot
        plt.figure(figsize=(12, 7))
        
        # Plot DRND
        plt.plot(drnd_steps, drnd_mean, color='blue', linewidth=2, label="DRND")
        plt.fill_between(
            drnd_steps,
            drnd_mean - drnd_std,
            drnd_mean + drnd_std,
            color='blue',
            alpha=0.2
        )
        
        # Plot RND
        plt.plot(rnd_steps, rnd_mean, color='red', linewidth=2, label="RND")
        plt.fill_between(
            rnd_steps,
            rnd_mean - rnd_std,
            rnd_mean + rnd_std,
            color='red',
            alpha=0.2
        )
        
        # Get custom labels if available
        clean_tag = tag.replace("/", "_").replace(" ", "_")
        
        # Set title - use custom title if available
        if f"{clean_tag}_title" in custom_labels:
            title = custom_labels[f"{clean_tag}_title"]
        else:
            title = f"Comparison of {tag.split('/')[-1]}"
        plt.title(title, fontsize=16)
        
        # Set x-axis label
        if f"{clean_tag}_xlabel" in custom_labels:
            xlabel = custom_labels[f"{clean_tag}_xlabel"]
        else:
            xlabel = "Training Steps"
        plt.xlabel(xlabel, fontsize=14)
        
        # Set y-axis label
        if f"{clean_tag}_ylabel" in custom_labels:
            ylabel = custom_labels[f"{clean_tag}_ylabel"]
        else:
            ylabel = tag.split('/')[-1]
        plt.ylabel(ylabel, fontsize=14)
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        
        # Save plot
        plot_path = os.path.join(output_dir, f"{clean_tag}_comparison.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        # Also save CSV with comparison data
        # First, need to align the data if step ranges differ
        all_steps = np.unique(np.concatenate([drnd_steps, rnd_steps]))
        all_steps.sort()
        
        # Interpolate DRND data to all steps
        drnd_interp = np.interp(
            all_steps,
            drnd_steps,
            drnd_mean,
            left=np.nan,
            right=np.nan
        )
        
        # Interpolate RND data to all steps
        rnd_interp = np.interp(
            all_steps,
            rnd_steps,
            rnd_mean,
            left=np.nan,
            right=np.nan
        )
        
        # Create and save dataframe
        comparison_df = pd.DataFrame({
            "steps": all_steps,
            "DRND_mean": drnd_interp,
            "RND_mean": rnd_interp
        })
        
        csv_path = os.path.join(output_dir, f"{clean_tag}_comparison.csv")
        comparison_df.to_csv(csv_path, index=False)
        
        print(f"  Saved plot to: {plot_path}")
        print(f"  Saved data to: {csv_path}")

def parse_custom_labels(label_file):
    """Parse custom labels from a file."""
    if not label_file or not os.path.exists(label_file):
        return {}
        
    try:
        custom_labels = {}
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                parts = line.split('=', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    custom_labels[key] = value
        
        return custom_labels
    except Exception as e:
        print(f"Error parsing custom labels file: {e}")
        return {}

def main():
    parser = argparse.ArgumentParser(description="Compare metrics between two algorithms (DRND vs RND)")
    parser.add_argument("--drnd_dir", type=str, required=True,
                       help="Directory containing DRND algorithm runs")
    parser.add_argument("--rnd_dir", type=str, required=True,
                       help="Directory containing RND algorithm runs")
    parser.add_argument("--output_dir", type=str, default="./algorithm_comparison",
                       help="Directory to save comparison plots and CSVs")
    parser.add_argument("--labels_file", type=str, default=None,
                       help="File containing custom axis labels and titles")
    parser.add_argument("--drnd_name", type=str, default="DRND",
                       help="Display name for DRND algorithm")
    parser.add_argument("--rnd_name", type=str, default="RND",
                       help="Display name for RND algorithm")
    
    args = parser.parse_args()
    
    # Process data for both algorithms
    drnd_data = process_algorithm_data(args.drnd_dir, args.drnd_name)
    rnd_data = process_algorithm_data(args.rnd_dir, args.rnd_name)
    
    # Parse custom labels if provided
    custom_labels = parse_custom_labels(args.labels_file)
    
    # Create comparison plots
    if drnd_data and rnd_data:
        create_comparison_plots(drnd_data, rnd_data, args.output_dir, custom_labels)
        print(f"\nComparison complete! Results saved to: {args.output_dir}")
    else:
        print("\nCould not create comparisons - missing data from one or both algorithms.")

if __name__ == "__main__":
    main()