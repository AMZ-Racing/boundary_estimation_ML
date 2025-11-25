import csv
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path
import sys
import numpy as np


def is_augmented_track(csv_file):
    """Check if CSV file contains augmented data (has aug_tag, aug_x, aug_y columns)"""
    with open(csv_file, "r") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        return (
            fieldnames is not None
            and "aug_tag" in fieldnames
            and "aug_x" in fieldnames
            and "aug_y" in fieldnames
        )


def plot_track_from_csv(csv_file, track_name, mode="both"):
    """
    Plot track from CSV file.
    
    Args:
        csv_file: Path to CSV file
        track_name: Name of the track for the title
        mode: 'original', 'augmented', or 'both' (default: 'both')
    """
    # Check if this is an augmented track
    is_augmented = is_augmented_track(csv_file)

    if not is_augmented and mode == "augmented":
        print(f"Warning: {csv_file} is not an augmented track. Showing original data.")
        mode = "original"

    # Data structures for original cones
    blue_cones = []
    yellow_cones = []
    blank_cones = []
    orange_cones = []

    # Data structures for augmented cones
    aug_blue_cones = []
    aug_yellow_cones = []
    aug_blank_cones = []
    aug_orange_cones = []

    # Read CSV file
    with open(csv_file, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Original data (ground truth)
            if mode in ["original", "both"]:
                tag = row["tag"]
                x, y = float(row["x"]), float(row["y"])
                if tag == "blue":
                    blue_cones.append([x, y])
                elif tag == "yellow":
                    yellow_cones.append([x, y])
                elif tag == "blank":
                    blank_cones.append([x, y])
                else:  # orange, big_orange, etc.
                    orange_cones.append([x, y])

            # Augmented data - only plot valid cones
            if is_augmented and mode in ["augmented", "both"]:
                # Check if cone is valid (not removed)
                is_valid = row.get("is_valid", "True")
                if is_valid in ["True", "true", "1", True, 1]:
                    aug_tag = row["aug_tag"]
                    aug_x, aug_y = float(row["aug_x"]), float(row["aug_y"])
                    if aug_tag == "blue":
                        aug_blue_cones.append([aug_x, aug_y])
                    elif aug_tag == "yellow":
                        aug_yellow_cones.append([aug_x, aug_y])
                    elif aug_tag == "blank":
                        aug_blank_cones.append([aug_x, aug_y])
                    else:  # orange, big_orange, etc.
                        aug_orange_cones.append([aug_x, aug_y])

    # Convert lists to numpy arrays if cones exist
    def to_array(cone_list):
        return np.array(cone_list) if cone_list else np.array([])

    blue_cones = to_array(blue_cones)
    yellow_cones = to_array(yellow_cones)
    orange_cones = to_array(orange_cones)
    blank_cones = to_array(blank_cones)

    aug_blue_cones = to_array(aug_blue_cones)
    aug_yellow_cones = to_array(aug_yellow_cones)
    aug_orange_cones = to_array(aug_orange_cones)
    aug_blank_cones = to_array(aug_blank_cones)

    # Plot the track
    plt.figure(figsize=(12, 12))

    # Plot original cones
    if mode in ["original", "both"]:
        # alpha = 0.4 if mode == "both" else 0.7
        alpha = 1.0 if mode == "both" else 0.7
        marker = "o" if mode == "original" else "x"
        size = 50 if mode == "original" else 100
        label_prefix = "Original " if mode == "both" else ""

        if len(orange_cones) > 0:
            plt.scatter(
                orange_cones[:, 0],
                orange_cones[:, 1],
                c="orange",
                s=size,
                alpha=alpha,
                marker=marker,
                label=f"{label_prefix}Orange Cones",
            )

        if len(blue_cones) > 0:
            plt.scatter(
                blue_cones[:, 0],
                blue_cones[:, 1],
                c="blue",
                s=size,
                alpha=alpha,
                marker=marker,
                label=f"{label_prefix}Blue Cones",
            )

        if len(yellow_cones) > 0:
            plt.scatter(
                yellow_cones[:, 0],
                yellow_cones[:, 1],
                c="yellow",
                s=size,
                alpha=alpha,
                marker=marker,
                edgecolors="orange",
                label=f"{label_prefix}Yellow Cones",
            )

        if len(blank_cones) > 0:
            plt.scatter(
                blank_cones[:, 0],
                blank_cones[:, 1],
                c="red",
                s=size,
                alpha=alpha,
                marker=marker,
                label=f"{label_prefix}False Positives",
            )

    # Plot augmented cones
    if is_augmented and mode in ["augmented", "both"]:
        alpha = 0.7
        marker = "o"
        size = 50
        label_prefix = "Augmented " if mode == "both" else ""

        if len(aug_orange_cones) > 0:
            plt.scatter(
                aug_orange_cones[:, 0],
                aug_orange_cones[:, 1],
                c="orange",
                s=size,
                alpha=alpha,
                marker=marker,
                label=f"{label_prefix}Orange Cones",
            )

        if len(aug_blue_cones) > 0:
            plt.scatter(
                aug_blue_cones[:, 0],
                aug_blue_cones[:, 1],
                c="blue",
                s=size,
                alpha=alpha,
                marker=marker,
                label=f"{label_prefix}Blue Cones",
            )

        if len(aug_yellow_cones) > 0:
            plt.scatter(
                aug_yellow_cones[:, 0],
                aug_yellow_cones[:, 1],
                c="yellow",
                s=size,
                alpha=alpha,
                marker=marker,
                edgecolors="orange",
                label=f"{label_prefix}Yellow Cones",
            )

        if len(aug_blank_cones) > 0:
            plt.scatter(
                aug_blank_cones[:, 0],
                aug_blank_cones[:, 1],
                c="red",
                s=size,
                alpha=alpha,
                marker=marker,
                label=f"{label_prefix}False Positives",
            )

    # Set axis labels and title
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    
    title_suffix = ""
    if is_augmented:
        if mode == "original":
            title_suffix = " (Original)"
        elif mode == "augmented":
            title_suffix = " (Augmented)"
        elif mode == "both":
            title_suffix = " (Original + Augmented)"
    
    plt.title(f"Track Layout - {track_name}{title_suffix}")
    plt.legend()
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Track Visualizer - Visualize original and augmented tracks"
    )

    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Directory containing track CSV files",
    )

    parser.add_argument(
        "--track_name",
        type=str,
        default=None,
        help="Base name for the track file (without .csv extension)",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["original", "augmented", "both"],
        help="Visualization mode: 'original' (only original cones), 'augmented' (only augmented cones), or 'both' (overlay both) - default: both",
    )

    args = parser.parse_args()

    # Determine the directory to use
    if args.dir is not None:
        tracks_directory = Path(args.dir)
        if not tracks_directory.exists():
            print(f"Error: Directory '{args.dir}' does not exist.")
            return
    else:
        # Try augmented_tracks first, then generated_tracks
        augmented_dir = Path(os.path.join(os.getcwd(), "data", "augmented_tracks"))
        generated_dir = Path(os.path.join(os.getcwd(), "data", "generated_tracks"))
        
        if augmented_dir.exists():
            tracks_directory = augmented_dir
            print(f"Using directory: {tracks_directory}")
        elif generated_dir.exists():
            tracks_directory = generated_dir
            print(f"Using directory: {tracks_directory}")
        else:
            print(f"Error: Neither '{augmented_dir}' nor '{generated_dir}' exist.")
            print("Please specify a valid directory with --dir or create one of the default directories.")
            return

    if args.track_name is not None:
        csv_file = tracks_directory / f"{args.track_name}.csv"
        if csv_file.exists():
            print(f"Plotting track from {csv_file} (mode: {args.mode})")
            plot_track_from_csv(csv_file, args.track_name, mode=args.mode)
        else:
            print(f"File {csv_file} does not exist.")
    else:
        # If no arguments, use all CSV files in the tracks directory
        file_list = list(tracks_directory.glob("*.csv"))
        if not file_list:
            print(f"No CSV files found in {tracks_directory}")
            return
        
        print(f"Found {len(file_list)} track(s) to visualize (mode: {args.mode})")
        # Iterate over all CSV files in the tracks directory
        for csv_file in file_list:
            track_name = csv_file.stem  # Get the filename without extension
            print(f"Plotting track from {csv_file}")
            plot_track_from_csv(csv_file, track_name, mode=args.mode)


if __name__ == "__main__":
    main()
