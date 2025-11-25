import csv
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path
import sys
import numpy as np


def plot_track_from_csv(csv_file, track_name):
    blue_cones = []
    yellow_cones = []
    black_cones = []
    orange_cones = []

    # Read CSV file
    with open(csv_file, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            tag = row["tag"]
            if tag == "blue":
                blue_cones.append([float(row["x"]), float(row["y"])])
            elif tag == "yellow":
                yellow_cones.append([float(row["x"]), float(row["y"])])
            elif tag == "blank":
                black_cones.append([float(row["x"]), float(row["y"])])
            else:
                orange_cones.append([float(row["x"]), float(row["y"])])

    # Convert lists to numpy arrays if cones exist
    if blue_cones:
        blue_cones = np.array(blue_cones)
    if yellow_cones:
        yellow_cones = np.array(yellow_cones)
    if orange_cones:
        orange_cones = np.array(orange_cones)
    if black_cones:
        black_cones = np.array(black_cones)

    # Plot the track
    plt.figure(figsize=(10, 10))

    if len(orange_cones) > 0:
        plt.scatter(
            orange_cones[:, 0],
            orange_cones[:, 1],
            c="orange",
            s=50,
            alpha=0.7,
            label="Orange Cones",
        )

    if len(blue_cones) > 0:
        plt.scatter(
            blue_cones[:, 0],
            blue_cones[:, 1],
            c="blue",
            s=50,
            alpha=0.7,
            label="Blue Cones",
        )

    if len(yellow_cones) > 0:
        plt.scatter(
            yellow_cones[:, 0],
            yellow_cones[:, 1],
            c="yellow",
            s=50,
            alpha=0.7,
            label="Yellow Cones",
        )

    if len(black_cones) > 0:
        plt.scatter(
            black_cones[:, 0],
            black_cones[:, 1],
            c="black",
            s=50,
            alpha=0.7,
            label="False Positives",
        )

    # Set axis labels and title
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Track Layout - {track_name}")
    plt.legend()

    plt.grid(True)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Track Generator Script")

    parser.add_argument(
        "--dir",
        type=str,
        default=os.path.join(os.getcwd(), "data", "generated_tracks"),
        help="Directory to save tracks",
    )

    parser.add_argument(
        "--track_name",
        type=str,
        default=None,
        help="Base name for the generated track files",
    )

    args = parser.parse_args()

    # Ensure the directory exists
    if not os.path.exists(args.dir):
        os.makedirs(args.dir)

    tracks_directory = Path(args.dir)

    if args.track_name is not None:
        csv_file = tracks_directory / f"{args.track_name}.csv"
        if csv_file.exists():
            print(f"Plotting track from {csv_file}")
            plot_track_from_csv(csv_file, args.track_name)
        else:
            print(f"File {csv_file} does not exist.")
    else:
        # If no arguments, use all CSV files in the tracks directory
        file_list = list(tracks_directory.glob("*.csv"))
        # Iterate over all CSV files in the tracks directory
        for csv_file in file_list:
            track_name = csv_file.stem  # Get the filename without extension
            print(f"Plotting track from {csv_file}")
            plot_track_from_csv(csv_file, track_name)


if __name__ == "__main__":
    main()
