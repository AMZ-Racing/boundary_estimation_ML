import os
import random
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


class TrackAugmentation:
    """
    Applies data augmentation to Formula Student track CSV files.
    Augmentations include:
    - Random cone removal
    - Random position noise
    - Random color changes
    """

    def __init__(self, config=None):
        default_cfg = {
            "seed": None,
            "removal_prob": 0.1,  # Probability of removing a cone
            "position_noise_std": 0.1,  # Standard deviation in meters (10cm default)
            "color_change_prob": 0.05,  # Probability of changing cone color
            "available_colors": ["blue", "yellow", "big_orange", "blank"],
        }
        self.config = {**default_cfg, **(config or {})}
        self.rng = random.Random(self.config["seed"])

    def augment_cone(self, tag, x, y):
        """
        Apply augmentation to a single cone.
        
        Returns:
        - (aug_tag, aug_x, aug_y, is_valid) tuple with augmented values and validity flag
        """
        # Check if cone should be marked as removed
        is_valid = self.rng.random() >= self.config["removal_prob"]

        # Apply position noise
        noise_x = self.rng.gauss(0, self.config["position_noise_std"])
        noise_y = self.rng.gauss(0, self.config["position_noise_std"])
        aug_x = x + noise_x
        aug_y = y + noise_y

        # Apply color change
        aug_tag = tag
        if self.rng.random() < self.config["color_change_prob"]:
            # Change to a different random color
            available = [c for c in self.config["available_colors"] if c != tag]
            if available:
                aug_tag = self.rng.choice(available)

        return (aug_tag, aug_x, aug_y, is_valid)

    def augment_track(self, input_csv_path, output_csv_path):
        """
        Augment a single track CSV file and save the result.
        
        Args:
            input_csv_path: Path to input CSV file
            output_csv_path: Path to output augmented CSV file
        """
        # Read original track
        df = pd.read_csv(input_csv_path)

        # Prepare augmented data
        augmented_rows = []

        for _, row in df.iterrows():
            tag = row["tag"]
            x = row["x"]
            y = row["y"]

            # Apply augmentation
            aug_tag, aug_x, aug_y, is_valid = self.augment_cone(tag, x, y)

            # Add all cones, including removed ones (marked with is_valid=False)
            augmented_rows.append(
                {
                    "tag": tag,
                    "x": x,
                    "y": y,
                    "aug_tag": aug_tag,
                    "aug_x": aug_x,
                    "aug_y": aug_y,
                    "is_valid": is_valid,
                }
            )

        # Create DataFrame and save
        augmented_df = pd.DataFrame(augmented_rows)
        augmented_df.to_csv(output_csv_path, index=False)

        num_valid = augmented_df["is_valid"].sum()
        num_removed = len(augmented_df) - num_valid
        print(
            f"Augmented: {input_csv_path.name} -> {output_csv_path.name} "
            f"({len(df)} cones: {num_valid} valid, {num_removed} removed)"
        )

    def augment_directory(self, input_dir, output_dir):
        """
        Augment all CSV files in a directory.
        
        Args:
            input_dir: Path to directory containing original tracks
            output_dir: Path to directory where augmented tracks will be saved
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)

        # Create output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)

        # Find all CSV files
        csv_files = list(input_path.glob("*.csv"))

        if not csv_files:
            print(f"No CSV files found in {input_dir}")
            return

        print(f"Found {len(csv_files)} tracks to augment")
        print(f"Output directory: {output_dir}")
        print(f"Configuration:")
        print(f"  - Removal probability: {self.config['removal_prob']:.1%}")
        print(f"  - Position noise std: {self.config['position_noise_std']*100:.1f}cm")
        print(f"  - Color change probability: {self.config['color_change_prob']:.1%}")
        print()

        # Process each file
        for csv_file in sorted(csv_files):
            output_file = output_path / csv_file.name
            self.augment_track(csv_file, output_file)

        print(f"\nAugmentation complete! {len(csv_files)} tracks processed.")


def main():
    parser = argparse.ArgumentParser(
        description="Apply data augmentation to Formula Student track files"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./data/generated_tracks",
        help="Directory containing original track CSV files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/augmented_tracks",
        help="Directory where augmented tracks will be saved",
    )
    parser.add_argument(
        "--removal_prob",
        type=float,
        default=0.1,
        help="Probability of removing a cone (default: 0.1)",
    )
    parser.add_argument(
        "--position_noise",
        type=float,
        default=0.1,
        help="Standard deviation of position noise in meters (default: 0.1 = 10cm)",
    )
    parser.add_argument(
        "--color_change_prob",
        type=float,
        default=0.05,
        help="Probability of changing cone color (default: 0.05)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (optional)",
    )

    args = parser.parse_args()

    # Create augmentation configuration
    config = {
        "seed": args.seed,
        "removal_prob": args.removal_prob,
        "position_noise_std": args.position_noise,
        "color_change_prob": args.color_change_prob,
    }

    # Create augmenter and process tracks
    augmenter = TrackAugmentation(config)
    augmenter.augment_directory(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
