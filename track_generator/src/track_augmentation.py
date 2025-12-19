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
            "false_positive_prob": 0.01,  # Probability of adding false positive per cone
            "false_positive_distance": (2.0, 8.0),  # Distance range from track (min, max) in meters
            "start_finish_false_positives": 6,  # Number of false positives around start/finish
        }
        self.config = {**default_cfg, **(config or {})}
        self.rng = random.Random(self.config["seed"])

    def augment_cone(self, tag, x, y):
        """
        Apply augmentation to a single cone.
        
        Returns:
        - (aug_tag, aug_x, aug_y, detected) tuple with augmented values and detection flag
        """
        # Check if cone should be marked as detected
        detected = self.rng.random() >= self.config["removal_prob"]

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

        return (aug_tag, aug_x, aug_y, detected)

    def generate_false_positives(self, track_df):
        """
        Generate false positive cones around the track.
        
        Args:
            track_df: DataFrame with original track cones (columns: tag, x, y)
        
        Returns:
            List of false positive dicts with keys: tag, x, y, aug_tag, aug_x, aug_y, detected
        """
        false_positives = []
        
        # Calculate number of false positives to add
        n_cones = len(track_df)
        n_false_positives = int(n_cones * self.config["false_positive_prob"])
        
        if n_false_positives == 0:
            return false_positives
        
        # Calculate track center (centroid)
        center_x = track_df["x"].mean()
        center_y = track_df["y"].mean()
        
        # Convert to numpy arrays for vectorized operations
        cone_positions = track_df[["x", "y"]].values
        
        # Sample random cones as reference points
        reference_indices = self.rng.choices(range(n_cones), k=n_false_positives)
        
        for idx in reference_indices:
            ref_cone = track_df.iloc[idx]
            ref_x, ref_y = ref_cone["x"], ref_cone["y"]
            
            # Calculate direction from center to reference cone (outward direction)
            dx = ref_x - center_x
            dy = ref_y - center_y
            distance_from_center = np.sqrt(dx**2 + dy**2)
            
            # Normalize direction vector (unit vector pointing outward)
            if distance_from_center > 0:
                dir_x = dx / distance_from_center
                dir_y = dy / distance_from_center
            else:
                # If cone is at center, use random direction
                angle = self.rng.uniform(0, 2 * np.pi)
                dir_x = np.cos(angle)
                dir_y = np.sin(angle)
            
            # Calculate distance to nearest neighbor cone in the outward direction
            # This ensures false positives are placed beyond the track boundary
            min_neighbor_dist = 0
            for other_x, other_y in cone_positions:
                if other_x == ref_x and other_y == ref_y:
                    continue
                # Check if neighbor is in the outward direction (dot product > 0)
                to_neighbor_x = other_x - ref_x
                to_neighbor_y = other_y - ref_y
                dot_product = to_neighbor_x * dir_x + to_neighbor_y * dir_y
                if dot_product > 0:  # In outward direction
                    neighbor_dist = np.sqrt(to_neighbor_x**2 + to_neighbor_y**2)
                    if min_neighbor_dist == 0 or neighbor_dist < min_neighbor_dist:
                        min_neighbor_dist = neighbor_dist
            
            # Generate random distance, ensuring it's beyond the nearest neighbor
            min_dist, max_dist = self.config["false_positive_distance"]
            # Add track width buffer to ensure we're outside
            safety_margin = 3.0  # meters (typical track width)
            base_distance = max(min_dist, min_neighbor_dist + safety_margin)
            distance = self.rng.uniform(base_distance, base_distance + max_dist - min_dist)
            
            # Calculate false positive position (always outward from reference cone)
            fp_x = ref_x + distance * dir_x
            fp_y = ref_y + distance * dir_y
            
            # Random color for false positive
            fp_tag = self.rng.choice(self.config["available_colors"])
            
            false_positives.append({
                "tag": "false_positive",
                "x": fp_x,
                "y": fp_y,
                "aug_tag": fp_tag,
                "aug_x": fp_x,
                "aug_y": fp_y,
                "detected": True,  # False positives are marked as detected (visible)
            })
        
        # Add false positives around start/finish line
        n_sf_fp = self.config.get("start_finish_false_positives", 0)
        if n_sf_fp > 0:
            # Randomize number of false positives (0 to n_sf_fp)
            n_sf_fp = self.rng.randint(0, n_sf_fp)
            
        if n_sf_fp > 0:
            # Find start/finish cones (orange/big_orange)
            orange_cones = track_df[track_df["tag"].isin(["orange", "big_orange"])]
            blue_cones = track_df[track_df["tag"] == "blue"]
            yellow_cones = track_df[track_df["tag"] == "yellow"]
            
            if len(orange_cones) >= 2 and (len(blue_cones) > 0 or len(yellow_cones) > 0):
                # Get first two orange cones (start line)
                start_cones = orange_cones.iloc[:2]
                
                # Calculate midpoint and track direction
                mid_x = start_cones["x"].mean()
                mid_y = start_cones["y"].mean()
                
                # Find nearest blue and yellow cones to determine track direction
                if len(blue_cones) > 0:
                    distances_blue = np.sqrt((blue_cones["x"] - mid_x)**2 + (blue_cones["y"] - mid_y)**2)
                    nearest_blue = blue_cones.iloc[distances_blue.argmin()]
                else:
                    nearest_blue = None
                
                if len(yellow_cones) > 0:
                    distances_yellow = np.sqrt((yellow_cones["x"] - mid_x)**2 + (yellow_cones["y"] - mid_y)**2)
                    nearest_yellow = yellow_cones.iloc[distances_yellow.argmin()]
                else:
                    nearest_yellow = None
                
                # Calculate track direction (tangent)
                if nearest_blue is not None and nearest_yellow is not None:
                    # Track direction is average of directions to nearest blue and yellow
                    track_dx = (nearest_blue["x"] + nearest_yellow["x"]) / 2 - mid_x
                    track_dy = (nearest_blue["y"] + nearest_yellow["y"]) / 2 - mid_y
                elif nearest_blue is not None:
                    track_dx = nearest_blue["x"] - mid_x
                    track_dy = nearest_blue["y"] - mid_y
                elif nearest_yellow is not None:
                    track_dx = nearest_yellow["x"] - mid_x
                    track_dy = nearest_yellow["y"] - mid_y
                else:
                    track_dx = 1
                    track_dy = 0
                
                # Normalize track direction
                track_length = np.sqrt(track_dx**2 + track_dy**2)
                if track_length > 0:
                    track_dx /= track_length
                    track_dy /= track_length
                
                # Calculate perpendicular direction (normal to track)
                # Rotate 90 degrees counterclockwise
                normal_dx = -track_dy
                normal_dy = track_dx
                
                # For each start/finish cone, add false positives
                for _, cone in start_cones.iterrows():
                    cone_x, cone_y = cone["x"], cone["y"]
                    
                    # Determine if this is left (blue) or right (yellow) side
                    # Using cross product to determine which side
                    to_cone_x = cone_x - mid_x
                    to_cone_y = cone_y - mid_y
                    cross = track_dx * to_cone_y - track_dy * to_cone_x
                    
                    # Generate false positives for this cone
                    n_per_cone = n_sf_fp // 2
                    for _ in range(n_per_cone):
                        # Distance along normal (perpendicular to track, outward)
                        normal_dist = self.rng.uniform(2.0, 5.0)
                        
                        # Distance along tangent (along track direction)
                        tangent_dist = self.rng.uniform(-3.0, 3.0)
                        
                        # If on left side (cross > 0), use positive normal
                        # If on right side (cross < 0), use negative normal
                        if cross > 0:  # Left side (blue)
                            fp_x = cone_x + normal_dist * normal_dx + tangent_dist * track_dx
                            fp_y = cone_y + normal_dist * normal_dy + tangent_dist * track_dy
                        else:  # Right side (yellow)
                            fp_x = cone_x - normal_dist * normal_dx + tangent_dist * track_dx
                            fp_y = cone_y - normal_dist * normal_dy + tangent_dist * track_dy
                        
                        fp_tag = self.rng.choice(self.config["available_colors"])
                        
                        false_positives.append({
                            "tag": "false_positive",
                            "x": fp_x,
                            "y": fp_y,
                            "aug_tag": fp_tag,
                            "aug_x": fp_x,
                            "aug_y": fp_y,
                            "detected": True,
                        })
        
        return false_positives

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
            aug_tag, aug_x, aug_y, detected = self.augment_cone(tag, x, y)

            # Add all cones, including removed ones (marked with detected=False)
            augmented_rows.append(
                {
                    "tag": tag,
                    "x": x,
                    "y": y,
                    "aug_tag": aug_tag,
                    "aug_x": aug_x,
                    "aug_y": aug_y,
                    "detected": detected,
                }
            )

        # Add false positives
        false_positives = self.generate_false_positives(df)
        augmented_rows.extend(false_positives)

        # Create DataFrame and save
        augmented_df = pd.DataFrame(augmented_rows)
        augmented_df.to_csv(output_csv_path, index=False)

        # Also save a numpy (.npy) file with the same stem next to the CSV
        out_path = Path(output_csv_path)
        if len(augmented_df) > 0:
            dtype = np.dtype([
                ("tag", "U16"),
                ("x", float),
                ("y", float),
                ("aug_tag", "U16"),
                ("aug_x", float),
                ("aug_y", float),
                ("detected", bool),
            ])
            arr = np.zeros(len(augmented_df), dtype=dtype)
            arr["tag"] = augmented_df["tag"].astype(str).values
            arr["x"] = augmented_df["x"].astype(float).values
            arr["y"] = augmented_df["y"].astype(float).values
            arr["aug_tag"] = augmented_df["aug_tag"].astype(str).values
            arr["aug_x"] = augmented_df["aug_x"].astype(float).values
            arr["aug_y"] = augmented_df["aug_y"].astype(float).values
            arr["detected"] = augmented_df["detected"].astype(bool).values

            npy_path = out_path.with_suffix(".npy")
            np.save(npy_path, arr)

        num_detected = augmented_df["detected"].sum()
        num_removed = (augmented_df["tag"] != "false_positive").sum() - num_detected
        num_false_positives = (augmented_df["tag"] == "false_positive").sum()
        print(
            f"Augmented: {input_csv_path.name} -> {output_csv_path.name} "
            f"({len(df)} cones: {num_detected} detected, {num_removed} removed, {num_false_positives} false positives)"
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
        print(f"  - False positive probability: {self.config['false_positive_prob']:.1%}")
        print(f"  - False positive distance: {self.config['false_positive_distance'][0]:.1f}-{self.config['false_positive_distance'][1]:.1f}m")
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
