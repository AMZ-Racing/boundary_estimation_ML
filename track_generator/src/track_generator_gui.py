import sys
import os
import re
import argparse
import random
import math
from PyQt5.QtCore import QPointF
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QSpinBox,
    QDoubleSpinBox,
    QGroupBox,
    QFormLayout,
    QPushButton,
    QSizePolicy,
    QHBoxLayout,
    QLabel,
    QFileDialog,
    QSplitter,
)
from PyQt5.QtGui import QBrush, QPainter, QPen, QColor
from track_generator import TrackGenerator

# Replace ROS Plugin with a normal PyQt5 application
constant_ranges = {
    "seed": {"min": 0, "max": 10000},
    "min_corner_radius": {"min": 2, "max": 50},
    "starting_straight_length": {"min": 0, "max": 12},
    "cone_spacing_bias": {"min": 0, "max": 2},
    "length": {"max": 10000},
    "margin": {"min": 0},
    "track_width": {"min": 0},
    "min_cone_spacing": {"min": 0.1},
    "max_cone_spacing": {"max": 10},
    "starting_cone_spacing": {"min": 0},
}

settings = {
    "seed": random.randint(constant_ranges["seed"]["min"], constant_ranges["seed"]["max"]),
    "min_corner_radius": 3,
    "length": 500,
    "margin": 0,
    "starting_straight_length": 6,
    "min_cone_spacing": 3 * math.pi / 16,
    "max_cone_spacing": 5,
    "track_width": 3,
    "cone_spacing_bias": 0.5,
    "starting_cone_spacing": 0.5,
}

ranges = {
    "min_corner_radius": {
        "length": {"min": 2 * math.pi},
        "track_width": {"max": "2*(settings['min_corner_radius'] - settings['margin'])"},
        "margin": {"max": "settings['min_corner_radius'] - settings['track_width']/2"},
    },
    "margin": {
        "track_width": {"max": "2*(settings['min_corner_radius'] - settings['margin'] - 0.01)"}
    },
    "track_width": {
        "margin": {"max": "settings['min_corner_radius'] - settings['track_width']/2 - 0.01"},
    },
    "min_cone_spacing": {"max_cone_spacing": {"min": 1}},
    "max_cone_spacing": {
        "starting_cone_spacing": {"max": 0.5},
        "min_cone_spacing": {"max": 1},
    },
}


def log_scaling(control, scaling):
    control.setSingleStep(control.value() * scaling)

    def adjust_step_size(value):
        control.setSingleStep(value * scaling)

    control.valueChanged.connect(adjust_step_size)
    return control


# Utility function to get the next incremental filename
def get_incremental_filename(directory, base_name="track", extension="csv"):
    # Use regex to extract numeric part from the filenames
    files = [
        f for f in os.listdir(directory) if f.startswith(base_name) and f.endswith(f".{extension}")
    ]

    # Sort files based on the numeric part extracted from the filename
    def extract_number(filename):
        match = re.search(r"_(\d+)\.", filename)  # Look for _number.
        return int(match.group(1)) if match else 0

    files.sort(key=extract_number, reverse=True)

    if not files:
        return os.path.join(directory, f"{base_name}_1.{extension}")

    last_file = files[0]
    last_number = extract_number(last_file)
    next_number = last_number + 1

    return os.path.join(directory, f"{base_name}_{next_number}.{extension}")


# Batch generation function


def generate_tracks_in_batch(directory, num_tracks):
    for _ in range(num_tracks):
        filename = get_incremental_filename(directory)
        # generate random settings
        settings["seed"] = random.randint(
            constant_ranges["seed"]["min"], constant_ranges["seed"]["max"]
        )
        #! TODO add more randomization in the track parameters.
        #         default_cfg = { # Default settings for track generation -> possible TODO -> make these randomizable
        #     "seed": random.random(),
        #     "min_corner_radius": 3,
        #     "max_frequency": 7,
        #     "amplitude": 1 / 3,
        #     "check_self_intersection": True,
        #     "starting_amplitude": 0.4,
        #     "rel_accuracy": 0.005,
        #     "margin": 0,
        #     "starting_straight_length": 6,
        #     "starting_straight_downsample": 2,
        #     "min_cone_spacing": 3 * math.pi / 16,
        #     "max_cone_spacing": 5,
        #     "track_width": 3,
        #     "cone_spacing_bias": 0.5,
        #     "starting_cone_spacing": 0.5,
        # }
        TrackGenerator.write_to_csv(filename, *TrackGenerator(settings)(), overwrite=True)
        print(f"Track saved as: {filename}")


class TrackControls(QWidget):
    """Widget containing controls for track generation."""

    def __init__(self, parent_gui, save_dir):
        super(TrackControls, self).__init__()
        self.parent_gui = parent_gui  # Store a reference to the parent GUI
        self.save_dir = save_dir  # Store the save directory
        print(self.save_dir)
        self.setMinimumSize(100, 480)

        layout = QVBoxLayout(self)

        controls = {
            "seed": QSpinBox(),
            "min_corner_radius": QDoubleSpinBox(),
            "length": QDoubleSpinBox(),
            "margin": QDoubleSpinBox(),
            "starting_straight_length": QDoubleSpinBox(),
            "min_cone_spacing": QDoubleSpinBox(),
            "max_cone_spacing": QDoubleSpinBox(),
            "track_width": QDoubleSpinBox(),
            "cone_spacing_bias": QDoubleSpinBox(),
            "starting_cone_spacing": QDoubleSpinBox(),
        }

        for ctrl in constant_ranges:
            if "min" in constant_ranges[ctrl]:
                controls[ctrl].setMinimum(constant_ranges[ctrl]["min"])
            if "max" in constant_ranges[ctrl]:
                controls[ctrl].setMaximum(constant_ranges[ctrl]["max"])

        def update_dependant_ranges(src):
            for dst in ranges[src]:
                if "min" in ranges[src][dst]:
                    new_min = (
                        eval(ranges[src][dst]["min"])
                        if isinstance(ranges[src][dst]["min"], str)
                        else settings[src] * ranges[src][dst]["min"]
                    )
                    controls[dst].setMinimum(new_min)
                if "max" in ranges[src][dst]:
                    new_max = (
                        eval(ranges[src][dst]["max"])
                        if isinstance(ranges[src][dst]["max"], str)
                        else settings[src] * ranges[src][dst]["max"]
                    )
                    controls[dst].setMaximum(new_max)

        for src in ranges:
            update_dependant_ranges(src)

        def on_value_changed(ctrl_name):
            def callback(value):
                settings[ctrl_name] = value
                if ctrl_name in ranges:
                    update_dependant_ranges(ctrl_name)
                self.parent_gui.redraw_track()

            return callback

        for ctrl in controls:
            controls[ctrl].setValue(settings[ctrl])
            controls[ctrl].valueChanged.connect(on_value_changed(ctrl))

        # Add some controls and default settings
        controls["seed"].setValue(random.randint(0, 10000))
        controls["min_corner_radius"].setValue(3)
        controls["length"].setValue(500)
        controls["margin"].setValue(0)
        controls["starting_straight_length"].setValue(6)
        controls["min_cone_spacing"].setValue(4.0)
        controls["max_cone_spacing"].setValue(5)
        controls["track_width"].setValue(3)
        controls["cone_spacing_bias"].setValue(0.5)
        controls["starting_cone_spacing"].setValue(0.5)

        # Set ranges for each control
        controls["seed"].setMinimum(0)
        controls["seed"].setMaximum(10000)
        controls["min_corner_radius"].setMinimum(2)
        controls["min_corner_radius"].setMaximum(50)

        generation_group = QGroupBox()
        generation_group.setTitle("Track Generation")

        group = QFormLayout(generation_group)
        randomize_seed_btn = QPushButton("Randomize")
        randomize_seed_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        def randomize_seed():
            settings["seed"] = random.randint(
                constant_ranges["seed"]["min"], constant_ranges["seed"]["max"]
            )
            controls["seed"].setValue(settings["seed"])
            self.parent_gui.redraw_track()

        randomize_seed_btn.clicked.connect(randomize_seed)
        g = QWidget()
        seed_layout = QHBoxLayout(g)
        seed_layout.setContentsMargins(0, 0, 0, 0)
        seed_layout.addWidget(controls["seed"])
        seed_layout.addWidget(randomize_seed_btn)

        group.addRow(QLabel("Seed"), g)
        group.addRow(QLabel("Length"), controls["length"])
        group.addRow(QLabel("Min Turn Radius"), controls["min_corner_radius"])
        group.addRow(QLabel("Margin"), controls["margin"])

        cone_placement_group = QGroupBox()
        cone_placement_group.setTitle("Cone Placement")
        group = QFormLayout(cone_placement_group)

        group.addRow(QLabel("Starting Straight Length"), controls["starting_straight_length"])
        group.addRow(QLabel("Starting Cone Spacing"), controls["starting_cone_spacing"])
        group.addRow(QLabel("Cone Spacing Min"), controls["min_cone_spacing"])
        group.addRow(QLabel("Cone Spacing Max"), controls["max_cone_spacing"])
        group.addRow(QLabel("Cone Spacing Bias"), controls["cone_spacing_bias"])
        group.addRow(QLabel("Track Width"), controls["track_width"])

        save_btn = QPushButton("Save")

        def save_track():
            filename = get_incremental_filename(self.save_dir, base_name="track", extension="csv")
            TrackGenerator.write_to_csv(filename, *TrackGenerator(settings)(), overwrite=True)
            print(f"Track saved to {filename}")

        save_btn.clicked.connect(save_track)

        # # Connect all controls to the redraw function
        # for control in controls.values():
        #   control.valueChanged.connect(self.on_control_update(control))

        layout.addWidget(generation_group)
        layout.addWidget(cone_placement_group)
        layout.addWidget(save_btn)


class TrackDisplay(QWidget):
    left_cone_fill = QBrush(QColor(49, 49, 226))
    right_cone_fill = QBrush(QColor(226, 220, 49))
    start_cone_fill = QBrush(QColor("#e28a31"))

    def __init__(self):
        super(TrackDisplay, self).__init__()
        self.resize(200, 200)
        self.setMinimumSize(480, 480)
        self.start_cones, self.left_cones, self.right_cones = TrackGenerator(settings)()

    def regenerate_path(self):
        self.start_cones, self.left_cones, self.right_cones = TrackGenerator(settings)()
        self.repaint(0, 0, -1, -1)
        self.update()

    def paintEvent(self, e):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        radius = 4
        stroke_width = 1
        margin = 0.1

        max_x = (
            max(
                max(self.left_cones.real),
                max(self.right_cones.real),
                max(self.start_cones.real),
            )
            + margin
        )
        min_x = (
            min(
                min(self.left_cones.real),
                min(self.right_cones.real),
                min(self.start_cones.real),
            )
            - margin
        )
        max_y = (
            max(
                max(self.left_cones.imag),
                max(self.right_cones.imag),
                max(self.start_cones.imag),
            )
            + margin
        )
        min_y = (
            min(
                min(self.left_cones.imag),
                min(self.right_cones.imag),
                min(self.start_cones.imag),
            )
            - margin
        )

        w, h = self.width(), self.height()

        def scale(p: QPointF) -> QPointF:
            return QPointF(
                (p.x() - min_x) / (max_x - min_x) * w,
                h - (p.y() - min_y) / (max_y - min_y) * h,
            )

        def draw_cones(cones, fill):
            painter.setPen(QPen(QColor(0, 0, 0), stroke_width))
            painter.setBrush(fill)
            for p in map(scale, map(lambda c: QPointF(c.real, c.imag), cones)):
                painter.drawEllipse(p, radius, radius)

        draw_cones(self.start_cones, self.start_cone_fill)
        draw_cones(self.left_cones, self.left_cone_fill)
        draw_cones(self.right_cones, self.right_cone_fill)


class EUFSTracksGUI(QWidget):
    """Main window for track visualization"""

    def __init__(self, save_dir):
        super().__init__()

        self.setWindowTitle("EUFS Tracks GUI")
        # self.setGeometry(100, 100, 800, 600)
        # Open in full screen
        self.showMaximized()

        layout = QVBoxLayout(self)

        splitter = QSplitter()
        layout.addWidget(splitter)

        self.track_display = TrackDisplay()

        splitter.addWidget(self.track_display)

        self.track_controls = TrackControls(self, save_dir)  # Pass a reference to the GUI
        splitter.addWidget(self.track_controls)

        splitter.setSizes([600, 200])

    def redraw_track(self):
        self.track_display.regenerate_path()


def main():
    parser = argparse.ArgumentParser(description="Track Generator Script")
    parser.add_argument("--batch", type=int, help="Generate multiple tracks in batch mode")
    parser.add_argument(
        "--dir",
        type=str,
        default=os.path.join(os.getcwd(), "data", "generated_tracks"),
        help="Directory to save tracks",
    )

    args = parser.parse_args()

    # Ensure the directory exists
    if not os.path.exists(args.dir):
        os.makedirs(args.dir)

    if args.batch:
        # Batch mode: Generate multiple tracks
        print(f"Generating {args.batch} tracks in batch mode...")
        generate_tracks_in_batch(args.dir, args.batch)
    else:
        app = QApplication(sys.argv)
        gui = EUFSTracksGUI(args.dir)
        gui.show()
        sys.exit(app.exec_())


if __name__ == "__main__":
    main()
