import sys
import numpy as np
from pyqtgraph.Qt import QtWidgets
import pyqtgraph as pg
from measure import Measures


def generate_random_data(num_points=10000, noise_level=0.05):
    # Define survey area dimensions
    x_range = 5000  # 5km range
    y_range = 5000  # 5km range

    # Generate initial points with jittered/non-uniform distribution
    # We'll generate more points than requested and then sample
    extra_factor = 1.2  # Generate 20% more points than needed for sampling flexibility
    points_to_generate = int(num_points * extra_factor)

    # Create several "flight lines" with varying density
    num_lines = 200
    points_per_line = points_to_generate // num_lines

    x_list = []
    y_list = []

    # Create non-uniform survey lines
    for i in range(num_lines):
        # Determine line direction (some horizontal, some vertical, some diagonal)
        line_type = i % 4

        if line_type == 0:  # Horizontal line with jitter
            x_base = np.linspace(-x_range / 2, x_range / 2, points_per_line)
            y_base = np.ones(points_per_line) * (np.random.uniform(-y_range / 2, y_range / 2))

            # Add meandering to flight line
            y_meander = 100 * np.sin(x_base / 500) + 50 * np.sin(x_base / 120)
            y_base += y_meander

        elif line_type == 1:  # Vertical line with jitter
            y_base = np.linspace(-y_range / 2, y_range / 2, points_per_line)
            x_base = np.ones(points_per_line) * (np.random.uniform(-x_range / 2, x_range / 2))

            # Add meandering to flight line
            x_meander = 100 * np.sin(y_base / 500) + 50 * np.sin(y_base / 120)
            x_base += x_meander

        else:  # Diagonal or curved line
            t = np.linspace(0, 1, points_per_line)

            # Create a curved path
            x_start = np.random.uniform(-x_range / 2, 0)
            x_end = np.random.uniform(0, x_range / 2)
            y_start = np.random.uniform(-y_range / 2, 0)
            y_end = np.random.uniform(0, y_range / 2)

            # Bezier control point
            cx = np.random.uniform(-x_range / 2, x_range / 2)
            cy = np.random.uniform(-y_range / 2, y_range / 2)

            # Quadratic Bezier curve
            x_base = (1 - t) ** 2 * x_start + 2 * (1 - t) * t * cx + t ** 2 * x_end
            y_base = (1 - t) ** 2 * y_start + 2 * (1 - t) * t * cy + t ** 2 * y_end

        # Add non-uniform point spacing by jittering the points
        # Points are closer in "interesting" areas
        point_jitter = np.random.exponential(scale=2.0, size=points_per_line)
        jitter_scale = 10  # Controls the amount of non-uniformity

        # Apply variable density - first add small random jitter to all points
        x_jitter = np.random.normal(0, 5, points_per_line)  # Small random noise
        y_jitter = np.random.normal(0, 5, points_per_line)  # Small random noise

        # Then add non-uniform spacing effect
        x_base += x_jitter + jitter_scale * (point_jitter - np.mean(point_jitter))
        y_base += y_jitter + jitter_scale * (point_jitter - np.mean(point_jitter))

        # Add the points to our lists
        x_list.append(x_base)
        y_list.append(y_base)

    # Combine all the line points
    x = np.concatenate(x_list)
    y = np.concatenate(y_list)

    # Clip points to the survey area
    valid_idx = (x >= -x_range / 2) & (x <= x_range / 2) & (y >= -y_range / 2) & (y <= y_range / 2)
    x = x[valid_idx]
    y = y[valid_idx]

    # Randomly sample to get exactly num_points
    if len(x) > num_points:
        idx = np.random.choice(len(x), size=num_points, replace=False)
        x = x[idx]
        y = y[idx]

    # Generate terrain-like elevation using multiple frequency components
    # Start with a base elevation
    z_base = 100 + 50 * np.sin(x / 1000) * np.cos(y / 1200)

    # Add medium scale features
    z_medium = 20 * np.sin(x / 400) * np.sin(y / 450)

    # Add small scale features
    z_small = 10 * np.sin(x / 100) * np.cos(y / 120)

    # Add micro features
    z_micro = 5 * np.sin(x / 30) * np.cos(y / 35)

    # Add random noise
    z_noise = 2 * np.random.normal(0, 1, size=len(x))

    # Combine all elevation components
    z_synthetic = z_base + z_medium + z_small + z_micro

    # This is the "perfect" synthetic model
    synthetic = z_synthetic

    # Create "measured" values by adding noise
    # Instead of using gradient on a grid, calculate local slope using neighbors
    # We'll use a simpler approximation for slope since we have irregular points

    # Calculate approximate slope using the rate of change of z with respect to x and y
    # For irregular points, we can use the derivatives of our analytical functions
    slope_x = (50 / 1000) * np.cos(x / 1000) * np.cos(y / 1200) + \
              (20 / 400) * np.cos(x / 400) * np.sin(y / 450) + \
              (10 / 100) * np.cos(x / 100) * np.cos(y / 120) + \
              (5 / 30) * np.cos(x / 30) * np.cos(y / 35)

    slope_y = (-50 / 1200) * np.sin(x / 1000) * np.sin(y / 1200) + \
              (20 / 450) * np.sin(x / 400) * np.cos(y / 450) + \
              (-10 / 120) * np.sin(x / 100) * np.sin(y / 120) + \
              (-5 / 35) * np.sin(x / 30) * np.sin(y / 35)

    # Calculate slope magnitude
    slope_magnitude = np.sqrt(slope_x ** 2 + slope_y ** 2)

    # Scale the noise by the slope and add a baseline noise
    noise = noise_level * (1 + 2 * slope_magnitude) * np.random.normal(0, 1, size=len(x))

    # Generate the measured values with noise
    measures = synthetic + noise

    # Actual elevation with noise (this is what we'll display as z)
    z = measures

    # Calculate error between synthetic and measured
    error = measures - synthetic

    # Calculate a quality score based on error
    # Higher score for lower error
    normalized_error = np.abs(error) / np.max(np.abs(error) + 0.001)
    score = 100 * np.exp(-5 * normalized_error)

    # Color parameter - we'll use slope as the coloring parameter
    c = slope_magnitude

    # Reduce the dataset if it's too large for interactive rendering
    # Keep every nth point if more than the specified limit
    max_plot_points = 100000  # Maximum number of points to plot
    if len(x) > max_plot_points:
        step = len(x) // max_plot_points
        x = x[::step]
        y = y[::step]
        z = z[::step]
        c = c[::step]
        measures = measures[::step]
        synthetic = synthetic[::step]
        error = error[::step]
        score = score[::step]
        print(f"Reduced dataset for plotting: {len(x)} points (from {num_points})")
    else:
        print(f"Using full dataset: {len(x)} points")

    return x, y, z, c


def main():
    # Initialize PyQtGraph
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)

    # Create a window with a GraphicsLayoutWidget
    win = pg.GraphicsLayoutWidget(show=True, title="Measures Test Example")
    win.resize(800, 600)

    # Generate random data
    x, y, z, c = generate_random_data()

    # Create a plot area for the data points
    plot_area = win.addPlot(title="Magnetic Survey Visualization")
    plot_area.setLabel('bottom', 'X', 'm')
    plot_area.setLabel('left', 'Y', 'm')

    # Create Measures object
    measures_plot = Measures(
        x=x,
        y=y,
        data={"z": z, "c": c},
        fig=win,
        ax=plot_area,
        name="Field Intensity",
        unit="nT"
    )

    # Display instructions
    print("Interactive Visualization Instructions:")
    print("- Hover over points to see detailed information")
    print("- Right-click and drag to select points for a Slice view")
    print("- Adjust color scale in the histogram")

    # Start Qt event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()