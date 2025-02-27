import pyqtgraph
import numpy
import utils
import scipy.interpolate

# Check if bezier module is available
try:
    import bezier
except ImportError:
    print("Warning: bezier module not available, bezier smoothing will not work")


class Slice:
    """A class to visualize a slice through data as a single plot with line-linked data"""

    def __init__(self, data):
        self.data_points = data

        # Get coordinates
        self.x = data.get('x', None)
        self.y = data.get('y', None)

        # Ensure we have valid x and y coordinates
        if self.x is None or self.y is None:
            raise ValueError("Slice requires 'x' and 'y' coordinates in data")

        self.a = utils.curviligne_abs(numpy.array([self.x, self.y]).T)

        # Add abscissa to data
        self.data_points['a'] = self.a

        # Create the plot window
        self.ax = pyqtgraph.plot(title="Slice View")
        self.legend = self.ax.addLegend()

        # Store plots for access later
        self.plots = {}
        self.smooth_plots = {}

        # Define a list of colors to cycle through for data fields
        self.color_cycle = ['r', 'g', 'b', 'c', 'm', 'y', 'w']

        # Create plots for all data fields
        self.create_plots()

        # Set up grid
        self.ax.showGrid(x=True, y=True, alpha=1)

    def create_plots(self):
        """Create plots for all data fields with lines instead of scatter points"""
        # Skip these fields as they're used for coordinates
        skip_fields = ['x', 'y', 'a']

        # Import the smoothing functions from utils if they don't exist in the global namespace
        # This ensures we have access to the smoothing methods
        for smooth_method in ['bezier_pipe', 'akima_pipe', 'spline_pipe', 'spline_smooth_pipe']:
            if smooth_method not in globals() and hasattr(utils, smooth_method):
                globals()[smooth_method] = getattr(utils, smooth_method)

        # Plot all numeric data fields
        color_index = 0
        for key, values in self.data_points.items():
            if key in skip_fields:
                continue

            if isinstance(values, numpy.ndarray) and values.dtype.kind in 'iuf':
                try:
                    # Get color for this field (cyclic)
                    color = self.color_cycle[color_index % len(self.color_cycle)]
                    color_index += 1

                    # Try to create smoothed line if needed
                    try:
                        # Use smooth_pipe function with a different method
                        if hasattr(utils, 'smooth_pipe'):
                            # Try different smoothing methods in order of preference
                            smoothing_methods = ["akima", "spline", "bezier", "spline_smooth"]

                            for method in smoothing_methods:
                                try:
                                    a_smooth, values_smooth = utils.smooth_pipe(
                                        numpy.array([self.a, values]).T, method=method
                                    ).T
                                    print(f"Successfully used '{method}' smoothing for '{key}'")
                                    break
                                except Exception as e:
                                    print(f"Method '{method}' failed for '{key}': {e}")
                                    continue

                            # Create smooth plot line
                            smooth_line = self.ax.plot(
                                x=a_smooth,
                                y=values_smooth,
                                pen=pyqtgraph.mkPen(color, width=2, style=pyqtgraph.QtCore.Qt.DashLine),
                                name=f"{key} Smooth"
                            )
                            self.smooth_plots[key] = smooth_line
                            print(f"Created smooth line for '{key}'")
                    except Exception as e:
                        print(f"Warning: Could not create smooth line for '{key}': {e}")

                except Exception as e:
                    print(f"Warning: Could not plot '{key}': {e}")

    def update_data(self, data):
        """Update the plot with new data"""
        self.data_points = data

        # Update coordinates
        self.x = data.get('x', None)
        self.y = data.get('y', None)

        if self.x is None or self.y is None:
            raise ValueError("Slice requires 'x' and 'y' coordinates in data")

        self.a = utils.curviligne_abs(numpy.array([self.x, self.y]).T)
        self.data_points['a'] = self.a

        # Clear existing plots
        self.ax.clear()
        self.legend = self.ax.addLegend()

        # Recreate plots
        self.plots = {}
        self.smooth_plots = {}
        self.create_plots()

        # Update grid
        self.ax.showGrid(x=True, y=True, alpha=1)
