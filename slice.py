import pyqtgraph
import numpy
import utils


class Slice:
    """A class to visualize a slice through data"""

    def __init__(self, data):
        self.data_points = data

        # Get coordinates
        self.x = data.get('x', None)
        self.y = data.get('y', None)
        self.z = data.get('z', numpy.zeros_like(self.x) if self.x is not None else None)

        # Ensure we have valid x and y coordinates
        if self.x is None or self.y is None:
            raise ValueError("Slice requires 'x' and 'y' coordinates in data")

        # Calculate curvilinear abscissa if we have coordinates
        if self.z is None:
            # Use flat z=0 if not provided
            self.z = numpy.zeros_like(self.x)

        # Calculate distance along the path
        self.a = utils.curviligne_abs(numpy.array([self.x, self.y, self.z]).T)

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
        """Create plots for all data fields"""
        # Skip these fields as they're used for coordinates
        skip_fields = ['x', 'y', 'z', 'a']

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

                    # Create scatter plot for raw data
                    scatter = pyqtgraph.ScatterPlotItem(
                        x=self.a,
                        y=values,
                        pen=pyqtgraph.mkPen(color),
                        brush=pyqtgraph.mkBrush(color),
                        name=f"{key}"
                    )
                    self.ax.addItem(scatter)
                    self.plots[key] = scatter

                    # Try to create smoothed line
                    try:
                        # Use smooth_pipe function if available
                        if 'smooth_pipe' in globals():
                            a_smooth, values_smooth = utils.smooth_pipe(
                                numpy.array([self.a, values]).T, method="convolve_fft"
                            ).T

                            # Create plot line
                            line = self.ax.plot(
                                x=a_smooth,
                                y=values_smooth,
                                pen=color,
                                name=f"{key} Smooth"
                            )
                            self.smooth_plots[key] = line
                            print(f"Created smooth line for '{key}'")
                    except Exception as e:
                        print(f"Warning: Could not create smooth line for '{key}': {e}")

                except Exception as e:
                    print(f"Warning: Could not plot '{key}': {e}")



