
import pyqtgraph
import numpy
from pyqtgraph.Qt import QtCore

import slice


class Measures(pyqtgraph.ScatterPlotItem):
    def __init__(self, x, y, data: dict = None, ax=None, fig=None, name="", unit=""):
        self.x = x
        self.y = y
        self.data_points = {} if data is None else data.copy()

        # Add x and y to data
        self.data_points["x"] = x
        self.data_points["y"] = y

        # Get first numeric data key for coloring (excluding x and y)
        self.color_key = None
        for key, value in self.data_points.items():
            if key not in ['x', 'y'] and isinstance(value, numpy.ndarray) and value.dtype.kind in 'iuf':
                self.color_key = key
                print(f"Using '{key}' for coloring")
                break

        # If no data fields, use zeros
        if self.color_key is None:
            self.color_key = 'x'

        # Initialize plot elements
        self.fig = fig
        self.ax = ax if ax is not None else self.fig.addPlot()
        self.name = lambda: name
        self.unit = unit
        self.lines = []
        self.lines_values = []
        self.slice = []

        # Data for point optimization
        self.all_points_idx = numpy.arange(len(self.x))
        self.visible_points_idx = self.all_points_idx  # Initially all points are visible
        self.min_distance = 5.0  # Minimum pixel distance between points

        # Set up color mapping
        self.cb = pyqtgraph.HistogramLUTItem(orientation='horizontal', gradientPosition=None, levelMode="mono")
        self.colormap = pyqtgraph.colormap.get("Turbo")

        # Set up histogram bins
        self.set_bins()

        # Initialize scatter plot
        super().__init__(
            x=self.x,
            y=self.y,
            data=numpy.array(range(len(self.x))),
            tip=lambda x, y, data: self.tooltip(x, y, data),
            pen=None,
            brush=[pyqtgraph.mkBrush(color) for color in self.colormap.mapToQColor(self.data_points[self.color_key])],
            hoverable=True,
            hoverSymbol="s",
            hoverSize=10,
            hoverPen=pyqtgraph.mkPen("w", width=1)
        )

        # Complete initialization
        self.init()

    def set_bins(self):
        """Set up histogram bins for the color data"""
        color_data = self.data_points[self.color_key]

        # Try different decimal places to get a reasonable number of bins
        for decimal in range(10):
            try:
                n = len(numpy.unique(numpy.round(color_data, decimal)))
                if n > 100:
                    y_hist, x_hist = numpy.histogram(color_data, bins=n)
                    self.cb.plot.setData(x_hist, y_hist, stepMode="center")
                    break
            except Exception as e:
                print(f"Warning: Error setting bins at decimal {decimal}: {e}")
                continue

    def init(self):
        """Initialize plot elements"""
        color_data = self.data_points[self.color_key]

        # Add percentile lines to histogram
        self.cb.vb.addItem(pyqtgraph.InfiniteLine(pos=numpy.nanpercentile(color_data, 50), angle=90,
                                                  pen=pyqtgraph.mkPen('g'), label="50%"))
        self.cb.vb.addItem(pyqtgraph.InfiniteLine(pos=numpy.nanpercentile(color_data, 25), angle=90,
                                                  pen=pyqtgraph.mkPen('r'), label="25%"))
        self.cb.vb.addItem(pyqtgraph.InfiniteLine(pos=numpy.nanpercentile(color_data, 75), angle=90,
                                                  pen=pyqtgraph.mkPen('r'), label="75%"))
        self.cb.vb.addItem(pyqtgraph.InfiniteLine(pos=numpy.nanpercentile(color_data, 5), angle=90,
                                                  pen=pyqtgraph.mkPen('r'), label="5%"))
        self.cb.vb.addItem(pyqtgraph.InfiniteLine(pos=numpy.nanpercentile(color_data, 95), angle=90,
                                                  pen=pyqtgraph.mkPen('r'), label="95%"))

        # Set up labels and color map
        self.cb.axis.setLabel(f"{self.name()} ({self.color_key})", self.unit)
        self.cb.gradient.setColorMap(self.colormap)

        # Set initial levels
        try:
            min_val = numpy.nanmin(color_data)
            max_val = numpy.nanmax(color_data)
            self.cb.setLevels(min_val, max_val)
        except (ValueError, TypeError) as e:
            print(f"Warning: Could not set initial levels: {e}")

        # Update scatter plot colors
        self.repaint()

        # Set to 5-95% percentile for better visualization
        try:
            p05 = numpy.nanpercentile(color_data, 5)
            p95 = numpy.nanpercentile(color_data, 95)
            self.cb.setLevels(p05, p95)
        except (ValueError, TypeError) as e:
            print(f"Warning: Could not set percentile levels: {e}")

        # Connect signals
        self.cb.sigLevelChangeFinished.connect(self.repaint)
        self.sigHovered.connect(lambda plot_item, indexes, event: self.on_hover(plot_item, indexes, event))

        # Connect to ViewBox range changes (zoom/pan events)
        self.ax.vb.sigRangeChanged.connect(self.on_view_changed)

        # Add plot items
        self.ax.addItem(self)
        self.ax.showGrid(x=True, y=True, alpha=1)
        self.fig.nextRow()
        self.fig.addItem(self.cb)

    def on_view_changed(self, view_box, range_change):
        """Called when the view range changes (zoom/pan)"""
        # Get the current view ranges
        view_range = view_box.viewRange()

        # Cancel any pending optimization timer
        if hasattr(self, '_optimization_timer'):
            self._optimization_timer.stop()

        # Use a longer delay (300ms instead of 100ms) to avoid recalculating during continuous panning
        self._optimization_timer = QtCore.QTimer()
        self._optimization_timer.setSingleShot(True)
        self._optimization_timer.timeout.connect(lambda: self.optimize_visible_points(view_range))
        self._optimization_timer.start(300)  # Increased delay for better performance

    def optimize_visible_points(self, view_range):
        """Optimize which points to display based on current view"""
        try:
            # Extract view boundaries
            x_range, y_range = view_range
            x_min, x_max = x_range
            y_min, y_max = y_range

            # Filter points that are within view bounds
            in_view = (
                    (self.x >= x_min) & (self.x <= x_max) &
                    (self.y >= y_min) & (self.y <= y_max)
            )
            points_in_view = self.all_points_idx[in_view]

            # If very few points, show all of them
            if len(points_in_view) < 500:
                self.visible_points_idx = points_in_view
                self.update_visible_points()
                return

            # If number of points is reasonable, show them all
            if len(points_in_view) < 3000:
                self.visible_points_idx = points_in_view
                self.update_visible_points()
                return

            # For very large datasets, use a more aggressive downsampling
            # Calculate a dynamic min_distance based on point count
            point_count = len(points_in_view)
            if point_count > 10000:
                # More aggressive for very large datasets
                self.min_distance = 10.0
            elif point_count > 5000:
                self.min_distance = 7.0
            else:
                self.min_distance = 5.0

            # If dataset is extremely large, do a simple random sampling first
            if point_count > 15000:
                # Take a random subset (20%) for extremely large datasets
                rng = numpy.random.default_rng()
                random_indices = rng.choice(len(points_in_view), size=int(len(points_in_view) * 0.2), replace=False)
                points_in_view = points_in_view[random_indices]

            # Map data coordinates to screen coordinates (only for the subset if used)
            x_filtered = self.x[points_in_view]
            y_filtered = self.y[points_in_view]

            # Use vectorized operations to convert coordinates where possible
            # or batch the coordinate conversion
            screen_points = []
            batch_size = 1000  # Process in batches to reduce overhead

            for i in range(0, len(x_filtered), batch_size):
                batch_end = min(i + batch_size, len(x_filtered))
                batch_points = []

                for j in range(i, batch_end):
                    screen_point = self.ax.vb.mapFromView(QtCore.QPointF(x_filtered[j], y_filtered[j]))
                    batch_points.append((screen_point.x(), screen_point.y()))

                screen_points.extend(batch_points)

            # Apply grid-based filtering
            coords = numpy.array(screen_points)
            mask = numpy.ones(len(points_in_view), dtype=bool)
            cell_size = self.min_distance
            grid = {}

            for i, (x, y) in enumerate(coords):
                cell_x = int(x / cell_size)
                cell_y = int(y / cell_size)
                cell_key = (cell_x, cell_y)

                if cell_key in grid:
                    mask[i] = False
                else:
                    grid[cell_key] = i

            # Apply the mask to get the visible points
            self.visible_points_idx = points_in_view[mask]

            # Update plot with only the visible points
            self.update_visible_points()

            print(f"Showing {len(self.visible_points_idx)} of {len(self.x)} points")

        except Exception as e:
            print(f"Error in optimize_visible_points: {e}")
            import traceback
            traceback.print_exc()

    def update_visible_points(self):
        """Update the scatter plot to show only the visible points"""
        try:
            # Extract visible points data
            visible_x = self.x[self.visible_points_idx]
            visible_y = self.y[self.visible_points_idx]

            # Get the color data for visible points
            color_data = self.data_points[self.color_key]
            visible_colors = color_data[self.visible_points_idx]

            # Get current color levels
            min_level, max_level = self.cb.getLevels()

            # Normalize data to 0-1 range based on color levels
            if max_level > min_level:
                normalized = (visible_colors - min_level) / (max_level - min_level)
            else:
                normalized = numpy.zeros_like(visible_colors)

            # Map to colors
            colors = [pyqtgraph.mkBrush(color) for color in
                      self.colormap.map(normalized, mode='qcolor')]

            # Update scatter plot data with original indices stored properly
            # Convert indices to integer to avoid any type issues
            data_indices = [int(idx) for idx in self.visible_points_idx]
            self.setData(
                x=visible_x,
                y=visible_y,
                data=data_indices,  # Use the index in the original dataset as data
                brush=colors
            )
        except Exception as e:
            print(f"Error in update_visible_points: {e}")
            import traceback
            traceback.print_exc()

    def tooltip(self, x, y, data):
        """Generate tooltip text for hover events"""
        # Note: 'data' contains the index in the original dataset
        txt = f"Point {data}" + "\n"
        txt += f"\tX = {round(x, 2)} m" + "\n"
        txt += f"\tY = {round(y, 2)} m" + "\n"

        # Add all data values from the dictionary
        for key, value in sorted(self.data_points.items()):
            if key in ['x', 'y']:  # Skip x,y as they're already shown
                continue

            if isinstance(value, numpy.ndarray) and len(value) > data:
                try:
                    # Try to format the value appropriately
                    val = value[data]
                    if isinstance(val, (int, float)) and not numpy.isnan(val):
                        decimal = 2
                        while True:
                            val_rounded = round(val, decimal)
                            decimal += 1
                            if val_rounded != 0 or decimal > 10:
                                break
                        txt += f"\t{key} = {val_rounded}"

                        # Add units if known
                        if key in ['measures', 'synth', 'error']:
                            txt += " nT"
                        elif key in ['z']:
                            txt += " m"

                        txt += "\n"
                except (IndexError, TypeError, ValueError):
                    continue

        return txt.rstrip()  # Remove trailing newline

    def on_hover(self, plot_item, indexes, event):
        """Handle hover events to show values in histogram"""
        color_data = self.data_points[self.color_key]

        # Get values directly from the hovered points
        # The hovered points will have x, y values corresponding to the data
        values_bars = []
        for point_idx in indexes[:min(len(indexes), 1)]:
            # Find the original index in the data array
            try:
                # If point has _data attribute as an integer index
                if hasattr(point_idx, '_data') and isinstance(point_idx._data, (int, numpy.integer)):
                    original_idx = point_idx._data
                    values_bars.append(color_data[original_idx])
                # Fallback: find the index by matching x,y coordinates
                else:
                    x_val = point_idx.pos().x()
                    y_val = point_idx.pos().y()
                    # Find closest point in original data
                    distances = numpy.sqrt((self.x - x_val) ** 2 + (self.y - y_val) ** 2)
                    original_idx = numpy.argmin(distances)
                    values_bars.append(color_data[original_idx])
            except Exception as e:
                print(f"Error getting hover value: {e}")
                continue

        # Skip if no change in values
        if self.lines_values == values_bars:
            return

        # Add new lines
        for value in values_bars:
            if value not in self.lines_values:
                value_line = pyqtgraph.InfiniteLine(angle=90, movable=False, pen=pyqtgraph.mkPen('w'))
                value_line.setValue(value)
                self.cb.vb.addItem(value_line)
                self.lines.append(value_line)
                self.lines_values.append(value)

        # Remove lines for points no longer hovered
        # Make copies to avoid modifying during iteration
        lines_to_remove = []
        values_to_remove = []

        for i, (line, value) in enumerate(zip(self.lines, self.lines_values)):
            if value not in values_bars:
                lines_to_remove.append(line)
                values_to_remove.append(value)

        for line, value in zip(lines_to_remove, values_to_remove):
            self.cb.vb.removeItem(line)
            self.lines.remove(line)
            self.lines_values.remove(value)

    def repaint(self):
        """Update scatter plot colors based on color data and levels"""
        try:
            self.update_visible_points()  # Update points with new color mapping
        except Exception as e:
            print(f"Error in repaint: {e}")

    def mouseDragEvent(self, ev):
        """Handle mouse drag events for creating slices"""
        if ev.button() != QtCore.Qt.MouseButton.RightButton:
            return

        # Get points at the current mouse position
        points_at_pos = self.pointsAt(ev.pos())

        if ev.isStart():
            print("Drag started with right button")
            self.slice = []
            if len(points_at_pos) > 0:
                print(f"Found {len(points_at_pos)} points at start position")
                self.slice.extend(points_at_pos)

        elif ev.isFinish():
            print(f"Drag finished with {len(self.slice)} points in slice")
            if len(self.slice) > 0:
                # Extract indexes from points in the slice
                indexes = []
                for point in self.slice:
                    try:
                        # If point has _data attribute as integer
                        if hasattr(point, '_data') and isinstance(point._data, (int, numpy.integer)):
                            indexes.append(int(point._data))
                        # Fallback: find the index by matching x,y coordinates
                        else:
                            x_val = point.pos().x()
                            y_val = point.pos().y()
                            # Find closest point in original data
                            distances = numpy.sqrt((self.x - x_val) ** 2 + (self.y - y_val) ** 2)
                            original_idx = numpy.argmin(distances)
                            indexes.append(int(original_idx))
                    except Exception as e:
                        print(f"Error getting point index: {e}")
                        continue

                print(f"Creating slice with {len(indexes)} points")

                # Create a dictionary to hold all sliced data
                slice_data = {}

                # Extract all data fields
                for key, value in self.data_points.items():
                    if isinstance(value, numpy.ndarray) and len(value) > 0:
                        try:
                            slice_data[key] = value[indexes]
                            print(f"Added {key} data to slice")
                        except (IndexError, TypeError) as e:
                            print(f"Warning: Could not slice '{key}' data: {e}")

                # Create the Slice object with the selected data
                try:
                    slice.Slice(data=slice_data)
                    print("Slice created successfully")
                except Exception as e:
                    print(f"Error creating slice: {str(e)}")
                    import traceback
                    traceback.print_exc()
            self.slice = []

        else:  # During drag
            for point in points_at_pos:
                # Check if the point is already in the slice using explicit comparison
                already_in_slice = False
                for existing_point in self.slice:
                    if id(point) == id(existing_point):  # Compare by identity
                        already_in_slice = True
                        break

                if not already_in_slice:
                    print(f"Adding point to slice")
                    self.slice.append(point)

        # Always accept the event to ensure it's processed
        ev.accept()