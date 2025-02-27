import pyqtgraph
import numpy
from pyqtgraph.Qt import QtCore, QtGui
import threading
import queue
import time
import weakref

import slice


class OptimizationWorker(QtCore.QObject):
    # Signal to pass results back to the main thread
    result_ready = QtCore.Signal(object)

    def __init__(self):
        super().__init__()
        self.task_queue = queue.Queue()
        self.measures_ref = None  # Weak reference to measures object
        self.running = True
        self.thread = threading.Thread(target=self._process_queue, daemon=True)
        self.thread.start()

    def set_measures(self, measures):
        """Set a weak reference to the measures object"""
        self.measures_ref = weakref.ref(measures)

    def add_task(self, view_range):
        """Add a task to the queue"""
        # Clear the queue first to avoid outdated tasks
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
                self.task_queue.task_done()
            except queue.Empty:
                break
        self.task_queue.put(view_range)

    def _process_queue(self):
        """Process tasks from the queue"""
        while self.running:
            try:
                # Get task with timeout to allow checking running flag periodically
                try:
                    view_range = self.task_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Get measures object from weak reference
                measures = self.measures_ref() if self.measures_ref else None
                if measures is None:
                    self.task_queue.task_done()
                    continue

                try:
                    # Extract view boundaries
                    x_range, y_range = view_range
                    x_min, x_max = x_range
                    y_min, y_max = y_range

                    # Get data from the measures object
                    x = measures.x
                    y = measures.y
                    all_points_idx = measures.all_points_idx

                    # Filter points that are within view bounds
                    in_view = (
                            (x >= x_min) & (x <= x_max) &
                            (y >= y_min) & (y <= y_max)
                    )
                    points_in_view = all_points_idx[in_view]

                    # Quick exit for small datasets
                    if len(points_in_view) < 1000:
                        result = {'visible_points_idx': points_in_view}
                        self.result_ready.emit(result)
                        self.task_queue.task_done()
                        continue

                    # For larger datasets, apply downsampling
                    point_count = len(points_in_view)

                    # Choose downsampling strategy based on point count
                    if point_count > 20000:
                        # Very aggressive downsampling for huge datasets
                        # Just use random sampling for speed
                        rng = numpy.random.default_rng()
                        sample_size = min(10000, int(point_count * 0.1))
                        indices = rng.choice(len(points_in_view), size=sample_size, replace=False)
                        filtered_points = points_in_view[indices]
                    elif point_count > 10000:
                        # Use grid-based filtering with larger cells
                        filtered_points = self._grid_filter(points_in_view, x, y, x_range, y_range, cell_factor=15.0)
                    elif point_count > 5000:
                        # Medium-sized dataset
                        filtered_points = self._grid_filter(points_in_view, x, y, x_range, y_range, cell_factor=10.0)
                    else:
                        # Smaller dataset
                        filtered_points = self._grid_filter(points_in_view, x, y, x_range, y_range, cell_factor=5.0)

                    # Send results back to the main thread
                    result = {'visible_points_idx': filtered_points}
                    self.result_ready.emit(result)

                except Exception as e:
                    print(f"Error in optimization worker: {e}")
                    import traceback
                    traceback.print_exc()

                # Mark task as done
                self.task_queue.task_done()

            except Exception as e:
                print(f"Error in worker thread: {e}")

    def _grid_filter(self, points_in_view, x, y, x_range, y_range, cell_factor=5.0):
        """Filter points using a grid-based approach in data space"""
        # Get x and y data for points in view
        x_filtered = x[points_in_view]
        y_filtered = y[points_in_view]

        # Calculate grid cell size based on view range
        x_min, x_max = x_range
        y_min, y_max = y_range

        # Compute data-to-screen scale factors (approximate)
        x_range_size = x_max - x_min
        y_range_size = y_max - y_min

        # Use cell factor to control grid density
        # Higher factor = fewer points (faster but less detail)
        cell_size_x = (x_range_size / 1000) * cell_factor
        cell_size_y = (y_range_size / 1000) * cell_factor

        # Apply grid-based filtering
        mask = numpy.ones(len(points_in_view), dtype=bool)
        grid = {}

        for i in range(len(x_filtered)):
            # Integer division for grid cell assignment
            cell_x = int(x_filtered[i] / cell_size_x)
            cell_y = int(y_filtered[i] / cell_size_y)
            cell_key = (cell_x, cell_y)

            if cell_key in grid:
                mask[i] = False
            else:
                grid[cell_key] = i

        # Return the filtered points
        return points_in_view[mask]

    def stop(self):
        """Stop the worker thread"""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)


# Shared worker instance that can be reused across all Measures instances
_SHARED_WORKER = None


def get_worker():
    """Get or create the shared worker instance"""
    global _SHARED_WORKER
    if _SHARED_WORKER is None or not _SHARED_WORKER.running:
        _SHARED_WORKER = OptimizationWorker()
    return _SHARED_WORKER


class Measures(pyqtgraph.ScatterPlotItem):
    def __init__(self, x, y, data: dict = None, ax=None, fig=None, name="", unit=""):
        # Initialize scatter plot first
        super().__init__(
            hoverable=True,
            hoverSymbol="s",
            hoverSize=10,
            hoverPen=pyqtgraph.mkPen("w", width=1)
        )

        # Initialize our instance attributes
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

        # Thread-related attributes
        self._last_view_range = None
        self._optimization_pending = False
        self._last_update_time = time.time()
        self._debounce_timeout = 100  # ms
        self._debounce_timer = QtCore.QTimer()
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self._do_schedule_optimization)

        # Get worker and connect to its signal
        self._worker = get_worker()
        self._worker.set_measures(self)
        self._worker.result_ready.connect(self._apply_optimization_results)

        # Set up color mapping
        self.cb = pyqtgraph.HistogramLUTItem(orientation='horizontal', gradientPosition=None, levelMode="mono")
        self.colormap = pyqtgraph.colormap.get("Turbo")

        # Set up histogram bins
        self.set_bins()

        # Update the data for scatter plot
        self.setData(
            x=self.x,
            y=self.y,
            data=numpy.array(range(len(self.x))),
            tip=lambda x, y, data: self.tooltip(x, y, data),
            pen=None,
            brush=[pyqtgraph.mkBrush(color) for color in self.colormap.mapToQColor(self.data_points[self.color_key])]
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

        # Initial point optimization
        self._do_schedule_optimization()

    def on_view_changed(self, view_box, range_change):
        """Called when the view range changes (zoom/pan)"""
        # Use Qt timer for debouncing (delay execution until user stops changing view)
        if self._debounce_timer.isActive():
            self._debounce_timer.stop()

        # Restart timer
        self._debounce_timer.start(self._debounce_timeout)

        # During active dragging/zooming, use a simplified representation
        # This gives immediate visual feedback while keeping the UI responsive
        if len(self.visible_points_idx) > 10000:
            # Use random sampling for temporary view during drag
            rng = numpy.random.default_rng()
            temp_indices = rng.choice(len(self.visible_points_idx), size=5000, replace=False)
            temp_visible = self.visible_points_idx[temp_indices]
            self._update_points_display(temp_visible)

    def _do_schedule_optimization(self):
        """Actually schedule the optimization after debounce timeout"""
        view_range = self.ax.vb.viewRange()

        # Don't reschedule if view hasn't changed
        if (self._last_view_range is not None and
                numpy.allclose(view_range[0], self._last_view_range[0]) and
                numpy.allclose(view_range[1], self._last_view_range[1])):
            return

        # Update last view range
        self._last_view_range = view_range

        # Flag that optimization is pending
        self._optimization_pending = True

        # Add task to worker queue
        self._worker.add_task(view_range)

    def _apply_optimization_results(self, result):
        """Apply the optimization results from the worker thread (runs in UI thread)"""
        try:
            self.visible_points_idx = result['visible_points_idx']
            self._update_points_display(self.visible_points_idx)
            self._optimization_pending = False
            print(f"Showing {len(self.visible_points_idx)} of {len(self.x)} points")
        except Exception as e:
            print(f"Error applying optimization results: {e}")

    def _update_points_display(self, visible_indices):
        """Update just the display data without recalculating visibility"""
        try:
            # Extract visible points data
            visible_x = self.x[visible_indices]
            visible_y = self.y[visible_indices]

            # Get the color data for visible points
            color_data = self.data_points[self.color_key]
            visible_colors = color_data[visible_indices]

            # Get current color levels
            min_level, max_level = self.cb.getLevels()

            # Normalize data to 0-1 range based on color levels
            if max_level > min_level:
                normalized = (visible_colors - min_level) / (max_level - min_level)
            else:
                normalized = numpy.zeros_like(visible_colors)

            # Map to colors - do this in batches for large datasets
            if len(visible_indices) > 10000:
                # For large datasets, use a faster approach with numpy vectorization
                # Convert colormap to numpy array for faster processing
                cmap_array = numpy.array(
                    [c.getRgb() for c in self.colormap.map(numpy.linspace(0, 1, 256), mode='qcolor')])

                # Map normalized values to colormap indices
                color_indices = numpy.clip((normalized * 255).astype(int), 0, 255)

                # Get colors from colormap
                colors = [pyqtgraph.mkBrush(*cmap_array[idx]) for idx in color_indices]
            else:
                # For smaller datasets, use the standard approach
                colors = [pyqtgraph.mkBrush(color) for color in
                          self.colormap.map(normalized, mode='qcolor')]

            # Update scatter plot data - use integers for the data indices
            data_indices = [int(idx) for idx in visible_indices]
            self.setData(
                x=visible_x,
                y=visible_y,
                data=data_indices,
                brush=colors
            )
        except Exception as e:
            print(f"Error updating points display: {e}")
            import traceback
            traceback.print_exc()

    def update_visible_points(self):
        """Update the scatter plot to show only the visible points"""
        self._update_points_display(self.visible_points_idx)

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

    def cleanup(self):
        """Clean up resources when the plot is closed"""
        # Stop debounce timer
        if self._debounce_timer.isActive():
            self._debounce_timer.stop()


# Function to clean up the worker when the application exits
def cleanup_worker():
    """Clean up the shared worker when the application exits"""
    global _SHARED_WORKER
    if _SHARED_WORKER is not None:
        _SHARED_WORKER.stop()
        _SHARED_WORKER = None