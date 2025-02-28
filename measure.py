import sys
import pyqtgraph
import numpy
from PyQt5 import QtWidgets
from pyqtgraph.Qt import QtCore, QtGui
import time

# Imports from your existing code
import optimize
import slice
import slider


class Measures(pyqtgraph.ScatterPlotItem):
    def __init__(self, x, y, data, name, sample_size=1000, response_time=100, ax=None, fig=None):
        super().__init__(
            hoverable=True,
            hoverSymbol="s",
            hoverSize=10,
            hoverPen=pyqtgraph.mkPen("w", width=1)
        )

        self.x = x
        self.y = y
        self.data_points = {} if data is None else data.copy()
        self.data_points["x"] = x
        self.data_points["y"] = y
        self.name_str = name  # Store name as string for persistence

        # Initialize persistence manager
        self.persistence = slider.SliderPersistence()

        # Load saved settings if they exist
        saved_settings = self.persistence.load_settings(name)
        if saved_settings:
            # Use saved values if available
            self.sample_size = saved_settings.get('sample_size', sample_size)
            self.response_time = saved_settings.get('response_time', response_time)
            print(
                f"Loaded saved settings for {name}: sample_size={self.sample_size}, response_time={self.response_time}")
        else:
            # Use default values
            self.sample_size = sample_size
            self.response_time = response_time

        self.color_key = None
        for key, value in self.data_points.items():
            if key not in ['x', 'y'] and isinstance(value, numpy.ndarray) and value.dtype.kind in 'iuf':
                self.color_key = key
                print(f"Using '{key}' for coloring")
                break

        # If no data fields, use zeros
        if self.color_key is None:
            raise Exception("No data provided")

        # Initialize plot elements
        self.fig = fig
        self.ax = ax
        if self.ax is None:
            if self.fig is not None:
                self.ax = self.fig.addPlot()
            else:
                pyqtgraph.setConfigOptions(antialias=True)
                self.fig = pyqtgraph.GraphicsLayoutWidget(show=True)
                self.ax = self.fig.addPlot()

        self.name = lambda: name
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
        self._debounce_timeout = self.response_time  # ms
        self._debounce_timer = QtCore.QTimer()
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self._do_schedule_optimization)

        # Get worker and connect to its signal
        self._worker = optimize.get_worker(self.sample_size, self.response_time)
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

    def mousePressEvent(self, ev):
        """
        Override mousePressEvent to intercept right clicks before they trigger context menus
        """
        # If it's a right-click, handle it here
        if ev.button() == QtCore.Qt.MouseButton.RightButton:
            # Block the event from propagating further
            ev.accept()

            # Call our custom method to show the context menu
            self.showCustomContextMenu(ev)
        else:
            # For all other mouse buttons, let the parent class handle it
            super().mousePressEvent(ev)

    def showCustomContextMenu(self, ev):
        """
        Show our custom context menu for right-click events
        """
        # Create a brand new menu
        menu = QtWidgets.QMenu()

        # ===== Add our Configuration menu at the very top =====
        config_menu = QtWidgets.QMenu("Configuration")
        menu.addMenu(config_menu)

        # Create custom widget actions for sliders
        sample_size_action = QtWidgets.QWidgetAction(config_menu)
        response_time_action = QtWidgets.QWidgetAction(config_menu)

        # Create slider widgets
        sample_slider = slider.LabeledSlider("Sample Size", 100, 10000, self.sample_size, 100)
        time_slider = slider.LabeledSlider("Response Time (ms)", 10, 1000, self.response_time, 10)

        # Set the widgets for the actions
        sample_size_action.setDefaultWidget(sample_slider)
        response_time_action.setDefaultWidget(time_slider)

        # Add actions to the config menu
        config_menu.addAction(sample_size_action)
        config_menu.addAction(response_time_action)

        # Connect sliders to update functions - update on release for performance
        sample_slider.sliderReleased.connect(lambda: self.update_sample_size(sample_slider.slider.value(), True))
        sample_slider.sliderReleased.connect(
            lambda: self.persistence.save_settings(
                'Data', {
                    "sample_size": self.sample_size,
                    "response_time": self.response_time
                }
            )
        )
        time_slider.sliderReleased.connect(lambda: self.update_response_time(time_slider.slider.value()))
        time_slider.sliderReleased.connect(
            lambda: self.persistence.save_settings(
                'Data', {
                    "sample_size": self.sample_size,
                    "response_time": self.response_time
                }
            )
        )
        # Connect valueChanged to update UI without triggering graph redraw)
        sample_slider.valueChanged.connect(lambda value: self.update_sample_size(value, False))

        # Add a separator and reset option
        config_menu.addSeparator()
        reset_action = config_menu.addAction("Reset to Defaults")
        reset_action.triggered.connect(lambda: self.reset_parameters(sample_slider, time_slider))

        # ===== Now manually add all the standard PyQtGraph plot options =====
        # View All
        view_all_action = menu.addAction("View All")
        view_all_action.triggered.connect(lambda: self.ax.autoRange())

        # X Axis submenu
        x_menu = QtWidgets.QMenu("X axis")
        menu.addMenu(x_menu)

        # X axis - Manual Range
        x_manual_action = x_menu.addAction("Manual Range")
        x_manual_action.triggered.connect(lambda: self.ax.vb.setMouseMode(pyqtgraph.ViewBox.RectMode))

        # X axis - Auto Range
        x_auto_action = x_menu.addAction("Auto Range")
        x_auto_action.triggered.connect(lambda: self.ax.enableAutoRange(axis=pyqtgraph.ViewBox.XAxis))

        # X axis - Auto Pan
        x_pan_action = x_menu.addAction("Auto Pan")
        x_pan_action.triggered.connect(lambda: self.ax.enableAutoPan(axis=pyqtgraph.ViewBox.XAxis))

        # X axis - Log Scale
        x_log_action = x_menu.addAction("Log Scale")
        x_log_action.setCheckable(True)
        x_log_action.setChecked(self.ax.getAxis('bottom').logMode)
        x_log_action.triggered.connect(lambda checked: self.ax.setLogMode(x=checked))

        # Y Axis submenu
        y_menu = QtWidgets.QMenu("Y axis")
        menu.addMenu(y_menu)

        # Y axis - Manual Range
        y_manual_action = y_menu.addAction("Manual Range")
        y_manual_action.triggered.connect(lambda: self.ax.vb.setMouseMode(pyqtgraph.ViewBox.RectMode))

        # Y axis - Auto Range
        y_auto_action = y_menu.addAction("Auto Range")
        y_auto_action.triggered.connect(lambda: self.ax.enableAutoRange(axis=pyqtgraph.ViewBox.YAxis))

        # Y axis - Auto Pan
        y_pan_action = y_menu.addAction("Auto Pan")
        y_pan_action.triggered.connect(lambda: self.ax.enableAutoPan(axis=pyqtgraph.ViewBox.YAxis))

        # Y axis - Log Scale
        y_log_action = y_menu.addAction("Log Scale")
        y_log_action.setCheckable(True)
        y_log_action.setChecked(self.ax.getAxis('left').logMode)
        y_log_action.triggered.connect(lambda checked: self.ax.setLogMode(y=checked))

        # Mouse Mode submenu
        mouse_menu = QtWidgets.QMenu("Mouse Mode")
        menu.addMenu(mouse_menu)

        # Mouse Mode - Rectangle Selection
        rect_action = mouse_menu.addAction("Rectangle Selection")
        rect_action.triggered.connect(lambda: self.ax.vb.setMouseMode(pyqtgraph.ViewBox.RectMode))

        # Mouse Mode - Pan/Zoom
        pan_action = mouse_menu.addAction("Pan/Zoom")
        pan_action.triggered.connect(lambda: self.ax.vb.setMouseMode(pyqtgraph.ViewBox.PanMode))

        # Plot Options submenu
        plot_menu = QtWidgets.QMenu("Plot Options")
        menu.addMenu(plot_menu)

        # Plot Options - Grid
        grid_action = plot_menu.addAction("Grid")
        grid_action.setCheckable(True)
        grid_action.setChecked(True)  # Assuming grid is on by default
        grid_action.triggered.connect(lambda checked: self.ax.showGrid(x=checked, y=checked))

        # Export option
        export_action = menu.addAction("Export...")
        export_action.triggered.connect(lambda: pyqtgraph.exportDialog(self.ax))

        # Show the menu at the position of the mouse event
        menu.exec_(ev.screenPos())

    def save_current_settings(self):
        """Save current slider settings to persistent storage"""
        settings = {
            'sample_size': self.sample_size,
            'response_time': self.response_time
        }

        success = self.persistence.save_settings(self.name_str, settings)
        if success:
            # Show a confirmation message using a message box
            msg_box = QtWidgets.QMessageBox()
            msg_box.setIcon(QtWidgets.QMessageBox.Information)
            msg_box.setText(f"Settings for {self.name_str} saved successfully.")
            msg_box.setWindowTitle("Settings Saved")
            msg_box.exec_()
        else:
            # Show error message
            msg_box = QtWidgets.QMessageBox()
            msg_box.setIcon(QtWidgets.QMessageBox.Warning)
            msg_box.setText("Failed to save settings.")
            msg_box.setWindowTitle("Save Error")
            msg_box.exec_()

    def update_sample_size(self, value, trigger_update=True):
        """Update the sample size parameter"""
        # Update the sample size parameter
        self.sample_size = value
        # Update the worker's sample size
        self._worker.sample_size = value
        print(f"Sample size updated to: {value}")

        # For immediate feedback, force a resampling of points
        if trigger_update:
            # Get current view range
            view_range = self.ax.vb.viewRange()
            # Clear any existing optimization tasks
            self._optimization_pending = False
            if self._debounce_timer.isActive():
                self._debounce_timer.stop()

            # Force the worker to process a new task immediately with the updated sample size
            self._worker.add_task(view_range)

            # Reset optimization pending flag to ensure next scheduled task is processed
            self._last_view_range = None

    def update_response_time(self, value):
        """Update the response time parameter"""
        self.response_time = value
        # Update the worker's response time
        self._worker.response_time = value
        # Update the debounce timeout
        self._debounce_timeout = value
        print(f"Response time updated to: {value}ms")
        # Trigger a repaint to ensure the changes are visible
        self.repaint()

    def reset_parameters(self, sample_slider, time_slider):
        """Reset parameters to default values"""
        # Default values (you can adjust these)
        default_sample_size = 1000
        default_response_time = 100

        # Update time slider first (less impactful change)
        time_slider.slider.setValue(default_response_time)
        self.update_response_time(default_response_time)

        # Update sample slider and force immediate update
        sample_slider.slider.setValue(default_sample_size)
        self.update_sample_size(default_sample_size, True)

        print(
            f"Parameters reset to defaults: Sample Size={default_sample_size}, Response Time={default_response_time}ms")

        # Optionally save the default values
        self.save_current_settings()

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

        self.cb.gradient.setColorMap(self.colormap)

        # Set initial levels
        try:
            min_val = numpy.nanmin(color_data)
            max_val = numpy.nanmax(color_data)
            self.cb.setLevels(min_val, max_val)
        except (ValueError, TypeError) as e:
            print(f"Warning: Could not set initial levels: {e}")
        try:
            p05 = numpy.nanpercentile(color_data, 5)
            p95 = numpy.nanpercentile(color_data, 95)
            self.cb.setLevels(p05, p95)
        except (ValueError, TypeError) as e:
            print(f"Warning: Could not set percentile levels: {e}")
        self.cb.sigLevelsChanged.connect(self.repaint)
        self.sigHovered.connect(lambda plot_item, indexes, event: self.on_hover(plot_item, indexes, event))
        self.ax.vb.sigRangeChanged.connect(self.on_view_changed)
        self.ax.addItem(self)
        self.ax.showGrid(x=True, y=True, alpha=1)
        self.fig.nextRow()
        self.fig.addItem(self.cb)
        self._do_schedule_optimization()

    def on_view_changed(self, view_box, range_change):
        if self._debounce_timer.isActive():
            self._debounce_timer.stop()
        self._debounce_timer.start(self._debounce_timeout)

    def _do_schedule_optimization(self):
        """Actually schedule the optimization after debounce timeout"""
        view_range = self.ax.vb.viewRange()
        if (self._last_view_range is not None and
                numpy.allclose(view_range[0], self._last_view_range[0]) and
                numpy.allclose(view_range[1], self._last_view_range[1])):
            return
        self._last_view_range = view_range
        self._optimization_pending = True
        self._worker.add_task(view_range)

    def _apply_optimization_results(self, result):
        """Apply the optimization results from the worker thread (runs in UI thread)"""
        try:
            # Check if this is a repaint operation
            is_repaint = isinstance(result, dict) and result.get('operation') == 'repaint'

            # Get the visible points indices from the result
            self.visible_points_idx = result['visible_points_idx']

            # Update the points display
            self._update_points_display(self.visible_points_idx)

            # Reset the optimization pending flag
            self._optimization_pending = False

            # Only log for non-repaint operations to avoid console spam
            if not is_repaint:
                print(f"Showing {len(self.visible_points_idx)} of {len(self.x)} points")

        except Exception as e:
            print(f"Error applying optimization results: {e}")
            import traceback
            traceback.print_exc()

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
            if len(visible_indices) > self.sample_size:
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

    def tooltip(self, x, y, data):
        """Generate tooltip text for hover events"""
        txt = f"Point {data}" + "\n"
        txt += f"\tX = {round(x, 2)} m" + "\n"
        txt += f"\tY = {round(y, 2)} m" + "\n"
        for key, value in sorted(self.data_points.items()):
            if key in ['x', 'y']:
                continue
            if isinstance(value, numpy.ndarray) and len(value) > data:
                try:
                    val = value[data]
                    if isinstance(val, (int, float)) and not numpy.isnan(val):
                        decimal = 2
                        while True:
                            val_rounded = round(val, decimal)
                            decimal += 1
                            if val_rounded != 0 or decimal > 10:
                                break
                        txt += f"\t{key} = {val_rounded}\n"
                except (IndexError, TypeError, ValueError):
                    continue

        return txt.rstrip()  # Remove trailing newline

    def on_hover(self, plot_item, indexes, event):
        """Handle hover events to show values in histogram"""
        color_data = self.data_points[self.color_key]
        values_bars = []
        for point_idx in indexes[:min(len(indexes), 1)]:
            try:
                if hasattr(point_idx, '_data') and isinstance(point_idx._data, (int, numpy.integer)):
                    original_idx = point_idx._data
                    values_bars.append(color_data[original_idx])
                else:
                    x_val = point_idx.pos().x()
                    y_val = point_idx.pos().y()
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
        """
        Queue a repaint operation to be processed by the optimization worker thread
        """
        # If optimization is already pending, no need to schedule another repaint
        if self._optimization_pending:
            return

        # Add a specific repaint task to the worker queue
        # We'll use the current view range but flag it as a repaint operation
        if hasattr(self, 'ax') and hasattr(self.ax, 'vb'):
            view_range = self.ax.vb.viewRange()
            self._optimization_pending = True

            # Create a special task for repaint that includes the current visible points
            repaint_task = {
                'view_range': view_range,
                'operation': 'repaint',
                'visible_points_idx': self.visible_points_idx
            }

            # Add the task to the worker queue
            self._worker.add_repaint_task(repaint_task)
        else:
            # Fallback to direct update if view box isn't available
            self._update_points_display(self.visible_points_idx)

    def mouseDragEvent(self, ev):
        """Handle mouse drag events for creating slices"""
        if ev.button() != QtCore.Qt.MouseButton.RightButton:
            return

        # Set flag to track if we're in a drag operation
        if ev.isStart():
            self._mouseDragging = True

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
            # Reset dragging flag when drag finishes
            self._mouseDragging = False

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