import numpy
from pyqtgraph.Qt import QtCore
import threading
import queue
import weakref


class OptimizationWorker(QtCore.QObject):
    # Signal to pass results back to the main thread
    result_ready = QtCore.Signal(object)

    def __init__(self, sample_size, response_time):
        super().__init__()
        self.task_queue = queue.Queue()
        self.measures_ref = None  # Weak reference to measures object
        self.running = True
        self.sample_size = sample_size
        self.response_time = response_time
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

    def add_repaint_task(self, task):
        """Add a repaint task to the queue with higher priority"""
        # Clear repaint tasks first to avoid outdated repaint operations
        # Keep optimization tasks in the queue
        new_queue = queue.Queue()
        while not self.task_queue.empty():
            try:
                existing_task = self.task_queue.get_nowait()
                # Only keep non-repaint tasks
                if isinstance(existing_task, dict) and existing_task.get('operation') != 'repaint':
                    new_queue.put(existing_task)
                self.task_queue.task_done()
            except queue.Empty:
                break

        # Replace the queue with the filtered queue
        self.task_queue = new_queue

        # Add the new repaint task
        self.task_queue.put(task)

    def _process_queue(self):
        """Process tasks from the queue"""
        while self.running:
            try:
                # Get task with timeout to allow checking running flag periodically
                try:
                    task = self.task_queue.get(timeout=self.response_time/1000)
                except queue.Empty:
                    continue

                # Get measures object from weak reference
                measures = self.measures_ref() if self.measures_ref else None
                if measures is None:
                    self.task_queue.task_done()
                    continue

                try:
                    # Check if this is a repaint task
                    if isinstance(task, dict) and task.get('operation') == 'repaint':
                        # For repaint tasks, we already have the visible points
                        visible_points_idx = task.get('visible_points_idx')

                        # Ensure we have valid indices
                        if visible_points_idx is not None and len(visible_points_idx) > 0:
                            result = {'visible_points_idx': visible_points_idx, 'operation': 'repaint'}
                            self.result_ready.emit(result)

                    else:
                        # Handle normal optimization task (view range change)
                        view_range = task if not isinstance(task, dict) else task.get('view_range')
                        if view_range is None:
                            self.task_queue.task_done()
                            continue

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

                        if len(points_in_view) < self.sample_size:
                            result = {'visible_points_idx': points_in_view}
                            self.result_ready.emit(result)
                            self.task_queue.task_done()
                            continue

                        rng = numpy.random.default_rng(seed=0)
                        indices = rng.choice(len(points_in_view), size=self.sample_size, replace=False)
                        filtered_points = points_in_view[indices]

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

    def stop(self):
        """Stop the worker thread"""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=self.response_time/1000)

_SHARED_WORKER = None


def get_worker(sample_size, response_time):
    """Get or create the shared worker instance"""
    global _SHARED_WORKER
    if _SHARED_WORKER is None or not _SHARED_WORKER.running:
        _SHARED_WORKER = OptimizationWorker(sample_size, response_time)
    return _SHARED_WORKER


def cleanup_worker():
    """Clean up the shared worker when the application exits"""
    global _SHARED_WORKER
    if _SHARED_WORKER is not None:
        _SHARED_WORKER.stop()
        _SHARED_WORKER = None
