import json
import os

from PyQt5 import QtWidgets
from pyqtgraph.Qt import QtCore


class LabeledSlider(QtWidgets.QWidget):
    valueChanged = QtCore.Signal(int)
    sliderReleased = QtCore.Signal(int)

    def __init__(self, title, min_val, max_val, value, step=1, parent=None):
        super().__init__(parent)
        self.setLayout(QtWidgets.QVBoxLayout())
        self.label = QtWidgets.QLabel(f"{title}: {value}")
        self.layout().addWidget(self.label)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(min_val)
        self.slider.setMaximum(max_val)
        self.slider.setValue(value)
        self.slider.setSingleStep(step)
        self.slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider.setTickInterval((max_val - min_val) // 10)
        self.layout().addWidget(self.slider)
        self.slider.valueChanged.connect(self.handleValueChanged)
        self.slider.sliderReleased.connect(self.handleSliderReleased)
        self.setFixedWidth(300)
        self.setFixedHeight(80)

    def handleValueChanged(self, value):
        self.label.setText(f"{self.label.text().split(':')[0]}: {value}")
        self.valueChanged.emit(value)

    def handleSliderReleased(self):
        self.sliderReleased.emit(self.slider.value())

    def value(self):
        return self.slider.value()


class SliderPersistence:
    """
    Class to handle saving and loading slider values to/from permanent storage.
    """

    def __init__(self, settings_file='slider_settings.json'):
        """
        Initialize the persistence manager with a settings file path.

        Args:
            settings_file (str): Path to the settings JSON file
        """
        # Use AppData or similar location based on platform
        app_data_path = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.AppDataLocation)

        # Debug output to help diagnose path issues
        print(f"App data path: {app_data_path}")

        # Create directory if it doesn't exist
        os.makedirs(app_data_path, exist_ok=True)

        # Full path to settings file
        self.settings_path = os.path.join(app_data_path, settings_file)

        # Debug: output the full settings path
        print(f"Settings will be stored at: {self.settings_path}")

    def save_settings(self, plot_name, settings):
        """
        Save slider settings to file.

        Args:
            plot_name (str): Name of the plot/visualization
            settings (dict): Dictionary containing settings

        Returns:
            bool: True if save was successful, False otherwise
        """
        print(f"Attempting to save settings for plot '{plot_name}': {settings}")

        # Load existing settings if file exists
        existing_settings = self.load_all_settings()

        # Update with new settings
        existing_settings[plot_name] = settings

        # Write to file
        try:
            with open(self.settings_path, 'w') as f:
                json.dump(existing_settings, f, indent=2)
            print(f"✓ Settings successfully saved to {self.settings_path}")
            # Debug: show contents of the file
            with open(self.settings_path, 'r') as f:
                print(f"File contents: {f.read()}")
            return True
        except Exception as e:
            print(f"✗ Error saving settings: {e}")
            return False

    def load_settings(self, plot_name):
        """
        Load settings for a specific plot.

        Args:
            plot_name (str): Name of the plot/visualization

        Returns:
            dict: Settings dictionary or None if not found
        """
        print(f"Attempting to load settings for plot '{plot_name}'")
        all_settings = self.load_all_settings()

        if plot_name in all_settings:
            print(f"✓ Found settings for plot '{plot_name}': {all_settings[plot_name]}")
            return all_settings.get(plot_name)
        else:
            print(f"✗ No settings found for plot '{plot_name}'")
            return None

    def load_all_settings(self):
        """
        Load all settings from file.

        Returns:
            dict: All settings or empty dict if file doesn't exist
        """
        try:
            if os.path.exists(self.settings_path):
                print(f"Settings file exists at {self.settings_path}")
                with open(self.settings_path, 'r') as f:
                    settings = json.load(f)
                    print(f"Loaded settings: {settings}")
                    return settings
            else:
                print(f"Settings file does not exist at {self.settings_path}")
                return {}
        except Exception as e:
            print(f"✗ Error loading settings: {e}")
            return {}
