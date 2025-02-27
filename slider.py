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