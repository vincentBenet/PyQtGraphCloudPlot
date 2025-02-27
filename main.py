import sys
from PyQt5 import QtWidgets
import measure


def main(x, y, data, name="", sample_size=2000, response_time=100, ax=None, fig=None):
    if ax is None and fig is None:
        app = QtWidgets.QApplication(sys.argv)
    obj = measure.Measures(
        x=x,
        y=y,
        data=data,
        name=name,
        ax=ax,
        fig=fig,
        sample_size=sample_size,
        response_time=response_time,
    )
    if ax is None and fig is None:
        sys.exit(app.exec_())
    return obj
