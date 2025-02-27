import numpy
import bezier
import scipy


def curviligne_abs(points: numpy.typing.NDArray[float]) -> numpy.typing.NDArray[float]:
    return numpy.insert(numpy.cumsum(numpy.sum((points[:-1] - points[1:]) ** 2, axis=1) ** 0.5), 0, 0)


def smooth_pipes(pipes, step=None, density=None, shape=None, method="bezier"):
    if step is None:
        if density is not None:
            step = 1 / density
    results = []
    for i, pipe in enumerate(pipes):
        n = None
        if shape is not None:
            n = shape[i]
        results.append(smooth_pipe(pipe=pipe, step=step, n=n, method=method))
    return results


def smooth_pipe(pipe, step=0.1, n=None, method="bezier"):
    return globals()[f"{method}_pipe"](pipe=pipe, step=step, n=n)


def bezier_pipe(pipe, step=0.1, n=None, degree=None):
    if degree is None:
        degree = len(pipe) - 1
    curve = bezier.curve.Curve(pipe.T, degree=degree)
    if n is None:
        n = int(curve.length / step)
    return curve.evaluate_multi(numpy.linspace(0.0, 1.0, n)).T


def akima_pipe(pipe, step=0.1, n=None):
    abs_cur = curviligne_abs(pipe)
    maxi_abs = abs_cur[-1]
    return scipy.interpolate.Akima1DInterpolator(abs_cur, pipe, method="akima")(numpy.linspace(0.0, maxi_abs, n if n is not None else max(2, int(maxi_abs / step))))


def spline_pipe(pipe, step=0.1, n=None):
    abs_cur = curviligne_abs(pipe)
    length = abs_cur[-1]
    if n is None:
        n = int(length / step)
    pipe_t = []
    abs_interp = numpy.linspace(min(abs_cur), max(abs_cur), n)
    for dim in pipe.T:
        tck = scipy.interpolate.splrep(abs_cur, dim, s=0)
        ynew = scipy.interpolate.splev(abs_interp, tck)
        pipe_t.append(ynew)
    return numpy.array(pipe_t).T


def spline_smooth_pipe(pipe, step=0.1, n=None):
    abs_cur = curviligne_abs(pipe)
    length = abs_cur[-1]
    if n is None:
        n = int(length / step)
    pipe_t = []
    abs_interp = numpy.linspace(min(abs_cur), max(abs_cur), n)
    for dim in pipe.T:
        tck = scipy.interpolate.splrep(abs_cur, dim, s=len(pipe))
        ynew = scipy.interpolate.splev(abs_interp, tck)
        pipe_t.append(ynew)
    return numpy.array(pipe_t).T
