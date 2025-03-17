
import typing as _t
import numpy as _np

def iter_window_values(window_size: int, window_step: int, values: _t.Iterable[float])->_t.Iterable[_np.ndarray]:
    """ 
    Given some values, apply a window to those values and yield the values for each window location.
    
    :param window_size: the size of the window (i.e. the number of values that window exposes at each position)
    :param window_step: the number of values that the window traverses for each step
    :param values: the values to apply the window to
    :raises ValueError: if either the window size or the window step are below 1
    :return: an iterable of Numpy arrays all of size 'window_size'
    """
    if window_size<1:
        raise ValueError(f"Window size should be above 0, but is currently {window_size}.")
    if window_step<1:
        raise ValueError(f"Window step should be above 0, but is currently {window_step}.")
    
    values = _np.array(values)
    max_index = len(values) - window_size
    
    index = 0
    while index <= max_index:
        yield values[index: index + window_size]
        index += window_step


