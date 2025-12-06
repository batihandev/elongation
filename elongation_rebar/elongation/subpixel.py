import numpy as np

def subpixel_refine(obj, y, mode='callable'):
    """
    Refine the best integer y position to subpixel accuracy using parabolic fitting.
    Parameters:
        obj: callable (score_func) or 1D array (e.g., gradient_strength)
        y: integer position of best match
        mode: 'callable' for score_func, 'array' for 1D array
    Returns:
        subpixel y position (float)
    Usage:
        subpixel_refine(score_func, y, mode='callable')
        subpixel_refine(arr, y, mode='array')
    """
    if mode == 'callable':
        score_func = obj
        y_vals = np.array([y-1, y, y+1])
        s_vals = np.array([score_func(y-1), score_func(y), score_func(y+1)])
    elif mode == 'array':
        arr = obj
        if y <= 0 or y >= len(arr) - 1:
            return float(y)
        y_vals = np.array([y-1, y, y+1])
        s_vals = arr[y-1:y+2]
    else:
        raise ValueError("mode must be 'callable' or 'array'")
    coeffs = np.polyfit(y_vals, s_vals, 2)
    a, b, c = coeffs
    if a == 0:
        return float(y)
    y_subpixel = -b / (2*a)
    return y_subpixel 