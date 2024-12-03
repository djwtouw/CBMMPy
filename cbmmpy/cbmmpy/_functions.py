from ._cbmmpy import _vec_sum, _square_cube


def vec_sum(x, y):
    """
    Add two vectors.

    Parameters
    ----------
    x : ndarray
        1D array containing data with `float` type.
    y : ndarray
        1D array containing data with `float` type.

    Returns
    -------
    ndarray
        1D array containing the result of x + y with `float` type.

    """
    res = _vec_sum(x, y)
    return res["vec"]


def square_cube(x):
    """
    Add the square and the cube of x.

    Parameters
    ----------
    x : float
        A parameter.

    Returns
    -------
    float
        The result of x*x + x*x*x.

    """
    return _square_cube(x)
