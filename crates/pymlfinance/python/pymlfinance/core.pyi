from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

def correlation_matrix(data):
    """
    Pearson correlation matrix.

    Parameters
    ----------
    data : numpy.ndarray
        2-D array of shape (n_observations, n_variables).

    Returns
    -------
    numpy.ndarray
        Symmetric correlation matrix of shape (n_variables, n_variables).
    """
    ...

def covariance_matrix(data):
    """
    Sample covariance matrix.

    Parameters
    ----------
    data : numpy.ndarray
        2-D array of shape (n_observations, n_variables).

    Returns
    -------
    numpy.ndarray
        Covariance matrix of shape (n_variables, n_variables).
    """
    ...

def cumsum(values):
    """
    Cumulative sum of an array.

    Parameters
    ----------
    values : numpy.ndarray
        Input array.

    Returns
    -------
    numpy.ndarray
        Cumulative sum with the same length as input.
    """
    ...

def ewma(values, span):
    """
    Exponentially weighted moving average.

    Parameters
    ----------
    values : numpy.ndarray
        Input time series.
    span : int
        Decay span (alpha = 2 / (span + 1)).

    Returns
    -------
    numpy.ndarray
        EWMA-smoothed series of the same length.

    Raises
    ------
    ValueError
        If span < 1 or values is empty.
    """
    ...

def ewma_std(values, span):
    """
    Exponentially weighted moving standard deviation.

    Parameters
    ----------
    values : numpy.ndarray
        Input time series.
    span : int
        Decay span (alpha = 2 / (span + 1)).

    Returns
    -------
    numpy.ndarray
        EWMA standard deviation series.

    Raises
    ------
    ValueError
        If span < 1 or values is empty.
    """
    ...

def kurtosis(values):
    """
    Excess kurtosis (fourth standardized moment minus 3).

    Parameters
    ----------
    values : numpy.ndarray
        Input array (length >= 4).

    Returns
    -------
    float
        Excess kurtosis (0 for a normal distribution).
    """
    ...

def log_returns(prices):
    """
    Logarithmic returns from a price series.

    Computes ``log(p[i] / p[i-1])`` for each consecutive pair.

    Parameters
    ----------
    prices : numpy.ndarray
        Price series of length n.

    Returns
    -------
    numpy.ndarray
        Log returns of length n-1.
    """
    ...

def matrix_inverse(matrix):
    """
    Invert a square matrix using LU decomposition.

    Parameters
    ----------
    matrix : numpy.ndarray
        Square matrix of shape (n, n).

    Returns
    -------
    numpy.ndarray
        Inverse matrix of shape (n, n).

    Raises
    ------
    ValueError
        If the matrix is singular or not square.
    """
    ...

def mean(values):
    """
    Arithmetic mean of an array.

    Parameters
    ----------
    values : numpy.ndarray
        Input array.

    Returns
    -------
    float
        Mean value.

    Raises
    ------
    ValueError
        If the array is empty.
    """
    ...

def power_iteration_eig(matrix, num_components, max_iter=1000, tol=1e-10):
    """
    Eigendecomposition via power iteration.

    Computes the top ``num_components`` eigenvalues and eigenvectors of a
    symmetric matrix using deflated power iteration.

    Parameters
    ----------
    matrix : numpy.ndarray
        Symmetric square matrix.
    num_components : int
        Number of leading eigenvalues/vectors to compute.
    max_iter : int, default 1000
        Maximum iterations per component.
    tol : float, default 1e-10
        Convergence tolerance.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(eigenvalues, eigenvectors)`` where eigenvalues has shape
        ``(num_components,)`` and eigenvectors has shape
        ``(n, num_components)`` with columns as eigenvectors.
    """
    ...

def simple_returns(prices):
    """
    Simple (arithmetic) returns from a price series.

    Computes ``(p[i] - p[i-1]) / p[i-1]`` for each consecutive pair.

    Parameters
    ----------
    prices : numpy.ndarray
        Price series of length n.

    Returns
    -------
    numpy.ndarray
        Simple returns of length n-1.
    """
    ...

def skewness(values):
    """
    Sample skewness (third standardized moment).

    Parameters
    ----------
    values : numpy.ndarray
        Input array (length >= 3).

    Returns
    -------
    float
        Skewness coefficient.
    """
    ...

def std_dev(values, ddof=1):
    """
    Sample standard deviation with configurable degrees of freedom.

    Parameters
    ----------
    values : numpy.ndarray
        Input array.
    ddof : int, default 1
        Delta degrees of freedom.

    Returns
    -------
    float
        Standard deviation.
    """
    ...

def variance(values, ddof=1):
    """
    Sample variance with configurable degrees of freedom.

    Parameters
    ----------
    values : numpy.ndarray
        Input array.
    ddof : int, default 1
        Delta degrees of freedom. Use 0 for population variance,
        1 for sample variance (Bessel's correction).

    Returns
    -------
    float
        Variance of the input.

    Raises
    ------
    ValueError
        If the array has fewer elements than ddof + 1.
    """
    ...

def weighted_mean(values, weights):
    """
    Weighted arithmetic mean.

    Parameters
    ----------
    values : numpy.ndarray
        Input values.
    weights : numpy.ndarray
        Non-negative weights (same length as values). Need not sum to 1.

    Returns
    -------
    float
        Weighted mean.
    """
    ...
