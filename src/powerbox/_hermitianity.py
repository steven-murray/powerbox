"""Functions for ensuring nd-arrays are Hermitian.

By hermitian, we mean that the IFFT of the array is real-valued.

We will assume that the fourier-space arrays are *centered* in the sense that the
zero-frequency component is at the center of the array. In this case, the Hermitian
condition is:

    A[i, j, k] = conj(A[-i, -j, -k])

Furthermore, for even-sized axes, the sub-space of that axis is also hermitian. For
example, if in the 3D case, the 2nd axis is even-sized, then we also have:

    A[i, 0, k] = conj(A[-i, 0, -k])
"""

import numpy as np


def conjugate_partner(arr: np.ndarray) -> np.ndarray:
    """Return the array re-indexed so that element ``k`` holds the element at ``-k``.

    The input is assumed to be in *centered* Fourier ordering. Along an odd-length axis
    every frequency has a distinct partner, so negating the frequency is a plain flip.
    An even-length axis carries one extra, unpaired Nyquist frequency at index 0, which
    is its own negative; rolling the flipped axis by one keeps it in place and pairs the
    rest correctly.

    Parameters
    ----------
    arr : np.ndarray
        A centered Fourier-space array of any shape, including 0-dimensional.

    Returns
    -------
    np.ndarray
        An array of the same shape, with element ``k`` taken from element ``-k``.
    """
    partner = np.flip(arr, axis=tuple(range(arr.ndim)))
    even_axes = tuple(axis for axis, n in enumerate(arr.shape) if n % 2 == 0)
    if even_axes:
        partner = np.roll(partner, shift=1, axis=even_axes)
    return partner


def hermitianize_full_array(arr: np.ndarray) -> None:
    """Make the input array Hermitian in place, without changing the power of any mode.

    Each mode is replaced by ``(A[k] + conj(A[-k])) / sqrt(2)``. This is a projection
    onto the Hermitian subspace which, unlike averaging the two, is norm-preserving: if
    the input modes are independent with unit variance, so are the output modes.

    * For a mode with a distinct partner, the two independent unit variances combine to
      ``(1 + 1) / 2 = 1``.
    * For a self-conjugate mode (every index at 0 or Nyquist), the result is
      ``sqrt(2) Re(A[k])``, which is real, as it must be, and again has unit variance.

    Averaging instead of this would halve the power of every self-conjugate *surface* --
    which in three dimensions is a substantial fraction of the lowest-|k| modes.

    The input array should NOT be an rfft array.

    Parameters
    ----------
    arr : np.ndarray
        The input array to be made Hermitian, in centered Fourier ordering.
    """
    arr[...] = (arr + np.conj(conjugate_partner(arr))) / np.sqrt(2)


def hermitianize_rfft_array(arr: np.ndarray, has_nyquist: bool) -> None:
    """Make the input rfft array Hermitian in place.

    The input array can have arbitrary number of dimensions and arbitrary shape, but
    must be in rfft ordering (i.e. the last axis is the "positive frequencies" axis).
    The output array will be in the same ordering as the input array, but will be
    modified in-place to be Hermitian.

    Parameters
    ----------
    arr : np.ndarray
        The input rfft array to be made Hermitian.
    has_nyquist : bool
        Whether the real-space axis corresponding to the reduced final axis has even
        length, in which case that axis carries a Nyquist surface as well.
    """
    # The only "special" frequencies in an rfft array are the first and last slices
    # along the last axis (the axis that has been halved). Every other mode already has
    # its conjugate partner outside the stored half-spectrum, so is unconstrained.
    special_indices = [0]
    if has_nyquist:
        special_indices.append(-1)

    for surface_index in special_indices:
        hermitianize_full_array(arr[..., surface_index])
