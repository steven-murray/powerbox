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


def hermitianize_full_array(arr: np.ndarray) -> None:
    """Make the input array Hermitian in place.

    The input array can have arbitrary number of dimensions and arbitrary shape. It
    may be in either standard or centered Fourier space ordering (either way, the
    'special' frequencies are the first and N//2 slices along even axes). The output
    array will be in the same ordering as the input array, but will be modified in-place
    to be Hermitian.

    The input array should NOT be an rfft array.

    Parameters
    ----------
    arr : np.ndarray
        The input array to be made Hermitian.
    """
    if arr.ndim == 0:
        arr[...] = np.real(arr)
        return

    axis_is_even = tuple(n % 2 == 0 for n in arr.shape)

    # Get the sub-space of the array that have negative partners. This excludes the
    # "nyquist" frequency (i.e. the first slice along even axes). These are easy to
    # make Hermitian by averaging with their conjugate partners.
    odd_subvolume = tuple(slice(1, None) if even else slice(None) for even in axis_is_even)
    view = arr[odd_subvolume]
    view += np.conj(np.flip(view, axis=tuple(range(arr.ndim))))
    view /= 2

    # Now we need to make the "nyquist" frequencies Hermitian. We can literally do this
    # by hermitianizing each sub-slice as if it were its own independent array.
    for axis in range(arr.ndim):
        if axis_is_even[axis]:
            subslice = tuple(slice(None) if ax != axis else 0 for ax in range(arr.ndim))
            view = arr[subslice]
            if np.ndim(view) == 0:
                arr[subslice] = np.real(view)
            else:
                hermitianize_full_array(view)


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
    """
    # The only "special" frequencies in an rfft array are the first and last slices
    # along the last axis (the axis that has been halved). We can make these Hermitian
    # by hermitianizing each sub-slice as if it were its own independent array.
    special_indices = [0]
    if has_nyquist:
        special_indices.append(-1)

    for surface_index in special_indices:
        view = arr[..., surface_index]
        if np.ndim(view) == 0:
            arr[..., surface_index] = np.real(view)
        else:
            hermitianize_full_array(view)
