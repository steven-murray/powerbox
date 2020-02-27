"""
A set of tools for dealing with structured boxes, such as those output by :mod:`powerbox`. Tools include those
for averaging a field angularly, and generating the isotropic power spectrum.
"""
from . import dft
import numpy as np
import warnings


def _getbins(bins, coords, log):
    mx = coords.max()
    if not np.iterable(bins):
        if not log:
            bins = np.linspace(coords.min(), mx, bins + 1)
        else:
            mn = coords[coords>0].min()
            bins = np.logspace(np.log10(mn), np.log10(mx), bins + 1)

    return bins


def angular_average(field, coords, bins, weights=1, average=True, bin_ave=True, get_variance=False, log_bins=False):
    r"""
    Average a given field within radial bins.

    This function can be used in fields of arbitrary dimension (memory permitting), and the field need not be centred
    at the origin. The averaging assumes that the grid cells fall completely into the bin which encompasses the
    co-ordinate point for the cell (i.e. there is no weighted splitting of cells if they intersect a bin edge).

    It is optimized for applying a set of weights, and obtaining the variance of the mean, at the same time as
    averaging.

    Parameters
    ----------
    field: nd-array
        An array of arbitrary dimension specifying the field to be angularly averaged.

    coords: nd-array or list of n arrays.
        Either the *magnitude* of the co-ordinates at each point of `field`, or a list of 1D arrays specifying the
        co-ordinates in each dimension.

    bins: float or array.
        The ``bins`` argument provided to histogram. Can be an int or array specifying radial bin edges.

    weights: array, optional
        An array of the same shape as `field`, giving a weight for each entry.
        
    average: bool, optional
        Whether to take the (weighted) average. If False, returns the (unweighted) sum.

    bin_ave : bool, optional
        Whether to return the bin co-ordinates as the (weighted) average of cells within the bin (if True), or
        the regularly spaced edges of the bins.

    get_variance : bool, optional
        Whether to also return an estimate of the variance of the power in each bin.

    log_bins : bool, optional
        Whether to create bins in log-space.

    Returns
    -------
    field_1d : 1D-array
        The angularly-averaged field.

    bins : 1D-array
        Array of same shape as field_1d specifying the radial co-ordinates of the bins. Either the mean co-ordinate
        from the input data, or the regularly spaced bins, dependent on `bin_ave`.

    var : 1D-array, optional
        The variance of the averaged field (same shape as bins), estimated from the mean standard error.
        Only returned if `get_variance` is True.

    Notes
    -----
    If desired, the variance is calculated as the weight unbiased variance, using the formula at
    https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Reliability_weights for the variance in each cell, and
    normalising by a factor of :math:`V_2/V_1^2` to estimate the variance of the average.

    Examples
    --------
    Create a 3D radial function, and average over radial bins:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(-5,5,128)   # Setup a grid
    >>> X,Y,Z = np.meshgrid(x,x,x)
    >>> r = np.sqrt(X**2+Y**2+Z**2) # Get the radial co-ordinate of grid
    >>> field = np.exp(-r**2)       # Generate a radial field
    >>> avgfunc, bins = angular_average(field,r,bins=100)   # Call angular_average
    >>> plt.plot(bins, np.exp(-bins**2), label="Input Function")   # Plot input function versus ang. avg.
    >>> plt.plot(bins, avgfunc, label="Averaged Function")

    See Also
    --------
    angular_average_nd : Perform an angular average in a subset of the total dimensions.

    """
    if len(coords) == len(field.shape):
        # coords are a segmented list of dimensional co-ordinates
        coords = _magnitude_grid(coords)

    indx, bins, sumweight = _get_binweights(coords, weights, bins, average, bin_ave=bin_ave, log_bins=log_bins)

    if np.any(sumweight==0):
        warnings.warn("One or more radial bins had no cells within it.")

    res = _field_average(indx, field, weights, sumweight)

    if get_variance:
        var = _field_variance(indx, field, res, weights, sumweight)
        return res, bins, var
    else:
        return res, bins


def _magnitude_grid(x, dim=None):
    if dim is not None:
        return np.sqrt(np.sum(np.meshgrid(*([x ** 2] * dim)), axis=0))
    else:
        return np.sqrt(np.sum(np.meshgrid(*([X ** 2 for X in x])), axis=0))


def _get_binweights(coords, weights, bins, average=True, bin_ave=True, log_bins=False):

    # Get a vector of bin edges
    bins = _getbins(bins, coords, log_bins)

    indx = np.digitize(coords.flatten(), bins)

    if average or bin_ave:
        if not np.isscalar(weights):
            if coords.shape != weights.shape:
                raise ValueError("coords and weights must have the same shape!")
            sumweights = np.bincount(indx, weights=weights.flatten(), minlength=len(bins)+1)[1:-1]
        else:
            sumweights = np.bincount(indx, minlength=len(bins)+1)[1:-1]

        if average:
            binweight = sumweights
        else:
            binweight = 1*sumweights
            sumweights = np.ones_like(binweight)

        if bin_ave:
            bins = np.bincount(indx, weights=(weights * coords).flatten(), minlength=len(bins)+1)[1:-1] / binweight

    else:
        sumweights = np.ones(len(bins)-1)

    return indx, bins, sumweights


def _field_average(indx, field, weights, sumweights):
    if not np.isscalar(weights) and field.shape != weights.shape:
        raise ValueError("the field and weights must have the same shape!")

    field = field * weights  # Leave like this because field is mutable

    rl = np.bincount(indx, weights=np.real(field.flatten()), minlength=len(sumweights)+2)[1:-1] / sumweights
    if field.dtype.kind == "c":
        im = 1j * np.bincount(indx, weights=np.imag(field.flatten()), minlength=len(sumweights)+2)[1:-1] / sumweights
    else:
        im = 0

    return rl + im


def _field_variance(indx, field, average, weights, V1):
    if field.dtype.kind == "c":
        raise NotImplementedError("Cannot use a complex field when computing variance, yet.")

    # Create a full flattened array of the same shape as field, with the average in that bin.
    # We have to pad the average vector with 0s on either side to account for cells outside the bin range.
    average_field = np.concatenate(([0],average, [0]))[indx]

    # Create the V2 array
    if not np.isscalar(weights):
        weights = weights.flatten()
        V2 = np.bincount(indx, weights=weights**2, minlength=len(V1)+2)[1:-1]
    else:
        V2 = V1

    field = (field.flatten()-average_field)**2 * weights

    # This res is the estimated variance of each cell in the bin
    res = np.bincount(indx, weights=field, minlength=len(V1)+2)[1:-1] / (V1 - V2/V1)

    # Modify to the estimated variance of the sum of the cells in the bin.
    res *= V2 / V1**2

    return res


def angular_average_nd(field, coords, bins, n=None, weights=1, average=True, bin_ave=True, get_variance=False,
                       log_bins=False):
    """
    Average the first n dimensions of a given field within radial bins.

    This function be used to take "hyper-cylindrical" averages of fields. For a 3D field, with `n=2`, this is exactly
    a cylindrical average. This function can be used in fields of arbitrary dimension (memory permitting), and the field
    need not be centred at the origin. The averaging assumes that the grid cells fall completely into the bin which
    encompasses the co-ordinate point for the cell (i.e. there is no weighted splitting of cells if they intersect a bin
    edge).

    It is optimized for applying a set of weights, and obtaining the variance of the mean, at the same time as
    averaging.

    Parameters
    ----------
    field : md-array
        An array of arbitrary dimension specifying the field to be angularly averaged.

    coords : list of n arrays
        A list of 1D arrays specifying the co-ordinates in each dimension *to be average*.

    bins : int or array.
        Specifies the radial bins for the averaged dimensions. Can be an int or array specifying radial bin edges.

    n : int, optional
        The number of dimensions to be averaged. By default, all dimensions are averaged. Always uses
        the first `n` dimensions.
        
    weights : array, optional
        An array of the same shape as the first `n` dimensions of `field`, giving a weight for each entry.
        
    average : bool, optional
        Whether to take the (weighted) average. If False, returns the (unweighted) sum.

    bin_ave : bool, optional
        Whether to return the bin co-ordinates as the (weighted) average of cells within the bin (if True), or
        the linearly spaced edges of the bins

    get_variance : bool, optional
        Whether to also return an estimate of the variance of the power in each bin.

    log_bins : bool, optional
        Whether to create bins in log-space.

    Returns
    -------
    field : (m-n+1)-array
        The angularly-averaged field. The first dimension corresponds to `bins`, while the rest correspond to the
        unaveraged dimensions.

    bins : 1D-array
        The radial co-ordinates of the bins. Either the mean co-ordinate from the input data, or the regularly spaced
        bins, dependent on `bin_ave`.

    var : (m-n+1)-array, optional
        The variance of the averaged field (same shape as `field`), estimated from the mean standard error.
        Only returned if `get_variance` is True.

    Examples
    --------
    Create a 3D radial function, and average over radial bins. Equivalent to calling :func:`angular_average`:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(-5,5,128)   # Setup a grid
    >>> X,Y,Z = np.meshgrid(x,x,x)  # ""
    >>> r = np.sqrt(X**2+Y**2+Z**2) # Get the radial co-ordinate of grid
    >>> field = np.exp(-r**2)       # Generate a radial field
    >>> avgfunc, bins, _ = angular_average_nd(field,[x,x,x],bins=100)   # Call angular_average
    >>> plt.plot(bins, np.exp(-bins**2), label="Input Function")   # Plot input function versus ang. avg.
    >>> plt.plot(bins, avgfunc, label="Averaged Function")

    Create a 2D radial function, extended to 3D, and average over first 2 dimensions (cylindrical average):
    
    >>> r = np.sqrt(X**2+Y**2)
    >>> field = np.exp(-r**2)    # 2D field
    >>> field = np.repeat(field,len(x)).reshape((len(x),)*3)   # Extended to 3D
    >>> avgfunc, avbins, coords = angular_average_nd(field, [x,x,x], bins=50, n=2)
    >>> plt.plot(avbins, np.exp(-avbins**2), label="Input Function")
    >>> plt.plot(avbins, avgfunc[:,0], label="Averaged Function")
    """
    if n is None:
        n = len(coords)

    if len(coords) != len(field.shape):
        raise ValueError("coords should be a list of arrays, one for each dimension.")

    if n == len(coords):
        return angular_average(field, coords, bins, weights, average, bin_ave, get_variance, log_bins=log_bins)

    coords = _magnitude_grid([c for i, c in enumerate(coords) if i < n])

    indx, bins, sumweights = _get_binweights(coords, weights, bins, average, bin_ave=bin_ave, log_bins=log_bins)

    n1 = np.product(field.shape[:n])
    n2 = np.product(field.shape[n:])

    res = np.zeros((len(sumweights), n2), dtype=field.dtype)
    if get_variance:
        var = np.zeros_like(res)

    for i, fld in enumerate(field.reshape((n1, n2)).T):
        try:
            w = weights.flatten()
        except AttributeError:
            w = weights

        res[:, i] = _field_average(indx, fld, w, sumweights)

        if get_variance:
            var[:, i] = _field_variance(indx, fld, res[:,i], w, sumweights)

    if not get_variance:
        return res.reshape((len(sumweights),) + field.shape[n:]), bins
    else:
        return res.reshape((len(sumweights),) + field.shape[n:]), bins, var


def get_power(deltax, boxlength, deltax2=None, N=None, a=1., b=1., remove_shotnoise=True,
              vol_normalised_power=True, bins=None, res_ndim=None, weights=None, weights2=None,
              dimensionless=True, bin_ave=True, get_variance=False, log_bins=False, ignore_zero_mode=False,
              k_weights = 1,
              ):
    r"""
    Calculate the isotropic power spectrum of a given field, or cross-power of two similar fields.

    This function, by default, conforms to typical cosmological power spectrum conventions -- normalising by the volume
    of the box and removing shot noise if applicable. These options are configurable.

    Parameters
    ----------
    deltax : array-like
        The field on which to calculate the power spectrum . Can either be arbitrarily n-dimensional, or 2-dimensional with the
        first being the number of spatial dimensions, and the second the positions of discrete particles in the field. 
        The former should represent a density field, while the latter
        is a discrete sampling of a field. This function chooses which to use by checking the value of `N` (see below).
        Note that if a discrete sampling is used, the power spectrum calculated is the
        "overdensity" power spectrum, i.e. the field re-centered about zero and rescaled by the mean.

    boxlength : float or list of floats
        The length of the box side(s) in real-space.

    deltax2 : array-like
        If given, a box of the same shape as deltax, against which deltax will be cross correlated.

    N : int, optional
        The number of grid cells per side in the box. Only required if deltax is a discrete sample. If given,
        the function will assume a discrete sample.

    a,b : float, optional
        These define the Fourier convention used. See :mod:`powerbox.dft` for details. The defaults define the standard
        usage in *cosmology* (for example, as defined in Cosmological Physics, Peacock, 1999, pg. 496.). Standard
        numerical usage (eg. numpy) is (a,b) = (0,2pi).

    remove_shotnoise : bool, optional
        Whether to subtract a shot-noise term after determining the isotropic power. This only affects discrete samples.

    vol_weighted_power : bool, optional
        Whether the input power spectrum, ``pk``, is volume-weighted. Default True because of standard cosmological
        usage.

    bins : int or array, optional
        Defines the final k-bins output. If None, chooses a number based on the input resolution of the box. Otherwise,
        if int, this defines the number of kbins, or if an array, it defines the exact bin edges.
        
    res_ndim : int, optional
        Only perform angular averaging over first `res_ndim` dimensions. By default, uses all dimensions. 

    weights, weights2 : array-like, optional
        If deltax is a discrete sample, these are weights for each point.

    dimensionless: bool, optional
        Whether to normalise the cube by its mean prior to taking the power.

    bin_ave : bool, optional
        Whether to return the bin co-ordinates as the (weighted) average of cells within the bin (if True), or
        the linearly spaced edges of the bins

    get_variance : bool, optional
        Whether to also return an estimate of the variance of the power in each bin.

    log_bins : bool, optional
        Whether to create bins in log-space.

    ignore_zero_mode : bool, optional
        Whether to ignore the k=0 mode (or DC term).

    k_weights : nd-array, optional
        The weights of the n-dimensional k modes. This can be used to filter out some modes completely.

    Returns
    -------
    p_k : array
        The power spectrum averaged over bins of equal :math:`|k|`.

    meank : array
        The bin-centres for the p_k array (in k). This is the mean k-value for cells in that bin.

    var : array
        The variance of the power spectrum, estimated from the mean standard error. Only returned if `get_variance` is
        True.

    Examples
    --------
    One can use this function to check whether a box created with :class:`PowerBox` has the correct
    power spectrum:

    >>> from powerbox import PowerBox
    >>> import matplotlib.pyplot as plt
    >>> pb = PowerBox(250,lambda k : k**-2.)
    >>> p,k = get_power(pb.delta_x,pb.boxlength)
    >>> plt.plot(k,p)
    >>> plt.plot(k,k**-2.)
    >>> plt.xscale('log')
    >>> plt.yscale('log')
    """

    # Check if the input data is in sampled particle format
    if N is not None:

        if deltax.shape[1] > deltax.shape[0]:
            raise ValueError("It seems that there are more dimensions than particles! Try transposing deltax.")

        if deltax2 is not None and deltax2.shape[1] > deltax2.shape[0]:
            raise ValueError("It seems that there are more dimensions than particles! Try transposing deltax2.")

        dim = deltax.shape[1]
        if deltax2 is not None and dim != deltax2.shape[1]:
            raise ValueError("deltax and deltax2 must have the same number of dimensions!")

        if not np.iterable(N):
            N = [N] * dim

        if not np.iterable(boxlength):
            boxlength = [boxlength] * dim

        Npart1 = deltax.shape[0]

        if deltax2 is not None:
            Npart2 = deltax2.shape[0]
        else:
            Npart2 = Npart1

        # Generate a histogram of the data, with appropriate number of bins.
        edges = [np.linspace(0, L, n + 1) for L, n in zip(boxlength, N)]

        deltax = np.histogramdd(deltax % boxlength, bins=edges, weights=weights)[0].astype("float")

        if deltax2 is not None:
            deltax2 = np.histogramdd(deltax2 % boxlength, bins=edges, weights=weights2)[0].astype("float")

        # Convert sampled data to mean-zero data
        if dimensionless:
            deltax = deltax / np.mean(deltax) - 1
            if deltax2 is not None:
                deltax2 = deltax2 / np.mean(deltax2) - 1
        else:
            deltax -= np.mean(deltax)
            if deltax2 is not None:
                deltax2 -= np.mean(deltax2)
    else:
        # If input data is already a density field, just get the dimensions.
        dim = len(deltax.shape)

        if not np.iterable(boxlength):
            boxlength = [boxlength] * dim

        if deltax2 is not None and deltax.shape != deltax2.shape:
            raise ValueError("deltax and deltax2 must have the same shape!")

        N = deltax.shape
        Npart1 = None

    V = np.product(boxlength)

    # Calculate the n-D power spectrum and align it with the k from powerbox.
    FT, freq, k = dft.fft(deltax, L=boxlength, a=a, b=b, ret_cubegrid=True)

    if deltax2 is not None:
        FT2 = dft.fft(deltax2, L=boxlength, a=a, b=b)[0]
    else:
        FT2 = FT

    P = np.real(FT * np.conj(FT2) / V ** 2)

    if vol_normalised_power:
        P *= V

    if res_ndim is None:
        res_ndim = dim

    # Determine a nice number of bins.
    if bins is None:
        bins = int(np.product(N[:res_ndim]) ** (1. / res_ndim) / 2.2)

    # Set k_weights so that k=0 mode is ignore if desired.
    if ignore_zero_mode:
        kmag = _magnitude_grid([c for i, c in enumerate(freq) if i < res_ndim])
        if np.isscalar(k_weights):
            k_weights = np.ones_like(kmag)

        k_weights[kmag == 0] = 0

    # res is (P, k, <var>)
    res = angular_average_nd(P, freq, bins, n=res_ndim, bin_ave=bin_ave, get_variance=get_variance, log_bins=log_bins,
                             weights=k_weights)
    res = list(res)
    # Remove shot-noise
    if remove_shotnoise and Npart1:
        res[0] -= np.sqrt(V ** 2 / Npart1 / Npart2)

    if res_ndim < dim:
        return res + [freq[res_ndim:]]
    else:
        return res
