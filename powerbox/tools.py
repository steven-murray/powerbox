"""
A set of tools for dealing with structured boxes, such as those output by :mod:`powerbox`. Tools include those
for averaging a field isotropically, and generating the isotropic power spectrum.
"""
import dft
import numpy as np

def angular_average(field,coords,bins):
    r"""
    Perform a radial histogram -- averaging within radial bins -- of a field.

    Parameters
    ----------
    field : array
        An array of arbitrary dimension specifying the field to be angularly averaged.

    coords : array
        The magnitude of the co-ordinates at each point of `field`. Must be the same size as field.

    bins : float or array.
        The ``bins`` argument provided to histogram. Can be a float or array specifying bin edges.

    Returns
    -------
    field_1d : array
        The field averaged angularly (finally 1D)

    binavg : array
        The mean co-ordinate in each radial bin.

    Examples
    --------
    Create a 3D radial function, and average over radial bins:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(-5,5,128)   # Setup a grid
    >>> X,Y,Z = np.meshgrid(x,x,x)  # ""
    >>> r = np.sqrt(X**2+Y**2+Z**2) # Get the radial co-ordinate of grid
    >>> field = np.exp(-r**2)       # Generate a radial field
    >>> avgfunc, bins = angular_average(field,r,bins=100)   # Call angular_average
    >>> plt.plot(bins, np.exp(-bins**2), label="Input Function")   # Plot input function versus ang. avg.
    >>> plt.plot(bins, avgfunc, label="Averaged Function")
    """
    weights,edges = np.histogram(coords.flatten(), bins=bins)
    binav = np.histogram(coords.flatten(),bins=edges,weights=coords.flatten())[0]/weights
    return np.histogram(coords.flatten(),bins=edges,weights=field.flatten())[0]/weights, binav


def get_power(deltax,boxlength,N=None,a=1.,b=1., remove_shotnoise=True,
              vol_normalised_power=True,bins=None):
    r"""
    Calculate the isotropic power spectrum of a given field.

    Parameters
    ----------
    deltax : array-like
        The field to calculate the power spectrum of. Can either be arbitrarily n-dimensional, with each dimension
        the same size, or 2-dimensional with the first being the number of spatial dimensions, and the second
        the positions of discrete particles in the field. The former should represent a density field, while the latter
        is a discrete sampling of a field. Note that if a discrete sampling is used, the power spectrum calculated is the
        "overdensity" power spectrum, i.e. the field re-centered about zero and rescaled by the mean.

    boxlength : float
        The length of the box side in real-space.

    N : int, optional
        The number of grid cells per side in the box. Only required if deltax is a discrete sample.

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

    Returns
    -------
    p_k : array
        The power spectrum averaged over bins of equal ``|k|``.

    meank : array
        The bin-centres for the p_k array (in k). This is the mean k-value for cells in that bin.

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
    if len(deltax.shape)==2 and deltax.shape[0]!=deltax.shape[1]:
        if deltax.shape[1]>deltax.shape[0]:
            raise ValueError("It seems that there are more dimensions than particles! Try transposing deltax.")

        dim = deltax.shape[1]
        Npart = deltax.shape[0]

        # If so, generate a histogram of the data, with appropriate number of bins.
        edges = np.linspace(deltax.min(),deltax.min()+boxlength,N+1)
        deltax = np.histogramdd(deltax,bins=[edges]*dim)[0].astype("float")

        # Convert sampled data to mean-zero data
        deltax = deltax/np.mean(deltax) - 1

    else:
        # If input data is already a density field, just get the dimensions.
        dim = len(deltax.shape)
        N = len(deltax)
        Npart = None

    # Calculate the n-D power spectrum and align it with the k from powerbox.
    FT, _, k = dft.fft(deltax, L=boxlength, a=a, b=b, ret_cubegrid=True)
    P = np.abs(FT/boxlength**dim)**2

    if vol_normalised_power:
        P *= boxlength**dim

    # Generate the bin edges
    if bins is None:
        bins = int(N/2.2)

    p_k, kbins = angular_average(P[k!=0], k[k!=0], bins)

    # Remove shot-noise
    if remove_shotnoise and Npart:
        p_k -= boxlength**dim / Npart

    return p_k, kbins
