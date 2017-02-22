import numpy as np
import warnings
from cached_property import cached_property

from fft import fft,ifft, fftfreq
#
# # Try importing the pyFFTW interface
# try:
#     from multiprocessing import cpu_count
#     THREADS = cpu_count()
#
#     from pyfftw.interfaces.numpy_fft import fftn as _fftn, ifftn as _ifftn, ifftshift, fftshift, fftfreq
#     from pyfftw.interfaces.cache import enable, set_keepalive_time
#     enable()
#     set_keepalive_time(10.)
#
#     def fftn(*args,**kwargs):
#         return _fftn(threads=THREADS,*args,**kwargs)
#
#     def ifftn(*args, **kwargs):
#         return _ifftn(threads=THREADS,*args, **kwargs)
#
#     HAVE_FFTW = True
#
# except ImportError:
#     warnings.warn("You do not have pyFFTW installed. Installing it should give some speed increase.")
#     HAVE_FFTW = False
#     from numpy.fft import fftn, ifftn, ifftshift, fftshift, fftfreq


#TODO: add hankel-transform version of LogNormal

class PowerBox(object):
    def __init__(self,N,pk, dim=2, boxlength=1.0, ensure_physical=False, a=1.,b=1.,
                 vol_normalised_power = True,seed=None):
        """
        An object which calculates and stores the real-space and fourier-space fields generated with a given power
        spectrum.

        Parameters
        ----------
        N : int
            Number of grid-points on a side for the resulting box (equivalently, number of wavenumbers to use).

        pk : func
            A function of a single (vector) variable k, which is the isotropic power spectrum.

        dim : int, default 2
            Number of dimensions of resulting box.

        boxlength : float, default 1.0
            Length of the final signal on a side. This may have arbitrary units, so long as `pk` is a function of a
            variable which has the inverse units.

        ensure_physical : bool, optional
            Interpreting the power spectrum as a spectrum of density fluctuations, the minimum physical value of the
            real-space field, :meth:`delta_x`, is -1. With ``ensure_physical`` set to ``True``, :meth:`delta_x` is
            clipped to return values >-1. If this is happening a lot, consider using a log-normal box.

        a,b : float, optional
            These define the Fourier convention used. See :module:`fft` for details. The defaults define the standard
            usage in *cosmology* (for example, as defined in Cosmological Physics, Peacock, 1999, pg. 496.). Standard
            numerical usage (eg. numpy) is (a,b) = (0,2pi).

        vol_weighted_power : bool, optional
            Whether the input power spectrum, ``pk``, is volume-weighted. Default True because of standard cosmological
            usage.


        Notes
        -----
        A number of conventions need to be listed.

        The conventions of using `x` for "real-space" and `k` for "fourier space" arise from cosmology, but this does
        not affect anything -- `x` could just as well stand for "time domain" and `k` for "frequency domain".

        The important convention is the relationship between `x` and `k`, or in other words, whether `k` is interpreted
        as an angular frequency or ordinary frequency. By default, because of cosmological conventions, `k` is an
        angular frequency, so that the fourier transform integrand is delta_k*exp(-ikx).

        The normalisation of the FT is set so as to return a statistically invariant real-space field with respect to
        the resolution of the grid. That is, increasing the resolution does not change the variance of the resulting
        field on a given scale. Again, this conforms to physical expectation in terms of cosmological usage.

        The primary quantity of interest is `delta_x`, which is a zero-mean Gaussian field with a power spectrum
        equivalent to that which was input. Being zero-mean enables its direct interpretation as a cosmological overdensity
        field (but should also be applicable to many other fields).
        Also, it means that the resulting field can be sampled such that the sampling densities in each grid-cell
        is given by n(1 + delta_x), with n some arbitrary mean number density across the field.

        Examples
        --------
        To create a 3-dimensional box of gaussian over-densities, with side length 1 Mpc, gridded equally into
        100 bins, and where k=2pi/x, with a power-law power spectrum, simply use

        >>> pb = PowerBox(100,lambda k : 0.1*k**-3., dim=3,boxlength=100.0)
        >>> overdensities = pb.delta_x
        >>> grid = pb.x
        >>> radii = pb.r

        To create a 2D turbulence structure, with arbitrary units, once can use

        >>> import matplotlib.pyplot as plt
        >>> pb = PowerBox(1000, lambda k : k**-7./5.)
        >>> plt.imshow(pb.delta_x)
        """

        self.N = N
        self.dim = dim
        self.boxlength = boxlength
        self.L = boxlength
        self.fourier_a = a
        self.fourier_b = b
        self.vol_normalised_power = vol_normalised_power
        self.V = self.boxlength**self.dim


        if self.vol_normalised_power:
            self.pk = lambda k : pk(k)/self.V
        else:
            self.pk = pk

        self.ensure_physical = ensure_physical
        self.Ntot = self.N**self.dim

        self.seed = seed

        # Our algorithm at this point only deals with even-length arrays.
        # assert N%2 == 0
        if N%2 == 0:
            self._even = True
        else:
            self._even = False

        self.n = N + 1 if self._even else N

        # Get the grid-size for the final real-space box.
        self.dx = float(boxlength)/N


    @cached_property
    def k(self):
        "The entire grid of wavenumber magitudes"
        k = self.kvec ** 2
        K = self.kvec ** 2
        for i in range(self.dim - 1):
            k = np.add.outer(K, k)
        return np.sqrt(k)

    @property
    def kvec(self):
        "The vector of wavenumbers along a side"
        return fftfreq(self.N, d=self.dx, b=self.fourier_b)

    @property
    def r(self):
        "The radial position of every point in the grid"
        X = self.x**2
        x = self.x**2
        for i in range(self.dim - 1):
            x = np.add.outer(X, x)
        return np.sqrt(x)

    @property
    def x(self):
        "The co-ordinates of the grid along a side"
        return np.arange(-self.boxlength/2,self.boxlength/2,self.dx)[:self.N]

    @cached_property
    def gauss_hermitian(self):
        "A random array which has Gaussian magnitudes and Hermitian symmetry"
        if self.seed:
            np.random.seed(self.seed)

        mag = np.random.normal(0, 1, size=[self.n]*self.dim)
        pha = 2*np.pi*np.random.uniform(size=[self.n]*self.dim)

        dk = self._make_hermitian(mag, pha)

        if self._even:
            cutidx = [slice(None, -1)]*self.dim
            dk = dk[cutidx]

        return dk

    @property
    def power_array(self):
        "The Power Spectrum (volume normalised) at `self.k`"
        k = self.k
        P = np.zeros_like(k)
        P[k != 0] = self.pk(k[k != 0])
        return P

    @cached_property
    def delta_k(self):
        "A realisation of the delta_k, i.e. the gaussianised square root of the power spectrum (i.e. the Fourier co-efficients)"
        return np.sqrt(self.power_array)*self.gauss_hermitian

    @cached_property
    def delta_x(self):
        "The realised field in real-space from the input power spectrum"
        # Here we multiply by V because the (inverse) fourier-transform of the (dimensionless) power has
        # units of 1/V and we require a unitless quantity for delta_x.
        deltax = self.V * ifft(self.delta_k,L=self.boxlength,a=self.fourier_a,b=self.fourier_b)[0]
        deltax = np.real(deltax)

        if self.ensure_physical:
            np.clip(deltax,-1,np.inf,deltax)

        return deltax

    def _make_hermitian(self,mag,pha):
        """
        Take random arrays and convert them to a complex hermitian array.

        Note that this assumes that mag is distributed normally.

        Parameters
        ----------
        mag : array
            Normally-distributed magnitudes of the complex vector.

        pha : array
            Uniformly distributed phases of the complex vector

        Returns
        -------
        kspace : array
            A complex hermitian array with normally distributed amplitudes.
        """
        revidx = [slice(None,None,-1)]*len(mag.shape)
        mag = (mag + mag[revidx])/np.sqrt(2)
        pha = (pha - pha[revidx])/2 + np.pi
        return mag*(np.cos(pha) + 1j*np.sin(pha))

    def create_discrete_sample(self,nbar,randomise_in_cell=True,min_at_zero=False,
                               store_pos=False,seed=None):
        """
        Assuming that the real-space signal represents an over-density with respect to some mean, create a sample
        of tracers of the underlying density distribution.

        Parameters
        ----------
        nbar : float
            Mean tracer density within the box.
        """
        if seed:
            np.random.seed(seed)

        n = (self.delta_x + 1)*self.dx ** self.dim * nbar
        self.n_per_cell = np.random.poisson(n)

        # Get all source positions
        args = [self.x]*self.dim
        X = np.meshgrid(*args)

        tracer_positions = np.array([x.flatten() for x in X]).T
        tracer_positions = tracer_positions.repeat(self.n_per_cell.flatten(), axis=0)

        if randomise_in_cell:
            tracer_positions += np.random.uniform(size=(np.sum(self.n_per_cell), self.dim))*self.dx

        if min_at_zero:
            tracer_positions += self.boxlength/2.0

        if store_pos:
            self.tracer_positions = tracer_positions

        return tracer_positions


class LogNormalPowerBox(PowerBox):
    """
    A subclass of :class:`PowerBox` which calculates Log-Normal density fields with given power spectra.

    Please read the documentation for :class:`PowerBox` for a detailed explanation. In brief, this class calculates
    an (over-)density field of arbitrary dimension given an input isotropic power spectrum. In this case, the field
    has a log-normal distribution of over-densities, always yielding a physically valid field.

    Examples
    --------
    To create a log-normal over-density field:

    >>> from powerbox import LogNormalPowerBox
    >>> lnpb = LogNormalPowerBox(100,lambda k : k**-7./5.,dim=2, boxlength=1.0)
    >>> overdensities = lnpb.delta_x
    >>> grid = lnpb.x
    >>> radii = lnpb.r

    To plot the overdensities:

    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(pb.delta_x)

    Compare the fields from a Gaussian and Lognormal realisation with the same power:

    >>> lnpb = LogNormalPowerBox(300,lambda k : k**-7./5.,dim=2, boxlength=1.0)
    >>> pb = PowerBox(300,lambda k : k**-7./5.,dim=2, boxlength=1.0)
    >>> fig,ax = plt.subplots(2,1,sharex=True,sharey=True,figsize=(12,5))
    >>> ax[0].imshow(lnpb.delta_x,aspect="equal",vmin=-1,vmax=lnpb.delta_x.max())
    >>> ax[1].imshow(pb.delta_x,aspect="equal",vmin=-1,vmax = lnpb.delta_x.max())

    To create and plot a discrete version of the field:

    >>> positions = lnpb.create_discrete_sample(nbar=1000.0, # Number density in terms of boxlength units
    >>>                                         randomise_in_cell=True)
    >>> plt.scatter(positions[:,0],positions[:,1],s=2,alpha=0.5,lw=0)

    """
    def __init__(self,*args,**kwargs):
        super(LogNormalPowerBox,self).__init__(*args,**kwargs)

    @cached_property
    def correlation_array(self):
        "The correlation function from the input power, on the grid"
        return self.V * np.real(ifft(self.power_array, L = self.boxlength,a=self.fourier_a, b=self.fourier_b)[0])

    @cached_property
    def gaussian_correlation_array(self):
        "The correlation function required for a Gaussian field to produce the input power on a lognormal field"
        return np.log(1 + self.correlation_array)

    @cached_property
    def gaussian_power_array(self):
        "The power spectrum required for a Gaussian field to produce the input power on a lognormal field"
        gpa =  np.abs(fft(self.gaussian_correlation_array,L=self.boxlength,a=self.fourier_a, b=self.fourier_b))[0]
        gpa[self.k==0] = 0
        return gpa

    @cached_property
    def delta_k(self):
        """
        A realisation of the delta_k, i.e. the gaussianised square root of the unitless power spectrum
        (i.e. the Fourier co-efficients)
        """
        return np.sqrt(self.gaussian_power_array) *self.gauss_hermitian

    @cached_property
    def delta_x(self):
        "The real-space over-density field, from the input power spectrum"
        deltax = np.sqrt(self.V)* ifft(self.delta_k, L=self.boxlength, a=self.fourier_a, b=self.fourier_b)[0]
        deltax = np.real(deltax)

        sg = np.var(deltax)
        return np.exp(deltax - sg/2) -1


def angular_average(field,coords,bins):
    """
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
    """
    weights,edges = np.histogram(coords.flatten(), bins=bins)
    binav = np.histogram(coords.flatten(),bins=bins,weights=coords.flatten())[0]/weights
    return np.histogram(coords.flatten(),bins=bins,weights=field.flatten())[0]/weights, binav


def get_power(deltax,boxlength,N=None,a=1.,b=1., remove_shotnoise=True,
              vol_normalised_power=True,bins=None):
    """
    Calculate the n-ball-averaged power spectrum of a given field.

    Parameters
    ----------
    deltax : array-like
        The field to calculate the power spectrum of. Can either be arbitrarily n-dimensional, with each dimension
        should have the same size, or 2-dimensional with the first being the number of spatial dimensions, and the second
        the positions of discrete particles in the field. The former should represent a density field, while the latter
        is a discrete sampling of a field. Note that if a discrete sampling is used, the power spectrum calculated is the
        "overdensity" power spectrum, i.e. the field re-centered about zero and rescaled by the mean.

    boxlength : float
        The length of the box side in real-space.

    ncells : int, optional
        The number of grid cells per side in the box. Only required if deltax is a discrete sample.

    angular_freq : bool, optional
        Whether the fourier-dual of `x` (called `k` in this code) is an angular frequency (i.e. k = 2pi/x) or not
        (i.e. k = 1/x).

    remove_shotnoise : bool, optional
        Whether to subtract a shot-noise term after determining the isotropic power. This only affects discrete samples.

    bins : int or array, optional
        Defines the final k-bins outputted. If None, chooses a number based on the input resolution of the box. Otherwise,
        if int, this defines the number of kbins, or if an array, it defines the exact bin edges.

    Returns
    -------
    p_k : array
        The power spectrum averaged over bins of equal |k|.

    meank : array
        The bin-centres for the p_k array (in k). This is the mean k-value for cells in that bin.

    Examples
    --------
    One can use this function to check whether a box created with :class:`PowerBox` has the correct
    power spectrum:

    >>> from powerbox import PowerBox, get_power
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
    FT, _, k = fft(deltax, L=boxlength, a=a, b=b,ret_cubegrid=True)
    P = np.abs(FT/boxlength**dim)**2

    if vol_normalised_power:
        P *= boxlength**dim

    # Generate the bin edges
    if bins is None:
        bins = int(N/2.2)

    p_k, kbins = angular_average(P[k!=0], k[k!=0], bins)

    # # Get matching flattened arrays
    # P = P[k!=0].flatten()
    # k = k[k!=0].flatten()
    #
    # # Generate the number of kgrid-cells in each bin
    # hist1 = np.histogram(k,bins=bins)[0]
    #
    # # Average the power spectrum in each bin
    # p_k = np.histogram(k,bins=bins,weights=P)[0]/hist1
    #
    # # Average the k-value in each bin.
    # meank = np.histogram(k,bins=bins,weights=k)[0]/hist1

    # Remove shot-noise
    if remove_shotnoise and Npart:
        p_k -= boxlength**dim / Npart

    return p_k, kbins
