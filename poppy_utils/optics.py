import astropy.units as u
import poppy
from poppy import utils, wfe
from poppy.accel_math import xp
from poppy.wfe import _wave_y_x_to_rho_theta

def get_conic_sag(roc, k, r):
    """
    Get the sag for a given conic

    Parameters
    ------------
    roc : float
        Radius of curvature for the conic
    k : float
        conic constant
    r : float or array-like
        Radial distance from the vertex

    Returns
    ------------
    sag : float or array-like
        Sag of the conic evaluated at r
    """
    c = 1/roc
    return c*(r**2) / (1 + xp.sqrt(1-(1+k)*(c*r)**2))

def get_conic_parent_foci(roc, k):
    """
    Get the parent foci of a general conic
    (i.e., measured from the vertex)

    Parameters
    ------------
    roc : float
        Radius of curvature for the conic
    k : float
        conic constant

    Returns
    ------------
    f1, f2 : tuple of floats
        Two foci (one may be infinity)
    """
    eps = xp.sqrt(-1*k)

    if k == -1: # avoid dividing by zero
        return roc/2, xp.inf
    return roc/(1+eps), roc/(1-eps)

def get_conic_effective_foci(roc, k, oad):
    """
    Get the effective foic of a general conic
    measured from the off-axis distance

    Parameters
    -----------
    roc : float
        Radius of curvature for the conic
    k : float
        conic constant
    oad : float
        Distance from the vertex to the center
        of the optic

    Returns
    ------------
    f1, f2 : tuple of floats
        Two effective foci measured from OAD (one may be infinity)
    """

    f1_parent, f2_parent = get_conic_parent_foci(roc, k)
    sag = get_conic_sag(roc, k, oad)

    # this gets the absolute distance to the foci but doesn't capture signs
    # (which matters for hyperbolae)
    #f1_eff = xp.sqrt( (f1_parent - sag)**2 + oad**2)
    #f2_eff = xp.sqrt( (f2_parent - sag)**2 + oad**2)

    # this captures sign and distance
    f1_eff = ( f1_parent - sag ) / xp.cos(xp.arctan(oad/(f1_parent - sag)))
    f2_eff = ( f2_parent - sag ) / xp.cos(xp.arctan(oad/(f2_parent - sag)))

    return f1_eff, f2_eff

def get_on_axis_conic_from_off_axis_conic(roc, k, oad):
    """
    Given an off-axis conic, create the on-axis
    equivalent (i.e., match the effective focal length
    of the input off-axis conic to an on-axis parent)

    Note that for a parabola, the on-axis equivalent is
    another parabola with a different effective ROC.

    For other conics, both the ROC and conic constant
    will change.

    Parameters
    -----------
    roc : float
        Radius of curvature for the conic
    k : float
        conic constant
    oad : float
        Distance from the vertex to the center
        of the optic

    Returns
    ------------
    roc_new : float
        Radius of curvature for on-axis conic
    k_new : float
        Conic constant for on-axis conic

    """
    f1_eff, f2_eff = get_conic_effective_foci(roc, k, oad)

    if k == -1: # special case parabola
        k_new = -1
        roc_new = 2*f1_eff
    else: # other conics (is this fully general?)
        roc_new = 2 * (f1_eff*f2_eff) / (f1_eff + f2_eff)
        eps = (f2_eff - f1_eff) / (f1_eff + f2_eff)
        k_new = - eps**2

    return roc_new, k_new

def get_effective_distance_from_conic_vertex(roc, k, oad, vertex_distance):
    """
    Zemax thickness values are vertex-to-vertex distances,
    which means---in the case of off-axis optics---they can't
    directly be used in poppy, which is expecting a distance
    from the off-axis center.

    This function converts from the thickness (vertex distance)
    to an effective distance.

    By way of example, if using an OAP to go from a collimated space
    to the image:
    * the zemax thickness is the focal length of the parent conic
    * this function returns the effective focal length of the OAP.
    """

    sag = get_conic_sag(roc, k, oad)
    y = oad
    x = vertex_distance - sag
    return xp.sqrt(x**2 + y**2)

class Model(object):
    """
    This is intended to be a lightweight wrapper around an
    optical system. How necessary is this?

    Things this could add:
    * Easy broadband (calc_psf does this)
    * Tracking DM state?
    * Tracking tip/tilt (not sure this is necessary)
    * Partial propagation?
    """

    def __init__(self):
        """
        sdf
        """
        pass

    def run_mono(self, end_idx=-1, return_intermediates=False, wf=None, tiptilt=(0,0)):
        if wf is None:
            wf = some_default
        
        wf.tilt(tiptilt) # what units?

        self.osys.propagate(wf) # other units?

    def run_broadband(self, waves, wf=None, tiptilt=(0,0)):
        pass


class ConicPhase(poppy.QuadraticLens):
    """
    General phase for a conic surface

    If an off-axis-distance (OAD) is specified, this
    computes an on-axis new conic whose parent focal lengths
    match the effective focal lengths of the off-axis conic.

    TO DO: Note that this assumes a quadratic approximation of the 
    on-axis conic. This should preserve the expected imaging properties
    when used at appropriate conjugates but won't capture aberrations that
    arise from using conics at the wrong conjugates (e.g., a spherical surface
    used at infinite conjugates will give perfect imaging--but in reality this
    scenario would give you spherical aberration.)

    TO DO: generalize this to include surface sag that would capture aberrations
    from wrong conjugates? (on-axis only)

    Parameters
    ----------
    roc : float or astropy.Quantity of type length
        Radius of curvature
    oad : float or astropy.Quantity of type length
        Off-axis distance of the OAP
    k : float or astropy.Quantity of type length
        Conic constant
    name : string
        Descriptive string name
    planetype : poppy.PlaneType constant
        plane type
    """

    def __init__(self,
                 roc,
                 k,
                 oad=0 * u.m,
                 planetype=poppy.optics.PlaneType.intermediate,
                 name='Conic Wavefront Curvature Operator',
                 **kwargs):

        self._roc_input = roc
        self._k_input = k

        if oad.value == 0:
            self.roc = roc
            self.k = k
        else:
            # for off-axis conics, construct an on-axis conic that
            # preserves the effective foci of the off-axis input
            roc_new, k_new = get_on_axis_conic_from_off_axis_conic(roc, k, oad)
            self.roc = roc_new
            self.k = k_new

        self.oad = oad # ignored for now

        # quadratic phase approximation for Fresnel propagation
        f_lens = self.roc / 2

        super().__init__(f_lens=f_lens,
                        planetype=planetype,
                        name=name,
                        **kwargs)


    # def get_opd(self, wave):
    #     """
    #     TO DO: not sure if we want this any more -- we're making quadratic approx for this
    #     anyway
    #     """

    #     y, x = wave.coordinates()

    #     r = xp.sqrt(x**2 + y**2)
    #     opd = 2*get_conic_sag(self.roc.to(u.m).value, self.k, r)

    #     return opd
    
class OAP(ConicPhase):
    """
    Thin wrapper for ConicPhase with conic constant k=-1
    """

    def __init__(self,
                 roc,
                 oad=0 * u.m,
                 planetype=poppy.optics.PlaneType.intermediate,
                 name='OAP',
                 **kwargs):
        super().__init__(roc,
                         -1,
                         oad=oad,
                         planetype=planetype,
                         name=name,
                         **kwargs)
    
# class OAP(poppy.QuadraticLens):
#     """
#     On-axis representation of an off-axis parabola (OAP)

#     Thin wrapper for QaudraticLens. This calculates
#     an effective focal length from a radius of curvature
#     and an off-axis distance and then represents
#     the OAP with a quadratic surface sag.

#     Parameters
#     ----------
#     roc : float or astropy.Quantity of type length
#         Radius of curvature
#     oad : float or astropy.Quantity of type length
#         Off-axis distance of the OAP
#     k : float or astropy.Quantity of type length
#         Conic constant
#     name : string
#         Descriptive string name
#     planetype : poppy.PlaneType constant
#         plane type

#     """

#     @utils.quantity_input(f_lens=u.m)
#     def __init__(self,
#                  roc=1.0 * u.m,
#                  oad=1.0 * u.m,
#                  planetype=poppy.optics.PlaneType.unspecified,
#                  name='OAP',
#                  **kwargs):
#         f_lens = self._get_oap_fl(roc, oad)
#         poppy.QuadPhase.__init__(self,
#                            f_lens,
#                            planetype=planetype,
#                            name=name,
#                            **kwargs)
#         self.fl = f_lens.to(u.m)
#         poppy.utils._log.debug("Initialized: " + self.name + ", fl ={0:0.2e}".format(self.fl))
        
#     def _get_oap_fl(self, roc, oad):
#         # conic focal lenght for conic constatn k=-1
#         vfl = roc/2
#         del0 = oad**2 / (2*roc)
#         a   = vfl - del0
#         efl = xp.sqrt(a**2 + oad**2)
#         return efl

#     def __str__(self):
#         return "OAP: {0}, with focal length {1}".format(self.name, self.fl)
    

# class ConicPhase(poppy.optics.AnalyticOpticalElement):
#     """
#     General phase for a conic surface

#     Parameters
#     --------------
#     TBD


#     TO DO:
#     * drop OAD? This is dangerous

#     """

#     def __init__(self,
#                  roc,
#                  k,
#                  oad=0 * u.m,
#                  planetype=poppy.optics.PlaneType.intermediate,
#                  name='Conic Wavefront Curvature Operator',
#                  **kwargs):
#         poppy.AnalyticOpticalElement.__init__(self,
#                                               name=name,
#                                               planetype=planetype,
#                                               **kwargs)
#         self.roc = roc
#         self.k = k
#         self.oad = oad # ignored for now
        
#     def get_opd(self, wave):
#         """
        
#         TO DO: enforce some limits to the curvature (can't go imaginary)
        
#         """
#         y, x = wave.coordinates()
        
#         opd = (x**2 + y**2) / ( self.roc + np.sqrt(self.roc**2 - (self.k+1)*(x**2 + y**2)) )
#         return opd
    
# class Conic(poppy.QuadraticLens):
#     """
#     On-axis representation of a general conic surface

#     Thin wrapper for QaudraticLens. This calculates
#     an effective focal length from a radius of curvature,
#     conic constant, and an off-axis distance and then represents
#     the OAP with a quadratic surface sag.
    
#     TO DO: This just calculate a single EFL, which can't be right
#     for a general conic.

#     Parameters
#     ----------
#     roc : float or astropy.Quantity of type length
#         Radius of curvature
#     oad : float or astropy.Quantity of type length
#         Off-axis distance of the OAP
#     k : float or astropy.Quantity of type length
#         Conic constant
#     name : string
#         Descriptive string name
#     planetype : poppy.PlaneType constant
#         plane type

#     """

#     @utils.quantity_input(roc=u.m, oad=u.m)
#     def __init__(self,
#                  roc=1.0 * u.m,
#                  k = -1,
#                  oad=1.0 * u.m,
#                  planetype=poppy.optics.PlaneType.unspecified,
#                  name='OAP',
#                  **kwargs):
#         f_lens = self._get_conic_fl(roc, k, oad)
#         poppy.QuadPhase.__init__(self,
#                            f_lens,
#                            planetype=planetype,
#                            name=name,
#                            **kwargs)
#         self.fl = f_lens.to(u.m)
#         poppy.utils._log.debug("Initialized: " + self.name + ", fl ={0:0.2e}".format(self.fl))

#     def _get_conic_fl(self, roc, k, oad):
#         f_parent = roc/2
#         sag = oad**2 / ( roc + xp.sqrt(roc**2 - (k+1)*oad**2) )
#         efl = ( f_parent - sag ) / xp.cos(xp.arctan(oad/(f_parent - sag)))
#         return efl

#     def __str__(self):
#         return "Conic: {0}, with focal length {1}".format(self.name, self.fl)


class ABCPSDWFE(poppy.WavefrontError):
    """
    Statistical ABC PSD WFE class from power law for optical noise.

    To do: eqn goes here.
    To do: parameter a is currently ignored
    To do: if seed is used, it'll generate the same surface for both amplitude and phase

    Parameters
    ----------
    name : string
        name of the optic
    psd_params: list of floats
        (a,b,c)
    wfe: astropy quantity
        RMS wfe in linear astropy units, defaults to 50 nm
    radius: astropy quantity
        radius of optic in linear astropy units, defaults to 1 m
    seed : integer
        seed for the random phase screen generator
    """

    @utils.quantity_input(wfe_rms=u.nm)
    def __init__(self, name='ABC PSD WFE',
                 wfe_params=(1.0, 3.0, 2.5),
                 wfe_rms=50*u.nm,
                 amp_params=(1.0, 3.0, 2.5),
                 amp_rms=0,
                 seed=None,
                 **kwargs):

        super().__init__(name=name, **kwargs)
        self.wfe_params = wfe_params
        self.wfe_rms = wfe_rms
        self.amp_params = amp_params
        self.amp_rms = amp_rms
        self.seed = seed

    @wfe._check_wavefront_arg
    def get_opd(self, wave):
        """
        Parameters
        ----------
        wave : poppy.Wavefront (or float)
            Incoming Wavefront before this optic to set wavelength and
            scale, or a float giving the wavelength in meters
            for a temporary Wavefront used to compute the OPD.
        """
        self.opd = get_abc_psd_surface(wave, self.wfe_params, self.wfe_rms, seed=self.seed).value
        return self.opd

    @wfe._check_wavefront_arg
    def get_transmission(self, wave):
        amp = get_abc_psd_surface(wave, self.amp_params, self.amp_rms, seed=self.seed)
        amp = 1 + amp # ABC PSD surface is centered at 0. This shifts to 1. 
        self.amp = amp
        return self.amp

def get_abc_psd_surface(wave, abc, rms, seed=None):
    """
    Given a poppy.Wavefront, generate a surface defined by an ABC PSD
    """
    f1 = xp.fft.fftfreq(wave.shape[0], d=wave.pixelscale).value
    y, x = xp.meshgrid(f1, f1)
    rho = xp.sqrt(y**2 + x**2) # radial spatial frequency

    a, b, c = abc
    psd = a / (1 + xp.power(rho/b, c))

    psd_random_state = xp.random.RandomState()
    psd_random_state.seed(seed)   # if provided, set a seed for random number generator
    rndm_phase = psd_random_state.normal(size=(len(y), len(x)))   # generate random phase screen
    rndm_psd = xp.fft.fft2(xp.fft.fftshift(rndm_phase))   # FT of random phase screen to get random PSD
    scaled = xp.sqrt(psd) * rndm_psd    # scale random PSD by power-law PSD
    phase_screen = xp.fft.ifftshift(xp.fft.ifft2(scaled)).real   # FT of scaled random PSD makes phase screen

    phase_screen -= xp.mean(phase_screen)  # force zero-mean
    phase_screen_normalized = phase_screen / xp.std(phase_screen) * rms  # normalize to wanted RMS
    return phase_screen_normalized




