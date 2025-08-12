from copy import deepcopy
from collections import OrderedDict
from pathlib import Path

import astropy.units as u
from astropy.io import fits

import numpy as np

import poppy
from poppy import utils, wfe
from poppy.accel_math import xp
from poppy.wfe import _wave_y_x_to_rho_theta
from poppy.utils import _log, pad_or_crop_to_shape

tukey =  poppy.accel_math._scipy.signal.windows.tukey

import tomlkit as tk

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
    return c*(r**2) / (1 + np.sqrt(1-(1+k)*(c*r)**2))

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
    eps = np.sqrt(-1*k)

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
    f1_eff = ( f1_parent - sag ) / np.cos(np.arctan(oad/(f1_parent - sag)))
    f2_eff = ( f2_parent - sag ) / np.cos(np.arctan(oad/(f2_parent - sag)))

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
    return np.sqrt(x**2 + y**2)

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

class SimpleTipTiltStage(poppy.TipTiltStage):
    """
    Lightweight wrapper around poppy.TipTiltStage
    that assumes a poppy.ScalarTransmission optic
    """

    def __init__(self, *args, **kwargs):
        name = kwargs.pop('name', None)
        optic = poppy.ScalarTransmission(name=name)
        super().__init__(optic=optic, *args, **kwargs)
        self.name = name

class FITSJonesOpticalElement(poppy.JonesMatrixOpticalElement):
    """
    Modification of FITSOpticalElement to handle Jones matrices

    The FITS file with the Jones pupils are expected to be formatted
    as 8 x Y x X cubes, where the elements are
    [Re(E_xx), Im(E_xx), Re(E_xy), Im(E_xy), Re(E_yx), Im(E_yx), Re(E_yy), Im(E_yy)]

    Parameters
    ----------
    jones_file : str or path-like
        Path to FITS file with Jones pupil.

    """

    def __init__(self, jones_file, pixelscale, **kwargs):
        self.jones_file = jones_file

        with fits.open(jones_file) as f:
            jones_cube = f[0].data

        # contrast 2x2xYxX jones pupil (TO DO: check if Exy and Eyx are flipped)
        jones_matrix = xp.asarray(
            [[jones_cube[0] + 1j*jones_cube[1], # Exx_re + 1j*Exx_im
              jones_cube[2] + 1j*jones_cube[3]], # Exy_re + 1j*Exy_im 
             [jones_cube[4] + 1j*jones_cube[5], # Eyx_re + 1j*Eyx_im
              jones_cube[6] + 1j*jones_cube[7]]] # Eyy_re + 1j*Eyy_im
        )

        super().__init__(jones_matrix)

        self.pixelscale = pixelscale

class MultiScaleCoronagraph(poppy.poppy_core.OpticalSystem):
    """ WIP port of hcipy.MultiScaleCoronagraph to poppy.
    
    Essentially a mod of the SemiAnalyticCoronagraph that allows
    (1) extended focal plane masks
    (2) multi-scale MFTs for sampling purposes
    
    To do:
    * Extend from two-scale to multi-scale

    """

    def __init__(self, existing_optical_system, oversample=8, fpm_box=1.0, window=tukey, to_fpm=False, **kwargs):
        super(MultiScaleCoronagraph, self).__init__(**kwargs)

        self.name = "SemiAnalyticCoronagraph for " + existing_optical_system.name
        self.verbose = existing_optical_system.verbose
        self.source_offset_r = existing_optical_system.source_offset_r
        self.source_offset_theta = existing_optical_system.source_offset_theta
        self.planes = existing_optical_system.planes
        self.npix = existing_optical_system.npix
        self.pupil_diameter = existing_optical_system.pupil_diameter
        self.to_fpm = to_fpm
        
        
        self.window_func = window

        
        self.pupil = self.planes[0]
        # assume occulter is the 2nd plane
        self.fpm = self.planes[1]

        # some tweaks for display
        self.fpm.wavefront_display_hint = 'intensity'

        #self.mask_function = self.occulter

        self.oversample = oversample

        if not np.isscalar(fpm_box):
            fpm_box = np.array(fpm_box)  # cast to numpy array so the multiplication by 2
                                                  # just below will work
        self.fpm_box = fpm_box
        

        self.fpm_highres = None
        self.window_lowres = None
        self.window_highres = None

    def propagate(self,
                  wavefront,
                  normalize='first',
                  return_intermediates=False,
                  display_intermediates=False):
        """ Core low-level routine for propagating a wavefront through an optical system

        See docstring of OpticalSystem.propagate for details

        """

        if self.verbose:
            _log.info(" Propagating wavelength = {0:g} meters using "
                      "Fast Semi-Analytic Coronagraph method".format(wavefront.wavelength))

        intermediate_wfs = []
        
        
        wavefront *= self.pupil

        #wavefront.history.append("Propagating using Fast Semi-Analytic Method")
        #wavefront.history.append(" for Coronagraphy (See Soummer et al. 2007).")


        # ------- differences from regular propagation begin here --------------
        
        nrows = len(self.planes) + 2  # there are some extra display planes
        wavefront._display_hint_expected_nplanes = nrows  # For display of intermediate steps nicely
        if (normalize.lower() != 'first') and (normalize.lower() != 'last'):
            raise NotImplementedError("Only normalizations 'first' or 'last' are implemented for SAMC")

        # Start with a two-step thing
        # 1. MFT for small region (small window)
        # 2. FFT over whole region (large window)

        pupil_diam = getattr(self.pupil, 'pupil_diam', None)
        if pupil_diam is None:
            pupil_diam = wavefront.diam
        
        # taken from MFT Coronagraph
        metadet_pixelscale = ((wavefront.wavelength / pupil_diam).decompose()
                              * u.radian).to(u.arcsec) / self.oversample / 2 / u.pixel
        self.fpm_highres = poppy.Detector(metadet_pixelscale, fov_arcsec=self.fpm_box * 2,
                                      name='Oversampled Occulter Plane')

        #if return_intermediates:
        #    intermediate_wfs.append(wavefront_cor.copy())

        #if display_intermediates:  # Display prior to the occulter
        #    wavefront_cor._display_after_optic(self.occulter_highres, default_nplanes=nrows)
    
        # FFT from pupil to full FPM
        wavefront_lres = wavefront.copy()
        wavefront_lres.propagate_to(self.fpm) # This should be an FFT?
        wavefront_lres *= self.fpm # apply low res VVC
        # apply the window (create if not defined yet)
        if self.window_lowres is None:
            npix = xp.int64( wavefront_lres.shape[0] * (self.fpm_highres.fov_arcsec / wavefront_lres.fov) )
            w1d = self.window_func(npix, alpha=1, sym=False)
            self.window_lowres = 1 - pad_or_crop_to_shape(xp.outer(w1d, w1d), wavefront_lres.shape)
        wavefront_lres.wavefront *= self.window_lowres
        wavefront_lres.propagate_to(self.pupil) # back to input pupil plane
        #return wavefront_lres

        
        # MFT from pupil to small FPM region
        wavefront_hres = wavefront.copy()
        #wavefront_hres.propagation_hint = 'MFT'
        wavefront_hres.propagate_to(self.fpm_highres)  # This will be an MFT propagation
        wavefront_hres *= self.fpm # apply high res VVC
        # apply the window (create if not defined yet) -- might need to redefine every time
        if self.window_highres is None:
            w1d = self.window_func(wavefront_hres.shape[0], alpha=1, sym=False)
            self.window_highres = xp.outer(w1d, w1d)
        wavefront_hres.wavefront *= self.window_highres
        # propagate back to the wavefront_lres plane (which is the pupil, sampled appropriately)
        wavefront_hres._propagate_mft_inverse(wavefront_lres)#, pupil_npix=wavefront_lres.shape[0]) # back to input pupil plane

                
        #return wavefront_hres
        wavefront = wavefront_lres + wavefront_hres
        
        # if requested, propagate back to FPM plane so that we end up in the expected plane at the end
        if self.to_fpm:
            wavefront.propagate_to(self.fpm)        
        
        #wavefront_cor *= self.mask_function
        #wavefront_cor.current_plane_index += 1
        if return_intermediates:
            intermediate_wfs.append(wavefront.copy())
        if display_intermediates:  # Display after the occulter (EXTRA PLANE)
            wavefront._display_after_optic(self.occulter_highres, default_nplanes=nrows,)

        # ------- differences from regular propagation end here --------------

        # prepare output arrays
        if normalize.lower() == 'last':
            wavefront.normalize()

        if return_intermediates:
            return wavefront, intermediate_wfs
        else:
            return wavefront

class ABCPSDWFE(poppy.WavefrontError):
    """
    Statistical ABC PSD WFE class from power law for optical noise.

    To do: eqn goes here.
    To do: parameter a is currently ignored
    To do: if seed is used, it'll generate the same surface for both amplitude and phase

    Note that this differs from other Statistical WFE classes in two ways:
    - Both amplitude and OPD maps are generated
    - Maps are only generated one time (NOT with every propagation) -- this can be circumvented
    by setting self.opd and self.amp to None between runs.
        - The exception to this is if the wavefront pixelscale changes -- this forces a recomputation

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
                 amp_seed=None,
                 wfe_seed=None,
                 **kwargs):

        super().__init__(name=name, **kwargs)
        self.wfe_params = wfe_params
        self.wfe_rms = wfe_rms
        self.amp_params = amp_params
        self.amp_rms = amp_rms
        self.amp_seed = amp_seed
        self.wfe_seed = wfe_seed

        self.opd = None
        self.amp = None
        self.pixelscale = None

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
        if (self.opd is None) or (wave.pixelscale != self.pixelscale):
            self.opd = get_abc_psd_surface(wave, self.wfe_params, self.wfe_rms.to(u.meter).value, seed=self.amp_seed)
            self.pixelscale = wave.pixelscale
        return self.opd

    @wfe._check_wavefront_arg
    def get_transmission(self, wave):
        if (self.amp is None) or (wave.pixelscale != self.pixelscale):
            amp = get_abc_psd_surface(wave, self.amp_params, self.amp_rms, seed=self.wfe_seed)
            amp = 1 + amp # ABC PSD surface is centered at 0. This shifts to 1.
            self.amp = amp
            self.pixelscale = wave.pixelscale
        return self.amp

def get_abc_psd_surface(wave, abc, rms, seed=None):
    """
    Given a poppy.Wavefront, generate a surface defined by an ABC PSD
    """
    f1 = xp.fft.fftfreq(wave.shape[0], d=wave.pixelscale.value)
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

def save_surfaces_and_reflectivities(model, outdir, orig_toml_path=None, overwrite=False):
    """
    Given a .toml optical system, generate and save surface and reflectivity maps.

    TO DO: I think this breaks if you have more than one WavefrontError in a compound optic

    Parameters
    ----------
    model : OpticalModel
        Optical model generated from a .toml file
    outdir : str
        Directory to save amplitude and OPD files to. Will be
        created if it does not exist.
    orig_toml_path : str, optional
        Path to original .toml file from which the optical model
        was created. If given, a new "static" version of the .toml
        will be written to outdir.
    overwrite : bool, optional
        Overwrite existing files at outdir? Default: False

    Returns
    --------
    Nothing
    """

    if not outdir.exists():
        outdir.mkdir()

    # first, run a propagation through optical system
    # to force phase/amplitude screens to be generated
    print('Running model to get pixelscales at each plane.')
    model.run_mono()

    if orig_toml_path is not None:
        with open(orig_toml_path, mode='r') as f:
            tkdict = tk.load(f)
        tk_new = tk.document()
        tk_new.add('optical_system', tkdict['optical_system'])
    else:
        tkdict = None

    # loop over elements of optical system
    for idx, optic in enumerate(model.osys.planes):
        # if it's compound, loop over the optics in the compound optic
        if isinstance(optic, poppy.CompoundAnalyticOptic):
            is_compound = True
            for optic_elem in optic.opticslist:
                if isinstance(optic_elem, poppy.WavefrontError):
                    amp = optic_elem.amp
                    opd = optic_elem.opd
                    name = optic.name
                    pixelscale = optic_elem.pixelscale
        elif isinstance(optic, poppy.WavefrontError):
            is_compound = False
            amp = optic.amp
            opd = optic.opd
            name = optic.name
            pixelscale = optic.pixelscale
        else:
            name = optic.name
            if tkdict is not None:
                tk_new.add(name, tkdict[name])
            continue # do nothing

        header = fits.Header({
            'PIXELSCL' : pixelscale.to(u.m / u.pix).value,
            'PIXUNIT' : 'meter',
            'OPTIC' : name,
            'NAME' : name,
        })

        # convert to numpy arrays if needed
        if not isinstance(amp, np.ndarray):
            amp = amp.get()
        if not isinstance(opd, np.ndarray):
            opd = opd.get()

        opd_path = Path(outdir / f'{name}_opd.fits')
        print(f'Writing out OPD file for {name} to {opd_path}')
        fits.writeto(opd_path, opd, header=header, overwrite=overwrite)

        amp_path = Path(outdir / f'{name}_amp.fits')
        print(f'Writing out amplitude file for {name} to {amp_path}')
        fits.writeto(amp_path, amp, header=header, overwrite=overwrite)


        # update toml dict
        if tkdict is not None:
            print(optic_elem)

            if is_compound:
                # get the new name to use for the non-compound surface
                new_name = optic_elem.name
                tkdict[name].pop(new_name.split('_')[-1])
                # add the original dictionary, sans the surface
                tk_new.add(name, tkdict[name])
            else:
                # not compound, so just remove the old entry
                tkdict.pop(name)
                new_name = name
            
            new_entry = tk.table()

            new_dict = {
                'optic_type' : 'poppy.FITSOpticalElement',
                'transmission' : str(amp_path),
                'opd' : str(opd_path),
                'opdunits' : 'meters',
                'planetype' : 'poppy.poppy_core.PlaneType.intermediate',
            }
            for key, val in new_dict.items():
                new_entry.add(key, val)

            tk_new.add(new_name, new_entry)

    if tkdict is not None:
        toml_path = Path(outdir / f"{tkdict['optical_system']['name']}_static.toml")
        with open(toml_path, 'w') as fp:
            tk.dump(tk_new, fp)








