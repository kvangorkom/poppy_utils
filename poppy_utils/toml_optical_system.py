"""
To do:
* X Generate surface/refl errors
* Auto-gen .toml w/out ABC PSDs from one w/ ABC PSD (i.e., create fixed-ab version)
* Handle jones pupils (fits optical elements?) (and measured DM map)
* DM/FSM discretization
* WFS&C stuff -- jacobian, pupil-corr, EFC, etc? <-- get from Lina? 
* VVC implementation <--- re-implement SAM or MatrixCoronagraph thing?

"""
import os
import pathlib
import re
from collections import OrderedDict
from copy import deepcopy
from importlib import import_module

import astropy.units as u
from astropy.io import fits

import poppy
from poppy.accel_math import xp
import numpy as np

import tomlkit as tk


def toml2dict(filename):
    """
    Read in toml file and create an ordered dictionary that
    can be parsed to create a poppy optical system
    """
    with open(filename, mode='r') as f:
        tkdict = tk.load(f).unwrap()

    optics = OrderedDict({})
    for key, val in tkdict.items():
        # parse the dict
        optics[key] = parse_dict(val)

    return optics

def parse_dict(tomldict):
    """
    Parse values from a toml dict.

    TO DO: load any fits files here too?
    """

    # from toml object to dict
    val = dict(tomldict)

    # extract values and units, if possible
    for skey, sval in val.items():
        if isinstance(sval, str):
            val[skey] = _extract_value_and_unit(sval, False)

        if isinstance(sval, str) and (sval.upper() == 'NONE'):
            val[skey] = None

        #if isinstance(sval, str) and os.path.isfile(sval):
        #    # assume any path is a FITS file for now
        #    filepath = pathlib.Path(sval)
        #    val[skey] = fits.getdata(filepath)

        # if there's a nested entry (optics_args, for example, recursively call this
        # function to parse it)
        #if isinstance(sval, tk.items.Table):
        if isinstance(sval, dict):
            val[skey] = parse_dict(sval)

        # parse optic_type to poppy object
        if skey in ['optic_type', 'planetype']:
            val[skey] = parse_class_str(sval)

    return val



def _extract_value_and_unit(value: str, values_only: bool):
        """
        Stolen from https://github.com/uasal/utils_config/blob/develop/src/utils_config/config_loader.py#L168

        Extracts a numerical value and unit from a string.

        Recognizes values like '10e-3arcsecond' or '0.024Kelvin/hour'.

        Parameters
        ----------
        value : str
            Input string to parse.
        values_only : bool
            If True, return only float; otherwise return dict with 'value' and 'unit'.

        Returns
        -------
            Parsed result or original string if no match.
        """
        match = re.match(r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)([a-zA-Z/%µ]+$)", value.strip())
        if match:
            num, unit = match.groups()
            return float(num) if values_only else float(num) * u.Unit(unit) if unit else float(num)
        return value  # Return as-is if no match

def parse_class_str(classstr):
    classstr_split = classstr.split('.')
    mname = classstr_split[0] # module is first element
    cname = classstr_split[1:] # path to class is all the rest
    mod = import_module(mname)
    
    cur = mod
    for elem in cname:
        cur = getattr(cur, elem)
    return cur

def parse_optic(optic):
    """
    For most optics, this should:
    * Get a component that's powered (quadraticlens) and a component that's the surface/refl map

    But there are many exceptions:
    * FPM (not powered, could still have surf/refl map)
    * Detector
    * ??
    """

    opticslist = []

    # get powered optic
    #if optic.get('roc', None) is not None:
    #    efl = conic_fl(optic['roc'], optic['k'], optic['oad'])
    #    popt_power = optic['optic_type'](f_lens=efl, name=f"{optic['name']}_power")
    #    opticslist.append(popt_power)
    #else:
    #    popt = optic['optic_type'](**optic.get('optics_args', {}), name=optic['name'])
    #    opticslist.append(popt)

    popt = optic['optic_type'](**optic.get('optics_args', {}), name=optic['name'])
    opticslist.append(popt)

    # get surface/refl map
    has_psd = optic.get('surf_psd', None) is not None
    has_map = optic.get('surf_map', None) is not None
    if has_psd or has_map:
        if has_map:
            popt_surf = poppy.FITSOpticalElement(name=f"{optic['name']}_surface") # TO DO
        if has_psd and (not has_map): # need to check that has_map and has_psd are not both defined, or raise a warning
            # generate surfaces dynamically?
            popt_surf = poppy.ArrayOpticalElement(name=f"{optic['name']}_surface") # TO DO
        opticslist.append(popt_surf)

    # get jones matrix
    if optic.get('jones_matrix', None) is not None:
        jm = poppy.JonesMatrixOpticalElement(optic['jones_matrix'], name=f"{optic['name']}_jones")
        opticslist.append()

    # can't use a compound analytic optic with array optical elements
    # so just return a list of all the components for now
    return opticslist #poppy.CompoundAnalyticOptic(opticslist=opticslist, name=optic['name'])

# def construct_optical_system(optics):
#     osys_constructor = parse_class_str(optics['optical_system'].pop('osys_type', 'poppy.FresnelOpticalSystem'))
#     osys = osys_constructor(**optics['optical_system']) # needs some more args
#     for optic_name, optic in list(optics.items())[1:]:
#         poppy_optic = parse_optic(optic) # this may be a list
#         if isinstance(poppy_optic, list):
#             for i, element in enumerate(poppy_optic):
#                 if i == 0: # distance from last optic 
#                     osys.add_optic(element, distance=optic.get('dz', 0*u.m)) 
#                 else:
#                     osys.add_optic(element)
#         else:
#             osys.add_optic(poppy_optic, distance=optic.get('dz', 0*u.m))             
#     return osys

def construct_optical_system(optics):
    """
    Given a dictionary parsed from a TOML file,
    construct a poppy optical system

    Parameters
    ----------
    optics : OrderedDict
        Ordered dictionary of dictionaries, each of which describes
        an optical element, i.e., the output of load_optical_system 

    Returns
    --------
        poppy.FresnelOpticalSystem, poppy.FresnelWavefront
    """

    optics = deepcopy(optics)

    # first, construct the optical system
    osys_dict = optics['optical_system']
    osys_constructor = parse_class_str(osys_dict.pop('osys_type', 'poppy.FresnelOpticalSystem'))
    wavelength = osys_dict.pop('wavelength') # needed for wavefront

    osys = osys_constructor(**osys_dict) # needs some more args

    # generate a wf
    wf = poppy.FresnelWavefront(osys_dict['pupil_diameter']*0.5,
                                npix=osys_dict['npix'],
                                wavelength=wavelength,
                                oversample=1.0/osys_dict['beam_ratio'])

    # then go optic-by-optic and build compound optics in order
    for optic_name, optic in list(optics.items())[1:]: # skip optical_system

        #print(f'loading {optic_name}...')

        # treat as a compound optic
        is_compound = optic.get('is_compound', 'False').upper() == 'TRUE'

        # is the distance a vertex-to-vertex distance? 
        #is_vertex_dz = optic.get('is_vertex_dz', 'False').upper() == 'TRUE'
        dz = optic.pop('dz', 0*u.m)

        if is_compound:
            # construct compound optic out of each optic
            opticslist = []
            for elem_name, elem_dict in optic.items():
                if not isinstance(elem_dict, dict):
                    continue # skip non-dictionary entries (e.g., dz)
                #print(f'loading {elem_name}')
                optic_type = elem_dict.pop('optic_type')
                cur_optic = optic_type(**elem_dict, name=f'{optic_name}_{elem_name}')
                opticslist.append(cur_optic)
            compound_optic = poppy.CompoundAnalyticOptic(opticslist=opticslist, name=optic_name)
        else:
            optic_type = optic.pop('optic_type')
            compound_optic = optic_type(**optic, name=optic_name)

        osys.add_optic(compound_optic, distance=dz)

    return osys, wf

def load_optical_system(filename):
    """
    Load a .toml file into a parsed dictionary

    Parameters
    ----------
    filename : str
        Filename of a .toml file

    Returns
    --------
        poppy.FresnelOpticalSystem, poppy.Wavefront, parsed OrderedDict
    """
    optics_dict = toml2dict(filename)
    osys, wf = construct_optical_system(optics_dict)
    return osys, wf, optics_dict

def load_optical_system_into_model(filename):
    """
    Load a .toml file into an OpticalModel

    Parameters
    ----------
    filename : str
        Filename of a .toml file

    Returns
    --------
        OpticalModel
    """
    osys, wf, optics_dict = load_optical_system(filename)
    return OpticalModel(osys, wavelength=wf.wavelength, wf0=wf)

class OpticalModel(object):
    """
    This is intended to be a lightweight wrapper around an
    optical system. How necessary is this?

    Things this could add:
    * Easy broadband (calc_psf does this)
    * Tracking DM state?
    * Tracking tip/tilt (not sure this is necessary)
    * Partial propagation?
    """

    def __init__(self, osys, wavelength=635*u.nm, wf0=None):
        """
        sdf
        """

        # poppy optical system
        self.osys = osys

        # default input wavefront
        if wf0 is not None:
            self._wf0 = deepcopy(wf0)

        self.wavelength = wavelength

        ttms = []
        dms = []
        # expose active optics
        for plane in osys.planes:
            # tip/tilt stages
            if isinstance(plane, poppy.TipTiltStage):
                ttms.append(plane)

            # deformable mirrors
            if isinstance(plane, poppy.ContinuousDeformableMirror):
                dms.append(plane)
        self.ttms = ttms
        self.dms = dms

    @poppy.utils.quantity_input(wavelength=u.meter)
    def run_mono(self, wf=None, wavelength=None, tiptilt=(0,0), return_intermediates=False, return_intensity=False):
        """
        input_tiptilt = at first plane, NOT TIP/TILT STAGES
        """
        wf = deepcopy(self._wf0) if wf is None else deepcopy(wf)

        if wavelength is None: # not given, set to model default
            wavelength = self.wavelength

        # should we warn about this? It's kind of the whole point of the wavelength override
        #if wavelength != wf.wavelength:
        #    poppy.utils._log.warning('Requested wavelength does not match wavelength of input wavefront! Changing wavefront wavelength to match.')
        #    wf.wavelength = wavelength
        wf.wavelength = wavelength

        if (xp.abs(tiptilt[0]) > 0) or (xp.abs(tiptilt[1]) > 0):
            wf.tilt(Xangle=tiptilt[0], Yangle=tiptilt[1]) # TO DO: what units?

        wf_out = self.osys.propagate(wf, return_intermediates=return_intermediates)
        if not return_intensity:
            return wf_out
        else:
            if return_intermediates:
                return wf_out[0].intensity, wf_out[1]
            else:
                return wf_out.intensity

    def run_broadband(self, cenwave, bw, nwaves=20, **kwargs):
        """
        Wrapper around run_mono

        Input = cenwave, bw, and num_waves? or just do list of wavelengths?
        """
        wavelens = np.linspace(cenwave*(1-bw/2.0), cenwave*(1+bw/2.0), num=nwaves)

        out = []
        for i, wavelen in enumerate(wavelens): # note -- currently does not let wavefront change with wavelength
            out.append( self.run_mono(wavelength=wavelen, **kwargs) )

        return out

    def inspect_roc(self, do_print=False):
        """
        Wrapper around inspect_osys_wf_roc
        """
        return inspect_osys_wf_roc(self.osys, self._wf0, do_print=do_print)


# ----- some analysis tools -----

def inspect_osys_wf_roc(osys, wf, do_print=False):
    """
    Given a poppy.FresnelOpticalSystem, propagate a poppy.FresnelWavefront
    through the system and print the wavefront radius of curvature at
    each plane

    Parameters
    ----------
    osys : poppy.FresnelOpticalSystem
        Input optical system to evaluate
    wf : poppy.FresnelWavefront
        Input wavefront to propagate through the system
    do_print : bool
        Print the radii of curvature? Default: False

    Returns
    --------
        list of radii of curvature
    """

    wf_out, wflist = osys.propagate(deepcopy(wf), return_intermediates=True)

    roc_list = []
    for idx, wf in enumerate(wflist):
        roc = wf.z_w0 - wf.z
        roc_list.append(roc)
        if do_print:
            print(f'Plane {idx}, {wf.location}: {roc}')

    return roc_list
