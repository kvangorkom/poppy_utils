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

from . import dm


def toml2dict(filename):
    """
    Read in toml file and create an ordered dictionary that
    can be parsed to create a poppy optical system
    
    For toml files containing multiple optical systems,
    this will be a list of dictionaries
    """
    with open(filename, mode='r') as f:
        tkdict = tk.load(f).unwrap()
       
    # extract optical system(s)
    systems_raw = tkdict['osys']
    
    wavefront = parse_dict(tkdict['wavefront'])
    
    systems = []
    for osys in systems_raw:
        # unparsed values from optical components
        optics_raw = osys.pop('optics')
        # everything else is an argument for the optical system itself
        osys_args = osys
        
        # parse osys
        osys_parsed = parse_dict(osys_args)
        
        # parse optics
        optics = []
        for optic in optics_raw:
            optics.append(parse_dict(optic))
            
        # each optical system is a dict
        osysdict = {
            'osys' : osys_parsed,
            'optics' : optics
        }
        systems.append(osysdict)

    return systems, wavefront

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
        if skey in ['optic_type']: #'planetype'
            val[skey] = parse_class_str(sval)

        # traverse lists to parse
        if skey == 'optics':
            vals = []
            for sval_cur in sval:
                vals.append(parse_dict(sval_cur))
            val[skey] = vals

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

def construct_optical_system(systems):
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
        poppy OpticalSystem
    """
    
    systems_parsed = []
    for system in systems:

        optics = deepcopy(system['optics'])

        # first, construct the optical system
        osys_dict = deepcopy(system['osys'])
        osys_wrapper = osys_dict.pop('wrapper', None)
        osys_constructor = osys_dict.pop('optic_type')

        osys = osys_constructor(**osys_dict) # needs some more args

        # then go optic-by-optic and build compound optics in order
        for optic in optics:

            #print(f'loading {optic_name}...')
            optic_name = optic['name']

            # treat as a compound optic
            is_compound = optic.get('is_compound', False)
            planetype = optic.get('planetype', None)

            # is the distance a vertex-to-vertex distance? 
            #is_vertex_dz = optic.get('is_vertex_dz', 'False').upper() == 'TRUE'
            dz = optic.pop('dz', 0*u.m)

            if is_compound:
                # construct compound optic out of each optic
                opticslist = []
                try:
                    for elem_dict in optic['optics']: # nested optics
                        elem_name = elem_dict.get('name', None)
                        if not isinstance(elem_dict, dict):
                            continue # skip non-dictionary entries (e.g., dz)
                        #print(f'loading {elem_name}')
                        optic_type = elem_dict.pop('optic_type')
                        cur_optic = optic_type(**elem_dict) # , name=f'{optic_name}_{elem_name}'
                        opticslist.append(cur_optic)
                    compound_optic = poppy.CompoundAnalyticOptic(opticslist=opticslist, name=optic_name)
                except Exception as e: # add info about optic being parsed to general errors
                    raise ValueError(f'Error while parsing optic {elem_name} in {optic_name}') from e
            else: # not compound
                try:
                    optic_type = optic.pop('optic_type')
                    compound_optic = optic_type(**optic) # name=optic_name
                except Exception as e: # add info about optic being parsed to general errors
                    raise ValueError(f'Error while parsing optic {optic_name}') from e
            if isinstance(osys, poppy.FresnelOpticalSystem):
                osys.add_optic(compound_optic, distance=dz)
            elif isinstance(osys, poppy.OpticalSystem): # Fraunhofer systems need plane types
                if planetype == 'pupil':
                    osys.add_pupil(compound_optic)
                elif planetype == 'image':
                    osys.add_image(compound_optic)
                elif planetype == 'detector':
                    osys.add_detector(compound_optic) # this will break
                    
        # special propagation requires a wrapper around a standard optical system
        if osys_wrapper is not None:
            wrapper_constructor = osys_wrapper.pop('optic_type')
            osys = wrapper_constructor(osys, **osys_wrapper)
            
        systems_parsed.append(osys)

    if len(systems_parsed) > 1:
        osys_out = poppy.CompoundOpticalSystem(optsyslist=systems_parsed)
    else:
        osys_out = systems_parsed[0] # just a singular optical system

    return osys_out

def construct_wavefront(wf_dict):
    """
    TO DO
    """
    wf_dict = deepcopy(wf_dict)
    wf_constructor = wf_dict.pop('optic_type')
    beam_radius = wf_dict.pop('beam_radius')
    return wf_constructor(beam_radius, **wf_dict)

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
    systems_dict, wavefront_dict = toml2dict(filename)
    osys = construct_optical_system(systems_dict)
    wf = construct_wavefront(wavefront_dict)
    return osys, wf, systems_dict

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
    osys, wf, systems_dict = load_optical_system(filename)
    return OpticalModel(osys, wavelength=wf.wavelength, wf0=wf, toml_dict=systems_dict)

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

    def __init__(self, osys, wavelength=635*u.nm, wf0=None, toml_dict=None):
        """
        sdf
        """

        # poppy optical system
        self.osys = osys

        # if given, save the toml dict
        self.toml_dict = toml_dict

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
            if isinstance(plane, poppy.ContinuousDeformableMirror) or isinstance(plane, dm.DeformableMirror):
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

        wf_out = self.osys.propagate(wf, normalize='first', return_intermediates=return_intermediates)
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

    wf_out, wflist = osys.propagate(deepcopy(wf), normalize='first', return_intermediates=True)

    roc_list = []
    for idx, wf in enumerate(wflist):
        if isinstance(wf, poppy.FresnelWavefront):
            roc = wf.r_c()
            dz = wf.z - wf.z_w0
            if xp.isclose(dz.to(u.m).value, 0.0):
                roc = np.inf * u.m
            roc_list.append(roc)
        else: # Fraunhofer
            roc = np.nan
            dz = np.nan
        if do_print:
            print(f'Plane {idx}, {wf.location}: {roc:.6f}, {dz:.6f}')

    return roc_list
