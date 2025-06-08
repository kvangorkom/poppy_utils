import os
import pathlib
import re
from collections import OrderedDict
from copy import deepcopy
from importlib import import_module

import astropy.units as u
from astropy.io import fits
import poppy
import tomlkit as tk

#from .optics import conic_fl


def toml2dict(filename):
    """
    Read in toml file and create an ordered dictionary that
    can be parsed to create a poppy optical system
    """
    with open(filename, mode='r') as f:
        tkdict = tk.load(f)

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

        if isinstance(sval, str) and os.path.isfile(sval):
            # assume any path is a FITS file for now
            filepath = pathlib.Path(sval)
            val[skey] = fits.getdata(filepath)

        # if there's a nested entry (optics_args, for example, recursively call this
        # function to parse it)
        if isinstance(sval, tk.items.Table):
            val[skey] = parse_dict(sval)

        # parse optic_type to poppy object
        if skey == 'optic_type':
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
    mname, fname = classstr.rsplit('.', 1)
    mod = import_module(mname)
    return getattr(mod, fname)

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
                optic_type = elem_dict.pop('optic_type')
                cur_optic = optic_type(**elem_dict)
                opticslist.append(cur_optic)
            compound_optic = poppy.CompoundAnalyticOptic(opticslist=opticslist, name=optic_name)
        else:
            optic_type = optic.pop('optic_type')
            compound_optic = optic_type(**optic)

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
            print(idx, roc)

    return roc_list
