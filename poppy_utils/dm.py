import astropy.units as u

import numpy as np

import poppy
from poppy import utils
from poppy.accel_math import xp


class QuantizedContinuousDeformableMirror(poppy.ContinuousDeformableMirror):
    """
    Lightweight wrapper around poppy.ContinuousDeformableMirror that supports
    quantization of the input DM command.

    Note that the quantization can be set by either directly supplying hmin
    or else supplying all of gain, vmax, and bits.

    Parameters
    -----------
    gain : float
        Surface displacement per volt
    vmax : float
        Maximum voltage
    bits : int
        Bit depth
    hmin : float
        Minimum surface displacement
    """
    @utils.quantity_input(gain=u.meter/u.V, vmax=u.V, hmin=u.meter)
    def __init__(self, gain=5*u.nm/u.V, vmax=200*u.V, bits=16, hmin=None, **kwargs):
        super().__init__(**kwargs)

        # check for valid arguments
        if hmin is None:
            if None in [gain, vmax, bits]:
                raise ValueError('If hmin is not supplied, gain, vmax, and bits must all be supplied!')
            vmin = vmax / 2**bits
            hmin = vmin * gain
        else:
            # when hmin is given, gain, vmax, and bits must all be None
            if not np.all([x is None for x in [gain, vmax, bits]]):
                raise ValueError('If hmin is supplied, gain, vmax, and bits must all be None!')
        self.hmin = hmin

    @utils.quantity_input(new_surface=u.meter)
    def set_surface(self, new_surface):
        new_surface = self._discretize_dm_cmd(new_surface, self.hmin)
        super().set_surface(new_surface)

    @utils.quantity_input(new_value=u.meter)
    def set_actuator(self, actx, acty, new_value):
        new_value = self._discretize_dm_cmd(new_value, self.hmin)
        super().set_actuator(actx, acty, new_value)

    @utils.quantity_input(cmd=u.meter, hmin=u.meter)
    def _discretize_dm_cmd(self, cmd, hmin):
        # this converts fractional steps, rounds to the nearest step, and then converts back to surface
        return np.rint(cmd.to(u.meter).value / hmin.to(u.meter).value ) * hmin