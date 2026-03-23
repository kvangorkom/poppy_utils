import astropy.units as u

import numpy as np

import poppy
from poppy import utils
from poppy.accel_math import xp, _scipy



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

def make_gaussian_inf_fun(act_spacing=300e-6*u.m, sampling=10, coupling=0.15, Nact=4):
    """
    Lifted from Kian's code: https://github.com/uasal/stp-psf

    Slight modification to add an actuator buffer on each side to avoid
    wrapping across the boundaries of the array

    """
    ng = int(sampling*(Nact+2)) # number of pixels across the entire array
    pxscl = act_spacing/(sampling*u.pix)

    xs = (xp.linspace(-ng/2,ng/2-1,ng)+1/2)*pxscl.to_value(u.m/u.pix)
    x,y = xp.meshgrid(xs,xs)
    r = xp.sqrt(x**2 + y**2)

    d = act_spacing.to_value(u.m)/np.sqrt(-np.log(coupling))

    inf_fun = np.exp(-(r/d)**2)

    return inf_fun

class DeformableMirror(poppy.AnalyticOpticalElement):
    """
    Lifted from Kian's code: https://github.com/uasal/stp-psf
    (with some minor modifications)
    """
    
    def __init__(self,
                 inf_fun,
                 inf_sampling,
                 Nact=34,
                 act_spacing=300e-6*u.m,
                 aperture=None,
                 include_reflection=True,
                 planetype=poppy.poppy_core.PlaneType.intermediate,
                 name='DM',
                 failure_type = 'none',
                 strk_flty = 0*u.m,
                 msk_flty = xp.zeros((34,34)),
                 Nbad = 0,
                ):

        self.failure_type = failure_type
        self.strk_flty = strk_flty
        self.msk_flty = xp.zeros((Nact,Nact))
        self.Nbad = xp.sum(self.msk_flty)
        
        self.inf_fun = inf_fun
        self.inf_sampling = inf_sampling

        self.Nact = Nact
        self.act_spacing = act_spacing
        self.include_reflection = include_reflection

        self.Nsurf = inf_fun.shape[0]
        self.pixelscale = self.act_spacing/(self.inf_sampling*u.pix)
        self.active_diam = self.Nact * self.act_spacing

        self.yc, self.xc = (xp.indices((Nact, Nact)) - Nact//2 + 1/2)
        self.rc = xp.sqrt(self.xc**2 + self.yc**2)
        self.dm_mask = self.rc<(Nact/2 + 1/2)
        self.Nacts = int(xp.sum(self.dm_mask))
        
        self._command = xp.zeros((self.Nact, self.Nact))
        self.actuators = xp.zeros(self.Nacts)

        self.aperture = aperture
        self.planetype = planetype
        self.name = name

        self.inf_fun_fft = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(self.inf_fun,)))
        fx = xp.fft.fftshift(xp.fft.fftfreq(self.Nsurf))
        fy = xp.fft.fftshift(xp.fft.fftfreq(self.Nsurf))
        x = self.inf_sampling*(xp.linspace(-self.Nact//2, self.Nact//2-1, self.Nact) + 1/2)
        y = self.inf_sampling*(xp.linspace(-self.Nact//2, self.Nact//2-1, self.Nact) + 1/2)

        self.Mx = xp.exp(-1j*2*np.pi*xp.outer(fx,x))
        self.My = xp.exp(-1j*2*np.pi*xp.outer(y,fy))

        self.pxscl_tol = 1e-6

    @property
    def command(self):
        return self._command

    @command.setter
    def command(self, command_values):
        command_values *= self.dm_mask
        self._actuators = self.map_command_to_actuators(command_values) # ensure you update the actuators if command is set
        self._command = command_values
    
    @property
    def actuators(self):
        return self._actuators

    @actuators.setter
    def actuators(self, act_vector):
        self._command = self.map_actuators_to_command(act_vector) # ensure you update the actuators if command is set
        self._actuators = act_vector
    
    def map_command_to_actuators(self, command_values, dm_msk = None):
        if dm_msk is None: dm_msk = self.dm_mask
        actuators = command_values.ravel()[dm_msk.ravel()]
        return actuators
        
    def map_actuators_to_command(self, act_vector, dm_msk = None):
        if dm_msk is None: dm_msk = self.dm_mask
        command = xp.zeros((self.Nact, self.Nact))
        command[dm_msk == True] = act_vector
        return command
    
    def get_surface(self):
        mft_command = self.Mx@self.command@self.My
        fourier_surf = self.inf_fun_fft * mft_command
        surf = xp.fft.ifftshift(xp.fft.ifft2(xp.fft.fftshift(fourier_surf,))).real
        return surf
    
    # METHODS TO BE COMPATABLE WITH POPPY
    def get_opd(self, wave):
        opd = self.get_surface()
        if self.include_reflection:
            opd *= 2

        pxscl_diff = xp.abs(wave.pixelscale.to_value(u.m/u.pix) - self.pixelscale.to_value(u.m/u.pix))
        if pxscl_diff > self.pxscl_tol:
            opd = interp_arr(opd, self.pixelscale.to_value(u.m/u.pix), wave.pixelscale.to_value(u.m/u.pix) )
        
        #opd = utils.pad_or_crop(opd, wave.shape[0])
        opd = utils.pad_or_crop_to_shape(opd, wave.shape)

        return opd

    def get_transmission(self, wave):
        if self.aperture is None:
            trans = xp.ones_like(wave.wavefront)
        else:
            trans = self.aperture.get_transmission(wave)
        return trans
    
    def get_phasor(self, wave):
        """
        Compute the amplitude transmission appropriate for a vortex for
        some given pixel spacing corresponding to the supplied Wavefront
        """

        assert (wave.planetype != poppy.poppy_core.PlaneType.image)

        dm_phasor = self.get_transmission(wave) * xp.exp(1j * 2*np.pi/wave.wavelength.to_value(u.m) * self.get_opd(wave))

        return dm_phasor

class QuantizedGaussianDeformableMirror(DeformableMirror):
    """
    Wrapper around DeformableMirror that assumes Gaussian influence
    functions and quantization error.
    """

    def __init__(self, gain=5*u.nm/u.V, vmax=200*u.V, bits=16, hmin=None, inf_sampling=30, **kwargs):

        # make the gaussian influence function
        act_spacing = kwargs.get('act_spacing', 300e-6*u.m)      
        Nact = kwargs.get('Nact', 34)  
        inf_fun = make_gaussian_inf_fun(act_spacing=act_spacing, sampling=inf_sampling, coupling=0.15, Nact=Nact)

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
        self.hmin = hmin.to(u.m).value # convert to meters

        # initialize parent
        super().__init__(inf_fun, inf_sampling, **kwargs)

        self.disable_quantization = False

    @property
    def command(self):
        return self._command

    @command.setter
    def command(self, command_values):
        if not self.disable_quantization:
            command_values = self._discretize_dm_cmd(command_values)
        #super().command(command_values)
        command_values *= self.dm_mask
        self._actuators = self.map_command_to_actuators(command_values) # ensure you update the actuators if command is set
        self._command = command_values

    def _discretize_dm_cmd(self, cmd):
        # this converts fractional steps, rounds to the nearest step, and then converts back to surface
        return xp.rint(cmd / self.hmin ) * self.hmin


def interp_arr(arr, pixelscale, new_pixelscale, order=1):
    """"
    Lifted from Kian's code: https://github.com/uasal/stp-psf
    """
    Nold = arr.shape[0]
    old_xmax = pixelscale * (Nold/2)
    Nnew = 2*int(np.round(old_xmax/new_pixelscale))
    new_xmax = new_pixelscale * (Nnew/2)

    x,y = xp.ogrid[-old_xmax:old_xmax - pixelscale:Nold*1j,
                   -old_xmax:old_xmax - pixelscale:Nold*1j]

    newx,newy = xp.mgrid[-new_xmax:new_xmax-new_pixelscale:Nnew*1j,
                         -new_xmax:new_xmax-new_pixelscale:Nnew*1j]
    
    x0 = x[0,0]
    y0 = y[0,0]
    dx = x[1,0] - x0
    dy = y[0,1] - y0

    ivals = (newx - x0)/dx
    jvals = (newy - y0)/dy

    coords = xp.array([ivals, jvals])

    interped_arr = _scipy.ndimage.map_coordinates(arr, coords, order=order)
    return interped_arr

class FaultyQuantizedGaussianDeformableMirror(QuantizedGaussianDeformableMirror):
    """
    Wrapper around poppy.QuantizedGaussianDeformableMirror to include the effects of dead/floating actuators.

    Parameters:
    -----------
    failure_type: string
        Either "dead" or "float" or "none"

    msk_flty: xp.array
        Mask of faulty actuators -- set the bad actuators to True
    
    Nbad: int
        Number of bad actuators - used to randomly generate a mask with Nbad act.
        If given with msk_flty, msk_flty is used and this is ignored. 

    strk_flty: float or xp.array
        Either a single value that all the dead actuators will be forced to or an array of values
        where each element corresponds to the stroke of one of the faulty act. 
        This input is mandatory when failure_type = 'dead', not needed for 'float'.
    """
    @utils.quantity_input(strk_flty=u.meter)


    def __init__(self, failure_type = 'none', msk_flty = None, Nbad = 0, strk_flty = 0*u.m, **kwargs):
    
        
        # check for arguments
        if failure_type == 'none':
            print("DM without any faulty actuators.")
            Nact = kwargs.get('Nact', 34)  
            msk_flty = xp.zeros((self.Nact,self.Nact))
            Nbad = 0
            strk_flty = 0*u.m

        elif failure_type == 'dead' and strk_flty is None:
            raise ValueError('Stroke to be set for dead actuators needed.')

        if failure_type == 'dead' and msk_flty is None and Nbad == 0:
            raise ValueError('Either a mask of faulty actuators or the number of faulty actuators needed.')
        
        super().__init__(failure_type = failure_type, strk_flty = strk_flty, msk_flty = msk_flty, Nbad = Nbad, **kwargs)

        if msk_flty is None and Nbad != 0:
            print("Mask of bad actuators not given. Selecting %d random ones."%Nbad)
            msk_flty = xp.zeros((self.Nact,self.Nact))
            # find the indices of illuminated act
            ind_il = xp.argwhere(self.dm_mask)
            # Generate Nbad values from 0 to Nbad to get the act which are faulty and set them to True in the mask
            flty_act_ind = xp.random.randint(low = 0, high = len(ind_il), size = Nbad)
            msk_flty[ind_il[flty_act_ind,0],ind_il[flty_act_ind,1]] = True
    
        self.failure_type = failure_type
        self.strk_flty = strk_flty
        self.msk_flty = msk_flty
        if msk_flty is None: self.Nbad = 0 
        else: self.Nbad = xp.sum(msk_flty)
    
    @property
    def command(self):
        return self._command

    @command.setter
    def command(self, command_values):

        # desired command values
        command_values *= self.dm_mask

        # if actuators are dead, force them to given stroke value(s)
        if self.failure_type == 'dead':
            # subtract the commands at the dead act positions
            command_values -= 1*(command_values*self.msk_flty) 
            # add the stroke for dead actuators
            command_values += (self.msk_flty*self.strk_flty.to_value(u.m))

        # set the command and update actuators as well
        self._actuators = self.map_command_to_actuators(command_values) # ensure you update the actuators if command is set
        self._command = command_values


    @property
    def actuators(self):
        return self._actuators
    @actuators.setter
    def actuators(self, act_vector):

        # if actuators are dead, force them to given stroke value(s)
        if self.failure_type == 'dead':
            command_values = self.map_actuators_to_command(act_vector)
            # subtract the commands at the dead act positions
            command_values -= 1*(command_values*self.msk_flty) 
            # add the stroke for dead actuators
            command_values += (self.msk_flty*self.strk_flty.to_value(u.m))

            # update act vector to new values
            act_vector = self.map_command_to_actuators(command_values)

        self._command = self.map_actuators_to_command(act_vector) # ensure you update the actuators if command is set
        self._actuators = act_vector
