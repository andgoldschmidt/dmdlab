#-----
# Spring-mass-damper
# @author Andy Goldschmidt
#-----
import numpy as np
from scipy import integrate, interpolate


class spring_mass_damper:
    def __init__(self, params):
        '''
        Simulate a spring-mass-damper system. Coordinates are position and velocity.
        
        parameters:
            params['mass']
            params['spring']
            params['damper']
        '''
        try:
            self.m = params['mass']
            self.k = params['spring']    
            self.d = params['damper']
        except:
            raise ValueError('Error initializing spring_mass_damper instance: Missing required parameter.')
        
        # State-space
        self.ndim = 2
        self.udim = 1
        
        self.A =  np.array([[0, 1],
                             [-self.k/self.m, -self.d/self.m]])
    
        self.B = np.array([[0],
                           [1/self.m]])
        
        self.u = None
        
        # Results
        self.x = None
        self.xdot = None
        
    def set_control(self, time, u):
        self.u = interpolate.interp1d(time, u, bounds_error=True)

    def rhs_with_control(self, t, x):    
        assert self.u, "Control not set."
        x1 = self.A@(x.reshape(-1,1)) + self.B@(self.u(t).reshape(-1,1))
        return x1.flatten()
    
    def rhs(self, t, x):
        x1 = self.A@(x.reshape(-1,1))
        return x1.flatten()
    
    def simulate(self, init, t_span, dt, control=False):
        timesteps = np.linspace(*t_span, int((t_span[1]-t_span[0])/dt))
        res = None
        if control:
            res = integrate.solve_ivp(self.rhs_with_control, t_span, init, t_eval=timesteps)
        else:
            res = integrate.solve_ivp(self.rhs, t_span, init, t_eval=timesteps)
        self.t = timesteps
        self.x = res.y[0,:]
        self.xdot = res.y[0,:]
        return res