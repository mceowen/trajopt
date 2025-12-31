import numpy as np
import jax 
import jax.numpy as jnp
import trajopt.core.utils.tools as tools
jax.config.update("jax_enable_x64", True)
import trajopt.core.model.obstacles     as obstacles

class Model:
    def __init__(self, config, mission):
        self.config = config
        self.mission = mission
    

    def dynamics_jax(t, z, nu, params, fcns):

        # Extract constant param values from struct
        Om = self.mission.planet["omega"]
        mu = self.mission.planet["mu"]

        # Extract states
        r, theta, phi, v, gamma, psi = z

        sigma = nu[0]
        alpha = nu[1]
        
        # Determine lift and drag coefficients from velocity
        aero = self.mission.nonlinear_aero_jax(rho, v)
        L    = aero["L"]
        D    = aero["D"]

        # Extract sines and cosines of various values
        cp  = jnp.cos(phi)
        sp  = jnp.sin(phi)
        tp  = jnp.tan(phi)
        cg  = jnp.cos(gamma)
        sg  = jnp.sin(gamma)
        tg  = jnp.tan(gamma)
        cps = jnp.cos(psi)
        sps = jnp.sin(psi)

        cs  = jnp.cos(sigma)
        ss  = jnp.sin(sigma)
        
        # state derivative function
        xDot = jnp.array([
            v * sg,
            v * cg * sps / (r * cp),
            v * cg * cps / r, 
            - D - mu * sg / r**2 + Om**2 * r * cp * (sg * cp - cg * sp * cps),
            (1 / v) * ( L * cs + (v**2 - mu / r) * cg / r ) + 2 * Om * cp * sps + Om**2 * r * (1 / v) * cp * (cg * cp + sg * cps * sp),
            (1 / v) * ( L * ss / cg + v**2 * cg * sps * tp / r ) - 2 * Om * (tg * cps * cp - sp) + Om**2 * r * (1 / (v * cg)) * sps * sp * cp
        ])

        return xDot

    def dynamic_pressure(self, t, z, nu):  #dynamic pressure
        
        rs = z[0]
        vs = z[3]

        rho = self.mission.atmosphere_model_jax(rs)

        return 0.5 * rho * (vs) ** 2 

    def heat_rate(self, t, z, nu): # heat rate

        r = z[0]
        v = z[3]

        rho = self.atmosphere_model_jax(r)

        return self.mission.vehicle["kQ"] * rho ** 0.5 * v ** 3

    def aero_load(self, t, z, nu): # normal load

        r = z[0]
        v = z[3]

        aero = self.mission.nonlinear_aero(t, z, nu)

        L = aero["L"]
        D = aero["D"]

        return jnp.sqrt(L ** 2 + D ** 2)

    def dynamic_pressure_nonjax(self, t, z, nu):
        r = z[0]
        v = z[3]

        rho = self.atmosphere_model_nonjax(r)

        return 0.5 * rho * v ** 2

    def heat_rate_nonjax(self, t, z, nu):
        r = z[0]
        v = z[3]

        rho = self.atmosphere_model_nonjax(r)

        return self.mission.vehicle["kQ"] * rho ** 0.5 * v ** 3

    def aero_load_nonjax(self, t, z, nu):
        r = z[0]
        v = z[3]

        aero = self.mission.nonlinear_aero_nonjax(t, z, nu)

        L = aero["L"]
        D = aero["D"]

        return (L ** 2 + D ** 2) ** 0.5