import numpy as np


class P:
    t       = 0 #NN hopping
    mu      = 1 #chemical potential
    tnnn    = 2 #NNN hopping
    tz      = 3 #interlayer hopping
    tH      = 4 #Haldane hopping
    hx      = 5 #Zeeman field along x
    hy      = 6 #Zeeman field along y
    hz      = 7 #Zeeman field along z
    rsoc    = 8 #Rashba SOC strength
    kmsoc   = 9 #Kane-Mele SOC strength
    delta_s = 10 #s-wave pairing amplitude
    Vcdw    = 11 #CDW potential strength
    Vstag   = 12 #Staggered potential strength
    Jex     = 13 #Exchange field strength
    Jimp    = 14 #Impurity strength
    t_hof   = 15
    phi     = 16
    Vx_aah  = 17
    Vy_aah  = 18
    Jfm     = 19
    Jafm    = 20
    delta_pip = 21
    Vbilayer  = 22

    n_param = 23


    @staticmethod
    def set(param_dict=None, **kwargs):
        lam = np.zeros(P.n_param, dtype=np.float64)
        
        # merge dictionaries
        all_params = {}
        if param_dict is not None:
            all_params.update(param_dict)
        all_params.update(kwargs)

        for name, value in all_params.items():
            if not hasattr(P, name):
                raise ValueError(f"Unknown parameter '{name}'")
            pid = getattr(P, name)
            lam[pid] = value
        return lam