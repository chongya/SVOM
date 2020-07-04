'''
Jiang, C., Ryu, Y., Wang, H., Keenan, T.: An optimality-based model explains seasonal variation in C3 plant photosynthetic capacity, Clobal Change Biology, 2020. 
Contact: chongya.jiang@gmail.com
'''

import numpy as np
from scipy.special import expi
from scipy.signal import savgol_filter

def Optimality(PAR,TA,VPD,CO2,PA,LAI,ALB,CC,CI,kn,npast):
    '''
    The optimality model to calculate daily Vcmax25 of top leaves.
    
    PAR: time series of incident photosynthetically active radiation (mol m-2 d-1).
    TA: time series of surface air temperature (K).
    VPD: time series of vapor pressure deficit (Pa).
    CO2: time seires of ambient CO2 concentration (umol mol-1).
    PA: time series of surface air pressure (Pa).
    LAI: time series of leaf area index (LAI).
    ALB: time series of albedo in visible range (a.u.).
    CC: time series of crown cover (a.u.).
    CI: time series of clumping index (a.u.).
    kn: model parameter: nitrogen distribution coefficient accounting for vertical variation in nitrogen within the plant canopy (a.u.).
    npast: model parameter: the past n days that are considered for the lag effect (day).
    '''
    
    # antecedent environment
    PARLag = np.array([np.nanmean(PAR[max(0,i-npast):i+1]) for i in range(PAR.size)])
    TALag = np.array([np.nanmean(TA[max(0,i-npast):i+1]) for i in range(TA.size)])
    VPDLag = np.array([np.nanmean(VPD[max(0,i-npast):i+1]) for i in range(VPD.size)])
    CO2Lag = np.array([np.nanmean(CO2[max(0,i-npast):i+1]) for i in range(CO2.size)])
    PALag = np.array([np.nanmean(PA[max(0,i-npast):i+1]) for i in range(PA.size)])
    
    # growth temperature, defined as mean air temperature over the past 30 days.
    tgrowth = np.array([np.nanmean(TA[max(0,i-30):i+1]) for i in range(TA.size)]) - 273.16

    # the maximum quantum yield of photosystem II for a light-adapted leaf 
    alf = 0.352 + 0.022*(TALag-273.16) - 0.00034*(TALag-273.16)**2    # (Eq (3))
    alf = alf * 1 * 0.5 / 4    # (Eq (2)) but leaf absorptance is set 1 as it is already embedded in FPAR.

    # gas constant
    R = 8.314e-3    # [kJ mol-1 K-1]
    # patial pressure of O2
    O = PALag * 0.21    # [Pa]
    # Michaelis–Menten constants of carboxylation (Supplementary Table 1)
    KC = np.exp(38.05-79.43/(R*TALag)) * 1e-6 * PALag    # [Pa]
    # Michaelis–Menten constants of oxygenation (Supplementary Table 1)
    KO = np.exp(20.30-36.38/(R*TALag)) * 1000 * 1e-6 * PALag    # [Pa]
    # CO2 compensation point (Supplementary Table 1)
    GammaS = np.exp(19.02-37.83/(R*TALag)) * 1e-6 * PALag    # [Pa]
    # Michaelis–Menten coefficient of Rubisco (Supplementary Table 1)
    K = KC*(1+O/KO)    # [Pa]
    # viscosity of water relative to its value at 25 °C (Supplementary Table 1)
    etaS = np.exp(-580/(-138+TALag)**2*(TALag-298))    # [-]
    # the ratio of intercellular CO2 to ambient CO2 (Eq (5))
    ksi = np.sqrt(240*(K+GammaS)/(1.6*etaS*2.4))
    chi = (ksi+GammaS*np.sqrt(VPDLag)/CO2Lag)/(ksi+np.sqrt(VPDLag))    # [-]

    # Landscape-level leaf area index
    LAI0 = LAI.copy()
    # Plant-level leaf area index (Eq (7))
    LAI = LAI0 / CC    # [-]
    # extinction coefficient under diffuse sky radiation (Eq (9))
    c = -0.5 * CI * LAI
    I = 0.5 * (np.exp(c)*(c+1)-c**2*expi(c))
    kd = -np.log(2*I) / (CI*LAI)    # [-]
    # plant-level fraction of absorbed PAR
    FPAR = (1-ALB) * (1-np.exp(-kd*CI*LAI))

    # antecedent FPAR
    FPARLag = np.array([np.nanmean(FPAR[max(0,i-npast):i+1]) for i in range(FPAR.size)])
    # intercellular CO2 concentration
    Ci = CO2Lag * chi * 1e-6 * PALag    # [Pa]
    # plant-level averaged Vcmax (Eq (6))
    m = (Ci-GammaS) / (Ci+2*GammaS)
    c = np.sqrt(1-(0.41/m)**(2./3))
    c = np.real(c)
    m_ = m * c
    A = alf * 12 * PARLag * FPARLag * m_    # [gC m-2 d-1] 
    V = A / ((Ci-GammaS)/(Ci+K)) / (60*60*24*1e-6*12)    # [umol m-2 s-1]

    # top-leaf Vcmax (Eq (10))
    fC = (1-np.exp(-kn*CI)) / (kn*CI)
    Vtoc = V / fC    # [umol m-2 s-1]

    # entropy factor of Vcmax (Eq (12))
    dS = (668.39+tgrowth*-1.07) / 1000
    # temperature correction function (Eq (11))
    fT = np.exp(72*(TALag-298.15)/(R*TALag*298.15)) * (1+np.exp((dS*298.15-200)/(R*298.15)))/(1+np.exp((dS*TALag-200)/(R*TALag)))
    fT[fT<0.08] = 0.08
    # top-leaf Vcmax at 25 °C (Eq (13))
    Vtoc25 = Vtoc / fT
    
    # constrains
    Vtoc25[LAI0<=1e-5] = np.nanmin(Vtoc25)
    data = savgol_filter(Vtoc25,31,1)
    data[LAI0<=1e-5] = np.nan

    return data
