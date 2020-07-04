'''
Run and evaluate the model, and generate the figure used in the Graphic Abstract (also Figure 6 in the paper).
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tk
from scipy.io import loadmat
from calendar import isleap
from Optimality import Optimality
np.warnings.filterwarnings('ignore')

folders = ['Cumberland','Tapajos','Borden','Hyytiala','Nonantola','Takayama','Cheorwon','Champaign','Harvard','Tonzi']
sites = ['AU-Cum','BR-Sa1','CA-Cbo','FI-Hyy','IT-Non','JP-TKY','KR-CRK','SoyFACE','US-Ha1','US-Ton']

def eva(x,y):
    R2 = np.corrcoef(x,y)[0,1]**2
    RMSE = np.sqrt(np.mean((y-x)**2))
    MBE = np.mean(y-x)
    return R2,RMSE,MBE
    

X = np.array([])
Y = np.array([])
    

fig = plt.figure()


## AU-Cum

k = 0
ax = fig.add_subplot(5,2,k+1)
folder = folders[k]
site = sites[k]

doy = np.arange(1,366)
doy_ = np.concatenate([doy[doy>=182],doy[doy<182]])
dt = pd.date_range('2008-07-01','2009-06-30')
doy = np.arange(-183,182)

ratio0 = np.array([ 0.4569,
                    0.4579,
                    0.4617,
                    0.4702,
                    0.4676,
                    0.4626,
                    0.4710,
                    0.4665,
                    0.4635,
                    0.4600,
                    0.4616,
                    0.4576])

ratio = np.full(dt.size,np.nan)
for i in range(12):
    msk = dt.month == (7+i if i < 6 else i-5)
    ratio[msk] = ratio0[i]

csv = pd.read_csv('%s/FLUXNET2015/FLUXNET2015.csv' % folder)
SW = csv['SW']
TA = csv['TA']
VPD = csv['VPD']
PA = csv['PA']
PAR = SW * ratio * 4.56
CO2 = csv['CA']

csv = pd.read_csv('%s/MODIS/MODIS.csv' % folder)
LAI = csv['LAI']
LAI[LAI<1e-5] = 1e-5
ALB = csv['ALB']

CC = 0.33
CI = 0.8112
LAI0 = 4.047
MAT = 17.3
kn = -0.62 * np.log(LAI0) + 0.98
nlag = 40
nlag = np.uint8(np.round(nlag))

simu = Optimality(PAR,TA,VPD,CO2,PA,LAI,ALB,CC,CI,kn,nlag)

csv = np.loadtxt('%s/Screenshot/Lin20082009.csv' % folder,delimiter=',')
temp = csv[:,0]
temp[temp==1] = -123
temp[temp==2] = -47
temp[temp==3] = 31
meas = np.array([csv[temp==i,1].mean() for i in doy])
unc = np.array([csv[temp==i,1].std() for i in doy])
msk = np.isfinite(meas)
err = eva(meas[msk],simu[msk])

ax.errorbar(doy,meas,yerr=unc,color='k',fmt='o',elinewidth=0.5,ecolor='gray',capsize=1,ms=2)
ax.plot(doy,meas,'ko',ms=2)
ax.plot(doy,simu,'black',lw=1)

ax.text(0.025,0.95,'(a) %s' % site,ha='left',va='top',size=9,transform=ax.transAxes)
ax.set_xlim(doy[0]-1,doy[-1]+1)
ax.set_ylim(0,125)
ax.xaxis.set_major_locator(tk.FixedLocator(locs=np.where(dt.day==16)[0]-184))
ax.xaxis.set_minor_locator(tk.FixedLocator(locs=np.where(dt.day==1)[0]-184)) 
[tl.set_markersize(0) for tl in ax.xaxis.get_ticklines(minor=False)]
[tl.set_markersize(3) for tl in ax.yaxis.get_ticklines(minor=False) + ax.yaxis.get_ticklines(minor=True) + ax.xaxis.get_ticklines(minor=True)]
ax.set_xticklabels(['Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May','Jun'])
ax.text(0.80,0.95,'R$^2$ = %.2f\nRMSE = %.1f\nbias = %.1f' % (err[0],err[1],err[2]),ha='left',va='top',transform=ax.transAxes)

ax.legend(['Measurements','Estimations'],loc='lower left',numpoints=3)

X = np.concatenate([X,meas])
Y = np.concatenate([Y,simu])

## BR-Sa1

k = 1
ax = fig.add_subplot(5,2,k+1)
folder = folders[k]
site = sites[k]

year = 2014
doy = np.arange(1,367 if isleap(year) else 366)
dt = pd.date_range('%d-01-01'%year,'%d-12-31'%year)                

csv = pd.read_csv('%s/FLUXNET2015/FLUXNET2015.csv' % folder)
PAR = csv['PAR']
TA = csv['TA']
VPD = csv['VPD']
CO2 = csv['CA']
PA = csv['PA']
ALB = csv['ALB']

csv = pd.read_csv('%s/Screenshot/Screenshot.csv' % folder)
LAI = csv['LAI']
LAI[LAI<1e-5] = 1e-5
FVC = 1 - csv['LC']/100

CC = FVC
CI = 0.5101
LAI0 = 4.863
MAT = 26.1
kn = -0.62 * np.log(LAI0) + 0.98
nlag = 40
nlag = np.uint8(np.round(nlag))

simu = Optimality(PAR,TA,VPD,CO2,PA,LAI,ALB,CC,CI,kn,nlag)

csv = np.loadtxt('%s/Albert.csv' % folder,delimiter=',')
meas = np.array([csv[csv[:,0]==i,1].mean() for i in doy])
unc = np.array([csv[csv[:,0]==i,1].std() for i in doy])
msk = np.isfinite(meas)
err = eva(meas[msk],simu[msk])

ax.errorbar(doy,meas,yerr=unc,color='k',fmt='o',elinewidth=0.5,ecolor='gray',capsize=1,ms=2)
ax.plot(doy,meas,'ko',ms=2)
ax.plot(doy,simu,'black',lw=1)

ax.text(0.025,0.95,'(b) %s' % site,ha='left',va='top',size=9,transform=ax.transAxes)
ax.set_xlim(doy[0]-1,doy[-1]+1)
ax.set_ylim(0,125)
ax.xaxis.set_major_locator(tk.FixedLocator(locs=np.where(dt.day==16)[0]))
ax.xaxis.set_minor_locator(tk.FixedLocator(locs=np.where(dt.day==1)[0])) 
[tl.set_markersize(0) for tl in ax.xaxis.get_ticklines(minor=False)]
[tl.set_markersize(3) for tl in ax.yaxis.get_ticklines(minor=False) + ax.yaxis.get_ticklines(minor=True) + ax.xaxis.get_ticklines(minor=True)]
ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
ax.text(0.80,0.95,'R$^2$ = %.2f\nRMSE = %.1f\nbias = %.1f' % (err[0],err[1],err[2]),ha='left',va='top',transform=ax.transAxes)

X = np.concatenate([X,meas])
Y = np.concatenate([Y,simu])


## CA-Cbo

k = 2
ax = fig.add_subplot(5,2,k+1)
folder = folders[k]
site = sites[k]

year = 2014
doy = np.arange(1,367 if isleap(year) else 366)
dt = pd.date_range('%d-01-01'%year,'%d-12-31'%year)                       

csv = pd.read_csv('%s/AmeriFlux/AmeriFlux.csv' % folder)
PAR = csv['PAR']
TA = csv['TA']
VPD = csv['VPD']
VPD[VPD<0] = 0
CO2 = csv['CO2']
PA = csv['PA']
ALB = csv['ALB']

csv = pd.read_csv('%s/Screenshot/Screenshot.csv' % folder)
LAI = csv['LAI']
LAI[LAI<1e-5] = 1e-5

CC = 0.87
CI = 0.7080
LAI0 = 4.965
MAT = 6.7
kn = -0.62 * np.log(LAI0) + 0.98
if kn < 0.01: kn = 0.01
nlag = 40
nlag = np.uint8(np.round(nlag))

simu = Optimality(PAR,TA,VPD,CO2,PA,LAI,ALB,CC,CI,kn,nlag)

csv = np.loadtxt('%s/Croft2014.csv' % folder,delimiter=',')
meas = np.array([csv[csv[:,0]==i,1].mean() for i in doy])
meas[meas>85] = np.nan
unc = np.array([csv[csv[:,0]==i,1].std() for i in doy])
msk = np.isfinite(meas)
err = eva(meas[msk],simu[msk])

ax.errorbar(doy,meas,yerr=unc,color='k',fmt='o',elinewidth=0.5,ecolor='gray',capsize=1,ms=2)
ax.plot(doy,meas,'ko',ms=2)
ax.plot(doy,simu,'black',lw=1)

ax.text(0.025,0.95,'(c) %s' % site,ha='left',va='top',size=9,transform=ax.transAxes)
ax.set_xlim(doy[0]-1,doy[-1]+1)
ax.set_ylim(0,125)
ax.xaxis.set_major_locator(tk.FixedLocator(locs=np.where(dt.day==16)[0]))
ax.xaxis.set_minor_locator(tk.FixedLocator(locs=np.where(dt.day==1)[0])) 
[tl.set_markersize(0) for tl in ax.xaxis.get_ticklines(minor=False)]
[tl.set_markersize(3) for tl in ax.yaxis.get_ticklines(minor=False) + ax.yaxis.get_ticklines(minor=True) + ax.xaxis.get_ticklines(minor=True)]
ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
ax.text(0.80,0.95,'R$^2$ = %.2f\nRMSE = %.1f\nbias = %.1f' % (err[0],err[1],err[2]),ha='left',va='top',transform=ax.transAxes)

X = np.concatenate([X,meas])
Y = np.concatenate([Y,simu])


## FI-Hyy

k = 3
ax = fig.add_subplot(5,2,k+1)
folder = folders[k]
site = sites[k]

year = 2011
doy = np.arange(1,367 if isleap(year) else 366)
dt = pd.date_range('%d-01-01'%year,'%d-12-31'%year)                  

csv = pd.read_csv('%s/FLUXNET2015/FLUXNET2015.csv' % folder)
PAR = csv['PAR']
TA = csv['TA']
VPD = csv['VPD']
CO2 = csv['CO2']
PA = csv['PA']
ALB = csv['ALB']

csv = pd.read_csv('%s/Screenshot/Screenshot.csv' % folder)
LAI = csv['LAI']
LAI[LAI<1e-5] = 1e-5

CC = 0.53
CI = 0.5519
LAI0 = 2.946
MAT = 3.8
kn = -0.62 * np.log(LAI0) + 0.98
nlag = 40
nlag = np.uint8(np.round(nlag))

simu = Optimality(PAR,TA,VPD,CO2,PA,LAI,ALB,CC,CI,kn,nlag)

csv = np.loadtxt('%s/Kolari2011.csv' % folder,delimiter=',')
meas = np.array([csv[csv[:,0]==i,1].mean() for i in doy])
unc = np.array([csv[csv[:,0]==i,1].std() for i in doy])
msk = np.isfinite(meas)
err = eva(meas[msk],simu[msk])

ax.errorbar(doy,meas,yerr=unc,color='k',fmt='o',elinewidth=0.5,ecolor='gray',capsize=1,ms=2)
ax.plot(doy,meas,'ko',ms=2)
ax.plot(doy,simu,'black',lw=1)

ax.text(0.025,0.95,'(d) %s' % site,ha='left',va='top',size=9,transform=ax.transAxes)
ax.set_xlim(doy[0]-1,doy[-1]+1)
ax.set_ylim(0,125)
ax.xaxis.set_major_locator(tk.FixedLocator(locs=np.where(dt.day==16)[0]))
ax.xaxis.set_minor_locator(tk.FixedLocator(locs=np.where(dt.day==1)[0])) 
[tl.set_markersize(0) for tl in ax.xaxis.get_ticklines(minor=False)]
[tl.set_markersize(3) for tl in ax.yaxis.get_ticklines(minor=False) + ax.yaxis.get_ticklines(minor=True) + ax.xaxis.get_ticklines(minor=True)]
ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
ax.text(0.80,0.95,'R$^2$ = %.2f\nRMSE = %.1f\nbias = %.1f' % (err[0],err[1],err[2]),ha='left',va='top',transform=ax.transAxes)

X = np.concatenate([X,meas])
Y = np.concatenate([Y,simu])


## IT-Non

k = 4
ax = fig.add_subplot(5,2,k+1)
folder = folders[k]
site = sites[k]

year = 2003
doy = np.arange(1,367 if isleap(year) else 366)
dt = pd.date_range('%d-01-01'%year,'%d-12-31'%year)
                    
ratio0 = np.array([     0.4684,
                        0.4529,
                        0.4571,
                        0.4569,
                        0.4540,
                        0.4556,
                        0.4595,
                        0.4576,
                        0.4589,
                        0.4604,
                        0.4587,
                        0.4515])                      
                    
CO20 = np.array([ 374.2936,
                  375.2486,
                  376.5358,
                  375.6290,
                  377.2791,
                  375.3595,
                  376.4950,
                  372.1979,
                  372.5915,
                  373.8950,
                  370.8306,
                  370.8306])
                    
ratio = np.full(dt.size,np.nan)
for i in range(12):
    msk = dt.month == i + 1
    ratio[msk] = ratio0[i]
    
CO2 = np.full(dt.size,np.nan)
for i in range(12):
    msk = dt.month == i + 1
    CO2[msk] = CO20[i]
    
csv = pd.read_csv('%s/EropeanFlux/EuropeanFlux.csv' % folder)
SW = csv['SW']
TA = csv['TA']
VPD = csv['VPD']
PA = csv['PA']
PAR = SW * ratio * 4.56

csv = pd.read_csv('%s/MODIS/MODIS.csv' % folder)
LAI = csv['LAI']
LAI[LAI<1e-5] = 1e-5
ALB = csv['ALB']

CC = 0.77
CI = 0.7710
LAI0 = 1.3
MAT = 14.5
kn = -0.62 * np.log(LAI0) + 0.98
nlag = 40
nlag = np.uint8(np.round(nlag))

simu = Optimality(PAR,TA,VPD,CO2,PA,LAI,ALB,CC,CI,kn,nlag)

csv = np.loadtxt('%s/Grassi2003.csv' % folder,delimiter=',')
meas = np.array([csv[csv[:,0]==i,1].mean() for i in doy])
unc = np.array([csv[csv[:,0]==i,1].std() for i in doy])
msk = np.isfinite(meas)
err = eva(meas[msk],simu[msk])

ax.errorbar(doy,meas,yerr=unc,color='k',fmt='o',elinewidth=0.5,ecolor='gray',capsize=1,ms=2)
ax.plot(doy,meas,'ko',ms=2)
ax.plot(doy,simu,'black',lw=1)

ax.text(0.025,0.95,'(e) %s' % site,ha='left',va='top',size=9,transform=ax.transAxes)
ax.set_xlim(doy[0]-1,doy[-1]+1)
ax.set_ylim(0,125)
ax.xaxis.set_major_locator(tk.FixedLocator(locs=np.where(dt.day==16)[0]))
ax.xaxis.set_minor_locator(tk.FixedLocator(locs=np.where(dt.day==1)[0])) 
[tl.set_markersize(0) for tl in ax.xaxis.get_ticklines(minor=False)]
[tl.set_markersize(3) for tl in ax.yaxis.get_ticklines(minor=False) + ax.yaxis.get_ticklines(minor=True) + ax.xaxis.get_ticklines(minor=True)]
ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
ax.text(0.80,0.95,'R$^2$ = %.2f\nRMSE = %.1f\nbias = %.1f' % (err[0],err[1],err[2]),ha='left',va='top',transform=ax.transAxes)

ax.set_ylabel('$V_{cmax,25C}$' + ' ' + '$(\mu mol\ m^{-2}\ s^{-1})$',labelpad=0)

X = np.concatenate([X,meas])
Y = np.concatenate([Y,simu])


## JP-TKY

k = 5
ax = fig.add_subplot(5,2,k+1)
folder = folders[k]
site = sites[k]

year = 2004
doy = np.arange(1,367 if isleap(year) else 366)
dt = pd.date_range('%d-01-01'%year,'%d-12-31'%year)                    

csv = pd.read_csv('%s/LaThuile/LaThuile.csv' % folder)
PAR = csv['PAR']
TA = csv['TA']
VPD = csv['VPD']
CO2 = csv['CO2']
PA = csv['PA']
ALB = csv['ALB']

csv = pd.read_csv('%s/Screenshot/Screenshot.csv' % folder)
LAI = csv['LAI']
LAI[LAI<1e-5] = 1e-5

CC = 0.72
CI = 0.5108
LAI0 = 4.459
MAT = 6.5
kn = -0.62 * np.log(LAI0) + 0.98
nlag = 40
nlag = np.uint8(np.round(nlag))

simu = Optimality(PAR,TA,VPD,CO2,PA,LAI,ALB,CC,CI,kn,nlag)

csv = np.loadtxt('%s/Muraoka2004All.csv' % folder,delimiter=',')
meas = np.array([csv[csv[:,0]==i,1].mean() for i in doy])
unc = np.array([csv[csv[:,0]==i,1].std() for i in doy])
msk = np.isfinite(meas)
err = eva(meas[msk],simu[msk])

ax.errorbar(doy,meas,yerr=unc,color='k',fmt='o',elinewidth=0.5,ecolor='gray',capsize=1,ms=2)
ax.plot(doy,meas,'ko',ms=2)
ax.plot(doy,simu,'black',lw=1)

ax.text(0.025,0.95,'(f) %s' % site,ha='left',va='top',size=9,transform=ax.transAxes)
ax.set_xlim(doy[0]-1,doy[-1]+1)
ax.set_ylim(0,125)
ax.xaxis.set_major_locator(tk.FixedLocator(locs=np.where(dt.day==16)[0]))
ax.xaxis.set_minor_locator(tk.FixedLocator(locs=np.where(dt.day==1)[0])) 
[tl.set_markersize(0) for tl in ax.xaxis.get_ticklines(minor=False)]
[tl.set_markersize(3) for tl in ax.yaxis.get_ticklines(minor=False) + ax.yaxis.get_ticklines(minor=True) + ax.xaxis.get_ticklines(minor=True)]
ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
ax.text(0.80,0.95,'R$^2$ = %.2f\nRMSE = %.1f\nbias = %.1f' % (err[0],err[1],err[2]),ha='left',va='top',transform=ax.transAxes)

X = np.concatenate([X,meas])
Y = np.concatenate([Y,simu])


## KR-CRK

k = 6
ax = fig.add_subplot(5,2,k+1)
folder = folders[k]
site = sites[k]

year = 2016
doy0 = np.arange(1,367)
doy = np.arange(121,247)
dt = pd.date_range('%d-01-01'%year,'%d-12-31'%year) 

CO20 = np.array([ 402.6573,
                  401.9472,
                  404.7636,
                  404.7440,
                  404.6871,
                  401.1481,
                  397.6090,
                  399.7267,
                  396.0789,
                  400.4265,
                  402.9648,
                  404.2444])

CO2 = np.full(dt.size,np.nan)
for i in range(12):
    msk = dt.month == i+1
    CO2[msk] = CO20[i]
CO2 = CO2[np.in1d(doy0,doy)]
    
PAR = loadmat('%s/Field/PAR_Daily.mat' % folder)['data'].ravel()
TA = loadmat('%s/Field/Ta_Daily.mat' % folder)['data'].ravel()
TA[doy>237] = TA[doy==237]
VPD = loadmat('%s/Field/VPD_Daily.mat' % folder)['data'].ravel()
ALT = 175
PA = 101325 * (1.0 - 0.0065*ALT/TA)**(9.807/(0.0065*287))

LAI = loadmat('%s/Field/LAI.mat' % folder)['data'].ravel()
LAI[LAI<1e-5] = 1e-5
FVC = 1 - np.exp(-0.5*LAI)

csv = pd.read_csv('%s/MODIS/MODIS.csv' % folder)
ALB = csv['ALB']

CC = FVC
CI = 0.7756
LAI0 = 3.844
MAT = 10.2
kn = 1.16
nlag = 40
nlag = np.uint8(np.round(nlag))

simu = Optimality(PAR,TA,VPD,CO2,PA,LAI,ALB,CC,CI,kn,nlag)

csv = np.loadtxt('%s/CW2016.csv' % folder,delimiter=',')
meas = np.array([csv[csv[:,0]==i,1].mean() for i in doy])
unc = np.array([csv[csv[:,0]==i,1].std() for i in doy])
msk = np.isfinite(meas)
err = eva(meas[msk],simu[msk])

ax.errorbar(doy,meas,yerr=unc,color='k',fmt='o',elinewidth=0.5,ecolor='gray',capsize=1,ms=2)
ax.plot(doy,meas,'ko',ms=2)
ax.plot(doy,simu,'black',lw=1)

ax.text(0.025,0.95,'(g) %s' % site,ha='left',va='top',size=9,transform=ax.transAxes)
ax.set_xlim(0,365)
ax.set_ylim(0,125)
ax.xaxis.set_major_locator(tk.FixedLocator(locs=np.where(dt.day==16)[0]))
ax.xaxis.set_minor_locator(tk.FixedLocator(locs=np.where(dt.day==1)[0])) 
[tl.set_markersize(0) for tl in ax.xaxis.get_ticklines(minor=False)]
[tl.set_markersize(3) for tl in ax.yaxis.get_ticklines(minor=False) + ax.yaxis.get_ticklines(minor=True) + ax.xaxis.get_ticklines(minor=True)]
ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
ax.text(0.80,0.95,'R$^2$ = %.2f\nRMSE = %.1f\nbias = %.1f' % (err[0],err[1],err[2]),ha='left',va='top',transform=ax.transAxes)

X = np.concatenate([X,meas])
Y = np.concatenate([Y,simu])


## SoyFACE

k = 7
ax = fig.add_subplot(5,2,k+1)
folder = folders[k]
site = sites[k]

year = 2001
doy0 = np.arange(1,367 if isleap(year) else 366)
doy = np.arange(117,271)
dt = pd.date_range('%d-01-01'%year,'%d-12-31'%year)
                        
ratio0 = np.array([     0.4599,
                        0.4526,
                        0.4471,
                        0.4557,
                        0.4681,
                        0.4634,
                        0.4622,
                        0.4629,
                        0.4636,
                        0.4660,
                        0.4551,
                        0.4588])

CO20 = np.array([ 377.7832,
                  377.7832,
                  377.7832,
                  377.9238,
                  376.7472,
                  373.5424,
                  371.7698,
                  368.3500,
                  371.5480,
                  373.6146,
                  374.9870,
                  375.4457])
                    
ratio = np.full(dt.size,np.nan)
for i in range(12):
    msk = dt.month == i + 1
    ratio[msk] = ratio0[i]
    
CO2 = np.full(dt.size,np.nan)
for i in range(12):
    msk = dt.month == i + 1
    CO2[msk] = CO20[i]

csv = pd.read_csv('%s/AmeriFlux/AmeriFlux.csv' % folder)
SW = csv['SW']
TA = csv['TA']
VPD = csv['VPD']
PA = csv['PA']
PAR = SW * ratio * 4.56

csv = pd.read_csv('%s/MODIS/MODIS.csv' % folder)
LAI = csv['LAI']
LAI[doy0>268] = 0
LAI[LAI<1e-5] = 1e-5
FVC = 1 - np.exp(-0.5*LAI)
ALB = csv['ALB']

msk = np.in1d(doy0,doy)
PAR = PAR[msk]
TA = TA[msk]
VPD = VPD[msk]
CO2 = CO2[msk]
PA = PA[msk]
LAI = LAI[msk]
ALB = ALB[msk]
FVC = FVC[msk]

CC = FVC
CI = 0.7671
LAI0 = 4.29
MAT = 11
kn = 1.16
nlag = 40
nlag = np.uint8(np.round(nlag))

simu = Optimality(PAR,TA,VPD,CO2,PA,LAI,ALB,CC,CI,kn,nlag)

csv = np.loadtxt('%s/Bernacchi2001.csv' % folder,delimiter=',')
meas = np.array([csv[csv[:,0]==i,1].mean() for i in doy])
unc = np.array([csv[csv[:,0]==i,1].std() for i in doy])
msk = np.isfinite(meas)
err = eva(meas[msk],simu[msk])

ax.errorbar(doy,meas,yerr=unc,color='k',fmt='o',elinewidth=0.5,ecolor='gray',capsize=1,ms=2)
ax.plot(doy,meas,'ko',ms=2)
ax.plot(doy,simu,'black',lw=1)

ax.text(0.025,0.95,'(h) %s' % site,ha='left',va='top',size=9,transform=ax.transAxes)
ax.set_xlim(0,365)
ax.set_ylim(0,125)
ax.xaxis.set_major_locator(tk.FixedLocator(locs=np.where(dt.day==16)[0]))
ax.xaxis.set_minor_locator(tk.FixedLocator(locs=np.where(dt.day==1)[0])) 
[tl.set_markersize(0) for tl in ax.xaxis.get_ticklines(minor=False)]
[tl.set_markersize(3) for tl in ax.yaxis.get_ticklines(minor=False) + ax.yaxis.get_ticklines(minor=True) + ax.xaxis.get_ticklines(minor=True)]
ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
ax.text(0.80,0.95,'R$^2$ = %.2f\nRMSE = %.1f\nbias = %.1f' % (err[0],err[1],err[2]),ha='left',va='top',transform=ax.transAxes)

X = np.concatenate([X,meas])
Y = np.concatenate([Y,simu])


## US-Ha1

k = 8
ax = fig.add_subplot(5,2,k+1)
folder = folders[k]
site = sites[k]

year = 2010
doy = np.arange(1,367 if isleap(year) else 366)
dt = pd.date_range('%d-01-01'%year,'%d-12-31'%year)                 

csv = pd.read_csv('%s/FLUXNET2015/FLUXNET2015.csv' % folder)
PAR = csv['PAR']
TA = csv['TA']
VPD = csv['VPD']
CO2 = csv['CO2']
PA = csv['PA']

csv = pd.read_csv('%s/AmeriFlux/AmeriFlux.csv' % folder)
LAI = csv['LAI']
LAI[LAI<1e-5] = 1e-5

csv = pd.read_csv('%s/MODIS/MODIS.csv' % folder)
ALB = csv['ALB']

CC = 0.61
CI = 0.6574
LAI0 = 3.372
MAT = 6.6
kn = -0.62 * np.log(LAI0) + 0.98
nlag = 40
nlag = np.uint8(np.round(nlag))

simu = Optimality(PAR,TA,VPD,CO2,PA,LAI,ALB,CC,CI,kn,nlag)

csv = np.loadtxt('%s/Dillen2010.csv' % folder,delimiter=',')
meas = np.array([csv[csv[:,0]==i,1].mean() for i in doy])
unc = np.array([csv[csv[:,0]==i,1].std() for i in doy])
msk = np.isfinite(meas)
err = eva(meas[msk],simu[msk])

ax.errorbar(doy,meas,yerr=unc,color='k',fmt='o',elinewidth=0.5,ecolor='gray',capsize=1,ms=2)
ax.plot(doy,meas,'ko',ms=2)
ax.plot(doy,simu,'black',lw=1)

ax.text(0.025,0.95,'(i) %s' % site,ha='left',va='top',size=9,transform=ax.transAxes)
ax.set_xlim(doy[0]-1,doy[-1]+1)
ax.set_ylim(0,125)
ax.xaxis.set_major_locator(tk.FixedLocator(locs=np.where(dt.day==16)[0]))
ax.xaxis.set_minor_locator(tk.FixedLocator(locs=np.where(dt.day==1)[0])) 
[tl.set_markersize(0) for tl in ax.xaxis.get_ticklines(minor=False)]
[tl.set_markersize(3) for tl in ax.yaxis.get_ticklines(minor=False) + ax.yaxis.get_ticklines(minor=True) + ax.xaxis.get_ticklines(minor=True)]
ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
ax.text(0.80,0.95,'R$^2$ = %.2f\nRMSE = %.1f\nbias = %.1f' % (err[0],err[1],err[2]),ha='left',va='top',transform=ax.transAxes)

X = np.concatenate([X,meas])
Y = np.concatenate([Y,simu])


## US-Ton

k = 9
ax = fig.add_subplot(5,2,k+1)
folder = folders[k]
site = sites[k]

year = 2001
doy = np.arange(1,367 if isleap(year) else 366)
dt = pd.date_range('%d-01-01'%year,'%d-12-31'%year)

csv = pd.read_csv('%s/FLUXNET2015/FLUXNET2015.csv' % folder)
PAR = csv['PAR']
TA = csv['TA']
VPD = csv['VPD']
CO2 = csv['CO2']
PA = csv['PA']
ALB = csv['ALB']

csv = pd.read_csv('%s/MODIS/MODIS.csv' % folder)
LAI = csv['LAI']
LAI[LAI<1e-5] = 1e-5

CC = 0.45
CI = 0.7968
LAI0 = 1.375
MAT = 15.8
kn = -0.62 * np.log(LAI0) + 0.98
nlag = 40
nlag = np.uint8(np.round(nlag))

simu = Optimality(PAR,TA,VPD,CO2,PA,LAI,ALB,CC,CI,kn,nlag)

csv = np.loadtxt('%s/Xu2001.csv' % folder,delimiter=',')
meas = np.array([csv[csv[:,0]==i,1].mean() for i in doy])
unc = np.array([csv[csv[:,0]==i,1].std() for i in doy])
msk = np.isfinite(meas)
err = eva(meas[msk],simu[msk])

ax.errorbar(doy,meas,yerr=unc,color='k',fmt='o',elinewidth=0.5,ecolor='gray',capsize=1,ms=2)
ax.plot(doy,meas,'ko',ms=2)
ax.plot(doy,simu,'black',lw=1)

ax.text(0.025,0.95,'(j) %s' % site,ha='left',va='top',size=9,transform=ax.transAxes)
ax.set_xlim(doy[0]-1,doy[-1]+1)
ax.set_ylim(0,125)
ax.xaxis.set_major_locator(tk.FixedLocator(locs=np.where(dt.day==16)[0]))
ax.xaxis.set_minor_locator(tk.FixedLocator(locs=np.where(dt.day==1)[0])) 
[tl.set_markersize(0) for tl in ax.xaxis.get_ticklines(minor=False)]
[tl.set_markersize(3) for tl in ax.yaxis.get_ticklines(minor=False) + ax.yaxis.get_ticklines(minor=True) + ax.xaxis.get_ticklines(minor=True)]
ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
ax.text(0.80,0.95,'R$^2$ = %.2f\nRMSE = %.1f\nbias = %.1f' % (err[0],err[1],err[2]),ha='left',va='top',transform=ax.transAxes)

X = np.concatenate([X,meas])
Y = np.concatenate([Y,simu])

plt.draw()
plt.rc('font',size=6)        
fig.subplots_adjust(left=0.06,right=0.99,bottom=0.03,top=0.99,hspace=0.15,wspace=0.1)
fig.set_size_inches(16.5/2.54,16.5/2.54*1.25)
plt.savefig('Results.png',dpi=300)
plt.close()

msk = np.isfinite(X) & np.isfinite(Y)
R2 = np.corrcoef(X[msk],Y[msk])[0,1]**2
RMSE = np.sqrt(np.mean((Y[msk]-X[msk])**2))
Bias = np.mean(Y[msk]-X[msk])
print(R2)
print(RMSE)
print(Bias)
