'''
########### Plotting code which stiches output DS files from DS_VI_fromRLvisib_MPI.py ######################
Plot combines DS for I, V and circ polarisation using pickle file data

'''
import pickle,os,glob,numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
from datetime import timedelta as Dt
from datetime import datetime as dt
from matplotlib import colors
from copy import deepcopy as cpy
from astropy import units as u
from astropy.constants import k_B
########### INPUT ##################################
V_pfildir='/Data/SplitMS/VDS_RLvisAvg/DS_Vtol0.5mJy_dt10_df50/DSpfils/' # Folder containing STOKES V DS pickle files
I_pfildir='/Data/SplitMS/IDS_RLvisAvg/DS_Vtol0.5mJy_dt10_df50/DSpfils/'  # Folder containing STOKES I DS pickle files
Pol_pfildir='/Data/SplitMS/polDS_RLvisAvg/DS_Vtol0.5mJy_dt10_df50/DSpfils/'  # Folder containing Circular Polariation DS pickle files
scans=[4,6,8,10]
fullVDS_pdir= '/Data/SplitMS/VDS_RLvisAvg/DS_Vtol0.5mJy_dt10_df50/' # Final location where the STOKES V DS has to be saved.
fullIDS_pdir= '/Data/SplitMS/IDS_RLvisAvg/DS_Vtol0.5mJy_dt10_df50/' # Final location where the STOKES I DS has to be saved.
fullpolDS_pdir= '/Data/SplitMS/polDS_RLvisAvg/DS_Vtol0.5mJy_dt10_df50/' # Final location where the STOKES V/I or circ. polarisation DS has to be saved.
cmap='nipy_spectral_r' # Colormap to be used in the plot.
unit='mJy' # Flux density unit in the plot or for computing the DS
tunit='min' # Plot time axis unit.
tconv=1/60. # Conversion from s to chosen time 
Vsign=-1 # Expected STOKES V sign
inverty_V=True # Invert y axis for stokes V plot? True or False
flux_ulim=26 # Flag any flux beyond this in unit of choice
Vmax=15 # Give a modulus of vmax for the flux colorscale in |vmax| unit. Leave as '' if you want the code to set this 
rot=45 # Degree by which X axis labels are to be rotated
flag_chans=[] # List of channels to be flagged fully in the final DS made. can be given as +ve and -ve indices representing DS columns
'''
######### INPUT for light curve data ########################################################################
'''
Poln_colcode=True # Only for light curve plots, if polarisation level is to be color-coded

'''
## INPUT for Plot axis units ################################################################################
'''
flxUnit= 2# 1. Flux density 2. TB. If 1, then the chosen flux density units will be used to plot the data

## Stellar parameters

R=0.328 #Radius of the star in Rsun
d=8.079	#Distance in pc to the star
#Tcor=2.8*10**6 # Stellar coronal temperature
'''
######## END of INPUT section ###############################################################################
##### DO NOT EDIT ANY portions further down in the code unless you are a developer ##########################
'''

conv=u.Jy.to(unit)
TBf=d**2/(2*k_B.value*np.pi*(R*u.Rsun.to('pc'))**2*conv*10**26)
###################################################
tflg=np.array(flag_chans).astype(str)
nmtg='_flgChn'+'.'.join(tflg) if len(tflg)>0 else ''
Cmap=cpy(cmap)
del tflg

if flxUnit==2:
	pltu=TBf
	collabs=[r'$\mathrm{T_B}$']*2+['Percentage (%)']
	nmtg+='_TB'
else:
	collabs=['Flux density ('+unit+')']*2+['Percentage (%)']
	pltu=1
del TBf


## Making dummy DS
Dp=pickle.load(open(glob.glob(V_pfildir+'*scan'+str(scans[0])+'_*.p')[0],'rb'))
DS1=Dp['DS']
freqs=Dp['nu']['Range']
fqunit=Dp['nu']['Unit']

dummy=np.zeros((DS1.shape[0],1))+9999
dummy=ma.masked_invalid(dummy)
VDS,IDS,polDS=dummy,dummy,dummy
times=np.array([np.nan])
Beg=Dp['Time']['Begin']
del DS1
del Dp
 
for scan in scans:
	Vs=sorted(glob.glob(V_pfildir+'*scan'+str(scan)+'_*.p'))
	inds=np.array([int(i.split('.p')[0].split('_')[-1].replace('t','')) for i in Vs])
	Match=dict(zip(inds,np.arange(len(Vs))))

	inds=np.sort(inds)
	VDSs=[Vs[Match[i]] for i in inds]

	IDSs=[glob.glob(I_pfildir+'*scan'+str(scan)+'_t'+str(i)+'.p')[0] for i in inds]
	polDSs=[glob.glob(Pol_pfildir+'*scan'+str(scan)+'_t'+str(i)+'.p')[0] for i in inds]

	for i in range(len(inds)):
		Vp=pickle.load(open(VDSs[i],'rb'))
		VDS=ma.hstack((VDS,Vp['DS']*conv,dummy))
		IDS=ma.hstack((IDS,pickle.load(open(IDSs[i],'rb'))['DS']*conv,dummy))
		polDS=ma.hstack((polDS,pickle.load(open(polDSs[i],'rb'))['DS'],dummy))
		delt=Vp['Time']['Begin']-Beg
		times=np.append(times,Vp['Time']['Range']+delt.seconds+delt.microseconds*10**-6)
			
		times=np.append(times,np.array([np.nan]))
		
VDS=VDS[:,1:-1]
IDS=IDS[:,1:-1]
polDS=polDS[:,1:-1]
times=times[1:-1]
nantims=np.where(np.isnan(times))[0]

VDS=ma.masked_inside(VDS,flux_ulim*Vsign,9998*Vsign)
IDS=ma.masked_inside(IDS,flux_ulim,9998)
## Flagging bad channels user mentioned in polDS alone since this will be transferred to others while plotting anyway##########

DSs=[VDS,IDS,polDS]
for i in flag_chans:
	polDS.mask[i,:]=True
	polDS.mask[i,nantims]=False

###############################################
## Name of figure file
fxtg='flxlim'+str(flux_ulim)+unit+nmtg

filnames=[fullVDS_pdir+'VDS_full_'+fxtg+'.png',fullIDS_pdir+'IDS_full_'+fxtg+'.png',fullpolDS_pdir+'polDS_full_'+fxtg+'.png']
stx=['Stokes V','Stokes I','Circ. Polarisation']
if Vsign==-1:
	cmaps=[cmap,cmap[:-2] if '_r' in cmap else cmap+'_r', cmap]
else:
	cmaps=[cmap]*3

del VDS
del IDS
del polDS

## Plotting section #######################################
## X axis setup
xp=list(nantims+1)
xp=[0]+xp
xp+=[len(times)-1]
step=len(xp)

xaxp=list(np.round(times[xp]*tconv,1))

inx=0
for DS in DSs:
	cmap=plt.cm.get_cmap(cmaps[inx])
	collab=collabs[inx]
	if inx>0:
		DS.mask=DS.mask+DSs[-1].mask # Apply polarisation mask only for STOKES I data. STOKES V is highly sensitive to stellar emission than STOKES I and any issue with polarisation not detected well should arise from poor STOKES I value. So STOKES I will be masked based on polarisation mask.
	else:
		for i in flag_chans:
			DS.mask[i,:]=True
			DS.mask[i,nantims]=False

	plt.figure(figsize=(16,8))
	td=cpy(DS)
	td[td==9999]=-9999
	vmax=ma.max(td)
	del td
	vmin=ma.min(DS)
	if Vmax!='':
		if inx==0:
			if Vsign==-1:
				vmin=-Vmax
			else:
				vmax=Vmax
		else:
			vmax=Vmax
	if Vsign==-1 and inx==0:
		DS[(DS==9999)]=-9999
	
	#Norm=colors.LogNorm(vmin=vmin,vmax=vmax)
	if inx!=2 and pltu!=1:
		vmax*=pltu
		vmin*=pltu
		Tscl=int(np.log10(np.abs(np.mean([vmax,vmin]))))
		vmax*=10**-Tscl
		vmin*=10**-Tscl		
		DS=DS*pltu*10**-Tscl	
		collab+=r'$\ (\times 10^{'+str(Tscl)+'}$'+' K)'

	if DS.shape[0]>1:
		if inx!=2:
			plt.imshow(DS,aspect='auto',origin='lower',cmap=cmap, extent=(0,DS.shape[1]-1,freqs[0],freqs[-1]),vmin=vmin-np.abs(vmin)*0.15,vmax=vmax-np.abs(vmax)*0.15)
		else:
			if Vsign==-1:
				vmax,vmin=0,-100
			else:
				vmax,vmin=100,0
			plt.imshow(DS,aspect='auto',origin='lower',cmap=cmap, extent=(0,DS.shape[1]-1,freqs[0],freqs[-1]),vmin=vmin,vmax=vmax)

		h=plt.colorbar(fraction=0.05)
		h.set_label(collab,size=20)
		h.ax.tick_params(labelsize=17)
		plt.ylabel('Frequency (MHz)',size=20)
	else:
		DS[0,nantims]=np.nan
		plt.plot(np.arange(DS.shape[1]),DS[0,:],'ko-',zorder=0)
		if Poln_colcode==True and inx!=2:
			plt.scatter(np.arange(DS.shape[1]),DS[0,:],c=DSs[-1][0,:],marker='o',cmap=Cmap,zorder=1)
			h=plt.colorbar(fraction=0.05)
			h.set_label(stx[-1]+' (%)',size=20)
			h.ax.tick_params(labelsize=17)			
		plt.ylabel(collab,size=20)
		if inverty_V==True and inx==0:
			plt.gca().invert_yaxis()
	plt.xlabel('Time ('+tunit+') +'+str(Beg).split('.')[0],size=20)
	plt.xticks(xp,xaxp,size=16,rotation=rot)
	plt.yticks(size=16)
	plt.title(stx[inx],size=21)
	plt.tight_layout()
	plt.savefig(filnames[inx])
	plt.close()
	inx+=1
