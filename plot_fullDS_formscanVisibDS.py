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

'''
########### INPUT ###########################################################################################
'''
basedir='/Data/' # Path to data folder
V_pfildir=basedir+'VDS_RLvisAvg/DS/DSpfils/' # Folder containing STOKES V DS pickle files
I_pfildir=basedir+'IDS_RLvisAvg/DS/DSpfils/'  # Folder containing STOKES I DS pickle files
Pol_pfildir=basedir+'polDS_RLvisAvg/DS/DSpfils/'  # Folder containing Circular Polariation DS pickle files
scans=[3,5,7,9,11,14,16,18,20,22] # scans to append to make VISAD
fullVDS_pdir= basedir+'VDS_RLvisAvg/' # Final location where the STOKES V DS has to be saved.
fullIDS_pdir= basedir+'IDS_RLvisAvg/' # Final location where the STOKES I DS has to be saved.
fullpolDS_pdir= basedir+'polDS_RLvisAvg/' # Final location where the STOKES V/I or circ. polarisation DS has to be saved.
cmap='nipy_spectral_r' # Colormap to be used in the plot.
unit='mJy' # Flux density unit in the plot or for computing the DS
tunit='min' # Plot time axis unit.
plot_UT=True # If plot should have UT in x axis. 
Vsign=-1 # Expected STOKES V sign
inverty_V=True # Invert y axis for stokes V plot? True or False
flux_ulim=26 # Flag any flux beyond this in unit of choice
Vmax='' # Give a modulus of vmax for the flux colorscale in |vmax| unit. Leave as '' if you want the code to set this 
rot=90 # Degree by which X axis labels are to be rotated
mark_only_scan_edges=True # Do you want to mark only the edges of scan times or at every split sub-scan edges?
interpol='none' # Python matplotlib dynamic spectrum smoothing function required. 'none' for no interpolation. Options can be found here: https://stackoverflow.com/questions/34230108/smoothing-imshow-plot-with-matplotlib

'''
### INPUT for plot label sizes 
'''
tick_size=20 # X and yaxis tick label mark size
axlabel_size=22 # X and Y axis label size
title_size=22 # Plot title size
'''
Optional Flagging section
'''
## Flagging in the final plotted DS matrix. Give the channel numbers based on the averaged DS of which plots are made. One can check the DS pickle files to get an idea of the channel numbers and frequencies.

flag_chans=[] # List of channels to be flagged fully in the final DS made along with a list of scan number as a tuple. Channels can be given as +ve and -ve indices representing DS columns. 
	#Eg: [(-2,[18,20,22]) flags -2 th channel (2nd last channel) for scans 18,20,22 
flag_val=True # Do you want to flag some regions of DS. This does clip flagging
flag_areas=[(16,'t2',3.8,''),(18,'',3.8,''),(20,'',3.8,''),(22,'',3.8,'')] # Give as (scan number, tN, flux upper limit, channel range) 
		# tN : '' --> all tNs will be considered for flag application. Else give eg: 't2'
		# channel range : egs:  [23,24] or [24]. Also '' can be give if all chans need to be considerd for clipping.
		# flux upper limit: 35 give as float in units specified in unit variable
		# Flagging will be in STOKES V DS.
        # Eg: [(16,'t2',3.5,''),(18,'',3.5,[23]),(20,'',3.5,[23])] --> flags all STOKES V flux areas with |V flux| > 3.5 in scan16_t2, |V flux [23,:]|>3.5 in scan18_t1, |V flux [23,:]|>3.5 in scan18_t2, |V flux [23,:]|>3.5 in scan20_t1, |V flux [23,:]|>3.5 in scan20_t2, where 23 is the 23rd channel and ':' denote all time series data in that channel.
'''
######### INPUT for light curve data ########################################################################
'''
Poln_colcode=True # Only for light curve plots, if polarisation level is to be color-coded
plot_vline=True # Want to plot vertical lines in light curve? True or False
Timstamps=['2021-12-03T23:09:00','2021-12-04T02:24:00'] # If plot_vline is True give the time stamps in UT where to put the lines. Format Eg: ['2021-12-03T23:09:00']
plot_xline=True # Want to plot horizontal line?
flx_val=[-1.93,3.8] # Give flux value in the specified unit in V and I as a list. eg: [-1.93,4]
'''
## INPUT for Plot axis units ################################################################################
'''
flxUnit=1 # 1. Flux density 2. TB. If 1, then the chosen flux density units will be used to plot the data

usertag='' # Optional input. Tag to be used for image name to distinguish previous runs if any.
## Stellar parameters

R=0.44 #Radius of the star in Rsun
d=4.7	#Distance in pc to the star
'''
######## END of INPUT section ###############################################################################

##############################################################################################################
##### DO NOT EDIT ANY portions further down in the code unless you are a developer ##########################
'''

conv=u.Jy.to(unit)
tconv=u.s.to(tunit)
TBf=d**2/(2*k_B.value*np.pi*(R*u.Rsun.to('pc'))**2*conv*10**26)
###################################################
tflg=np.array(flag_chans).astype(str)
nmtg='_flgChn'+'.'.join(tflg) if len(tflg)>0 else ''
Cmap=cpy(cmap)
del tflg

if flag_val==True:
	nmtg+='_flgArea'+str(len(flag_areas))

if flxUnit==2:
	pltu=TBf
	collabs=[r'$\mathrm{T_B}$']*2+['Percentage (%)']
	nmtg+='_TB'
else:
	collabs=['Flux density ('+unit+')']*2+['Percentage (%)']
	pltu=1
del TBf

if plot_UT==True:
	nmtg+='_UT'

fl_k=[]
fl_v=[]
if flag_val==True:
	for i in flag_areas:
		if i[1]!='':
			fl_k+=['scan'+str(i[0])+'_'+i[1]]
		else:
			fl_k+=['scan'+str(i[0])]
		fl_v+=[{'flux':i[2],'chans':i[3]}]
fl_regs=dict(zip(fl_k,fl_v))

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
		if 'scan'+str(scan) in fl_k or 'scan'+str(scan)+'_t'+str(i) in fl_k:
			b_reg=''
			try:
				b_reg=fl_regs['scan'+str(scan)]
			except:
				b_reg=fl_regs['scan'+str(scan)+'_t'+str(i)]
			print('Flagging regions in ','scan'+str(scan)+'_t'+str(i),' : ',b_reg)
			if b_reg['chans']!='':
				for chn in b_reg['chans']:
					bads=np.where(np.abs(Vp['DS'][chn,:]*conv)>b_reg['flux'])
					Vp['DS'].mask[bads]=True
			else:
				bads=np.where(np.abs(Vp['DS']*conv)>b_reg['flux'])
				Vp['DS'].mask[bads]=True

		if mark_only_scan_edges==True:
			VDS=ma.hstack((VDS,Vp['DS']*conv))	
			IDS=ma.hstack((IDS,pickle.load(open(IDSs[i],'rb'))['DS']*conv))
			polDS=ma.hstack((polDS,pickle.load(open(polDSs[i],'rb'))['DS']))
			delt=Vp['Time']['Begin']-Beg
			times=np.append(times,Vp['Time']['Range']+delt.seconds+delt.microseconds*10**-6)
			if i==len(inds)-1:
				VDS=ma.hstack((VDS,dummy))
				IDS=ma.hstack((IDS,dummy))
				polDS=ma.hstack((polDS,dummy))
				times=np.append(times,np.array([np.nan]))
		else:
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
fxtg='scn'+str(scans[0])+'-'+str(scans[-1])+'_flxlim'+str(flux_ulim)+unit+nmtg+usertag

filnames=[fullVDS_pdir+'VDS_'+fxtg+'.png',fullIDS_pdir+'IDS_'+fxtg+'.png',fullpolDS_pdir+'polDS_'+fxtg+'.png']
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
#step=len(xp)

if plot_UT==False:
	xaxp=list(np.round(times[xp]*tconv,1))
	XLAB='Time ('+tunit+')   +'+str(Beg).split('.')[0]
else:
	xaxp=[str((Beg+Dt(seconds=times[xi])).time()).split('.')[0] for xi in xp]
	XLAB='Time (UT)   +'+str(Beg.date())
################ V line plot x tick setting 
ppos=[]
poss=[]
if plot_vline==True and DSs[0].shape[0]==1:
	for tim in Timstamps:
		secs=(np.datetime64(tim)-np.datetime64(Beg)).astype('timedelta64[s]').astype(float)
		minpos=np.nanargmin(np.abs(times-secs))
		xp+=[minpos]
		poss+=[minpos]
		xaxp+=[tim.split('T')[1].split('.')[0]]
	xdict=dict(zip(xp,xaxp))
	xp=np.sort(np.array(xp))
	xaxp=[]
	xlj=0
	for xx in xp:
		xaxp+=[xdict[xx]]
		if xx in poss:
			ppos+=[xlj]
		xlj+=1
#############################################
inx=0
print(xp,xaxp)
#if plot_vline==True:

for DS in DSs:
	cmap=plt.cm.get_cmap(cmaps[inx])
	collab=collabs[inx]
	if inx>0:
		DS.mask=DS.mask+DSs[-1].mask # Apply polarisation mask only for STOKES I data. STOKES V is highly sensitive to stellar emission than STOKES I and any issue with polarisation not detected well should arise from poor STOKES I value. So STOKES I will be masked based on polarisation mask.
	else:
		for i in flag_chans:
			DS.mask[i,:]=True
			DS.mask[i,nantims]=False

	plt.figure(figsize=(16,8),dpi=90)
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
	if inx!=2:
		yvl=flx_val[inx]
	#Norm=colors.LogNorm(vmin=vmin,vmax=vmax)
	if inx!=2 and pltu!=1:
		vmax*=pltu
		vmin*=pltu
		Tscl=int(np.log10(np.abs(np.mean([vmax,vmin]))))
		vmax*=10**-Tscl
		vmin*=10**-Tscl		
		DS=DS*pltu*10**-Tscl
		yvl=yvl*pltu*10**-Tscl
		collab+=r'$\ (\times 10^{'+str(Tscl)+'}$'+' K)'

	if DS.shape[0]>1:
		if inx!=2:
			plt.imshow(DS,aspect='auto',origin='lower',cmap=cmap, interpolation=interpol,extent=(0,DS.shape[1]-1,freqs[0],freqs[-1]),vmin=vmin-np.abs(vmin)*0.15,vmax=vmax-np.abs(vmax)*0.15)
		else:
			if Vsign==-1:
				vmax,vmin=0,-100
			else:
				vmax,vmin=100,0
			plt.imshow(DS,aspect='auto',origin='lower',cmap=cmap, extent=(0,DS.shape[1]-1,freqs[0],freqs[-1]),vmin=vmin,vmax=vmax)

		h=plt.colorbar(fraction=0.05)
		h.set_label(collab,size=axlabel_size+1)
		h.ax.tick_params(labelsize=tick_size+1)
		plt.ylabel('Frequency (MHz)',size=axlabel_size)
	else:
		DS[0,nantims]=np.nan
		plt.plot(np.arange(DS.shape[1]),DS[0,:],'ko-',zorder=0)
		if Poln_colcode==True and inx!=2:
			plt.scatter(np.arange(DS.shape[1]),DS[0,:],c=DSs[-1][0,:],marker='o',cmap=Cmap,zorder=1)
			h=plt.colorbar(fraction=0.05)
			h.set_label(stx[-1]+' (%)',size=axlabel_size+1)
			h.ax.tick_params(labelsize=tick_size+1)
		plt.xlim([0,DS.shape[1]-1])
	
		if len(poss)>0:
			for minpos in poss:
				plt.axvline(minpos,linewidth=4,linestyle='--')
		if plot_xline==True and inx<2:
			plt.axhline(yvl,linewidth=4,linestyle='--',color='k')			
		plt.ylabel(collab,size=axlabel_size+1)
		if inverty_V==True and inx==0:
			plt.gca().invert_yaxis()
	plt.xlabel(XLAB,size=axlabel_size)		
	plt.xticks(xp,xaxp,size=tick_size,rotation=rot)
	if len(ppos)>0:
		for jj in ppos:
			plt.gca().get_xticklabels()[jj].set_color("red")		
	plt.yticks(size=tick_size)
	plt.tick_params(direction='out', length=6, width=2)
	plt.title(stx[inx],size=title_size)
	plt.tight_layout()
	plt.savefig(filnames[inx])
	plt.savefig(filnames[inx].replace('.png','.pdf'))
	plt.close()
	inx+=1
