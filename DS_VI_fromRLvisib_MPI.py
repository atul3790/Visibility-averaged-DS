'''
## Visibility averaged DS maker ########################################################################
Purpose: Make Stokes V, I, RR and LL DS from RR LL CASA Measurement sets (MS)

___________________________________________________________________
## IMPORTANT: Run MS_prep_forVisavgDS.py before running this code. 
--------------------------------------------------------------------

Code takes in splitted MS file data made by MS_prep_forVisavgDS.py. It generated a visibility averaged dynamic spectrum for each scan per polarisation (RR and LL) and also generate STOKES V, I and V/I dynamic spectra. The outputs will be saved in folders user specify.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Run in python 3.6 after bridging with CASA 6.4
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Just fill in the INPUT sections and set the code to run.

#############################################################################################################
'''

from multiprocessing import Pool
import numpy as np,os,glob
import matplotlib.pyplot as plt
from datetime import timedelta as Dt
from datetime import datetime as dt
import time,pickle
from matplotlib import colors
import numpy.ma as ma
import casatasks,casatools
from casatools import ms,msmetadata
from copy import deepcopy as cpy

'''
######### INPUT #############################################################################################
'''
R_MSdir='/Data/SplitMS/RR/MSpfils_V1/' # Directory containing STOKES RR MS pickle file data
L_MSdir='/Data/SplitMS/LL/MSpfils_V1/' # Directory containing STOKES LL MS pickle file data
DS_dir='/Data/SplitMS/' # RR and LL DS will be saved here. Separate folders will be made 
VDS_dir='/Data/SplitMS/VDS_RLvisAvg/' # Directory to save STOKES V DS. Code will make VDS_dir, IDS_dir and polDS_dir if they dont already exist.
IDS_dir='/Data/SplitMS/IDS_RLvisAvg/' # Directory to save STOKES I DS.
polDS_dir='/Data/SplitMS/polDS_RLvisAvg/' # Directory to save V/I (circ. polarisation) DS.
need_GMRTcorr=True #Correct for GMRT wrong tagging of RR and LL. Set to True if data is from GMRT. 
Nproc=15 # Number of processor cores to use
Nice=1 # CPU nice value
whichDS=1 # Either 1 or 2. 
	  # 1. |Mean(visibility)| [Absolute value is taken after visibility averaging/compression in baseline axis] 
	  # 2. Real(Mean(visibility)) [Real part of baseline averaged visibility is taken.] 
MStag='target*' #[Tag name of MS file names to seach for. Output from MS_prep_forVisavgDS.py will ensure that the names will start by 'target*.p'. But amend this if you have modified this.]
Epoch='1858/11/16 23:59:57.274466' # Give in format YYYY/MM/DD hh:mm:ss.s --> seconds in decimal upto microsecond precision. This is the observatory epoch to convert CASA MS time axis to UTC time. For GMRT this is '1858/11/16 23:59:57.274466'.
tunit='s' # Data time unit
frequnit='MHz' # Data frequency unit
del_t=10 # Averaging in time needed for Dynamic spectrum. Give in timesteps. Eg. 10 - averages 10 time steps of data
delf=50 # Averaging in channels needed. Give in channel width unit. 'full': for full band averaged lightcurve

'''
## INPUTS to worry about if you need error Dynamic spectra #################
'''
err_std=False # Do you want errors estimated based on standard deviation of the complex visibility data? If False, errors will be estimated just based on imaginary part of visibility data. This is based on the idea that imaginary part should ideally be 0 for a phase centre source.
'''
## INPUTS for flagging MS data further ##############################################################
## This section helps to clip out RFI or bad data regions with spurious high flux levels ############
'''
Vsign=-1 # -1 if expected STOKES V <0 for all flares. 
	 # 0 if STOKES V can be either +ve or -ve depening on flare events. 
	 # 1 if expected STOKES V>0 for all flares.
V_tol=0.0005 # Absolute value of the tolerable error flux limit, based on expected STOKES V flux in Jy. 
	     # Any flux value in DS above this will be flagged in the following manner:
	     # If Vsign == 1 => +ve STOKES V flares are expected. So stokes V flux < -V_tol are flagged across data
	     # If Vsign ==-1 => -ve STOKES V flares are expected. So stokes V flux > V_tol are flagged across data
	     # If Vsign == 0 => no preference in STOKES V. So stokes V flux < |V_tol| are flagged across data		
MaxVf='' # Maximum absolute flux expected in Jy from the source beyond which should be flagged in the original abs(visibility data).
	# this will be done on complex visibility data across baselines, before making the required averaging in time and spectral axes.
	# Condition used to flag: |RR(baseline,freq,time) - LL(baseline,freq,time)|/2<|MaxVf|
	#Leave as '' if no constraints are to be applied
MaxIf='' # Maximum absolute flux expected in Jy from the source beyond which should be flagged in the original abs(visibility data). Same as for MaxVf.
	# Condition used to flag: |RR(baseline,freq,time) + LL(baseline,freq,time)|/2<|MaxIf|
	#Leave as '' if no constraints are to be applied
'''
## Input for Masking scheme based on polarised Flare flux expectated ##################################################################
## This section will work only if Msk_wrtI==True ######################################################################################
### Idea is to further crop the non-flare high flux spurious data based on expected stellar flare levels & polarisation levels ########
'''
Msk_wrtI=False # True or False. Do you want to apply mask to DS based on criteria that strong flares are atleast > minPol % polarised
Stflare=0.04 # Expected minimum STOKES I flux of the strong flare in Jy. Used only is Msk_wrtI == True
minPol=4 #in % . sign of polarisation will be autoset as per Vsign. So DON'T provide signed %.  
fracmode=True #True or False. 
	      #If True masking will be done to freq-time regions of all STOKES DSs and circ. pol DS which satisfy criteria: I flux>Stflare & I flux/(Circ. Pol) > Stflare/minPol So that flares are atleast polarised to the expected ratio
	      #If False masking will be done as per: 	I flux>Stflare & Circ. poln > minpol
'''
###### INPUTS for Dynamic spectrum (DS) plotting section ########################
'''
Flx_u='mJy' # Unit of flux in the plot. eg: mJy, Jy
flxscl=1000	# Flux scaling required to convert Jy units to unit of choice in the plot.
plt_tunit='min' # Plot time unit required
plt_tconv=1/60. # Conversion from data time units to plot time
data_scl='' # Data scaling to apply. Choices: 'log','sqrt','pow_N' - where N is the power to which data is to be raised to. Leave as '' if no special normalisation is needed
cmap='nipy_spectral_r' # Color map to use in the DS plot.

'''
########################### END of INPUT SECTION ############################################################
##### DO NOT EDIT ANY portions further down in the code unless you are a developer ##########################
'''

#save_unavgdata=False # Save MS data array as pickle file for future ease of use.
#unavgdata_fold='MS_data_pfils/' #Name of the folder to which unavg MSdata is to be stored inside the R_MSdir and L_MSdir

collab='Flux density' # Flux axis label without units
collab=collab+' ('+Flx_u+')'

MS=casatools.ms()
msmd=casatools.msmetadata()
if need_GMRTcorr==True:
	gmrt=1
else:
	gmrt=-1

if whichDS==1:
	adg=''
	fun=ma.abs
else:
	adg='Real'
	fun=np.real

os.system('mkdir '+DS_dir)
os.system('mkdir '+DS_dir+'RRDS_visAvg/')
os.system('mkdir '+DS_dir+'LLDS_visAvg/')


def namer(tags,vs):
	name=''
	i=0
	for tg in tags:
		if vs[i]!='':
			name+=tg+str(vs[i])
		i+=1
	return name

if Msk_wrtI==True:
	if fracmode==False:
		DSfold=adg+'DS'+namer(['_Vtol','_dt','_df','_MxI','_MxV'],[str(np.round(V_tol*flxscl,2))+Flx_u,del_t,delf,MaxIf,MaxVf])+'_minPol'+str(minPol)+'_minFlr'+str(Stflare)+'/'
	else:
		DSfold=adg+'DS'+namer(['_Vtol','_dt','_df','_MxI','_MxV'],[str(np.round(V_tol*flxscl,2))+Flx_u,del_t,delf,MaxIf,MaxVf])+'_minPol'+str(minPol)+'_minFlr'+str(Stflare)+'_I-by-PolMask/'

else:
	DSfold=adg+'DS'+namer(['_Vtol','_dt','_df','_MxI','_MxV'],[str(np.round(V_tol*flxscl,2))+Flx_u,del_t,delf,MaxIf,MaxVf])+'/'

#if save_unavgdata==True:
#	RMSdloc=R_MSdir+unavgdata_fold
#	LMSdloc=L_MSdir+unavgdata_fold
#	os.system('mkdir '+RMSdloc)
#	os.system('mkdir '+LMSdloc)

os.system('mkdir '+VDS_dir)
os.system('mkdir '+IDS_dir)
os.system('mkdir '+polDS_dir)

VDS_dir+=DSfold
IDS_dir+=DSfold
RDS_dir=DS_dir+'RRDS_visAvg/'+DSfold
LDS_dir=DS_dir+'LLDS_visAvg/'+DSfold
polDS_dir=polDS_dir+DSfold

os.system('mkdir '+VDS_dir)
os.system('mkdir '+IDS_dir)
os.system('mkdir '+RDS_dir)
os.system('mkdir '+LDS_dir)
os.system('mkdir '+polDS_dir)

os.system('mkdir '+RDS_dir+'DSpfils/')
os.system('mkdir '+RDS_dir+'DSpngs/')
os.system('mkdir '+RDS_dir+'DSpngs/ErrDSs/')

os.system('mkdir '+LDS_dir+'DSpfils/')
os.system('mkdir '+LDS_dir+'DSpngs/')
os.system('mkdir '+LDS_dir+'DSpngs/ErrDSs/')

os.system('mkdir '+VDS_dir+'DSpfils/')
os.system('mkdir '+VDS_dir+'DSpngs/')
os.system('mkdir '+VDS_dir+'DSpngs/ErrDSs/')

os.system('mkdir '+IDS_dir+'DSpfils/')
os.system('mkdir '+IDS_dir+'DSpngs/')
os.system('mkdir '+IDS_dir+'DSpngs/ErrDSs/')

os.system('mkdir '+polDS_dir+'DSpfils/')
os.system('mkdir '+polDS_dir+'DSpngs/')
os.system('mkdir '+polDS_dir+'DSpngs/ErrDSs/')

Epoch=dt.strptime(Epoch,"%Y/%m/%d %H:%M:%S.%f")

RMSs=sorted(glob.glob(R_MSdir+MStag+'*.p')) 
LMSs=[glob.glob(L_MSdir+MStag+'_scan'+i.split('/')[-1].split('_scan')[1])[0] for i in RMSs]

plt.ioff()
def Analyse(i):
	global MS
	global msmd
	RMS=RMSs[i]
	LMS=LMSs[i] 
	scn=RMS.split('_scan')[1]
	scan=int(scn.split('_')[0])
	tn=RMS.split('_t')[1].split('.')[0]

	print('Analysing: \n',RMS.split('/')[-1],', ',LMS.split('/')[-1])
	bt=time.time()
	
	print('Loading RR MS data...')

	data=pickle.load(open(RMS,'rb'))
	Rdata=data['MS_data']
	Rflag=data['Flag']
	times=data['Time']['Range']
	begt=data['Time']['Begin']
	endt=data['Time']['End']
	
	freqs=data['Frequency']
	
	del data

	data=pickle.load(open(LMS,'rb'))
	Ldata=data['MS_data']
	Lflag=data['Flag']
	del data

	Flag=Rflag+Lflag #Regions of data to be flagged

#	if save_unavgdata==True:
#		pickle.dump({'MS_data':Rdata,'Flag':Rflag, 'Time':{'Range':times, 'Begin':begt, 'End':endt}, 'Frequency': freqs},open(RMSdloc+RMS.split('/')[-1].replace('.ms','.p'),'wb'))
#		pickle.dump({'MS_data':Ldata,'Flag':Lflag, 'Time':{'Range':times, 'Begin':begt, 'End':endt}, 'Frequency':freqs},open(LMSdloc+LMS.split('/')[-1].replace('.ms','.p'),'wb'))

		
	Rdata=ma.array(Rdata,mask=Flag)
	Ldata=ma.array(Ldata,mask=Flag)

	print('Flagged DS made in ',np.round((time.time()-bt)/60,2),' min..!',' MS: ',RMS.split('/')[-1])

	del Flag
	del Rflag
	del Lflag

	#amp=np.abs(np.nanmean(data,axis=2)) #average all baselines

	pname=adg+'DS'+'_scan'+scn.replace('.ms','.p')
	
	Vdata=0.5*(Ldata-Rdata)*gmrt
	print('STOKES V data made!!, MS: ',RMS.split('/')[-1])
	Idata=0.5*(Rdata+Ldata)
	print('STOKES I data made!!, MS: ',RMS.split('/')[-1])
#### Flagging bad visibilities based on Max flux limit in the STOKES V and I plane (MaxVf,MaxIf).
	if MaxIf!='':
		Idm=ma.masked_greater_equal(ma.abs(Idata),MaxIf)
		Rdata.mask=Rdata.mask+Idm.mask
		Ldata.mask=Ldata.mask+Idm.mask
		Idata.mask=Idata.mask+Idm.mask
		print('Mask made for I data based on abs flux limit, ',MaxIf,' Jy')
		del Idm
	if MaxVf!='':
		Vdm=ma.abs(Vdata)
		Vdm=ma.masked_greater_equal(Vdm,MaxVf)
		Vdata.mask=Vdm.mask
		print('Mask made for V data based on abs flux limit, ',MaxVf,' Jy')
		del Vdm

#Applying Vdata mask to all other DSs ##########################
	Rdata.mask=Rdata.mask+Vdata.mask
	Ldata.mask=Ldata.mask+Vdata.mask
	Idata.mask=Idata.mask+Vdata.mask
###########################################################
	Rdata=ma.masked_invalid(Rdata)
	Ldata=ma.masked_invalid(Ldata)
	Vdata=ma.masked_invalid(Vdata)
	Idata=ma.masked_invalid(Idata)

	RDS=ma.mean(Rdata,axis=2)[0]
	LDS=ma.mean(Ldata,axis=2)[0]
	VDS=ma.mean(Vdata,axis=2)[0]
	IDS=ma.mean(Idata,axis=2)[0]
	if err_std==True:
		eRDS_th =ma.mean(Rdata**2,axis=2)[0]
		eLDS_th =ma.mean(Ldata**2,axis=2)[0]
		eVDS_th =ma.mean(Vdata**2,axis=2)[0]
		eIDS_th =ma.mean(Idata**2,axis=2)[0]
	else:
		eRDS_th=ma.mean(np.angle(Rdata)**2,axis=2)[0]
		eLDS_th=ma.mean(np.angle(Ldata)**2,axis=2)[0]
		eIDS_th=ma.mean(np.angle(Idata)**2,axis=2)[0]
		eVDS_th=ma.mean(np.angle(Vdata)**2,axis=2)[0]

	del Rdata
	del Ldata
	del Vdata
	del Idata


	badflg=0
	if delf=='full':
		df=len(freqs)
		badflg=1
	else:
		df=delf
	if del_t!=0 or df!=0:
		DSm={0:RDS,1:LDS,2:VDS,3:IDS}
		eDSm=[eRDS_th,eLDS_th,eVDS_th,eIDS_th]
		## Making new time range array ##########################
		trs=np.int(np.ceil(len(times)/del_t))
		frs=np.int(np.ceil(len(freqs)/df))
		timn=np.zeros(trs)
		frqn=np.zeros(frs)

		for ti in range(trs):
			if ti<trs-1:
				timn[ti]=np.mean(times[ti*del_t:(ti+1)*del_t])
			else:
				timn[ti]=np.mean(times[ti*del_t:])
		Begt=begt+Dt(seconds=timn[0])
		endt=begt+Dt(seconds=timn[-1])
		begt=Begt
		times=timn-timn[0]
		for ti in range(frs):
			if ti<frs-1:
				frqn[ti]=np.mean(freqs[ti*df:(ti+1)*df])
			else:
				frqn[ti]=np.mean(freqs[ti*df:])
		freqs=cpy(frqn)
		del frqn
		#########################################################
		stk={0:'RR',1:'LL',2:'V',3:'I'}	
		for inx in range(4):
			Rw=np.int(np.ceil(DSm[inx].shape[0]/df))
			Cl=np.int(np.ceil(DSm[inx].shape[1]/del_t))
			DS_n=np.zeros((Rw,Cl))+np.nan*(1+1j)
			eDS_n=np.zeros((Rw,Cl))+np.nan*(1+1j)

			for M in range(Rw):
				sr=M*df
				if M==Rw-1:
					lr=DSm[inx].shape[0]-1
				else:
					lr=sr+df
				if sr==lr:
					lr+=1

				for N in range(Cl):
					sc=N*del_t
					if N==Cl-1:
						lc=DSm[inx].shape[1]-1
					else:
						lc=sc+del_t
					if sc==lc:
						lc+=1
						
					DS_n[M,N]=ma.mean(DSm[inx][sr:lr,sc:lc])
					eDS_n[M,N]=ma.mean(eDSm[inx][sr:lr,sc:lc])
					
			if inx!=2:
				DS_n=ma.masked_less_equal(DS_n,0)				
			else:
	### Amending V DS to cut out any bad error points since we know V fluxes should be either +ve or -ve. V_tol is the RMS error allowed 
				if Vsign==-1:
					DS_n=ma.masked_greater_equal(DS_n,V_tol)

				elif Vsign==1:
					DS_n=ma.masked_less_equal(DS_n,-V_tol)
				else:
					DS_n=ma.masked_less_equal(DS_n,-V_tol)
					DS_n=ma.masked_greater_equal(DS_n,V_tol)

				print('V DS modified')

			eDS_n=ma.array(eDS_n,mask=DS_n.mask)
			eDS_sys=np.abs(np.imag(DS_n))
			eDS_sys=ma.array(eDS_sys,mask=DS_n.mask)

			if err_std==True:
				eDS_n=fun(eDS_n-DS_n**2)
				print('Stokes ',stk[inx],', <V^2> - <V>^2: ',eDS_n)
				eDS_n=np.sqrt(eDS_n)
			else:
				eDS_n=np.sin(np.sqrt(np.abs(eDS_n)-np.angle(DS_n)**2))*fun(DS_n) #np.abs(eDS_th) cuts out the +0j imaginary term arising due to the nan+nan*j initialisation of the arrays. 
				
			if inx==0:
				RDS=fun(DS_n)
				eRDS_th=ma.abs(eDS_n)
				eRDS_sys =eDS_sys
				print('Shapes: RDS -',RDS.shape,' err random RDS: ',eRDS_th.shape,' err Sys RDS: ',eRDS_sys.shape)
				print('Made averaged RR DS and error DS')
			if inx==1:
				LDS=fun(DS_n)
				eLDS_th=ma.abs(eDS_n)
				eLDS_sys =eDS_sys
				print('Shapes: LDS -',LDS.shape,' err random DS: ',eLDS_th.shape,' err Sys DS: ',eLDS_sys.shape)
				print('Made averaged LL DS and error DS')
			if inx==2:
				VDS=fun(DS_n)
				if Vsign==-1 and whichDS==1:
					VDS=-VDS
				VDS=ma.masked_equal(VDS,0) # Exact 0 value is imposssible
				eVDS_sys =eDS_sys
				eVDS_th=ma.abs(eDS_n)
				eVDS_th.mask=eVDS_th.mask+VDS.mask
				eVDS_sys.mask=eVDS_sys.mask+VDS.mask
				print('Shapes: VDS -',VDS.shape,' err random DS: ',eVDS_th.shape,' err Sys DS: ',eVDS_sys.shape)
				print('Made averaged V DS and error DS')
			if inx==3:
				IDS=fun(DS_n)
				eIDS_th=np.abs(eDS_n)
				eIDS_sys =eDS_sys
				print('Shapes: IDS -',IDS.shape,' err random DS: ',eIDS_th.shape,' err Sys DS: ',eIDS_sys.shape)
				print('Made averaged I DS and error DS')

			del DS_n
			del eDS_n
			del eDS_sys
		del DSm
		del eDSm
### If no averaging in time spectral space is to be done
	else:

		eRDS_sys =np.abs(np.imag(RDS))
		eLDS_sys =np.abs(np.imag(LDS))
		eVDS_sys =np.abs(np.imag(VDS))
		eIDS_sys =np.abs(np.imag(IDS))

		if err_std==True:
			eRDS_th=fun(np.sqrt(eRDS_th-RDS**2))
			eLDS_th=fun(np.sqrt(eLDS_th-LDS**2))
			eIDS_th=fun(np.sqrt(eIDS_th-IDS**2))
			eVDS_th=fun(np.sqrt(eVDS_th-VDS**2))
		else:
			eRDS_th=np.sin(np.sqrt(np.abs(eRDS_th)-np.angle(RDS)**2))*fun(RDS) # np.abs(eDS_th) cuts out the +0j imaginary term arising due to the nan+nan*j initialisation of the arrays. 
			eLDS_th=np.sin(np.sqrt(np.abs(eLDS_th)-np.angle(LDS)**2))*fun(LDS)
			eIDS_th=np.sin(np.sqrt(np.abs(eIDS_th)-np.angle(IDS)**2))*fun(IDS)
			eVDS_th=np.sin(np.sqrt(np.abs(eVDS_th)-np.angle(VDS)**2))*fun(VDS)

		RDS=fun(RDS)
		LDS=fun(LDS)
		if whichDS==1 and Vsign==-1:
			VDS=Vsign*fun(VDS)
		else:
			VDS=fun(VDS)
		IDS=fun(IDS)

		print('All DSs and the random and systematic error DSs made')
		print('DS shapes, DS:',RDS.shape,', Random err DS:',eRDS_th.shape,', Sys error DS:', eRDS_sys.shape) 
## Making circular polarisation DS
	polDS=VDS/IDS*100
## Masking based on circular polarisation values

	if Vsign==-1:
		polDS=ma.masked_greater_equal(polDS,0)
		polDS=ma.masked_less(polDS,-100)
		print('Data with +ve polarisation and absolute polarisation >100% are masked in frequency-time plane in all DSs.')

	elif Vsign==1:
		polDS=ma.masked_less_equal(polDS,0)
		polDS=ma.masked_greater(polDS,100)
		print('Data with -ve and >100% polarisation are masked in frequency-time plane in all DSs.')
	else:
		polDS=ma.masked_less(polDS,-100)
		polDS=ma.masked_greater(polDS,100)
		print('Data with polarisation >100% in mudulus are masked in frequency-time plane in all DSs.')

	RDS.mask=RDS.mask+polDS.mask
	LDS.mask=LDS.mask+polDS.mask
	IDS.mask=IDS.mask+polDS.mask
	#VDS.mask=VDS.mask+polDS.mask # Pol mask isnt important STOKES V since this is less affected by systematic noise. Poor polarisation is usually a result of bad STOKES I estimate. Already MaxVf should well constrain STOKES V

## Masking based on the knowledge that strong flares need to be polarised. Conditions are supplied by the user
	if Msk_wrtI==True:
		if fracmode==False:
			bReg=np.where((IDS>Stflare) & (ma.abs(polDS)<minPol))
		else:
			bReg=np.where((IDS>Stflare) & (IDS/ma.abs(polDS)>Stflare/minPol))
		RDS.mask[bReg]=True
		LDS.mask[bReg]=True
		IDS.mask[bReg]=True
		polDS.mask[bReg]=True

		#VDS.mask[bReg]=True	# Pol mask isnt important STOKES V since this is less affected by systematic noise. Poor polarisation is usually a result of bad STOKES I estimate. Already MaxVf should well constrain STOKES V
	ePolDS=np.sqrt(((eVDS_th+eVDS_sys)/VDS)**2+((eIDS_th+eIDS_sys)/IDS)**2)*polDS
	ePolDS.mask=polDS.mask
	print('Pol error DS made')

	if err_std==True:
		eVDS_th.mask=eVDS_th.mask+VDS.mask
		eVDS_sys.mask=eVDS_sys.mask+VDS.mask
		eIDS_th.mask=eIDS_th.mask+IDS.mask
		eIDS_sys.mask=eIDS_sys.mask+IDS.mask
		eRDS_th.mask=eRDS_th.mask+RDS.mask
		eRDS_sys.mask=eRDS_sys.mask+RDS.mask
		eLDS_th.mask=eLDS_th.mask+LDS.mask
		eLDS_sys.mask=eLDS_sys.mask+LDS.mask

#########################################
## Saving final DS pickle files.

	pickle.dump({'Time':{'Begin':begt,'End':endt,'Range':times,'Unit':tunit},'nu':{'Range':freqs,'Unit':frequnit},'DS':polDS,'eDS':ePolDS}, open(polDS_dir+'DSpfils/CircPol'+pname,'wb'))
	print('Polarisation DS pickle file saved!')

	pickle.dump({'Time':{'Begin':begt,'End':endt,'Range':times,'Unit':tunit},'nu':{'Range':freqs,'Unit':frequnit},'DS':RDS, 'eDS_th':eRDS_th, 'eDS_sys':eRDS_sys}, open(RDS_dir+'DSpfils/R'+pname,'wb'))
	print('RR DS pickle file saved!')

	pickle.dump({'Time':{'Begin':begt,'End':endt,'Range':times,'Unit':tunit},'nu':{'Range':freqs,'Unit':frequnit},'DS':LDS,'eDS_th':eLDS_th, 'eDS_sys':eLDS_sys}, open(LDS_dir+'DSpfils/L'+pname,'wb'))
	print('LL DS pickle file saved!')
	
	pickle.dump({'Time':{'Begin':begt,'End':endt,'Range':times,'Unit':tunit},'nu':{'Range':freqs,'Unit':frequnit},'DS':VDS,'eDS_th':eVDS_th, 'eDS_sys':eVDS_sys}, open(VDS_dir+'DSpfils/V'+pname,'wb'))
	print('V DS pickle file saved!')

	pickle.dump({'Time':{'Begin':begt,'End':endt,'Range':times,'Unit':tunit},'nu':{'Range':freqs,'Unit':frequnit},'DS':IDS,'eDS_th':eIDS_th, 'eDS_sys':eIDS_sys}, open(IDS_dir+'DSpfils/I'+pname,'wb'))
	print('I DS pickle file saved!')
#######################################

## Plotting ##############################################

## STOKES DSs
	stokes=['R','L','I','V']
	match={0:[RDS,RDS_dir,eRDS_th,eRDS_sys],1:[LDS,LDS_dir,eLDS_th,eLDS_sys], 2:[IDS,IDS_dir,eIDS_th,eIDS_sys], 3:[VDS,VDS_dir,eVDS_th,eVDS_sys]}
	Ad={0:'',2:'eTh',3:'eSys',4:'eTot'}
	del RDS
	del LDS
	del VDS
	del IDS
	del eRDS_th
	del eLDS_th
	del eVDS_th
	del eIDS_th
	del eRDS_sys
	del eLDS_sys
	del eVDS_sys
	del eIDS_sys

	for j in range(4):
		poln=stokes[j]
		#print('Unmasked DS data count: ',ma.count(DS))
		print('Plotting DS for poln: ',poln) 
		DSjpg_loc=match[j][1]
		epos=''
		for P in [0,2,3,4]:
			if P==2:
				epos='ErrDSs/'
			if P<4:
				DS=match[j][P]*flxscl
			else:
				DS=(match[j][2]+match[j][3])*flxscl
#			if ma.count(DS)==0:
#				badflg=1
#				print('Data fully masked for the DS in poln: ',poln)
#				continue

			plt.figure(figsize=(10,8),dpi=100)
			if DS.shape[0]>1:
				if data_scl=='' or poln=='V':
					plt.imshow(DS,origin='lower',extent= (times[0]*plt_tconv,times[-1]*plt_tconv,freqs[0],freqs[-1]), aspect='auto',cmap=cmap)
				else:
					if data_scl=='log':
						Norm=colors.LogNorm(vmin=np.nanmin(DS),vmax=np.nanmax(DS))
					elif 'pow_' in data_scl:
						Norm=colors.PowerNorm(gamma=float(data_scl.split('pow_')[1]))
					plt.imshow(DS,origin='lower',extent= (times[0]*plt_tconv,times[-1]*plt_tconv,freqs[0],freqs[-1]), aspect='auto',cmap=cmap,norm=Norm)
	
				plt.ylabel('Frequency ('+frequnit+')',size=20)
				h=plt.colorbar()
				h.set_label(collab,size=20)
				h.ax.tick_params(labelsize=17)
	##### Plotting if DS is a light curve
			else:
				plt.plot(times*plt_tconv,DS[0,:],'o-')
				plt.ylabel(collab,size=20)
	######## Axes definitions ######################
			plt.xlabel('Time ('+plt_tunit+') +'+str(begt).split('.')[0],size=20)
			plt.title('Poln: '+poln+'  Scan: '+str(scan)+' t'+tn,size=21)
			plt.xticks(size=16)
			plt.yticks(size=16)
			plt.tight_layout()
			plt.savefig(DSjpg_loc+'DSpngs/'+epos+Ad[P]+pname.replace('.p','.png'))
			plt.close()
			print(DSjpg_loc+'DSpngs/'+epos+Ad[P]+pname.replace('.p','.png'),' written out')
			del DS
	del match

####### Plotting Circular polarisation DS
	Ad=['','err']
	epos=['','ErrDSs/']
	PDS=[polDS,ePolDS]
	del polDS
	del ePolDS

	for P in range(2):
		plt.figure(figsize=(10,8),dpi=100)
		polDS=PDS[P]		
		if badflg==0:
			plt.imshow(polDS,origin='lower',extent=(times[0]*plt_tconv,times[-1]*plt_tconv, freqs[0],freqs[-1]),aspect='auto',cmap=cmap)
			plt.ylabel('Frequency ('+frequnit+')',size=20)
			h=plt.colorbar()
			h.set_label('Percentage (%)',size=20)
			h.ax.tick_params(labelsize=17)
		else:
			plt.plot(times*plt_tconv,polDS[0,:],'o-')
			plt.ylabel('Percentage (%)',size=20)

		plt.xlabel('Time ('+plt_tunit+') +'+str(begt).split('.')[0],size=20)
		plt.title('Circ. Polarisation   Scan: '+str(scan)+' t'+tn,size=21)
		plt.xticks(size=16)
		plt.yticks(size=16)
		plt.tight_layout()
		plt.savefig(polDS_dir+'DSpngs/'+epos[P]+Ad[P]+pname.replace('.p','.png'))
		plt.close()
		print('Analysis completed in ',np.round((time.time()-bt)/60,2),' min.')

if __name__=='__main__':
	p=Pool(Nproc)
	os.nice(Nice)
	Tstt=time.time()
	p.map(Analyse,np.arange(len(RMSs)))
	print('Analysis done by ',os.uname().nodename,'\nTime taken: ',np.round((time.time()-Tstt)/60,2),' min.')
	os.system('rm -rf *.log')
