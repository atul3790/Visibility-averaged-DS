'''
#### Prepares MS and makes input files which will be used by DS_VI_fromRLvisib_MPI.py ####################

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Run in python 3.6 after bridging with CASA 6.4
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Pre-requisite:
1. A model for the background sky field with star location masked/zeroed out
2. A calibrated Measurement Set (MS) with data column as the corrected data. (Just split out the corrected data column from your final calibrated MS)

This code does the following initial steps: 
1. Plants the background sky model in the model column 
2. Does uvsub so that corrected data column has Corrected data - model.
3. Splits out the new corrected data column of the target field observations.

The code then further splits out the target field data for each scan across various polarisations. 
The code can further splits the data for each scan into finer time windows as per user's requirement.
The MS visibility data in each of these finer time windows and across polarisations will be extracted and saved in pickle file format readable by DS_VI_fromRLvisib_MPI.py.
###########################################################################################################
'''

import os,numpy as np,glob
import casatasks,casatools
from casatasks import split,uvsub,ft
from casatools import ms,msmetadata
from datetime import timedelta as Dt
from time import time
from datetime import datetime as dt
import pickle
'''
###################### INPUT ##############################################################################
'''
basedir='/Data/' # Directory with the calibrated MS data.
Ms='target_avg.ms' # final selfcal MS file with just the target field.
GMRT_pip_model=['../GMRT_pipeline_result/GJ486-selfcalimg8.model.tt0', '../GMRT_pipeline_result/GJ486-selfcalimg8.model.tt1'] # List of model components (.tt0, .tt1)
Nterms=2 # Number of Taylor terms (.tt0, .tt1) in background sky model
split_dir='SplitMSs/' # Folder where scan-split MS pickle files has to be saved. Code will make this folder if it does not exist already. 
spw=0 # SPW containing the data
Analysis_mode=1 # Integer choices:
		# 1 : Plant background sky model in madel data column, do uvsub, then split MS based on scans/sub-scans
		# 2 : Just split MS based on scans /sub-scans (Assumes that uvsub is already done on MS or is not needed.)
overwrite=False # Do you want to overwrite any existing pickle files resulted from previous code runs?
		# If False, code will not redo analysis, if results already exist.
Nsplits=1 # Number of times each target scan has to be further splitted. Egs. 0 means no further split. 1 will split each scan into 2 time windows further i.e scan start - scan duration/2 & scan duration/2 - scan end. 
Epoch='1858/11/16 23:59:57.274466' # Give in format YYYY/MM/DD hh:mm:ss.s --> seconds in decimal upto microsecond precision. This is the start epoch of the clock for the observatory. This will be used to convert scan time in CASA MS to real UT time. For GMRT this start Epoch is '1858/11/16 23:59:57.274466'
'''
############# END of INPUT Section ########################################################################
##### DO NOT EDIT ANY portions further down in the code unless you are a developer ##########################
'''
hme=os.getcwd()
os.chdir(basedir)
MS=casatools.ms()
msmd=casatools.msmetadata()
Epoch=dt.strptime(Epoch,"%Y/%m/%d %H:%M:%S.%f")

os.system('mkdir '+split_dir)
if Analysis_mode==1:
	## Planting model
	print('Planting GMRT pipeline model')
	ft(vis=Ms,model=GMRT_pip_model,nterms=Nterms)

	## UVsub
	print('Doing uvsub with the pipeline model planted')
	uvsub(vis=Ms)

nMS='target_avg_uvsub.ms'
if not os.path.isdir(nMS):
	## Split out the corrected MS
	print('Splitting out the uvsub data column. Making new MS: ',nMS)
	split(vis=Ms, outputvis='target_avg_uvsub.ms',field='0',width=1,timebin='0s')

msmd.open(nMS)
scans=msmd.scannumbers()
freqs=msmd.chanfreqs(spw)/1e6 #freq is in MHz
msmd.close()
print('Scans and frequencies extracted!!')

cors=['RR','LL','RL','LR']

for scn in scans:
	print('Analysing scan: ',scn)
	
	for cor in cors:		
		exp_filnames=['target_scan'+str(scn)+'_t'+str(i)+'.p' for i in np.arange(Nsplits+1)+1]	
		loc=split_dir+cor+'/'
		locp=loc+'MSpfils_V1/'		
		os.system('mkdir '+split_dir+cor)
		os.system('mkdir '+locp)
		ext_fs=glob.glob(locp+'*_scan'+str(scn)+'*.p')
		ext_fs=[i.split('/')[-1] for i in ext_fs]
		if ext_fs==exp_filnames and overwrite==False:
			print('All of ',exp_filnames,' exist in ',locp,'. So continuing with next correlation!!')
			continue

		print('Analysing correlation: ',cor)
		if not os.path.isdir(loc+'target_scan'+str(scn)+'_t0.ms'):
			split(vis=nMS,outputvis=loc+'target_scan'+str(scn)+'_t0.ms',scan=str(scn), correlation=cor,field='0', datacolumn='data')
			print(loc+'target_scan'+str(scn)+'_t0.ms made!!')

		else:
			print(loc+'target_scan'+str(scn)+'_t0.ms already exists!!')

		sMS=loc+'target_scan'+str(scn)+'_t0.ms'

		ldt=time()
		MS.open(sMS)
		data=MS.getdata('data',ifraxis=True)['data']
		print('loaded data...',sMS)
		flag=MS.getdata('flag',ifraxis=True)['flag']
		MS.close()
		print('Data and flags loaded in ',(time()-ldt)/60.,' min. MS: ',sMS)
		
		msmd.open(sMS)
		Times=msmd.timesforscans(scn)
		print('Time details of the scan extracted!!')
		msmd.close()
		loc=0
		dX=int(np.round(len(Times)/(Nsplits+1)))

		for i in np.arange(Nsplits+1)+1:
			pname=exp_filnames[i-1]
			if os.path.isfile(locp+pname) and overwrite==False:
				print(locp+pname+' exits. So continueing to next iteration!')
				continue				
			l1=loc
			if i==Nsplits+1:
				l2=len(Times)
			else:
				l2=loc+dX
			times=Times[l1:l2]					
			begt=Epoch+Dt(seconds=times[0])
			endt=Epoch+Dt(seconds=times[-1])
			times-=times[0]

			print('Full dataset is split in time to get scan'+str(scn)+'_t'+str(i)+' data.')
			pickle.dump({'MS_data':data[:,:,:,l1:l2],'Flag':flag[:,:,:,l1:l2], 'Time':{'Range':times, 'Begin':begt, 'End':endt}, 'Frequency': freqs},open(locp+pname,'wb'))
			loc+=dX
os.chdir(hme)
