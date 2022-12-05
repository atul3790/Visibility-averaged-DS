# Visibility-averaged-DS
This is a package which takes in calibrated measurement set and a background sky model, and generate visibility averaged dynamic spectrum for the central source after subtracting the visibilities of the background sky model. The package is primarily intended for long-duration multi-scan monitoring observations stellar/other variable radio sources.
The package uses python 3.6 and calls CASA tools/tasks via python. This bridge works between python 3.6 and CASA 6 versions (See, https://casadocs.readthedocs.io/en/v6.2.1/notebooks/usingcasa.html#Modular-Installation-of-CASA-6)
The package has three codes: 

1. **MS_prep_forVisavgDS.py**

   A prep code which prepares the CASA measurement set (MS) data to be passed to the dynamic spectrum making code. It splits the MS data into multiple target source scans as a function of polarisations. Data in each scan per polarisation will be divided into sub scan time periods as per user's needs to produce low volume sub data sets, easier to analyse parallely. The code can be used to also average the MS in time and frequency prior to splitiing. The splitted out MS data will be saved as numpy arrays in pickle file format.

2. **DS_VI_fromRLvisib_MPI.py**

   The dynamic spectrum (DS) making code. It uses parallel processing tools in python to operate on the splitted out sub-MS data pickle files- each file per core for each polarisation. It writes out visibility averaged dynamic spectra (pickled numpy ndarrays) in ***STOKES RR, LL, V, I and circular polarisation (V/I)*** for each sub data set. The cade is also equipped with simple flagging options based on clipping fluxes beyond some level in different polarisations (V & I). The code also masks bad regions of data based on expected stellar polarisation levels and sense of polarisation. The code can be used to average data in frequency and time as per need, which will help to also obtain band averaged light curve for the source.

   ***Note:*** The error analysis section of the code is idea wise in beta phase. The 2 ways it tries to infer error in DS fluxes are:
   - *Standard deviation in complex visibility data*

      |<V^2> - \<V\>^2|, where V = V(u,v) are the complex visibilities across baselines. <> is averaging across baselines. The Dynamic spectrum is basically <V(u,v)>. 
   - *Standard deviation in the argument of the complex visibilities*

      <Angle(V)^2> - Angle(\<V\>)^2
   - *Systematic error*
      
      Systematic measurement error is obtained as |Im(\<V\>)|. In ideal case since we expect a source at phase centre only, the Im(\<V\>) = 0.
   
   Code outputs both systematic and \`thermal' or random noise (standard deviation noise) separately. User may use this statistics to obtain one's own noise estimates. 
   
3. **plot_fullDS_formscanVisibDS.py** 

   The plotter code which stiches together the outputs from DS making code for each polarisation to generate 1 single plot for the entire observing period. The code can generate both flux density and brightness temperature dynamic spectra/light curve as per user's choice.
