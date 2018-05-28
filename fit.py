import numpy as np
import matplotlib.pyplot as plt
import corner
import mcmc_funct as mcmc
import subprocess
from astropy.io import fits


# Physical constants:
yr = 31556926.0               #  s 
psc = 3.09e18                 #  cm 
msolar = 1.99e33              #  g 
m_proton = 1.67262158e-24     #  g 
psi_0 = 2.026                 #
clight = 3.e10                #  cm/s 
m_electron = 9.10938188e-28   #  g 
e_charge = 4.8032068e-10      #  esu 
#erg = 624.150974             #  GeV 
hcgs = 6.626068e-27           #  ergs s [Planck's Constant] 
hev = 4.1356668e-15           #  eV s [Planck's Constant] 
kb = 1.380658e-16             #  ergs / K [Boltzmann's constant] 
mec2 = 510.998910             #  electron rest mass energy in keV 
sigmat = 6.6524586e-25        #  Thomson cross-section in cm^2 
mjy = 1.e-26                  #  ergs / s / cm^2 / Hz  
hev = 4.135667662*10**(-15)   #  Planck's constant[eV.s]
herg = 6.626068*10**-27       #  Planck's constant [ergs.s]
erg = 6.424*10**(11)          #  Erg to eV[eV/erg]
kpc = 3.086e+21               #  kpc to cm


probs=[]
pars=[]



with open('prob.dat') as f1:
    strprob= f1.readlines()
    for prob in strprob:
        probs.append(float(prob))
with open('predictobs.dat') as f3:
    predictobs=f3.readlines()
maxprob=max(probs)
print(maxprob)
ndx=np.argmax(probs)
with open ('chain.dat') as f2:
    for i in f2.readlines():
        pars.append([float(x) for x in i.split()])
        
# with open('chi2_ind.dat') as f4:
#     strchi2=f4.readlines()



print(pars[ndx])
print('\n')
print(predictobs[ndx])
print('\n')
#print(strchi2[ndx])

with open ('bestfit.txt','w') as fit:
    fit.write('Likelihood: '+str(maxprob)+'\n')
    fit.write('Pars: '+str(pars[ndx])+'\n')
    fit.write('Predictobs: '+str(predictobs[ndx])+'\n')

# par_sp=np.asarray(pars).reshape((12,len(pars)))
# steps=np.arange(len(pars))
        
# fig1=plt.figure()

# plt.subplot(321)
# plt.plot(steps,par_sp[0],color='k',alpha=0.5)
# plt.ylabel('Log[Esn]')

# plt.subplot(322)
# plt.plot(steps,par_sp[1],color='k',alpha=0.5)
# plt.ylabel('Log[Mej]')

# plt.subplot(323)
# plt.plot(steps,par_sp[2],color='k',alpha=0.5)
# plt.ylabel('Log[nism]')

# plt.subplot(324)
# plt.plot(steps,par_sp[3],color='k',alpha=0.5)
# plt.ylabel('brakind')

# plt.subplot(325)
# plt.plot(steps,par_sp[4],color='k',alpha=0.5)
# plt.ylabel('Log[tau]')

# plt.subplot(326)
# plt.plot(steps,par_sp[5],color='k',alpha=0.5)
# plt.ylabel('Log[Etab]')

# fig1.tight_layout(h_pad=0.0)
# plt.savefig('space-covered1.png')

# fig2=plt.figure()

# plt.subplot(321)
# plt.plot(steps,par_sp[6],color='k',alpha=0.5)
# plt.ylabel('Log[Emin]')

# plt.subplot(322)
# plt.plot(steps,par_sp[7],color='k',alpha=0.5)
# plt.ylabel('Log[Emax]')

# plt.subplot(323)
# plt.plot(steps,par_sp[8],color='k',alpha=0.5)
# plt.ylabel('Log[Ebreak]')

# plt.subplot(324)
# plt.plot(steps,par_sp[9],color='k',alpha=0.5)
# plt.ylabel('p1')

# plt.subplot(325)
# plt.plot(steps,par_sp[10],color='k',alpha=0.5)
# plt.ylabel('p2')

# plt.subplot(326)
# plt.plot(steps,par_sp[11],color='k',alpha=0.5)
# plt.ylabel('dist')

# fig2.tight_layout(h_pad=0.0)
# plt.savefig('space-covered2.png')


command=mcmc.runmodel(pars[ndx])
print(command)

with open ('bestfit.txt','a') as fit:
    fit.write('Command: '+str(command)+ '\n')

ExtProcess=subprocess.Popen(command,shell=True)
ExtProcess.wait()

photfreqlist,photspeclist,dyninfo=mcmc.loadfits(pars[ndx])

fluxdens = {'freq1':0.327,'flux1':7.3, 'flux1err':0.7,'freq2':1.43,'flux2':7.,'flux2err':0.4,
            'freq3':4.8,'flux3':6.54,'flux3err':0.37,'freq4':70.,'flux4':4.3,'flux4err':0.6,
            'freq5':84.2,'flux5':3.94,'flux5err':0.7,'freq6':90.7,'flux6':3.8,'flux6err':0.4,
            'freq7':94.,'flux7':3.5,'flux7err':0.4,'freq8':100.,'flux8':2.7,'flux8err':0.5,
            'freq9':141.9,'flux9':2.5,'flux9err':1.2,'freq10':143.,'flux10':3.0,'flux10err':0.4}
#Radio
obsfreq=[]
obsflux=[]
obsfluxerr=[]
for i in range(1,11):
    obsfreq.append(fluxdens['freq'+str(i)]*1.e9)
    obsflux.append(fluxdens['flux'+str(i)]*fluxdens['freq'+str(i)]*1.e9/(1.e23))
    obsfluxerr.append(fluxdens['flux'+str(i)+'err']*fluxdens['freq'+str(i)]*1.e9/(1.e23)) 

#X-ray
obsxray={'minfreq':15.e3/hev, 'maxfreq':50.e3/hev, 'gamma':2.093, 'gammaerr':0.008,
        'flux':5.11e-11, 'fluxerr':0.05e-11} 

obsgamma1=obsxray['gamma']
obsgamma2=obsxray['gamma']+obsxray['gammaerr']
obsgamma3=obsxray['gamma']-obsxray['gammaerr']
obsxrayflux=obsxray['flux']
xemin=obsxray['minfreq']*hcgs #erg
xemax=obsxray['maxfreq']*hcgs #erg
keverg=1.e3/erg #erg

term=1./(2-obsgamma1)*(xemax**(2-obsgamma1)-xemin**(2-obsgamma1))
xnorm=obsxrayflux/term*keverg**(-obsgamma1)

logfluxdens15kev1=(np.log10(xnorm)-obsgamma1*
                (np.log10(xemin/keverg))+np.log10(xemin)+np.log10(hcgs))

logfluxdens50kev1=(np.log10(xnorm)-obsgamma1*
                (np.log10(xemax/keverg))+np.log10(xemax)+np.log10(hcgs))


term2=1./(2-obsgamma2)*(xemax**(2-obsgamma2)-xemin**(2-obsgamma2))
xnorm2=obsxrayflux/term2*keverg**(-obsgamma2)


logfluxdens15kev2=(np.log10(xnorm2)-obsgamma2*
                (np.log10(xemin/keverg))+np.log10(xemin)+np.log10(hcgs))

logfluxdens50kev2=(np.log10(xnorm2)-obsgamma2*
                (np.log10(xemax/keverg))+np.log10(xemax)+np.log10(hcgs))


term3=1./(2-obsgamma3)*(xemax**(2-obsgamma3)-xemin**(2-obsgamma3))
xnorm3=obsxrayflux/term3*keverg**(-obsgamma3)


logfluxdens15kev3=(np.log10(xnorm3)-obsgamma3*
                (np.log10(xemin/keverg))+np.log10(xemin)+np.log10(hcgs))

logfluxdens50kev3=(np.log10(xnorm3)-obsgamma3*
                (np.log10(xemax/keverg))+np.log10(xemax)+np.log10(hcgs))



obsxray_x1 = [obsxray['minfreq'],obsxray['maxfreq']]
obsxray_y1 = [obsxray['minfreq']*10**(logfluxdens15kev1),obsxray['maxfreq']*10**(logfluxdens50kev1)]

obsxray_x2 = [obsxray['minfreq'],obsxray['maxfreq']]
obsxray_y2 = [obsxray['minfreq']*10**(logfluxdens15kev2),obsxray['maxfreq']*10**(logfluxdens50kev2)]

obsxray_x3 = [obsxray['minfreq'],obsxray['maxfreq']]
obsxray_y3 = [obsxray['minfreq']*10**(logfluxdens15kev3),obsxray['maxfreq']*10**(logfluxdens50kev3)]



#Gamma-ray
obsgammaray={'minfreq':0.2e+12/hev,'maxfreq':5.e+12/hev,'gamma':2.08,'gammaerr':0.22,
'freq':1.e+12/hev,'photdens':4.59e-13,'photdenserr':1.00e-13}

gemin=obsgammaray['minfreq']*hcgs #erg
gemax=obsgammaray['maxfreq']*hcgs #erg

gfluxdens1=mcmc.phot2fluxdens(obsgammaray['photdens'],12,1.e12/hev) #erg/s/cm^2/Hz
gfluxdenserr=mcmc.phot2fluxdens(obsgammaray['photdenserr'],12,1.e12/hev)


term2=1./(2.-obsgammaray['gamma'])*(gemax**(2.-obsgammaray['gamma'])-gemin**(2.-obsxray['gamma']))

#print(obsxray_x)
#print(obsxray_y)

#print(photfreqlist)

fig1=plt.figure()
fig1.set_size_inches(16,9)
#ax1=fig.add_subplot(111)
plt.plot(photfreqlist[:-1],photspeclist[:-1])
plt.scatter(obsfreq,obsflux,color='red',s=10)
plt.errorbar(obsfreq,obsflux,yerr=obsfluxerr,ls='none',color='red')
plt.errorbar(obsgammaray['freq'],gfluxdens1*obsgammaray['freq'],
    yerr=gfluxdenserr*obsgammaray['freq'],ls='none',color='purple')
plt.scatter(obsgammaray['freq'],gfluxdens1*obsgammaray['freq'],color='purple',s=10)
plt.plot(obsxray_x1,obsxray_y1,color='green')
plt.plot(obsxray_x2,obsxray_y2,color='green')
plt.plot(obsxray_x3,obsxray_y3,color='green')
plt.xlabel(r'Frequency $\nu$ [Hz]')
plt.ylabel(r'$\nu F_{\nu}$ [ergs s$^{-1}$ cm $^{-2}$]')
plt.title('Photon Spectrum')
plt.xscale('log')
plt.yscale('log')
plt.savefig('photspec.png',dpi=200)
#plt.show()

#samples=np.asarray(pars)

samples=[]
for i in range(len(probs)):
    if probs[i]>-4:
        samples.append(pars[i])
print(len(samples))

# #Corner plot
fig2=corner.corner(samples,labels=[r'$log[E_{sn,51}]$',r'$log[M_{ej}]$',r'$log[N_{ism}]$',r'$p$',r'$log[\tau]$',r'$log[e_{tab}]$',r'$log[E_{min}]$',r'$log[E_{max}]$',r'$log[E_{break}]$',r'$p_1$',r'$p_2$',r'$dist$'])
plt.savefig('corner.png',dpi=200)
plt.show()




# goodsamples=[]
# goodchi2=[]
# for i in range(len(samples)):
#     if float(probs[i])<6:
#         goodchi2.append(-2*float(probs[i]))
#         goodsamples.append(samples[i])



# goodsamples=np.asarray(goodsamples).reshape([12,len(goodsamples)])
# test=goodsamples[0:2].reshape(len(goodchi2),2)
# fig3=corner.corner(test,weights=goodchi2,labels=[r'$log[E_{sn 51}]$',r'$log[M_{ej}]$'])
# #plt.colorbar()
# plt.savefig('color1.png',dpi=200)
# plt.show()

# fig3=corner.hist2d(samples_ind[0],samples_ind[1],labels=[r'$log[E_{sn 51}]$',r'$log[M_{ej}]$'])
# plt.savefig('corner1.png',dpi=200)
# plt.show()
