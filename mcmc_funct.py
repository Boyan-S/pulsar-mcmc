import numpy as np
#import emcee
#from emcee.utils import MPIPool
#from mpi4py import MPI
from astropy.io import fits
import subprocess
import warnings
#import sys
warnings.filterwarnings('ignore')

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

#Initial Input Parameters (Some from papers, some from G54.1+0.3 as starting point)
esn_51 = 0.933254  
mej = 12 #Solar Masses
nism = 0.0051286 #G54.1+0.3
brakind = 3.
tau =  3980.
age = 870.
charage=4850.
edot_37 = 3.37
e0dot_37 = edot_37*(1+age/tau)**((brakind+1)/(brakind-1))
velpsr = 0
etag = 0 #fraction of SDL lost as radiation
etab = 0.0007 #Magnetization of Pulsar Wind
emin = 1. #GeV
ebreak = 1.e3 #GeV
emax = 1.e6 #GeV
p1 = 1.84 #Camilo et al, 2006
p2 = 2.77   #Camilo et al, 2006
f_max = 0 #fraction of pulsar wind particle energy in Maxwellian component
kT_max = 0 #Energy of Maxwellian component
nic = 0 #Background photon fields in addition to CMB
dist_global = 4.7



pwnpars=[np.log10(esn_51), np.log10(mej), np.log10(nism), brakind,
np.log10(tau), np.log10(age), np.log10(e0dot_37), velpsr, np.log10(etag), 
np.log10(etab), np.log10(emin), np.log10(emax), np.log10(ebreak), p1, p2, 
np.log10(f_max), np.log10(kT_max), nic,dist_global]


pwninput = []
for i in range(len(pwnpars)):
    if i<5 or (i>8 and i<15):
        pwninput.append(pwnpars[i])
pwninput.append(dist_global)
pwninput = np.asarray(pwninput)
#print(pwninput)




def loadfits(pars):
    global badpars#, photspeclist,elecspeclist,dyninfo
    badpars=False
    dist=pars[11]
    #rank = pool.rank
    elecfile = fits.open('modelres.elecspec.fits',ignore_missing_end=True)
    photfile = fits.open('modelres.photspec.fits',ignore_missing_end=True)
    dyninfofile = fits.open('modelres.dyninfo.fits',ignore_missing_end=True)
    try:
        dyninfo=dyninfofile[1].data
        photfreq=photfile[1].data
        photspec=photfile[2].data

        #elecnrg=elecfile[1].data
        #elecspec=elecfile[2].data

        #photfreq[1] - min 
        #photfreq[2] - max 
        #photfreq[3] - avg 
        #photfreq[4] - width

        photfreqlist=[]
        photspeclist=[]
        
        #elecnrglist=np.empty(len(photfreq))
        #elecspeclist=np.empty(len(photfreq))
        
        for i in range(len(photfreq)): 
            #elecnrglist[i] = elecnrg[i][3]
            #elecspeclist[i] = elecspec[0][i+1]
            
            if np.isnan(photspec[0][i+1]) == False and np.isnan(photfreq[i][3]) == False and photfreq[i][3]!=0:
                photfreqlist.append(photfreq[i][3])
                temp=photspec[0][i+1]*photfreq[i][3]*10**(40)/(4*np.pi*(dist*kpc)**2) #Ergs/s/cm^2
                photspeclist.append(temp)


        photfreqlist=np.delete(photfreqlist,len(photfreqlist)-1) #Remove last entry (0)
        photspeclist=np.delete(photspeclist,len(photspeclist)-1)



        elecfile.close()
        photfile.close()
        dyninfofile.close()


        return photfreqlist,photspeclist,dyninfo
    except:
        badpars=True
        return 0.,0.,0.

def calcfluxdens(freq,pars): #returns value in Jy
    dist = pars[11]
    #photfreqlist,photspeclist,dyninfo=loadfits(pars)
    try:
        if freq>np.amin(photfreqlist) and freq<np.amax(photfreqlist):
            diff = np.empty(len(photfreqlist))
            for i in range(len(photfreqlist)):
                diff[i] = np.abs(freq -photfreqlist[i])
            ndx = np.argmin(diff)
            fluxdens = photspeclist[ndx]*10**23/photfreqlist[ndx]
            #print('model fluxdens is',fluxdens)
            return fluxdens
        else:
            with open ('out.of.range.txt','a') as bad:
                bad.write(str(pars)+'\n')
            with open ('bad.freq.txt','a') as badfreq:
                badfreq.write(str(photfreqlist)+'\n')
            print('!!!Frequency out of range, returning 0!!!')
            return 0
    except ValueError:
        return -np.inf

def calcphotdens(energy,pars,power): #power refers to output (in 10^power eV)
    dist=pars[11]
    nbins=0

    try:
        #PhotonDensity = np.empty(len(photfreqlist))
        #Energies=np.empty(len(photfreqlist))
        #diff=np.empty(len(photfreqlist))
        freq=energy/hev
        for i in range(len(photfreqlist)):
            if freq>=photfreq[i][1] and freq<=photfreq[i][2]:      
                nbins+=1      
                fluxdens=calcfluxdens(photfreqlist[i], pars)
                logphotenergy=np.log10(hcgs)+np.log10(photfreqlist[i]) # ergs
                logphotdens=np.log10(fluxdens)-23.+power-np.log10(hev)-logphotenergy
                photdens=10.**logphotdens
        if nbins==1:
            return photdens
    except:
        return 0

def phot2fluxdens(photdens,power,freq): #power: 10^power eV (input units)
    logphotnrg=np.log10(hcgs)+np.log10(freq) #erg
    logfluxdens=np.log10(photdens)+logphotnrg-power+np.log10(hev) #
    fluxdens=10.**(logfluxdens)
    return fluxdens

def calcflux(minenergy,maxenergy):
    global flux
    flux=0.
    minfreq = minenergy/hev
    maxfreq= maxenergy/hev
    freq=[]
    for i in range(len(photfreqlist)):
        if photfreq[i][3]>=minfreq and photfreq[i][3]<=maxfreq:
            if photfreq[i][1]<minfreq:
                minbinfreq=minfreq
            else:
                minbinfreq=photfreq[i][1]
            if photfreq[i][2]>maxfreq:
                maxbinfreq=maxfreq
            else:
                maxbinfreq=photfreq[i][2]
            freqwidth=maxbinfreq-minbinfreq
            flux+=photspeclist[i]/photfreqlist[i]*freqwidth
        elif photfreq[i][3]>maxfreq:
            break
    return flux

def calcphotindex(minenergy,maxenergy,pars): #This function crashes sometimes, fix!
    #photfreqlist,photspeclist,dyninfo = loadfits(pars)
    minfreq = minenergy/hev
    maxfreq= maxenergy/hev
    photenergy=[]
    modphotdens=[]
    for i in range(len(photfreqlist)):
        if photfreqlist[i]>=minfreq and photfreqlist[i]<=maxfreq:
            logphotenergy=np.log10(herg) + np.log10(photfreqlist[i])
            photenergy.append(10.**(logphotenergy)*erg*10**(-3))
            logmodphotdens=np.log10(photspeclist[i]/photfreqlist[i])+3.*(-np.log(hev))-logphotenergy
            modphotdens.append(10.**(logmodphotdens))
    try:
        photinfo,err,_,_,_=np.polyfit(np.log10(photenergy),np.log10(modphotdens),1,full=True)
        gamma = -1.*photinfo[0]
        gammaerr = err[0]
        return gamma,gammaerr
    except TypeError:
        return np.inf,1
    
def calcrad(pars): #in arcminutes
    #photfreqlist,photspeclist,dyninfo = loadfits(pars)
    dist = pars[len(pars)-1] #kpc
    Rpwn = dyninfo['RADPWN']  #parsec
    #Rsnr = dyninfo['RADSNR']
    #thetaSNR = np.rad2deg(np.arcsin(Rsnr/(dist*10**3)))*60
    thetaPWN = float(np.rad2deg(np.arcsin(Rpwn/(dist*10**3)))*3600.)#arcseconds
    return thetaPWN
    
def runmodel(pars):
    strtstep='-1'
    stresn=str(10.**pars[0])
    strmej=str(10.**pars[1])
    strn=str(10.**pars[2])
    strp=str(pars[3])
    strtau=str(10.**pars[4])
    #Set age and e0dot based on tau and p
    age=2.*charage/(pars[3]-1.)-10.**pars[4]
    strage=str(age)
    stre0=str(edot_37*(1+age/10.**pars[4])**((pars[3]+1)/(pars[3]-1)))
    strvelpsr=str(pwnpars[7])
    if pwnpars[8] < -1.e4 or np.isfinite(pwnpars[8]) == False:
        streitag = '0'
    else:
        streitag = str(pwnpars[8])
    streitab=str(10.**pars[5])
    stremin=str(10.**pars[6])
    stremax=str(10.**pars[7])
    strebreak=str(10.**pars[8])
    strinjinx1=str(pars[9])
    strinjinx2=str(pars[10])
    if pwnpars[15] < -1.e4:
        strfmax='0'
        strktmax='0'
    else:
        strfmax = str(pwnpars[15])
        strktmax = str(pwnpars[16])
    strnic = str(pwnpars[17])
    nic = pwnpars[17]
    if nic>0:
        strictemp=''
        stricnorm=''
        index=17
        for i in range(nic):
            index+=1
            strictemp = str(strictemp+' '+str(10.**pwnpars[index]))
        for i in range(nic):
            index+=1
            stricnorm = str(stricnorm+' '+str(10.**pwnpars[index]))
        command=('./pwnmodel.exe '+ strtstep +' '+ stresn+ ' '+ strmej +
                  ' '+strn+' '+strp+' '+strtau+' '+strage+' '+
                  stre0+' '+strvelpsr+' '+streitag+' '+streitab +' '+
                  stremin+' '+stremax+' '+strebreak+' '+strinjinx1 +' '+
                  strinjinx2+' '+strfmax+' '+strktmax+' '+strnic+' '+strictemp+' '
                  +stricnorm) 

    else:
        command=('./pwnmodel.exe '+ strtstep +' '+ stresn+ ' '+ strmej +
                      ' '+strn+' '+strp+' '+strtau+' '+strage+' '+
                      stre0+' '+strvelpsr+' '+streitag+' '+streitab +' '+
                      stremin+' '+stremax+' '+strebreak+' '+strinjinx1 +' '+
                      strinjinx2+' '+strfmax+' '+strktmax+' '+strnic)

    return command

def prior(pars): #Parameter constraints go here:
    logmaxtau=np.log10(2.*charage/(pars[3]-1)-0.5)
    if (pars[9]<0 or pars[10]<0 or pars[2]<-6 or pars[0]<-3 or pars[0]>4 or pars[1]>3 or pars[6]<-3.3 
     or pars[2]>-0.1 or pars[6]>pars[7] or pars[6]>pars[8] or pars[7]<pars[8] or pars[5]>0 or pars[11]<0
     or pars[3]<1 or pars[3]>5 or pars[4]>logmaxtau):
        #print('Parameter check failed, returning -inf')
        return -np.inf
    else:
        return 0

    
    
def pwnevol(pars):
    global photfreqlist,photspeclist,dyninfo
    rank = pool.rank

    #Check parameter constraints
    pr = prior(pars)
    if np.isfinite(pr)==False:
        return -np.inf,'bad'
    #Run model, wait for it to finish
    else:
        command = runmodel(pars)
        ExtProcess = subprocess.Popen(command, cwd = '/home/bks311/astro/code/'+str(rank),shell=True)
        ExtProcess.wait()

        #Parameter/file output check
        photfreqlist,photspeclist,dyninfo=loadfits(pars)
        if badpars==True:
            return -np.inf,'bad'
        else:
            dist = pars[len(pars)-1]
            npwnpars = len(pars)
            strpwnpars=''
            for par in pars:
                strpar = str(par)
                strpwnpars = strpwnpars+strpar+' '
            
            chi2 = 0
            npar = 0
            strmodpred=''
            #obsthetapwn = 40 #[arcmin]
            
        #    #PWN radius
        #    radpwn = dyninfo['RADPWN']
        #    thetapwn, thetasnr = Radii() 
        #    npar+=1
        #    logthetapwn = np.log10(thetapwn)
        #    strobs = strobs+''+str(thetapwn)
        #    #chi2 += 
        #    
        #    #SNR radius
        #    radsnr = dyninfo['RADSNR']
        #    logthetasnr - np.log10(thetapwn)
        #    #chi2 +=
        #    npar+=1
        #    strobs=strobs+''+str(thetasnr)
            
            #Radio
            for i in range(1,11):
                obsfreq = fluxdens['freq'+str(i)]*10**9 #in Hz
                obsfluxdens = fluxdens['flux'+str(i)]
                obsfluxdenserr = fluxdens['flux'+str(i)+'err']
                modfluxdens = calcfluxdens(obsfreq,pars) #Jy
                chi2 += ((modfluxdens-obsfluxdens)/obsfluxdenserr)**2
                npar+=1
                strmodpred = strmodpred+' '+str(modfluxdens) #Predicted Observables
            

            modthetapwn=calcrad(pars)
            chi2 += ((modthetapwn-obsthetapwn)/obsthetapwnerr)**2
            npar+=1
            strmodpred=strmodpred+' '+str(modthetapwn)

            modgamm,modgammerr = calcphotindex(2.e11,5.e12,pars)
            chi2 += ((modgamm-obsgamm)/obsgammerr)**2
            npar+=1
            strmodpred=strmodpred+' '+str(modgamm)

            lnprob=-0.5*chi2
            strchi2 = str(chi2)
            strlnprob = str(lnprob)

            #with open('predictobs.dat','a') as predictobs:
                #predictobs.write(strmodpred+'\n')

            
            #print('# of observables= '+str(npar))
            #dof = npar - nvar
            #strdof = str(dof)
            #txtoutput = "{0:4d} {1:s}\n".format(strlnprob)
            
           # with open ('trials.txt','a') as mcmc:
            #    mcmc.write(str(lnprob)+'\n')
            #with open ('parameters.txt','a') as parfile:
             #   parfile.write(str(pars)+'\n')
            #with open ('predicted.observables.txt','a') as obs:
             #   obs.write(strobs)
            #print(lnprob)
                
            if np.isnan(lnprob) == True:
                print('NAN check failed, returning -inf')
                return -np.inf,'bad'

            else:
                return lnprob,strmodpred
        
        



