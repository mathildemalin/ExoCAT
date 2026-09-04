#### Author : Flavier Kiefer (Kiefer et al. 2024)

import numpy as np

# Function to correctly degrade a spectrum at a given resolution

def degrade_spectrum(lam,flux,res):

    newlam,newflux=convert_to_logspace(lam,flux) # convert to log-spaced wavelengths
    convlam,convflux=LSF_conv_log(newlam,newflux,res)  # outputs have same size as inputs

    return convlam,convflux



#######################################################################################################################  

def convert_to_logspace(lambdas,flux):
    # convert spectrum on linear lambda-grid to log-spaced lambda-grid
    # it keeps the same number of points
    # the flux is summed
 
    Nlam=len(lambdas)
    
    loglam=np.logspace(np.log10(np.min(lambdas)),np.log10(np.max(lambdas)),Nlam)
    logflux=interp_cumsum(lambdas,flux,loglam)
    
    return loglam,logflux
    


#######################################################################################################################    
    
def LSF_conv_log(wave,spectrum,resolution):
# Broadening of a non-broaden spectrum by the LSF
# The broadening is made in a [!!!!!] uniformly spaced grid in log(lambda) [!!!!!]

    # kernel formula
    sigma = 1/resolution/np.sqrt(8*np.log(2))

    def kernel(x):
        return 1/(np.sqrt(2*np.pi*sigma**2))*np.exp(-(1-np.exp(x))**2/(2*sigma**2))

    # transformation to log(wave)
    logwave    = np.log(wave)
    # integration measures (assumes uniformity)
    dlogwave   = np.nanmedian(np.diff(logwave))
    # grid for the Kernel
    xgrid      = np.arange(int(np.floor(np.log(1-4*sigma)/dlogwave)),int(np.ceil(np.log(1+4*sigma)/dlogwave)))*dlogwave
    # Kernel calculation [!!!] NO FLIPPING, THIS IS REAL CONVOLUTION [!!!]
    kernelcalc = kernel(xgrid)
    # define the paddings of the kernel
    leftpad    = np.sum(xgrid<0)
    rightpad   = np.sum(xgrid>0)

    # computation of the convolution
    broadspec  = np.convolve(spectrum,kernelcalc*dlogwave)/np.sum(kernelcalc*dlogwave)
    #print(len(logwave),len(broadspec),leftpad,rightpad)

    # cut the broaden spectrum properly to recover the good wavelength range
    broadspec   = broadspec[leftpad+rightpad:-rightpad-leftpad]
    broadwave   = wave[rightpad:-leftpad]

    return broadwave,broadspec

    



#############################################################################################################################

def interp_cumsum(x,y,xq):
# Function to compute the interpolation of the cumulative sum of y weighted
# by the spacings of variable x.
# Then the derivative of this interpolated function is computed. It does
# not assume uniform steps of xq, HOWEVER a uniform grid would be better.

    lx  = len(x)
    lxq = len(xq)
    x   = np.reshape(x,(lx,))
    y   = np.reshape(y,(lx,))
    xq  = np.reshape(xq,(lxq,))
    
    # initialisation
    xinf=x*1.
    xsup=x*1.
    
    # computation of the inf and sup boundaries of the integration steps
    xinf[0]    = x[0]-(x[1]-x[0])/2
    xinf[1:]   = x[0:-1]+(x[1:]-x[0:-1])/2
    xsup[0:-1] = xinf[1:]
    xsup[lx-1] = x[-1]+(x[-1]-x[-2])/2

    # computation of the integration steps (weights)
    spacings = xsup-xinf
    spacings = np.reshape(spacings,(len(spacings),))

    # padding of supplementary points for inferior edge effects
    sxq      = (xq[1]-xq[0])/2
    Nsq      = 10
    paddflag = 0
    if xinf[0] < xq[0]-sxq:
        Nsq      = int(np.ceil(((xq[0]-sxq) - xinf[0])/sxq/2))
        padd     = xq[0]+2*sxq*np.arange(-Nsq,0)
        xq       = np.concatenate((padd, xq))
        lxq      = lxq+Nsq
        paddflag = 1
        
    # initialisation
    xqinf=xq*1.
    xqsup=xq*1.
    yq=xq*1.


    # computation of the inf and sup boundaries for the interpolations
    xqinf[0]       = xq[0]-(xq[1]-xq[0])/2
    xqinf[1:]      = xq[0:-1]+(xq[1:]-xq[0:-1])/2
    xqsup[0:-1]    = xqinf[1:]
    xqsup[lxq-1]   = xq[lxq-1]+(xq[lxq-1]-xq[lxq-2])/2

    # computation of the integration steps (weights)
    qspacings = xqsup-xqinf
    qspacings = np.reshape(qspacings,(len(qspacings),))

    # computation of the weighted cumulative sum
    y = np.interp(x,x[np.isfinite(y)],y[np.isfinite(y)])
    xysum = np.concatenate(([0],np.cumsum(y*spacings)))
    xsum  = np.concatenate(([xinf[0]],xsup))

    # interpolation
    xysumq = np.interp(xqsup,xsum,xysum)
    

    # differentiation
    yq[0]     = xysumq[0]/qspacings[0]
    yq[1:]    = (xysumq[1:]-xysumq[0:-1])/qspacings[1:]
    if paddflag==1:
        xq=xq[Nsq:]
        yq=yq[Nsq:]

    return yq
  