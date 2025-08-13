import torch
import numpy as np
import math
from torch import unsqueeze as UNZ
from functions._functions import *

# for interpolation
import torchvision.transforms.functional as f
import torch.nn.functional as F


def get_modPhasor(fovPx,samp,mod, precision,device,**kwargs):
    """
    This function generates the modulation tensor which is used in propagation:
    output: T[1,nTheta,fovPx,fovPx]
    """
    x = torch.arange(0,fovPx,1, device=device,dtype=precision.real)/torch.tensor(fovPx,device=device)
    vv, uu = torch.meshgrid(x,x,indexing='ij')
    r,o = cart2pol_torch(uu,vv)
    nTheta = torch.tensor(np.round(2*np.pi*samp*mod), dtype=precision.int,device=device)
    if nTheta == 0:
        ModPhasor = torch.exp( -1j*torch.zeros((1,1,fovPx,fovPx), dtype=precision.real,device=device) )
    else:
        ModPhasor = torch.zeros((1,nTheta,fovPx,fovPx), dtype=precision.complex,device=device)
        for kTheta in range(nTheta):
            theta = 2*(kTheta)*torch.pi/nTheta
            ph = 4*torch.pi*mod*samp*r*torch.cos(o+theta)# type promotion int->float
            ModPhasor[0,kTheta,:,:] = torch.exp(-1j*ph)
    return nTheta,ModPhasor


def addNoise(iwfs, vNoise):  
    if vNoise[1]:# camilo lo tiene [1,1,1e1,1] 
        iwfs = torch.poisson(iwfs*vNoise[2])/vNoise[2]
    iwfs = vNoise[3]*iwfs
    if isinstance(vNoise[0],list):
        rn = torch.tensor(np.random.uniform(vNoise[0][0],vNoise[0][1])).to(iwfs.dtype)
    else:
        rn = vNoise[0]
    owfs = iwfs + torch.normal(0,rn,size=iwfs.shape,device=iwfs.device)
    owfs.clamp_(min=0)
    return owfs 

def Propagation(obj,phi,fourier_mask):
    pyrPupil = (obj.Piston)*torch.exp(1j*phi)
    sx = torch.tensor(np.round(obj.fovPx*(1/(2*obj.samp))), dtype=obj.precision.int)# doesnt require device 
    npv = torch.tensor( (obj.fovPx-sx)/2, dtype=obj.precision.int)
    PyrQ = torch.nn.functional.pad(pyrPupil, (npv,npv,npv,npv), "constant", 0)
    u2 = torch.fft.fft2( torch.fft.fft2(PyrQ*obj.ModPhasor) * fourier_mask ) 
    I = torch.sum( torch.abs(u2)**2 , dim=1, keepdim=True)# T[b,1,N,M]
    if obj.nTheta>0:    I = I/obj.nTheta
    I = f.resize(I,(sx,sx),antialias=True)*(2*obj.samp)
    return I



def poscrop(obj,I):
    if obj.nHead == 4:
        if obj.alpha == 2:
            if obj.nPx == 128:
                w = 35
                p1 = I[...,28:28+w,28:28+w]# sino con 25 y 40
                p2 = I[...,28:28+w,65:65+w]
                p3 = I[...,65:65+w,28:28+w]
                p4 = I[...,65:65+w,65:65+w]
            elif obj.nPx == 256:
                w = 70
                p1 = I[...,55:55+w,55:55+w]# sino con 25 y 40
                p2 = I[...,55:55+w,130:130+w]
                p3 = I[...,130:130+w,55:55+w]
                p4 = I[...,130:130+w,130:130+w]
        elif obj.alpha == 3:
            if obj.nPx==128:
                p1 = I[...,19:53,19:53]
                p2 = I[...,19:53,75:109]
                p3= I[...,75:109,19:53]
                p4 = I[...,75:109,75:109]
            elif obj.nPx==256:
                p1 = I[...,38:105,38:105]
                p2 = I[...,38:105,151:218]
                p3= I[...,151:218,38:105]
                p4 = I[...,151:218,151:218]
        h1 = torch.concat((p1,p2),dim=-1)
        h2 = torch.concat((p3,p4),dim=-1)
        out = torch.concat((h1,h2),dim=-2)   
    elif obj.nHead == 3:
        if obj.alpha == 3:
            if obj.nPx == 128:
                w = 35
                p1 = I[...,27:27+w,13:13+w]
                p2 = I[...,27:27+w,81:81+w]
                p3 = I[...,86:86+w,47:47+w]
                out = torch.zeros((*I.shape[:2],2*w,2*w), device=obj.device)
                out[...,0:w,0:w] = p1
                out[...,0:w,w:2*w] = p2
                out[...,w:2*w,w//2:3*w//2] = p3 
    return out









def fourier_geometry(alpha,nhead,**kwargs):
    """
    Fourier geometry function generated a general geometry given nhead and alpha
    """
    def Heaviside(x):
        out = torch.zeros_like(x)
        out[x==0] = 0.5
        out[x>0] = 1 
        return out.to(torch.bool)
    def angle_wrap(X,Y):# consider that X is Y and Y is X
        O = torch.zeros_like(X)
        mask1 = Heaviside(X)*Heaviside(Y)#(X>=0) & (Y>=0)
        O[mask1] = torch.atan2(torch.abs(Y[mask1]),torch.abs(X[mask1]))
        mask2 = Heaviside(-X)*Heaviside(Y)#(X<0) & (Y>=0)
        O[mask2] = torch.atan2(Y[mask2],X[mask2])
        mask3 = Heaviside(-X)*Heaviside(-Y)#(X<0) & (Y<0)
        O[mask3] = torch.atan2(torch.abs(Y[mask3]),torch.abs(X[mask3])) + torch.pi
        mask4 = Heaviside(X)*Heaviside(-Y)#(X>=0) & (Y<0)
        O[mask4] = 2*torch.pi-torch.atan2(torch.abs(Y[mask4]),torch.abs(X[mask4]))
        return O
    precision = kwargs.get('precision',get_precision(type='double'))
    device = kwargs.get('device', 'cpu')
    nPx = torch.tensor(kwargs.get('nPx',512), dtype=precision.int)
    wvl = torch.tensor(kwargs.get('wvl',635), dtype=precision.real)
    alpha = torch.tensor(alpha*torch.pi/180, dtype=precision.real,device=device)
    ps = torch.tensor(kwargs.get('ps',3.74e3), dtype=precision.real,device=device)   
    rooftop = (torch.tensor(kwargs.get('rooftop',0),dtype=precision.real,device=device)*ps)
    nhead = torch.tensor(nhead, dtype=precision.int)# con float funca mal
    # grid
    if nhead >= 2:
        # frequency grid
        x = torch.linspace( -(nPx-1)//2,(nPx-1)//2,nPx , dtype=precision.real,device=device)*ps
        X,Y = torch.meshgrid(x,x, indexing='ij')
        #
        step = torch.tensor(2*np.pi/nhead, dtype=precision.real)
        nTheta = torch.arange(0,2*np.pi+step,step, dtype=precision.real)
        k = 2*np.pi/wvl
        O_wrap = angle_wrap(X,Y)
        pyr = torch.zeros((nPx,nPx), dtype=precision.complex,device=device)
        beforeExp = torch.zeros_like(pyr, dtype=precision.real,device=device)
        for i in range( len(nTheta)-1 ):
            mask = (Heaviside( O_wrap-nTheta[i] )*Heaviside( nTheta[i+1]-O_wrap ))#< mask bool
            phase = (nTheta[i]+nTheta[i+1])/2
            cor = torch.cos(phase)*X + torch.sin(phase)*Y
            cor = ( (cor-rooftop)*(Heaviside(cor-rooftop).to(precision.real)) )# heaviside is made real}
            beforeExp += (mask.to(precision.real))*(k*torch.tan(alpha)*cor)
    elif nhead==0:
        beforeExp = torch.zeros((nPx,nPx),dtype=precision.complex,device=device)
    else:
        raise KeyError('Incorrect number of heads')
    afterExp = torch.exp( 1j*( beforeExp ) )
    fourierMask = UNZ( UNZ( torch.fft.fftshift( afterExp/torch.sum( torch.abs( afterExp.flatten() ) ) ) ,0) ,0)
    return fourierMask,beforeExp


  

def zernike_geometry(obj, h,lamD, **kwargs):
    precision = kwargs.get('precision',get_precision(type='double')) 
    nPx = torch.tensor(kwargs.get('nPx',128), dtype=precision.int)
    ps = torch.tensor(kwargs.get('ps',3.74e3), dtype=precision.real,device=obj.device)   
    #
    pyrPupil = (obj.Piston.to(precision.complex))*torch.exp(1j*obj.Piston.to(precision.real))
    subscale = torch.tensor( (1/(2*obj.samp)) ,dtype=precision.real)
    sx = torch.tensor(np.round(nPx*subscale), dtype=precision.int)# doesnt require device 
    npv = torch.tensor( (nPx-sx)/2 ,dtype=precision.int)
    PyrQ = torch.nn.functional.pad(pyrPupil, (npv,npv,npv,npv), "constant", 0)
    psf = torch.fft.fftshift( torch.abs(torch.fft.fft2(PyrQ))**2 ).squeeze()# T[b,c,NM]->T[N,M]

    # grid
    x = torch.linspace( -(nPx-1)//2,(nPx-1)//2,nPx, dtype=precision.real,device=obj.device)
    # compute FWHM
    FWHM = torch.sum( ( psf[psf.shape[0]//2,:]/torch.max(psf[psf.shape[0]//2,:]) )>=.5 )# sum of pixels 1D
    R_FWHM = (FWHM)/2# radius from the center (-1) 1D
    #
    h = torch.tensor(h, dtype=precision.real)
    lamD = torch.tensor(lamD, dtype=precision.real)
    X,Y = torch.meshgrid(x,x, indexing='ij')
    R = torch.sqrt(X**2+Y**2)
    phase = (R<=lamD*R_FWHM).to(precision.real)
    afterExp = torch.exp(1j*h*phase).to(precision.complex).to(obj.device)# T[N,M]
    fourierMask = UNZ( UNZ( torch.fft.fftshift( afterExp/torch.sum( torch.abs( afterExp.flatten() ) ) ) ,0) ,0)
    #print(afterExp.shape,fourierMask.shape)
    return fourierMask
