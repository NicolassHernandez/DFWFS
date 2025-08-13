import numpy as np
import math
import scipy.special as scp
import torch
import matplotlib.pyplot as plt
from functions.atmosphere import *
from functions.utils import *

class dC: pass

cart2pol_torch = lambda x,y: (torch.sqrt(x**2 + y**2), torch.atan2(y,x))
pol2car_torch = lambda r,o: (r*torch.cos(o), r*torch.sin(o))

def get_precision(type):
    precision = dC()
    if type=='hsingle':
        precision.int = torch.int16
        precision.real = torch.float16
        precision.complex = torch.complex64
    elif type=='single':
        precision.int = torch.int32
        precision.real = torch.float32
        precision.complex = torch.complex64
    elif type=='double':
        precision.int = torch.int64
        precision.real = torch.float64
        precision.complex = torch.complex128
    else:
        sys.exit('precision not recognized')
    return precision
    


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)    
def cart2pol_torch(x,y):
    rho = torch.sqrt(x**2 + y**2)
    phi = torch.atan2(y, x)
    return(rho, phi)


def get_pupil(N):
    D = N + 1# in pixels
    x = torch.linspace(-N/2,N/2,N)
    X,Y = torch.meshgrid(x,x)
    r,_ = cart2pol_torch(X,Y)
    pupil = r<=D/2
    return pupil

def CreateTelescopePupil(Npx):
    x = np.arange(-(Npx-1)/2,(Npx-1)/2+1,1)  # change to +1 if torch of tf
    x = x/np.max(x)
    u = x
    v = x
    x,y = np.meshgrid(u,v,indexing='ij')
    r,o = cart2pol(x,y)      
    out = r <= 1
    return(out)
 
 
def freqspace(n,flag):
    n = [n,n]
    a = np.arange(0,n[1],1)
    b = np.arange(0,n[0],1)
    f1 = (a-np.floor(n[1]/2))*(2/(n[1]))
    f2 = (b-np.floor(n[0]/2))*(2/(n[0]))
    f1,f2 = np.meshgrid(f1,f2,indexing='ij')
    return(f1,f2)
#4
def graduatedHeaviside(x,n):
    dx = x[0,1]-x[0,0]
    if dx == 0:
       dx = x[1,1]-x[1,0]
    out = np.zeros((np.size(x,0),np.size(x,1)))   
    out[-dx*n<=x] = 0.5
    out[x>dx*n] = 1
    return(out)
#5


def CreateZernikePolynomials_camilo(wfs):
    nPx = wfs.nPx
    jModes = wfs.jModes
    pupilLogical = wfs.pupilLogical
    u = nPx
    u = np.linspace(-1,1,u)
    v = u
    x,y = np.meshgrid(u,v,indexing='ij')
    r,o = cart2pol(x,y) 
    mode = jModes
    nf = [0]*len(mode)
    mf = [0]*len(mode)
    for cj in range(len(mode)):
        j = jModes[cj]
        n  = 0
        m  = 0
        j1 = j-1
        while j1 > n:
            n  = n + 1
            j1 = j1 - n
            m  = (-1)**j * (n%2 + 2*np.floor((j1+(n+1)%2)/2))
        nf[cj] = np.int32(n)
        mf[cj] = np.int32(np.abs(m))
    nv = np.array(nf)
    mv = np.array(mf)
    nf  = len(jModes)
    fun = np.zeros((np.size(r),nf))
    r = np.transpose(r)
    o = np.transpose(o)
    r = r[pupilLogical]
    o = o[pupilLogical]
    pupilVec = pupilLogical.flatten()
    def R_fun(r,n,m):
        R=np.zeros(np.size(r))
        sran = int((n-m)/2)+1# ver como poner np.int32
        for s in range(sran):
            Rn = (-1)**s*np.prod(np.arange(1,(n-s)+1,dtype=float))*r**(n-2*s)
            Rd = (np.prod(np.arange(1,s+1))*np.prod(np.arange(1,((n+m)/2-s+1),dtype=float))*np.prod(np.arange(1,((n-m)/2-s)+1)))
            R = R + Rn/Rd
        return(R)    
    ind_m = list(np.array(np.nonzero(mv==0))[0])
    for cpt in ind_m:
        n = nv[cpt]
        m = mv[cpt]
        fun[pupilLogical.flatten(),cpt] = np.sqrt(n+1)*R_fun(r,n,m)
    mod_mode = jModes%2
    ind_m = list(np.array(np.nonzero(np.logical_and(mod_mode==0,mv!=0))))
    for cpt in ind_m:
        n = nv[cpt]
        m = mv[cpt]
        fun[pupilLogical.flatten(),cpt] = np.sqrt(n+1)*R_fun(r,n,m)*np.sqrt(2)*np.cos(m*o)
    ind_m = list(np.array(np.nonzero(np.logical_and(mod_mode==1,mv!=0))))
    for cpt in ind_m:
        n = nv[cpt]
        m = mv[cpt]
        fun[pupilLogical.flatten(),cpt] = np.sqrt(n+1)*R_fun(r,n,m)*np.sqrt(2)*np.sin(m*o)
    modes = fun
    return(modes)





def factorial(n): return math.factorial(n)

def CreateZernikePolynomials1(wfs):
    nPx = wfs.nPx
    jModes = wfs.jModes
    pupilLogical = wfs.pupilLogical
    pupil = torch.tensor(wfs.pupil).numpy()
    u = np.linspace(-1,1,nPx)
    x,y = np.meshgrid(u,u,indexing='ij')
    #x = x/max(x[pupilLogical])
    #y = y/max(y[pupilLogical])
    r,theta = cart2pol(x,y) 
    #mode = jModes   
    def zrf(n,m,r):# Zernike radial function
        R = np.zeros_like(r)
        for k in range(0, int((n-m)/2)+1):
            num = (-1)**k * factorial( int(n)-k)
            den = factorial(k) * factorial( int((n+m)/2)-k ) * factorial( int((n-m)/2)-k )
            R += (num/den) * r**(n-2*k)
        return R
    def j2nm(j):
        n = int( ( -1.+np.sqrt( 8*(j-1)+1 ) )/2. )
        p = ( j-( n*(n+1) )/2. )
        k = n%2
        m = int((p+k)/2.)*2 - k
        if m!=0:
            if j%2==0:
                s = 1
            else:
                s=-1
            m *= s 
        return n,m
    zMode = np.zeros((np.prod(wfs.pupil.shape),len(jModes)))
    for i,j in enumerate(jModes):
        # nm2j
        n,m = j2nm(j)
        #
        if m == 0:
            Z = np.sqrt( (n+1) )*zrf(n,0,r)
        else:
            if m > 0:# j is even
                Z = np.sqrt(2*(n+1))*zrf(n,m,r) * np.cos( m*theta )
            else:# j is odd
                m = np.abs(m)
                Z = np.sqrt(2*(n+1))*zrf(n,m,r) * np.sin( m*theta )
        zMode[:,i] = (Z*pupil).flatten()
    return zMode


