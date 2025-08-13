import torch.nn as nn
import torch
import numpy as np
from functions._functions import *
from functions.utils import *
from torch import unsqueeze as UNZ
from torch import squeeze as NZ
from PIL import Image
from model.functions_model import *


class PWFS(nn.Module):
    def __init__(self, wfs,**kwargs):
        super().__init__()
        self.tag = 'major_pwfs'
        self.device = kwargs.get('device', 'cpu')
        self.precision = wfs.precision
        #
        self.D = wfs.D
        self.nPx = wfs.nPx
        self.fovPx = (wfs.fovPx)
        self.samp = wfs.samp
        #
        self.jModes = wfs.jModes
        self.modes = torch.tensor(wfs.modes, dtype=self.precision.real,device=self.device)
        self.bModes = torch.zeros((len(self.jModes),1,self.nPx,self.nPx), dtype=self.precision.real,device=self.device)
        for k in range(len(wfs.jModes)):           
            self.bModes[k,0:1,:,:] = self.modes[:,k].reshape(self.nPx,self.nPx)
        #
        self.pupil = torch.tensor(wfs.pupil)# T[N,M]###
        self.pupilLogical = torch.tensor( wfs.pupilLogical, dtype=torch.bool,device=self.device)
        #
        self.mInorm = wfs.mInorm
        self.amp_cal = torch.tensor(wfs.amp_cal, dtype=self.precision.real,device=self.device)#
        self.vNoise = wfs.vNoise# T2
        self.Piston = UNZ(UNZ(self.pupil,0),0).to(self.device).to(self.precision.real)# T[1,1,N,M]
        # FOURIER MASK
        self.wvl = (wfs.wvl)
        self.ps = (wfs.ps)
        self.rooftop = (wfs.rooftop)# [0]asi debe ser
        self.alpha = (wfs.alpha)
        self.nHead = ( wfs.nHead )
        self.crop = wfs.crop if hasattr(wfs,'crop') else False
        self.modulation = wfs.modulation
        print(f'PYR OBJECT INITIALIZED: nPx={self.nPx} | nZ={len(wfs.jModes)} | crop={self.crop} | M={self.modulation} | precision={wfs.precision_name}')

    # FORWARD
    def forward(self, phi, **kwargs):
        if (phi.is_complex() or (len(phi.shape)!=4)): raise ValueError('Input cannot be complex')
        fourier_mask = kwargs.get('fourier_mask', self.fourier_mask)
        vNoise = kwargs.get('vNoise', self.vNoise)
        I = Propagation(self, phi, fourier_mask=fourier_mask)# T[b,1,nPx,nPx]
        I = addNoise(I, vNoise)
        if self.crop:   I = poscrop(self,I)
        return I    
    # CALIBRATION
    def Calibration(self, **kwargs):
        fourier_mask = kwargs.get('fourier_mask', self.fourier_mask)
        mInorm = kwargs.get('mInorm', self.mInorm)# sum normalization
        vNoise = kwargs.get('vNoise',[0,0,0,1])# asumed average of frame/time exposure
        piston = self.Piston*self.amp_cal
        I0_v = self.forward(piston, fourier_mask=fourier_mask, vNoise=vNoise)[:,0,self.idx]# T[z,1,N,M]->T[z,NM]
        z = self.bModes*self.amp_cal
        mIp_v = self.forward(z, fourier_mask=fourier_mask, vNoise=vNoise)[:,0,self.idx]# T[z,1,N,M]->T[z,NM]
        mIn_v = self.forward(-z, fourier_mask=fourier_mask, vNoise=vNoise)[:,0,self.idx]# T[z,1,N,M]->T[z,NM]
        mIn_v,mIp_v = self.mI(mIn_v,I0_v, dim=1,mInorm=mInorm),self.mI(mIp_v,I0_v, dim=1,mInorm=mInorm)# T[z,NM]-I0
        IMat = ( mIp_v-mIn_v )/(2*self.amp_cal)# T[z,NM]
        PIMat = torch.linalg.pinv( IMat.t() )
        return I0_v, PIMat
    # META-INTENSITY
    def mI(self, I,I0, mInorm=1,dim=1):  
        if mInorm:
            return (I-I0)/torch.sum(I, dim=dim, keepdim=True)# T[b,NM]
        else:
            return (I-I0)
    # SETTERS
    # modulation
    @property 
    def modulation(self):
        return self._modulation
    @modulation.setter
    def modulation(self, value):
        self.nTheta,self.ModPhasor = get_modPhasor(fovPx=self.fovPx,samp=self.samp,mod=value,precision=self.precision,device=self.device)
        self._modulation = value
        
    # crop of pupil
    @property
    def crop(self):
        return self._crop
    @crop.setter
    def crop(self, value):
        assert isinstance(value,bool), 'crop must be a boolean variable'
        if value:
            if self.nHead == 4: 
                if self.alpha == 2:
                    if self.nPx == 128:
                        self.idx = (torch.ones((70,70), dtype=torch.bool,device=self.device)>0)# T[N,M]
                    elif self.nPx == 256:
                        self.idx = (torch.ones((140,140), dtype=torch.bool,device=self.device)>0)# T[N,M]
                elif self.alpha == 3:
                    if self.nPx == 128:
                        self.idx = (torch.ones((68,68), dtype=torch.bool,device=self.device)>0)# T[N,M]
                    elif self.nPx == 256:
                        self.idx = (torch.ones((134,134), dtype=torch.bool,device=self.device)>0)# T[N,M]
            elif self.nHead == 3:
                if self.alpha == 3:
                    if self.nPx == 128:
                        self.idx = (torch.ones((70,70), dtype=torch.bool,device=self.device)>0)# T[N,M]
        else:
            self.idx = (torch.ones((self.nPx,self.nPx), dtype=torch.bool,device=self.device)>0)# T[N,M]
        self._crop = value
    # alpha
    @property
    def alpha(self):
        return getattr(self, '_alpha', 0)
    @alpha.setter
    def alpha(self,value):
        self._alpha = value
        self.nHead = self.nHead# call the nHead setter
    # nhead
    @property
    def nHead(self):
        return getattr(self, '_nHead', 0)
    @nHead.setter
    def nHead(self,value):
        if -6<value<0:
            print('MAKING A ZWFS')
            self.fourier_mask = zernike_geometry(self,nPx=self.fovPx, h=self.alpha,lamD=np.abs(value),device=self.device,
                                                 precision=get_precision(type='double')).to(self.precision.complex)
        if value==-10:# ragazzoni
            img_pil = Image.open("./functions/ragazzoni.jpg").convert("RGB").convert("L")
            ar = torch.tensor(np.array(img_pil)[45:255,100:310]).unsqueeze(0).unsqueeze(0)
            amp_factor = (self.ps)*((self.fovPx-1)//2) * (2*torch.pi/self.wvl) * np.tan(self.alpha*np.pi/180)# 0.2-> 32.93
            afterExp = torch.exp( 1j*(f.resize(ar,(self.fovPx,self.fovPx),antialias=True).to(torch.float64)/255)*amp_factor   )
            self.fourier_mask = (torch.fft.fftshift( afterExp/torch.sum( torch.abs( afterExp.flatten() ) ) )).to(self.precision.complex).to(self.device)
        if value==-2:# random mask
            afterExp = torch.exp(1j*torch.normal(mean=0.,std=math.sqrt(2/self.fovPx),size=(self.fovPx,self.fovPx),dtype=self.precision.real))
            self.fourier_mask = (torch.fft.fftshift( afterExp/torch.sum( torch.abs( afterExp.flatten() ) ) )).to(self.precision.complex).to(self.device)
        if value>=0:
            self.fourier_mask,_ = fourier_geometry(nPx=self.fovPx,nhead=value,alpha=self.alpha,wvl=self.wvl,
                            ps=self.ps,rooftop=self.rooftop,precision=get_precision(type='double'),device=self.device)
            self.fourier_mask = self.fourier_mask.to(self.precision.complex)
        self._nHead = value