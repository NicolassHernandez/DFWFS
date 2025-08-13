import importlib
import torch.nn as nn
import torch
import numpy as np

from torch import unsqueeze as UNZ
from model.pyr import *
from model.functions_model import *



class FWFS(nn.Module):
    def __init__(self,wfs,**kwargs):
        super(FWFS,self).__init__() 
        self.fp = wfs.fp if hasattr(wfs,'fp') else [(0,0),(0,0)]
        self.train_state = [second_value > 0 for _,second_value in self.fp]# T[True,True,False]
        self.init = wfs.init if hasattr(wfs,'init') else ['constant',0]
        self.device = wfs.device if hasattr(wfs,'device') else kwargs.get('device','cpu')
        #
        self.precision = wfs.precision if hasattr(wfs,'precision') else get_precision(type='single')
        # PWFS MODEL
        self.pwfs = PWFS(wfs, device=self.device)
        self.mInorm = self.pwfs.mInorm
        self.norm_nn = wfs.norm_nn if hasattr(wfs,'norm_nn') else None
        self.pupil_sum = torch.sum(self.pwfs.pupil).to(self.precision.real)
        self.nRr = wfs.nRr if hasattr(wfs,'nRr') else False
        # DE element
        if self.train_state[0]:
            DE = torch.zeros((self.pwfs.fovPx,self.pwfs.fovPx), dtype=self.precision.real,device=self.device)
            self.DE  = nn.Parameter(DE)# T2
            if self.init[0]=='kaiming':
                print('KAIMING')
                nn.init.kaiming_normal_(self.DE, mode='fan_in')
            else: 
                init_val = float(self.init[1])
                print('CONSTANT',init_val)
                if init_val < 0:
                    print(f'Setting nH={np.abs(init_val)} as init')
                    self.pwfs.nHead = np.abs(init_val)
                    fourier_mask = torch.angle((self.pwfs.fourier_mask)).to(self.precision.real).to(self.device).squeeze()
                    self.pwfs.nHead = wfs.nHead
                    with torch.no_grad():
                        self.DE.copy_(fourier_mask)
                else:
                    print('normal')
                    nn.init.constant_(self.DE, init_val )
        self.DE_dummy = torch.zeros((self.pwfs.fovPx,self.pwfs.fovPx), dtype=self.precision.real,device=self.device)
        #DE_dummy = torch.exp( 1j*torch.zeros((self.pwfs.fovPx,self.pwfs.fovPx),dtype=self.precision.real,device=self.device) )# T[N,M]
        #self.DE_dummy = torch.angle(  DE_dummy/torch.sum( torch.abs(DE_dummy.flatten()) ) ) 
        # nn
        if self.train_state[1]:
            method = importlib.import_module("model.GcVit")
            if self.pwfs.nPx==256:
                self.NN = method.GCViT(num_classes=len(wfs.jModes),depths=[2,2,6,2],num_heads=[2,4,8,16],window_size=[8,8,16,8],
                                    resolution=wfs.nPx,in_chans=1,dim=64,mlp_ratio=3,drop_path_rate=0.2).to(self.precision.real).to(self.device)
            elif self.pwfs.nPx==128:
                if self.nRr:
                    self.NN = method.GCViT(num_classes=len(wfs.jModes),depths=[2,2,6,2],num_heads=[2,4,8,16],window_size=[4,4,8,4],
                              resolution=wfs.nPx,in_chans=1,dim=64,mlp_ratio=3,drop_path_rate=0.2).to(self.precision.real).to(self.device)
                else:# original
                    #self.NN = method.GCViT(num_classes=len(wfs.jModes),depths=[2,2,6,2],num_heads=[2,4,8,16],window_size=[7,7,14,7],
                    #          resolution=224,in_chans=1,dim=64,mlp_ratio=3,drop_path_rate=0.2).to(self.precision.real).to(self.device)
                    self.NN = method.gc_vit_xxtiny(num_classes=len(wfs.jModes)).to(self.precision.real).to(self.device)
        #
        self.k = torch.tensor( (2*torch.pi)/wfs.wvl ,dtype=self.precision.real,device=self.device)
        self.I0 = self.pwfs.forward(self.pwfs.Piston)# T[1,1,N,M]
    # MAIN FORWARD FUNCTION
    def forward(self, phi, **kwargs):
        vNoise = kwargs.get('vNoise',self.pwfs.vNoise)
        mInorm = kwargs.get('mInorm', self.mInorm)
        norm_nn = kwargs.get('norm_nn', self.norm_nn)
        # propagation
        de = self.DE if hasattr(self,'DE') else self.DE_dummy
        fourier_mask = kwargs.get('fourier_mask', self.pwfs.fourier_mask*torch.exp(1j*de) )
        I = self.pwfs.forward(phi, fourier_mask=fourier_mask, vNoise=vNoise)# T[b,1,N,M]
        I_deg = I.view(I.shape[0],1,-1)# T[b,1,NM]
        if hasattr(self,'NN'):
            I = self.norm_I(I,norm_nn)# T[b,1,N,M]
            Zest = self.NN(I)# T[b,z]
        else:
            I0_v,PIMat = self.pwfs.Calibration(fourier_mask=fourier_mask,vNoise=[0,0,0,1], mInorm=mInorm)
            mI_v = self.pwfs.mI(I[:,0,self.pwfs.idx],I0_v, dim=1, mInorm=mInorm)
            Zest = ( PIMat @ mI_v.t() ).t()# T[z,b]-> T[b,z]    
        return Zest,I_deg
    
    def norm_I(self,I,norm=None):
            if norm==None:
                return I
            elif norm=='max':
                return I/torch.amax(I,dim=(-2,-1),keepdim=True)
            elif norm=='sum':
                return (I/torch.sum(I,dim=(-2,-1),keepdim=True))*self.pupil_sum
            elif norm=='rms':
                return I/torch.sqrt( torch.sum(I**2, dim=(-2,-1),keepdim=True)/self.pupil_sum )
            elif norm=='zscore':
                return (I-torch.mean(I,dim=(-2,-1),keepdim=True))/torch.std(I,dim=(-2,-1),keepdim=True)
            elif norm=='zmI':
                return (I-self.I0)/torch.std((I-self.I0),dim=(-2,-1),keepdim=True)
    
    def __repr__(self):
        print(f'FWFS class: device={self.device} | fp={self.fp} | norm_nn={self.norm_nn}')
        return ''