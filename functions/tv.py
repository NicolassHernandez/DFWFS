import torch.optim as optim
import torch
import numpy as np
from tqdm import tqdm
from model  import *
from mpmath import *
from tqdm import tqdm
from functions._functions import *
from functions.atmosphere import *
from functions.utils import *

bar_format = "{l_bar}{bar}| {n}/{total} [{elapsed}<{remaining}, {rate_fmt}]"


def train(p, model, train_data):
    epoch_loss = 0
    if hasattr(model, 'DE'):
        model.DE.requires_grad = bool(p['fp'][0])
        optimizer_de = optim.AdamW([{'params': model.DE}], lr=p['lr'][0])
    if hasattr(model, 'NN'):
        model.NN.requires_grad = bool(p['fp'][1])
        optimizer_nn = optim.AdamW([{'params': model.NN.parameters()}], lr=p['lr'][1])
    #
    modes = model.pwfs.modes
    for _,data in tqdm(enumerate(train_data),total=len(train_data),desc='Trainning data', bar_format=bar_format):
        phi = data[0].to(model.device).to(model.precision.real)# T[b,1,N,M]
        zgt = data[1].to(model.device).to(model.precision.real)[:,:]# T[b,z]
        #
        if hasattr(model, 'DE'):
            optimizer_de.zero_grad()
        if hasattr(model, 'NN'):
            optimizer_nn.zero_grad()
        # CL
        if (not p['fp'][0]):# if DE freezed or not in the model
            vNoise = model.pwfs.vNoise
        else:
            vNoise = [model.pwfs.vNoise[0],0.,0.,model.pwfs.vNoise[-1]]# neglect poisson noise
        if p['cost'][0] in ['std','mad','var']:
            for _ in range(int(p['cl'][0])):
                zest_tmp,_ = model(phi, vNoise=vNoise)
                zgt = zgt-p['cl'][1]*zest_tmp# T[b,z]
                for i in range(phi.shape[0]): 
                    phi[i,0:1,:,:] = phi[i,0:1,:,:] - p['cl'][1]*(torch.reshape(modes@zest_tmp[i,:],(1,1,model.pwfs.nPx,model.pwfs.nPx))) 
            loss = torch.mean( p['cost'][1]( phi[:,:,model.pwfs.pupilLogical]*(1/model.k) ) )# select only values inside the pupil T[b]-> T[1]
        elif p['cost'][0] in ['mse','rmse','mae']:
            for _ in range(int(p['cl'][0])):
                with torch.no_grad():
                    zest_tmp,_ = model(phi, vNoise=vNoise)
                    zgt = zgt-p['cl'][1]*zest_tmp# T[b,z]
                    for i in range(phi.shape[0]):
                        phi[i,0:1,:,:] = phi[i,0:1,:,:] - p['cl'][1]*(torch.reshape(modes@zest_tmp[i,:],(1,1,model.pwfs.nPx,model.pwfs.nPx))) 
            zest,_ = model(phi, vNoise=vNoise)# T[b,z] 
            loss = torch.mean( p['cost'][1](zgt*(1/model.k),zest*(1/model.k)) )# T[b]-> T[1]            
        loss.backward()
        # step optimizer
        if p['fp'][0] and hasattr(model, 'DE'):
            optimizer_de.step()
        if p['fp'][1] and hasattr(model, 'NN'):
            optimizer_nn.step()
        epoch_loss+=loss.item()
    avg_loss = epoch_loss/len(train_data)
    #
    return avg_loss
#
def validation(p, model, val_data):# test with the vNoise set regardless it type because this is testing 
    rmse_nn = torch.zeros(len(val_data)*p['batch'], dtype=model.precision.real,device=model.device)
    Zest = torch.zeros((len(val_data)*p['batch'],len(model.pwfs.jModes)), dtype=model.precision.real,device='cpu')
    Zgt = torch.zeros_like( Zest )
    modes = model.pwfs.modes
    for i,data in tqdm(enumerate(val_data),total=len(val_data),desc='Validation data', bar_format=bar_format):
        phi = data[0].to(model.device).to(model.precision.real)# T[b,1,N,M]
        zgt = data[1].to(model.device).to(model.precision.real)[:,:]# T[b,z]##
        b = np.arange(p['batch']*i,(i+1)*p['batch'])
        with torch.no_grad():# force the test mode, gradients frozen
            # CL
            if p['cost'][0] in ['std','mad','var']:
                for _ in range(int(p['cl'][0])):
                    zest_tmp,_ = model(phi)
                    zgt = zgt-p['cl'][1]*zest_tmp# T[b,z]
                    for i in range(phi.shape[0]):
                        phi_corr = p['cl'][1]*(torch.reshape(modes@zest_tmp[i,:],(1,1,model.pwfs.nPx,model.pwfs.nPx))) 
                        phi[i,0:1,:,:] = phi[i,0:1,:,:] - phi_corr
                loss = ( p['cost'][1]( phi[:,:,model.pwfs.pupilLogical]*(1/model.k) ) )# select only values inside the pupil T[b]-> T[1]
            elif p['cost'][0] in ['mse','rmse','mae']:
                for _ in range(int(p['cl'][0])):
                    zest_tmp,_ = model(phi)
                    zgt = zgt-p['cl'][1]*zest_tmp# T[b,z]
                    for i in range(phi.shape[0]):
                        phi_corr = p['cl'][1]*(torch.reshape(modes@zest_tmp[i,:],(1,1,model.pwfs.nPx,model.pwfs.nPx))) 
                        phi[i,0:1,:,:] = phi[i,0:1,:,:] - phi_corr
                zest,_ = model(phi)# T[b,z] 
                loss = ( p['cost'][1](zgt*(1/model.k),zest*(1/model.k)) )# T[b]-> T[1]             
                #
                Zest[b,:] = zest.detach().cpu()
                Zgt[b,:] = zgt.detach().cpu()
            rmse_nn[b] = loss
            if int(p['cl'][0]):
                phis = {'phi_res':phi[-1,...].detach().cpu().numpy().squeeze(), 'phi_corr':phi_corr.detach().cpu().numpy().squeeze()} 
            else:
                phis = {'phi_res':np.zeros((model.pwfs.nPx,model.pwfs.nPx)), 'phi_corr':np.zeros((model.pwfs.nPx,model.pwfs.nPx))} 
    return torch.mean(rmse_nn),phis