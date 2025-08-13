####################################### LYBRARIES AND LAMBDA FUNCTIONS #############################################
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import scipy.io as scio
import time
import datetime
import os
import numpy as np
import argparse
import sys
from matplotlib.colors import Normalize
from mpmath import *

from model.dpwfs  import *
from functions._functions import *
from functions.utils import *
from functions.atmosphere import *
from functions.tv import *
from model.pyr import *
# dynamic class
dClass = type("dClass", (object,), {})
command = ' '.join(sys.argv)

# PARSE
parser = argparse.ArgumentParser(description='Settings, Training and Pyramid Wavefron Sensor parameters')
# main parameters
parser.add_argument('--D', default=3, type=float)
parser.add_argument('--nPx', default=128, type=int)
parser.add_argument('--modulation', default=0, type=int)
parser.add_argument('--samp', default=2, type=float)
#
# exp performance
parser.add_argument('--precision_name', default="single", type=str)
parser.add_argument('--routine', default='ns', type=str)
parser.add_argument('--device', default="cuda:0", type=str)
parser.add_argument('--expName', default="DEBUG", type=str)
parser.add_argument('--evol_save', default=1,type=int,help='save diffractive evolve on a gif')
wfs = parser.parse_args()

# DEVICE
wfs.wvl = 635
wfs.k = (2*np.pi)/wfs.wvl
wfs.fovPx = 2*wfs.samp*wfs.nPx
wfs.rooftop = 0
wfs.ps = 3.74e3
wfs.amp_cal = .1
wfs.device = torch.device(wfs.device if torch.cuda.is_available() else "cpu")
wfs.precision = get_precision(type=wfs.precision_name)# define the dtype for the experiment
wfs.fovPx = 2*wfs.samp*wfs.nPx
wfs.pupil = CreateTelescopePupil(wfs.nPx)
wfs.pupilLogical = wfs.pupil!=0
wfs.pid = os.getpid()

# ROUTINES
routine_lists = select_routine(wfs.routine)# extract a dictionary of a given routine to train

# LOG BEFORE TRAIN
current_date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
wfs.expName = f'{wfs.expName}-nPx{wfs.nPx}_{wfs.routine}_{wfs.precision_name}'
Log_path = f"./train/{wfs.precision_name}/{wfs.expName}"
os.makedirs(Log_path, exist_ok=True)
pars_log = {'D':wfs.D, 'nPx':wfs.nPx, 'samp':wfs.samp, 'rooftop':wfs.rooftop, 'fovPx':wfs.fovPx,'ampCal':wfs.amp_cal,
            'wvl':wfs.wvl, 'ps':wfs.ps, 'expName':wfs.expName, 'pid':wfs.pid, 'device':wfs.device,'precision name':wfs.precision_name,'evol save':wfs.evol_save,'command':command,'date':current_date_str}
Log(pars_log, routine_lists, path=Log_path, name=f'Log')

# MAIN TRAIN
print(f'TRAINING: device={wfs.device} | precision={wfs.precision_name} | expName={wfs.expName}')
for i,ro in enumerate(routine_lists, start=0):# rutines [{}]
    path_routine = Log_path + f'/routine_{i}'
    os.makedirs(path_routine, exist_ok=True)
    for j,p in enumerate(ro, start=0):# fine tunnings {}
        pars = wfs# copy of wfs into par
        path_train = path_routine + f'/train_{j}'
        os.makedirs(path_train, exist_ok=True)
        for key in p:
            if isinstance(p[key], list) and not(key=='vNoise'):
                p[key] = np.array( p[key] )
        # Assuming all these parameters to be equal from one fine tunning to another
        p['r0'] = np.round(pars.D/p['Dr0'][::-1], 2).astype(np.float64)
        pars.vNoise = p['vNoise']
        pars.zModes = p['zModes']
        pars.mInorm = p['mInorm']
        pars.init = p['init']
        pars.device = wfs.device
        pars.crop = p['crop']
        pars.norm_nn = p['norm_nn']
        pars.nRr = p['nRr']# nResolution respect
        #
        pars.zModes = torch.tensor(pars.zModes, dtype=pars.precision.real)
        pars.jModes = torch.arange(pars.zModes[0], pars.zModes[1]+1)
        pars.modes = torch.tensor(CreateZernikePolynomials1(wfs)).to(pars.device).to(pars.precision.real)
        #
        ab = ''.join(map(str, p['ab']))#p['ab']
        fp_str = "".join([str(item) for tup in p['fp'] for item in tup])
        freeze_mask = getTrainParam(p['fp'], p['epoch'])
        pars.fp = p['fp']
        # create folders
        checkpoint_path = path_train + f'/Checkpoint'
        os.makedirs(checkpoint_path, exist_ok=True)          
        # DATASET
        atm_path = f"./dataset/D{int(pars.D)}_R{int(pars.nPx)}_Dro{p['Dr0'][0]}-{p['Dr0'][1]}_Z{p['zModes'][-1]}_T{sum(p['nData'])}_αβ{ab}_{pars.precision_name}"
        if not os.path.exists(atm_path):
            os.makedirs(atm_path, exist_ok=True)
            getATM(pars, p,atm_path)
        if pars.evol_save:
            evol_path = f"{path_train}/evol"
            os.makedirs(evol_path, exist_ok=True)
        seed = 42  # You can choose any seed value

        dataset = importDataset(atm_path)
        train_dataset, test_dataset = random_split(dataset, p['nData'])
        train_data = DataLoader(train_dataset, batch_size=p['batch'], shuffle=True)
        val_data = DataLoader(test_dataset, batch_size=p['batch'], shuffle=False)
        # cost functions
        if p['cost'] in ['mse','rmse','mae']:
            cost = COST_modal(mode=p['cost'], dim=1).to(pars.device)
        elif p['cost'] in ['std','mad','var']:
            cost = COST_spatial(mode=p['cost'], dim=(-2,-1)).to(pars.device)
        #
        if p['fine tunning']:# direct fine tunning
            model = torch.load(p['fine tunning'], map_location=pars.device).to(pars.device)
            model.pwfs.vNoise = pars.vNoise
            model.device = pars.device
            print(model.device)
            lr,dlr = p['lr'],p['dlr']
            best_loss_v = float('Inf')
            epoch_check = 0
            print('Importing pretrained model')
        else:
            print('Training cero-model')
            if j == 0:# first training
                epoch_check = 0
                pars.nHead,pars.alpha = p['nHead'],p['alpha']
                best_loss_v = float('inf')
                lr,dlr = p['lr'],p['dlr']
                model = FWFS(pars, device=pars.device) 
                print(f'FIRST EXECUTION')
            else:
                finetuning_path = path_routine + f'/train_{j-1}/Checkpoint/checkpoint_best-v.pth'
                model = torch.load(finetuning_path, map_location=pars.device)
                lr = p['lr']
                dlr = p['dlr']
                best_loss_v = float('inf')
                epoch_check = 0
                print(f'FINE TUNNING {j}')
        # piston propagation
        with torch.no_grad():
            _,piston = model.eval()(model.pwfs.Piston)# T[b,1,NM]
            pos = np.sqrt(piston.shape[-1]).astype(np.int32)
            piston = piston.reshape( pos,pos ).detach().cpu()
            plt.imshow(piston)
            plt.axis('off')
            plt.colorbar()
            plt.savefig(path_train+f'/I0_e0.png', dpi=300, bbox_inches='tight')
            plt.close()
        # random zernike
        n = len(wfs.jModes)
        irand =  np.random.randint(0, n, n)
        amp = 2*torch.rand(n,dtype=pars.precision.real)-1
        input = torch.sum(amp*wfs.modes[:,irand].cpu(),dim=-1).reshape(1,1,pars.nPx,pars.nPx)
        Irand = model.pwfs(input.to(pars.device)).detach().cpu().squeeze()
        figs,axs = plt.subplots(1,2,figsize=(5*2,5))
        im0=axs[0].imshow(input.squeeze())
        im1=axs[1].imshow(Irand)
        axs[0].set_axis_off()
        axs[1].set_axis_off()
        plt.colorbar(im0,ax=axs[0])
        plt.colorbar(im1,ax=axs[1])
        plt.savefig(path_train+f'/Irand.png', dpi=300, bbox_inches='tight')
        plt.close()

    # TRAIN
        loss_t,loss_v = torch.zeros(p['epoch']),torch.zeros(p['epoch'])
        t0_total = time.time()
        tiempo = torch.zeros((p['epoch']))
        for e in range(epoch_check,epoch_check+p['epoch']):
            t0 = time.time()
            par = {'epoch':e, 'batch':p['batch'], 'lr':lr, 'dlr':dlr, 'cost':(p['cost'],cost), 'fp':freeze_mask[e], 'cl':p['cl'], 'vN':p['vNoise']}
            loss_t[e] = train(par, model.train(),train_data)    # loss
            loss_v[e],phis = validation(par, model.eval(),val_data) # error 
            # checkpoint
            if loss_v[e]<best_loss_v:
                best_loss_v = loss_v[e]
                torch.save(model, f'{checkpoint_path}/checkpoint_best-v.pth')
                if hasattr(model,'DE'):
                    DE_actual = model.state_dict()['DE'].detach().cpu()
                    plt.imshow(torch.fft.fftshift(DE_actual), vmin=-torch.pi,vmax=torch.pi,cmap='hsv')
                    plt.axis('off')
                    plt.savefig(path_train+f'/DE_actual.png', dpi=300, bbox_inches='tight')
                    plt.close()
                else:
                    plt.imshow(torch.fft.fftshift(model.DE_dummy.cpu()), vmin=-torch.pi,vmax=torch.pi,cmap='hsv')
                    plt.axis('off')
                    plt.savefig(path_train+f'/DE_actual.png', dpi=300, bbox_inches='tight')
                    plt.close()   
            if pars.evol_save:# de
                #torch.save(model, f'{evol_path}/checkpoint_e{e}.pth')
                with torch.no_grad():
                    _,piston = model.eval()(model.pwfs.Piston)# T[b,1,NM]
                    pos = np.sqrt(piston.shape[-1]).astype(np.int32)
                    piston = piston.reshape( pos,pos ).detach().cpu()
                    plt.imshow(piston)
                    plt.axis('off')
                    plt.colorbar()
                    plt.savefig(path_train+f'/I0.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    norm = Normalize(vmin=phis['phi_corr'].min(), vmax=phis['phi_corr'].max())
                    fig,ax = plt.subplots(1,2,figsize=(10,5))
                    ax[0].imshow( phis['phi_corr'], norm=norm )
                    ax[1].set_title(f'std(phi_res)={np.std(phis["phi_res"][pars.pupil]):.2f}')
                    ax[1].imshow( phis['phi_res'], norm=norm )
                    plt.savefig(path_train+f'/phis.png', dpi=300, bbox_inches='tight')
                    plt.close()
            # lr decreasing
            if (e+1)%5==0:
                lr = dlr*lr
                print('lr changed to: ' + ' '.join([format(value, '.3f') for value in lr]))
            t1 = time.time()
            tiempo[e] = t1-t0
            # images
            trainFigure(loss_t.detach(), loss_v.detach(), path_train)
            print(f'epoch finished={e} | loss_t={loss_t[e]:.3f} | loss_v={loss_v[e]:.3f}/{best_loss_v:.3f} | time={t1-t0:.3f}')
        t1_total = time.time()        
    
        
###################################################################