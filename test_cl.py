import torch
import scipy.io as scio
import time
import datetime
import os
import sys
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
import argparse
from tqdm import tqdm
from torch import unsqueeze as UNZ
from PIL import Image
from io import BytesIO
import imageio.v3 as iio
from model  import *
from functions._functions import *
from functions.utils import *
from functions.atmosphere import *
from model.functions_model import *
from model.dpwfs import *
import torchvision.transforms.functional as f

bar_format = "{l_bar}{bar}| {n}/{total} [{elapsed}<{remaining}, {rate_fmt}]"
command = ' '.join(sys.argv)

parser = argparse.ArgumentParser(description='Settings, Training and Pyramid Wavefront Sensor parameters')
# physical parameters
parser.add_argument('--D', default=3, type=float)# I think this does not affect
parser.add_argument('--nPx', default=128, type=int)
parser.add_argument('--samp', default=2, type=float)
parser.add_argument('--amp_cal', default=.1, type=float)
parser.add_argument('--vNoise',default=[[0.04, 0.55],0.,0.,1.],type=float,nargs='+')
parser.add_argument('--modulation', default=0, type=float)
#
parser.add_argument('--wvl', default=635, type=float)
parser.add_argument('--ps', default=3.74e3, type=float)
parser.add_argument('--rooftop', default=0, type=float)
# model configuration
parser.add_argument('--expName', default="paper_vNT0", type=str)
parser.add_argument('--device', default="cuda:2", type=str)
parser.add_argument('--precision_name', default="single", type=str)
# cl parameters
parser.add_argument('--Dr0v',default=[5],type=float,nargs='+') # 150,145,140,135,130,125,120,115,110,105,100,95,90,85,80,75,70,65,60,55,50,45,40,35,30,25,20,15,10,5,1
parser.add_argument('--kp', default=0.3, type=float)
parser.add_argument('--cl', default=50, type=int)
parser.add_argument('--t',default=300,type=int)
parser.add_argument('--layer',default=2,type=int)
parser.add_argument('--freq', default=200, type=int)
parser.add_argument('--seed', default=1, type=int)# 1=pulpos
parser.add_argument('--psf_samp', default=2, type=int)
parser.add_argument('--verbose', default=1, type=int)
parser.add_argument('--cmap', default='hot', type=str)
parser.add_argument('--plot_grid', default=1, type=int)
parser.add_argument('--fps', default=20, type=int)

parser.add_argument('--ipsf', default=[100,300],nargs='+', type=int)# integration range
parser.add_argument('--wpsf', default=64, type=int)# width of crop for images

# variation of PWFS
parser.add_argument('--mInorm', default=1, type=float)
parser.add_argument('--zModes',default=[2,200],type=int,nargs='+')
parser.add_argument('--mod',default=[0],type=int,nargs='+')
parser.add_argument('--nHeads',default=[],type=int,nargs='+')
parser.add_argument('--nHead',default=4,type=float)
parser.add_argument('--alpha',default=3,type=float)
parser.add_argument('--crop',default=False,type=bool)
parser.add_argument('--routine',default='n4_vNM0',type=str)
#
wfs = parser.parse_args()

mods = ''.join(map(str, wfs.mod)) if len(wfs.mod)!=0 else ''
nhs = ''.join(map(str, wfs.nHeads))

color_dpwfs = [
    # Original colors (with 'orange' replaced by '#00bfff')
    '#d62728',  # red
    '#1f77b4',  # blue
    '#228B22',  # forest green
    '#00bfff',  # deep sky blue (replaces orange)
    '#9467bd',  # purple
    '#ff7f0e',  # orange
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    '#c49c94',  # tan
    '#dbdb8d',  # light olive
    # Additional distinct colors
    '#17becf',  # blue-cyan
    '#bcbd22',  # olive
    '#8c564b',  # brown
    '#aec7e8',  # light blue
    '#ffbb78',  # light orange
    '#98df8a',  # light green
    '#c5b0d5',  # light purple
    '#f7b6d2',  # pink
    '#c7c7c7',  # light gray
    '#9edae5',  # pale cyan
    '#393b79',  # dark navy
    '#637939',  # dark olive
    '#8c6d31',  # dark mustard
    '#843c39',  # dark red
    '#7b4173',  # plum
    '#17bebb',  # teal
    '#f29e4c',  # peach
    '#e71d36',  # crimson
]
#

routine = []   

if wfs.routine == 'nD50k_vNM0':
    names = ['$\Lambda_2$','$\Lambda_3$','$\Lambda_4$','$\emptyset$','$\Lambda_{4i}$','$\Lambda_{4f}$','Z','R']
    routine = [
        {'tag':'$\Lambda_2$-IM','checkpoint':'./train/single/n2_zN0-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'$\Lambda_3$-IM','checkpoint':'./train/single/n3_zN0-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'$\Lambda_4$-IM','checkpoint':'./train/single/n4_zN0-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'$\emptyset$-IM','checkpoint':'./train/single/n0_zN0-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'$\Lambda_4i$-IM','checkpoint':'./train/single/n4i_zN0-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'$\Lambda_4f$-IM','checkpoint':'./train/single/n4f01_zN0-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'$Z$-IM','checkpoint':'./train/single/z_zN0-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'$R$-IM','checkpoint':'./train/single/r_zN0-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'},

        {'tag':'P+$\Lambda_2$-IM','checkpoint':'./train/single/n2_zN0-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'P+$\Lambda_3$-IM','checkpoint':'./train/single/n3_zN0-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'P+$\Lambda_4$-IM','checkpoint':'./train/single/n4_zN0-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'P+$\emptyset$-IM','checkpoint':'./train/single/n0_zN0-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'$\Lambda_4i$-IM','checkpoint':'./train/single/n4i_zN0-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'P+$\Lambda_4f$-IM','checkpoint':'./train/single/n4f01_zN0-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'P+$Z$-IM','checkpoint':'./train/single/z_zN0-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'P+$R$-IM','checkpoint':'./train/single/r_zN0-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'},

        {'tag':'$\Lambda_2$-NN','checkpoint':'./train/single/n2_zN0-nPx128_ns_all_single/routine_1/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'$\Lambda_3$-NN','checkpoint':'./train/single/n3_zN0-nPx128_ns_all_single/routine_1/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'$\Lambda_4$-NN','checkpoint':'./train/single/n4_zN0-nPx128_ns_all_single/routine_1/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'$\emptyset$-NN','checkpoint':'./train/single/n0_zN0-nPx128_ns_all_single/routine_1/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'$\Lambda_4i$-NN','checkpoint':'./train/single/n4i_zN0-nPx128_ns_all_single/routine_1/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'$\Lambda_4f$-NN','checkpoint':'./train/single/n4f01_zN0-nPx128_ns_all_single/routine_1/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'$Z$-NN','checkpoint':'./train/single/z_zN0-nPx128_ns_all_single/routine_1/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'$R$-NN','checkpoint':'./train/single/r_zN0-nPx128_ns_all_single/routine_1/train_0/Checkpoint/checkpoint_best-v.pth'},

        {'tag':'P+$\Lambda_2$-NN','checkpoint':'./train/single/n2_zN0-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'P+$\Lambda_3$-NN','checkpoint':'./train/single/n3_zN0-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'P+$\Lambda_4$-NN','checkpoint':'./train/single/n4_zN0-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'P+$\emptyset$-NN','checkpoint':'./train/single/n0_zN0-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'P+$\Lambda_4i$-NN','checkpoint':'./train/single/n4i_zN0-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'P+$\Lambda_4f$-NN','checkpoint':'./train/single/n4f01_zN0-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'P+$Z$-NN','checkpoint':'./train/single/z_zN0-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'P+$R$-NN','checkpoint':'./train/single/r_zN0-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'},
    ]    
    ncol = 8




# DEVICE
current_date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
wfs.device = torch.device(wfs.device if torch.cuda.is_available() else "cpu")
wfs.precision = get_precision(type=wfs.precision_name)# define the dtype for the experiment

wfs.k = ( (2*np.pi)/wfs.wvl )
wfs.dx = wfs.D/wfs.nPx
wfs.fovPx = 2*wfs.samp*wfs.nPx
#
wfs.pupil = torch.tensor(CreateTelescopePupil(wfs.nPx))
wfs.pupilLogical = wfs.pupil!=0
wfs.jModes = torch.arange(wfs.zModes[0], wfs.zModes[1]+1)
wfs.modes = torch.tensor(CreateZernikePolynomials1(wfs)).to(wfs.device).to(wfs.precision.real)

wfs.Dr0v = np.array(wfs.Dr0v)
wfs.r0v = np.round(wfs.D/wfs.Dr0v,4)

## create folders
wfs.expName = wfs.expName + f'_{wfs.precision_name}'
main_path = f'./tests/CL/'
os.makedirs(main_path, exist_ok=True)
# EXPERIMENT PATH
vNs = ''.join(map(str, wfs.vNoise))
Dros = ''.join(map(str,wfs.Dr0v))
exp_path = main_path + f'/EXP/{wfs.expName}-seed{wfs.seed}-R{wfs.routine}_M{mods}_Mo{len(routine)}_Dros{Dros}_kp{wfs.kp}_nZ{wfs.zModes[-1]}_vN{vNs}'
os.makedirs(exp_path, exist_ok=True)
# verbose
if wfs.verbose: 
    verbose_path = exp_path + f'/verbose'
    os.makedirs(verbose_path,exist_ok=True)
# I0 
I0_path = exp_path + f'/I0'
os.makedirs(I0_path,exist_ok=True)

# CLOSED-LOOP
print( f'CLOSED LOOP: nH={wfs.nHead} | dvc={wfs.device} | t={wfs.t} | ms={len(routine)} | Dr0={wfs.Dr0v} | M={wfs.mod}')

## Import evolving ATM
if wfs.seed == 1:
    atm_path = f'./Experiment/atm_pulpos/Phase_560px_Dr0_30_part_1.mat'
else:
    atm_path = f'./atmev/{wfs.seed}/Phase_128Npx_Dr0_30_frq_{wfs.freq}_seed_{wfs.seed}.mat'     
atm_file = scio.loadmat(atm_path)
PHI_atm = torch.tensor(atm_file['X_phase']).permute(2,0,1).unsqueeze(1).to(wfs.precision.real)
PHI_atm = f.resize(PHI_atm,(wfs.nPx,wfs.nPx), antialias=True)

# for pupil compact support
modes = wfs.modes.to('cpu').to(wfs.precision.real)# T[NM,nZ]
pmodes = torch.linalg.pinv(modes)
pupilL = (wfs.pupilLogical & (PHI_atm[0,0,...]!=0))
pupil = pupilL.to(wfs.precision.real)
pupilLV = pupilL.flatten()
PHI_atm *= pupil 
# compute SNR
model = FWFS(wfs, device=wfs.device).eval()
_,Iref = model( model.pwfs.Piston, vNoise=[0,0,0,1]) 
_,Inoise = model( model.pwfs.Piston, vNoise=wfs.vNoise)
snr = (torch.std(Iref,dim=-1)/torch.std(Iref-Inoise, dim=-1 )).squeeze().cpu().numpy()

def get_psf(u1, scale=wfs.psf_samp):# 2 is the nyquist criterion
    assert not u1.is_complex(), 'iwfs must be real: the phasor is constructed internally'
    if u1.ndim==2: u1=u1.unsqueeze(0).unsqueeze(0)
    u1 = pupil*torch.exp(1j*u1)
    npv = (u1.shape[-1]*(scale-1))//2
    u1big = torch.nn.functional.pad(u1, (npv,npv,npv,npv), "constant", 0)
    psf = torch.fft.fftshift( torch.abs(torch.fft.fft2( u1big ))**2 )
    H,W = u1.shape[-2:]
    x,y = (psf.shape[-2]-H)//2,(psf.shape[-1]-W)//2
    psf = psf[...,x:x+H,y:y+W]
    return psf.real

id_psf = get_psf(pupil).squeeze()
norm = Normalize(vmin=torch.min(id_psf),vmax=torch.max(id_psf))

# log
pars_log = {'D':wfs.D, 'nPx':wfs.nPx, 'samp':wfs.samp, 'amp_cal':wfs.amp_cal, 'vNoise':wfs.vNoise, 'modulation':wfs.modulation, 'wvl':wfs.wvl, 
            'ps':wfs.ps, 'rooftop':wfs.rooftop, 'expName':wfs.expName, 'precision':wfs.precision_name, 'command':command, 'crop':wfs.crop,
            'atm_path':atm_path}
pars_atm = {'Dr0v':wfs.Dr0v, 'kp':wfs.kp, 'cl':wfs.cl, 't':wfs.t, 'layer':wfs.layer, 'freq':wfs.freq, 'ipsf':wfs.ipsf, 'wpsf':wfs.wpsf, 'seed':wfs.seed}
pars_test = {'mInorm':wfs.mInorm, 'zModes':wfs.zModes, 'mod':wfs.mod, 'nHeads':wfs.nHeads, 'alpha':wfs.alpha, 'routine':wfs.routine, 'psf_sampling':wfs.psf_samp,
             'fps':wfs.fps, 'snr':snr, 'date':current_date_str}
pars = pars_log | pars_atm | pars_test# combine dictionary
Log(pars, [routine], path=exp_path, name=f'log_{datetime.date.today()}' )

# PWFS
if len(wfs.mod)!=0 and len(wfs.nHeads)!=0:
    nhs = ''.join(map(str, wfs.nHeads))
    pwfs_name = f'/pwfs_nhs{nhs}_M{mods}_T{wfs.t}_kp{wfs.kp}_cl{wfs.cl}_layer{wfs.layer}_freq{wfs.freq}'
    pwfs_path = exp_path + pwfs_name + '.npz'
    if not os.path.exists(pwfs_path):
        total_pwfs = wfs.t*len(wfs.mod)*len(wfs.r0v)*len(wfs.nHeads)
        # phasemaps
        ol_std = torch.zeros( (len(wfs.r0v),wfs.t),dtype=wfs.precision.real).to('cpu')
        phi_res_gt = torch.zeros( (len(wfs.r0v),wfs.t,wfs.nPx,wfs.nPx),dtype=wfs.precision.real).to('cpu')
        raw_pwfs_phi_res = torch.zeros((len(wfs.r0v),len(wfs.mod),len(wfs.nHeads),wfs.t,wfs.nPx**2),dtype=wfs.precision.real, device='cpu') 
        psf_pwfs = torch.zeros( (len(wfs.r0v),len(wfs.mod),len(wfs.nHeads),wfs.t,wfs.nPx,wfs.nPx), dtype=wfs.precision.real,device='cpu' )
        psf_gt = torch.zeros( (len(wfs.r0v),wfs.t,wfs.nPx,wfs.nPx), dtype=wfs.precision.real,device='cpu' )
        with tqdm(total=total_pwfs, desc='', bar_format=bar_format) as pbar_pwfs:
            for id_r,r in enumerate(wfs.r0v):
                PHI = ((wfs.Dr0v[id_r]/30.)**(5/6)) * PHI_atm
                for id_m,m in enumerate(wfs.mod):
                    model.pwfs.modulation = m
                    for id_nh,nh in enumerate(wfs.nHeads):
                        phi_res = torch.zeros( (1,1,wfs.nPx,wfs.nPx),dtype=wfs.precision.real, device='cpu')
                        phi_corr = torch.zeros( (1,1,wfs.nPx,wfs.nPx),dtype=wfs.precision.real, device='cpu')
                        z_corr = torch.zeros((1,len(wfs.jModes)),dtype=wfs.precision.real, device='cpu')
                        z_corr_gt = torch.zeros((1,len(wfs.jModes)),dtype=wfs.precision.real, device='cpu')
                        z_est = torch.zeros((1,len(wfs.jModes)),dtype=wfs.precision.real, device='cpu')
                        z_est_gt = torch.zeros((1,len(wfs.jModes)),dtype=wfs.precision.real, device='cpu')
                        #
                        model.pwfs.nHead = nh
                        _,piston = model(model.pwfs.Piston, vNoise=wfs.vNoise)
                        pos = np.sqrt(piston.shape[-1]).astype(np.int32)
                        piston = piston.reshape( pos,pos ).detach().cpu()
                        plt.imshow(piston.cpu().squeeze())
                        plt.colorbar()
                        plt.savefig(I0_path+f'/I0_pwfs_m{m}_nh{nh}.png', dpi=90, bbox_inches='tight')
                        plt.close()
                        I0_v,PIMat = model.pwfs.Calibration(mInorm=wfs.mInorm,vNoise = wfs.vNoise)
                        # ATMOSPHERE
                        for t in range(wfs.t):                                             
                            pbar_pwfs.set_description(f'Testing PWFS: Dro={wfs.Dr0v[id_r]} | m={m} | nh={nh} | t={t}')
                            z_corr = z_corr+z_est if t>=wfs.cl else z_corr # T[1,z]
                            phi_corr = torch.reshape( modes@(wfs.kp*z_corr[0,:]), (1,1,wfs.nPx,wfs.nPx) )*pupil
                            phi_res[0:1,0:1,:,:] = PHI[t:t+1,0:1,:,:] - phi_corr
                            if id_nh==0:
                                ol_std[id_r,t] = torch.std((PHI[t,0,...].flatten())[pupilLV], dim=-1 )# T[]
                                z_corr_gt = z_corr_gt + z_est_gt if t>=wfs.cl else z_corr_gt
                                phi_res_gt[id_r,t,...] = PHI[t,0,...] - torch.reshape((modes@(wfs.kp*z_corr_gt[0,:])),(wfs.nPx,wfs.nPx))*pupil# T[NM]
                                psf_gt[id_r,t,...] = get_psf( phi_res_gt[id_r,t,:,:] ).squeeze()
                                z_est_gt = (pmodes@(phi_res_gt[id_r,t,...].flatten())).unsqueeze(0)# T[1,Z]
                            # forward
                            I = model.pwfs.forward(phi_res.to(wfs.device), vNoise=wfs.vNoise)
                            mI_v = model.pwfs.mI(I[:,0,model.pwfs.idx],I0_v,dim=1,mInorm=wfs.mInorm)
                            z_est = ( ( PIMat @ mI_v.t() ).t() ).cpu()
                            # psf
                            psf = get_psf( phi_res.squeeze() ).squeeze()
                            if t%40==0 and wfs.verbose: 
                                plt.imshow( psf, cmap=wfs.cmap, norm=norm )  
                                plt.title(f'psf: nH={nh} | M={m} | t={t} | SR={torch.max(psf)/torch.max(id_psf):.3f}')
                                plt.axis('off')
                                plt.savefig( verbose_path + f'/psf_Dro{int(wfs.Dr0v[id_r])}_M{m}_t{t}.png', dpi=90, bbox_inches='tight')
                                plt.close()                          
                            # metrics
                            raw_pwfs_phi_res[id_r,id_m,id_nh,t,:] = phi_res.flatten()
                            psf_pwfs[id_r,id_m,id_nh,t,...] = psf
                            #
                            pbar_pwfs.update(1)
        np.savez(pwfs_path, raw_pwfs_phi_res=raw_pwfs_phi_res.numpy(), psf_pwfs=psf_pwfs.numpy(),psf_gt=psf_gt.numpy(),
                 ol_std=ol_std.numpy(),phi_res_gt=phi_res_gt.numpy())
    else:
        print(f'Importing {pwfs_name} PWFS')
        pwfs_file = np.load(pwfs_path)
        raw_pwfs_phi_res = torch.from_numpy(pwfs_file['raw_pwfs_phi_res'])# T[r0,m,nh,t,NM]
        psf_pwfs = torch.from_numpy(pwfs_file['psf_pwfs'])# T[r0,m,nh,t,N,M]
        ol_std = torch.from_numpy(pwfs_file['ol_std'])# T[r,t]
        phi_res_gt = torch.from_numpy(pwfs_file['phi_res_gt'])# T[r,t,N,M]
        psf_gt = torch.from_numpy(pwfs_file['psf_gt'])
    # computation of metrics
    pwfs_std = torch.std(raw_pwfs_phi_res[:,:,:,:,pupilLV].to(dtype=wfs.precision.real), dim=-1)# T[ro,m,nh,t]
    gt_std = torch.std(phi_res_gt[:,:,pupilL], dim=-1)# T[r,t]
    # NORMALIZATION
    psfn_pwfs = (psf_pwfs/torch.sum(psf_pwfs,dim=(-2,-1),keepdim=True))*(torch.sum(id_psf)/torch.max(id_psf))# T[ro,m,nh,t,N,M]
    ipsf_pwfs = torch.mean( psfn_pwfs[:,:,wfs.ipsf[0]:wfs.ipsf[-1],], dim=-3 )# T[ro,m,nh,N,M]
    sr_pwfs = torch.amax(psfn_pwfs,dim=(-2,-1))# T[ro,m,nh,t]
    psfn_gt = (psf_gt/torch.sum(psf_gt,dim=(-2,-1),keepdim=True))*(torch.sum(id_psf)/torch.max(id_psf))
    ipsf_gt = torch.mean(psfn_gt[:,wfs.ipsf[0]:wfs.ipsf[-1],:,:], dim=-3)## T[r0,N,M] 
    sr_gt = torch.amax(psfn_gt,dim=(-2,-1))# T[r0,t]
else:
    print('Running without PWFS')
    pwfs_std = []
del model

# DPWFS
if len(routine)!=0:
    dpwfs_name = f'/dpwfs_Mo{len(routine)}_T{wfs.t}_kp{wfs.kp}_cl{wfs.cl}_layer{wfs.layer}_freq{wfs.freq}'
    dpwfs_path = exp_path + dpwfs_name + '.npz'
    # test
    if not os.path.exists(dpwfs_path):
        total_dpwfs = len(routine)*len(wfs.r0v)*(wfs.t)
        ol_std = torch.zeros( (len(wfs.r0v),wfs.t),dtype=wfs.precision.real).to('cpu')
        phi_res_gt = torch.zeros( (len(wfs.r0v),wfs.t,wfs.nPx,wfs.nPx),dtype=wfs.precision.real).to('cpu')
        # phasemaps
        raw_dpwfs_phi_res = torch.zeros((len(wfs.r0v),len(routine),wfs.t,wfs.nPx**2), dtype=wfs.precision.real, device='cpu')  
        psf_dpwfs = torch.zeros( (len(wfs.r0v),len(routine),wfs.t,wfs.nPx,wfs.nPx), dtype=wfs.precision.real,device='cpu' )
        psf_gt = torch.zeros( (len(wfs.r0v),wfs.t,wfs.nPx,wfs.nPx), dtype=wfs.precision.real,device='cpu' )
        with tqdm(total=total_dpwfs, desc='', bar_format=bar_format) as pbar_dpwfs: # Barra
            for id_r,r in enumerate(wfs.r0v):
                PHI = ((wfs.Dr0v[id_r]/30.)**(5/6)) * PHI_atm
                with torch.no_grad():
                    for id_mo,mo in enumerate(routine):
                        phi_res = torch.zeros( (1,1,wfs.nPx,wfs.nPx),dtype=wfs.precision.real, device='cpu')
                        phi_corr = torch.zeros( (1,1,wfs.nPx,wfs.nPx),dtype=wfs.precision.real, device='cpu' )
                        z_corr = torch.zeros((1,len(wfs.jModes)),dtype=wfs.precision.real, device='cpu')
                        z_corr_gt = torch.zeros( (1,len(wfs.jModes)),dtype=wfs.precision.real, device='cpu' )
                        z_est = torch.zeros( (1,len(wfs.jModes)),dtype=wfs.precision.real, device='cpu' )
                        z_est_gt = torch.zeros( (1,len(wfs.jModes)),dtype=wfs.precision.real, device='cpu' )
                        #
                        model = torch.load(mo['checkpoint'], map_location=wfs.device).eval()
                        if 0<=id_mo<ncol:   del model.DE
                        if 0<=id_mo<2*ncol:
                            de = model.DE if hasattr(model,'DE') else model.DE_dummy
                            fourier_mask = UNZ(UNZ( torch.exp(1j*de) ,0),0)*model.pwfs.fourier_mask
                            I0_v,PIMat = model.pwfs.Calibration(mInorm=1,fourier_mask=fourier_mask,vNoise=[0.,0.,0.,1.])
                        #
                        _,piston = model(model.pwfs.Piston, vNoise=wfs.vNoise)
                        pos = np.sqrt(piston.shape[-1]).astype(np.int32)
                        piston = piston.reshape( pos,pos ).detach().cpu()
                        plt.imshow( piston.cpu().squeeze(), cmap=wfs.cmap)
                        plt.colorbar()
                        plt.savefig(I0_path+f'/I0_mo{id_mo}.png', dpi=90, bbox_inches='tight')
                        plt.close()
                        for t in range(wfs.t):
                            pbar_dpwfs.set_description(f'Testing DPWFS: Dro={wfs.Dr0v[id_r]} | model={mo["tag"]} | id_mo={id_mo} | t={t}')
                            z_corr = z_corr+z_est if t>=wfs.cl else z_corr # T[1,z]
                            phi_corr = torch.reshape( modes@(wfs.kp*z_corr[0,:]), (1,1,model.pwfs.nPx,model.pwfs.nPx) )*pupil
                            phi_res[0:1,0:1,:,:] = PHI[t:t+1,0:1,:,:] - phi_corr
                            # forward
                            if id_mo==0:
                                ol_std[id_r,t] = torch.std((PHI[t,0,...].flatten())[pupilLV], dim=-1 )# T[]
                                z_corr_gt = z_corr_gt + z_est_gt if t>=wfs.cl else z_corr_gt
                                phi_res_gt[id_r,t,...] = PHI[t,0,...] - torch.reshape((modes@(wfs.kp*z_corr_gt[0,:])),(wfs.nPx,wfs.nPx))*pupil# T[NM]
                                psf_gt[id_r,t,...] = get_psf( phi_res_gt[id_r,t,:,:] ).squeeze()
                                z_est_gt = (pmodes@(phi_res_gt[id_r,t,...].flatten())).unsqueeze(0)# T[1,Z]
                            if 0<=id_mo<2*ncol:# IM reconstruction
                                I = model.pwfs.forward(phi_res.detach().to(wfs.device), fourier_mask=fourier_mask, vNoise=wfs.vNoise)
                                mI_v = model.pwfs.mI(I[:,0,model.pwfs.idx],I0_v,dim=1,mInorm=wfs.mInorm)
                                zest = ( ( PIMat @ mI_v.t() ).t() ).cpu()
                            else:
                                zest,_ = model(phi_res.detach().to(wfs.device), vNoise=wfs.vNoise)
                            z_est = zest[:,:wfs.zModes[-1]-1].detach().cpu()
                            # psf
                            psf = get_psf( phi_res.squeeze() ).squeeze()
                            if t%40==0 and wfs.verbose: 
                                plt.imshow( psf, cmap=wfs.cmap, norm=norm)
                                plt.title(f'PSF: mo={mo["tag"]} | t={t} | SR={torch.max(psf)/torch.max(id_psf):.3f}')
                                plt.axis('off')
                                plt.savefig( verbose_path + f'/psf_Dro{int(wfs.Dr0v[id_r])}_mo{id_mo}_t{t}.png', dpi=90, bbox_inches='tight')
                                plt.close()   
                            # metrics
                            raw_dpwfs_phi_res[id_r,id_mo,t,:] = phi_res.flatten()
                            psf_dpwfs[id_r,id_mo,t,...] = psf
                            pbar_dpwfs.update(1)
                        del model
        np.savez(dpwfs_path, raw_dpwfs_phi_res=raw_dpwfs_phi_res.numpy(), psf_dpwfs=psf_dpwfs.numpy(),psf_gt=psf_gt.numpy(),
                    ol_std=ol_std.numpy(),phi_res_gt=phi_res_gt.numpy())
    else:
        print(f'Importing {dpwfs_name} DPWFS')
        dpwfs_file = np.load(dpwfs_path)
        raw_dpwfs_phi_res = torch.from_numpy(dpwfs_file['raw_dpwfs_phi_res'])# T[r0,mos,t,NM]
        psf_dpwfs = torch.from_numpy(dpwfs_file['psf_dpwfs'])# T[r0,mos,t,N,M]
        ol_std = torch.from_numpy(dpwfs_file['ol_std'])# T[r,t]
        phi_res_gt = torch.from_numpy(dpwfs_file['phi_res_gt'])# T[r0,t,N,M]
        psf_gt = torch.from_numpy(dpwfs_file['psf_gt'])# T[r0,t,N,M]
    # computation of metrics
    dpwfs_std = torch.std(raw_dpwfs_phi_res[:,:,:,pupilLV], dim=-1)# T[r0,mos,t]
    gt_std = torch.std(phi_res_gt[:,:,pupilL], dim=-1)# T[r,t]
    # NORMALIZATION
    psfn_dpwfs = (psf_dpwfs/torch.sum(psf_dpwfs,dim=(-2,-1),keepdim=True))*(torch.sum(id_psf)/torch.max(id_psf))# T[r0,mos,t,N,M]
    ipsf_dpwfs = torch.mean( psfn_dpwfs[:,:,wfs.ipsf[0]:wfs.ipsf[-1],:,:], dim=-3)# T[ro,mo,N,M] 
    sr_dpwfs = torch.amax(psfn_dpwfs,dim=(-2,-1))# T[r0,mos,t]
    psfn_gt = (psf_gt/torch.sum(psf_gt,dim=(-2,-1),keepdim=True))*(torch.sum(id_psf)/torch.max(id_psf))# T[r0,t,N,M]
    ipsf_gt = torch.mean(psfn_gt[:,wfs.ipsf[0]:wfs.ipsf[-1],:,:], dim=-3)## T[r0,N,M] 
    sr_gt = torch.amax(psfn_gt,dim=(-2,-1))# T[r0,t]
else:
    print('Running without DPWFS')
    dpwfs_std = []


######################################### PLOT ##################################   
fs = 16
ls = 1
# PLOT STD
x = np.arange(1,wfs.t+1)#np.arange(100,199+1)
for id_r,r in enumerate(wfs.r0v):
    plt.plot(x, ol_std[id_r,:], label='OL', color='gray', linestyle='--', linewidth=ls)
    ylim = ol_std[id_r,:].max()+2
    plt.plot(x, gt_std[id_r,:], label='gt', linewidth=ls,linestyle='-',color='black')
    for id_m,m in enumerate(wfs.mod):
        for id_nh,nh in enumerate(wfs.nHeads):
            plt.plot(x, pwfs_std[id_r,id_m,id_nh,:], label=fr'$\Lambda_{nh}$-m{m}-MVM',linestyle ='--', linewidth=ls)
    for id_mo,mo in enumerate(routine):
        lstyle_dpwfs = '--' if 0<=id_mo<ncol else '-'
        plt.plot(x, dpwfs_std[id_r,id_mo,:], label=fr'{mo["tag"]}', linewidth=ls,linestyle=lstyle_dpwfs)#,color=color_dpwfs[id_mo])
    # settings
    plt.rc('font', size=fs)
    plt.tick_params(axis="x",labelsize=fs)
    plt.tick_params(axis="y",labelsize=fs)
    plt.legend(fontsize=fs-4, loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.ylim(0,ylim)
    plt.xlabel('Iterations',fontsize=fs)
    plt.ylabel(r'$\sigma(\phi_{\mathrm{res}})\quad \mathrm{(rad)}$', fontsize=fs)
    plt.grid(True)
    plt.savefig(exp_path + f'/plot-CL_Dro{int(wfs.Dr0v[id_r])}_nZ{wfs.zModes[-1]}_t{wfs.t}_M{mods}_Mo{len(routine)}.png', dpi=300, bbox_inches='tight')
    plt.close()

# PLOT SR
for id_r,r in enumerate(wfs.r0v):
    plt.plot(x, sr_gt[id_r,:], label='gt', color='gray', linestyle='--', linewidth=ls)
    for id_m,m in enumerate(wfs.mod):
        for id_nh,nh in enumerate(wfs.nHeads):
            plt.plot(x, sr_pwfs[id_r,id_m,id_nh,:], label=fr'$\Lambda_{nh}$-m{m}-MVM',linestyle ='--', linewidth=ls)
    for id_mo,mo in enumerate(routine):
        lstyle_dpwfs = '--' if 0<=id_mo<ncol else '-'
        plt.plot(x, sr_dpwfs[id_r,id_mo,:], label=fr'{mo["tag"]}', linewidth=ls,linestyle=lstyle_dpwfs)#,color=color_dpwfs[id_mo])
    # settings
    plt.rc('font', size=fs)
    plt.tick_params(axis="x",labelsize=fs)
    plt.tick_params(axis="y",labelsize=fs)
    plt.legend(fontsize=fs-4, loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.ylim(0,1)
    plt.xlabel('Iterations',fontsize=fs)
    plt.ylabel(r'SR (-)', fontsize=fs)
    plt.grid(True)
    plt.savefig(exp_path + f'/plot-CL_sr_Dro{int(wfs.Dr0v[id_r])}_nZ{wfs.zModes[-1]}_t{wfs.t}_M{mods}_Mo{len(routine)}.png', dpi=300, bbox_inches='tight')
    plt.close()



if wfs.plot_grid:   
    psf_path = exp_path + f'/psf'
    os.makedirs(psf_path,exist_ok=True)
    psf_pwfs_path = psf_path + f'/pwfs'
    os.makedirs(psf_pwfs_path,exist_ok=True)
    psf_dpwfs_path = psf_path + f'/dpwfs'
    os.makedirs(psf_dpwfs_path,exist_ok=True)
    nrow = len(routine)//ncol
    sl = slice(wfs.nPx//2-wfs.wpsf,wfs.nPx//2+wfs.wpsf)
    norm = Normalize(vmin=0, vmax=1)
  
    
    # TABLE AND IMAGE GRID PLOT
    header_sr = [fr'$D/r_0$'] + ['Versions'] + [f'{mo}' for mo in names]
    rows_sr = []
    row_sr = []
    header_std = [fr'$D/r_0$'] + ['Versions'] + ['gt'] + [f'{mo}' for mo in names]
    rows_std = []
    row_std = []
    std_gt = torch.mean(gt_std[:,wfs.ipsf[0]:wfs.ipsf[-1]],dim=-1)# T[ro]
    std_mo = torch.mean(dpwfs_std[:,:,wfs.ipsf[0]:wfs.ipsf[-1]],dim=-1)# T[ro,mo]
    label_map = ['IM', 'P-IM', 'NN', 'P-NN']
    fs = 30
    dpi = 300
    fontfamily = 'monospace'#'DejaVu Serif'
    # GRID PLOTS
    for id_r,r in enumerate(wfs.r0v):
        fig_psf,ax_psf = plt.subplots(nrow+1,ncol+1,figsize=(4*(ncol+1),4*(nrow+1)))#, gridspec_kw={'width_ratios': [0.6, 1.0] + [1.0]*(ncol)})
        fig_psfp,ax_psfp = plt.subplots(nrow,ncol+1, figsize=(4*(ncol+1),4*nrow))
        # ideal PSF
        ipsf_gt_crop = ipsf_gt[id_r, sl,sl]# T[N,M]
        sr_gt = torch.max(ipsf_gt[id_r,...])#sr_gt = torch.exp(-std_gt[id_r]**2)
        for id_mo,mo in enumerate(routine):#
            row_psf = id_mo//ncol+1
            col_psf = id_mo%ncol+1  # +1 because ideal image is in column 0
            if col_psf==1:  # Only set once per row
                # VERSION OF MODELS
                ax_psf[row_psf,0].axis('off')
                label = label_map[row_psf-1] if row_psf <(len(label_map)+1) else ''
                ax_psf[row_psf,0].text(0.5, 0.5, label,ha='center', va='center',fontsize=40,fontfamily=fontfamily,rotation=90)
            if row_psf==1:# put models
                ax_psf[0,0].axis('off')
                for i,nms in enumerate(names):
                    ax_psf[0,i+1].text(0.5, 0.5, f'{nms}',ha='center', va='center',fontsize=40,fontfamily=fontfamily)
                    ax_psf[0,i+1].axis('off')
            sr_model = torch.max(ipsf_dpwfs[id_r,id_mo,...])
            psf_model = ipsf_dpwfs[id_r,id_mo, sl,sl]
            ax_psf[row_psf,col_psf].imshow((psf_model), cmap=wfs.cmap)#, norm=norm)
            ax_psf[row_psf,col_psf].set_title(f'SR={sr_model*100:.1f}', fontsize=fs)#,fontfamily='Times New Roman')
            ax_psf[row_psf,col_psf].axis('off')
            #----------------------
            # gt
            max_idx = torch.argmax(ipsf_gt_crop)
            row,col = divmod( max_idx.item(), ipsf_gt_crop.size(1) )
            ax_psfp[id_mo//ncol,0].plot( (ipsf_gt_crop[row,:]) )
            ax_psfp[id_mo//ncol,0].set_title(f'Ideal | SR={sr_gt*100:.1f}')
            # dpwfs
            ipsf = ipsf_dpwfs[id_r,id_mo, sl,sl]
            sr = torch.max(ipsf_dpwfs[id_r,id_mo,...])#.clamp_(min=0,max=sr_gt)# T[]#sr = torch.exp(-std_mo[id_r,id_mo]**2)
            max_idx = torch.argmax(ipsf)
            row,col = divmod( max_idx.item(), ipsf.size(1) )
            #
            ax_psfp[id_mo//ncol,id_mo%ncol+1].plot( (ipsf[row,:]) )
            ax_psfp[id_mo//ncol,id_mo%ncol+1].set_title(f'{mo["tag"]} | SR={sr:.3f}')
            #
            row_sr.append( f'{sr*100:.1f}')
            row_std.append( f'{std_mo[id_r,id_mo]:.3f}' )
            if (id_mo+1)%ncol==0: # next row
                if 0<=id_mo<ncol:# MVMD
                    rows_sr.append( [str((wfs.Dr0v[id_r]))] + ['IM'] + row_sr )
                    rows_std.append( [str((wfs.Dr0v[id_r]))] + ['IM'] + [f'{std_gt[id_r]:.3f}'] + row_std )
                if ncol<=id_mo<2*ncol:# D
                    rows_sr.append( [str((wfs.Dr0v[id_r]))] + ['P'] + row_sr )
                    rows_std.append( [str((wfs.Dr0v[id_r]))] + ['P'] + [f'{std_gt[id_r]:.3f}'] + row_std )
                elif 2*ncol<=id_mo<3*ncol:# NN
                    rows_sr.append( [str((wfs.Dr0v[id_r]))] + ['NN'] + row_sr )
                    rows_std.append( [str((wfs.Dr0v[id_r]))] + ['NN'] + [f'{std_gt[id_r]:.3f}'] + row_std )
                elif 3*ncol<=id_mo<4*ncol:# DNN
                    rows_sr.append( [str((wfs.Dr0v[id_r]))] + ['P-NN'] + row_sr )
                    rows_std.append( [str((wfs.Dr0v[id_r]))] + ['P-NN'] + [f'{std_gt[id_r]:.3f}'] + row_std )
                row_sr = []
                row_std = []
        if wfs.Dr0v[id_r]==7.5:
            fig_psf.suptitle( fr"$D/r_0={wfs.Dr0v[id_r]:.1f}$", fontsize=40,x=0.5, y=0.9)
        else:
            fig_psf.suptitle( fr"$D/r_0={int(wfs.Dr0v[id_r])}$", fontsize=40,x=0.5, y=0.9)
        fig_psf.savefig( exp_path + f'/grid_psf_dpwfs_Dro{int(wfs.Dr0v[id_r])}.png', dpi=dpi,bbox_inches='tight' )
        fig_psf.savefig( exp_path + f'/grid_psf_dpwfs_Dro{int(wfs.Dr0v[id_r])}.pdf', dpi=dpi,bbox_inches='tight' )
        plt.close(fig_psf)
        fig_psfp.suptitle( f"Dro={(wfs.Dr0v[id_r])}", fontsize=40)
        fig_psfp.savefig( exp_path + f'/grid_psfp_dpwfs_Dro{int(wfs.Dr0v[id_r])}.png', dpi=dpi,bbox_inches='tight' )
        plt.close(fig_psfp)

    # sr table
    fig,ax = plt.subplots()
    ax.axis('off')
    table = ax.table(
        cellText=rows_sr,                # 2-D list of cell values
        colLabels=header_sr,            # header row
        loc='center',                 # centre inside the Axes
        cellLoc='center'              # centre-align all text
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    fig.savefig(exp_path+f'/sr_table.png',dpi=300,bbox_inches='tight')
    plt.close(fig)
    # std table
    fig,ax = plt.subplots()
    ax.axis('off')
    table = ax.table(
        cellText=rows_std,                # 2-D list of cell values
        colLabels=header_std,            # header row
        loc='center',                 # centre inside the Axes
        cellLoc='center'              # centre-align all text
    )
    fig.savefig(exp_path+f'/std_table.png',dpi=300,bbox_inches='tight')


    ## GIF------------------------------------------------------------
    imgs_psf = []
    dpi = 90
    transparency = False
    with tqdm(total=len(wfs.r0v)*wfs.t, desc='', bar_format=bar_format) as pbar_gif_psf:
        for id_r,r in enumerate(wfs.r0v):
            imgs_psf.clear()  # clear image list for each r0 case
            for t in range(wfs.t):
                pbar_gif_psf.set_description(f'Making gif psf: Dr0={wfs.Dr0v[id_r]} | t={t}')
                fig, ax = plt.subplots(nrow+1,ncol+1,figsize=(4*(ncol+1),4*(nrow+1)))
                sr_gt = torch.max(psfn_gt[id_r,t, ...])
                psf_gt_crop = psfn_gt[id_r,t, sl,sl]
                for id_mo,mo in enumerate(routine):
                    row = id_mo//ncol+1
                    col = id_mo%ncol+1  # +1 because ideal image is in column 0
                    if col==1:  # Only set once per row
                        # VERSION OF MODELS
                        ax[row,0].axis('off')
                        label = label_map[row-1] if row <(len(label_map)+1) else ''
                        ax[row,0].text(0.5, 0.5, label,ha='center', va='center',fontsize=40, fontfamily='monospace',rotation=90)
                        ####
                    # Integrated PSF
                    sr_model = torch.max(psfn_dpwfs[id_r,id_mo,t,...])
                    psf_model = psfn_dpwfs[id_r,id_mo,t, sl,sl]
                    ax[row,col].imshow((psf_model), cmap=wfs.cmap)#, norm=norm)
                    ax[row,col].set_title(f'SR={sr_model*100:.1f}', fontsize=fs)
                    ax[row,col].axis('off')
                    if row==1:# put models
                        ax[0,0].axis('off')
                        for i,nms in enumerate(names):
                            ax[0,i+1].text(0.5, 0.5, f'{nms}',ha='center', va='center',fontsize=40, fontfamily='monospace')
                            ax[0,i+1].axis('off')
                fig.suptitle(fr'$D/r_0$={int(wfs.Dr0v[id_r])} | Iteration={t}', fontsize=40,x=0.5,y=0.9)
                buf = BytesIO()
                plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0, transparent=transparency)
                plt.close(fig)
                buf.seek(0)
                img = Image.open(buf)
                time.sleep(1e-2)
                imgs_psf.append(np.array(img))
                buf.close()
                pbar_gif_psf.update(1)
        # save gif
            iio.imwrite(exp_path + f"/psf_Dro{int(wfs.Dr0v[id_r])}_t{wfs.t}_dpi{dpi}.gif",
                    imgs_psf, duration=(1/wfs.fps), loop=1)

    ## FIGURE FM,P,FMP,I-------------------------------------------------------------------------------------------------------
    fig,ax = plt.subplots(5,ncol, figsize=(5*(ncol),5*5))
    for id_mo,mo in enumerate(routine[(nrow-1)*ncol:]):# consider only the PNN
        model = torch.load(mo['checkpoint'],map_location=wfs.device).eval()
        # name of the model
        ax[0,id_mo].axis('off')
        ax[0,id_mo].text(0.5, .5, f'{names[id_mo]}', ha='center', va='center',fontsize=40,fontfamily=fontfamily)
        # fm
        fm = torch.fft.fftshift(torch.angle(model.pwfs.fourier_mask)).detach().cpu().squeeze()
        ax[1,id_mo].imshow(fm, vmin=-np.pi,vmax=np.pi, cmap='hsv')
        ax[1,id_mo].axis('off')
        # p
        p = torch.fft.fftshift(model.DE).detach().cpu()
        ax[2,id_mo].imshow(p, vmin=-np.pi,vmax=np.pi, cmap='hsv')
        ax[2,id_mo].axis('off')       
        # fmp
        fmp = fm+p
        ax[3,id_mo].imshow(fmp, vmin=-np.pi,vmax=np.pi, cmap='hsv')
        ax[3,id_mo].axis('off')  
        # I
        _,I = model(model.pwfs.Piston)
        I = I.reshape(wfs.nPx,wfs.nPx).detach().cpu()
        ax[4,id_mo].imshow(I, cmap='hot')
        ax[4,id_mo].axis('off')  
    fig.subplots_adjust(wspace=0.05, hspace=0.05) 
    fig.savefig( exp_path + f'/preconditioners.png', dpi=dpi,bbox_inches='tight', transparent=True, pad_inches=0 )
    fig.savefig( exp_path + f'/preconditioners.pdf', dpi=dpi,bbox_inches='tight', transparent=True, pad_inches=0 )
    plt.close(fig)




