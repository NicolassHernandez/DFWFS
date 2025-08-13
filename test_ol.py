import importlib
import torch.optim as optim
import torch.nn as nn
import torch
import scipy.io as scio
import time
import datetime
import os
import sys
import numpy as np
import argparse
from tqdm import tqdm
from torch import unsqueeze as UNZ
from torch import squeeze as NZ
from model  import *
from functions._functions import *
from functions.utils import *
from functions.atmosphere import *
from model.functions_model import *
from model.dpwfs import *
# dynamic class
dClass = type("dClass", (object,), {})
bar_format = "{l_bar}{bar}| {n}/{total} [{elapsed}<{remaining}, {rate_fmt}]"

command = ' '.join(sys.argv)


parser = argparse.ArgumentParser(description='Settings, Training and Pyramid Wavefront Sensor parameters')
# physical parameters
parser.add_argument('--D', default=3, type=float)
parser.add_argument('--nPx', default=128, type=int)
parser.add_argument('--samp', default=2, type=float)
parser.add_argument('--amp_cal', default=.1, type=float)
parser.add_argument('--vNoise',default=[0,0,0,1],type=float,nargs='+')
parser.add_argument('--modulation', default=0, type=float)
#
parser.add_argument('--wvl', default=635, type=float)
parser.add_argument('--ps', default=3.74e3, type=float)
parser.add_argument('--rooftop', default=0, type=float)
# noise
# model configuration
parser.add_argument('--expName', default="default", type=str)
parser.add_argument('--device', default="cuda:7", type=str)
parser.add_argument('--precision_name', default="single", type=str)
#
#[50,45,40,35,30,25,20,15,10,5,1] [20,15,10,8,6,4,2,1,.1,.01]
parser.add_argument('--Dr0v',default='normal',type=str) # 150,145,140,135,130,125,120,115,110,105,100,95,90,85,80,75,70,65,60,55,50,45,40,35,30,25,20,15,10,5,1
parser.add_argument('--T',default=1000,type=int)
parser.add_argument('--batch', default=10, type=int, help='Batch size for training')
parser.add_argument('--cost',default='rmse',type=str)
# variation of PWFS
parser.add_argument('--mInorm', default=1, type=float)
parser.add_argument('--zModes',default=[2,200],type=float,nargs='+')
parser.add_argument('--mod',default=[0],type=int,nargs='+')
parser.add_argument('--nHeads',default=[],type=int,nargs='+')
parser.add_argument('--nHead',default=4,type=float)
parser.add_argument('--alpha',default=3,type=float)
parser.add_argument('--crop',default=False,type=bool)
parser.add_argument('--routine',default='ns_nD50k_b50',type=str)
#
wfs = parser.parse_args()

mods = ''.join(map(str, wfs.mod)) if len(wfs.mod)!=0 else -1
nhs = ''.join(map(str, wfs.nHeads))

# from paper folder
# old replication of paper folder models: nPx-2, nRr0, nZ24


# FIG D)
if wfs.routine == 'P_vNM0':
    routine = [ 
        {'tag':'P+$\Lambda_2$—IM','checkpoint':'./MODELS/nD50k/b10/nZ200/n2_all_cte_nRr1_nD50k_b10_nZ200-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'},  
        {'tag':'P+$\Lambda_3$—IM','checkpoint':'./MODELS/nD50k/b10/nZ200/n3_all_cte_nRr1_nD50k_b10_nZ200-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'}, 
        {'tag':'P+$\Lambda_4$—IM','checkpoint':'./MODELS/nD50k/b10/nZ200/n4_all_cte_nRr1_nD50k_b10_nZ200-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'P+$\emptyset$—IM','checkpoint':'./MODELS/nD50k/b10/nZ200/n0_all_cte_nRr1_nD50k_b10_nZ200-nPx128_ns_all_single/routine_2/train_0/Checkpoint/checkpoint_best-v.pth'},
        ]
if wfs.routine == 'NN_vNM0':
    routine = [ 
        {'tag':'$\Lambda_2$—NN','checkpoint':'./MODELS/nD50k/b10/nZ200/n2_all_cte_nRr1_nD50k_b10_nZ200-nPx128_ns_all_single/routine_1/train_0/Checkpoint/checkpoint_best-v.pth'},  
        {'tag':'$\Lambda_3$—NN','checkpoint':'./MODELS/nD50k/b10/nZ200/n3_all_cte_nRr1_nD50k_b10_nZ200-nPx128_ns_all_single/routine_1/train_0/Checkpoint/checkpoint_best-v.pth'}, 
        {'tag':'$\Lambda_4$—NN','checkpoint':'./MODELS/nD50k/b10/nZ200/n4_all_cte_nRr1_nD50k_b10_nZ200-nPx128_ns_all_single/routine_1/train_0/Checkpoint/checkpoint_best-v.pth'},
        #{'tag':'D—NN','checkpoint':'./MODELS/nD50k/b10/nZ200/n0_all_cte_nRr1_nD50k_b10_nZ200-nPx128_ns_all_single/routine_1/train_0/Checkpoint/checkpoint_best-v.pth'},
        ]
if wfs.routine == 'PNN_vNM0':
    routine = [ 
        {'tag':'P+$\Lambda_2$—NN','checkpoint':'./MODELS/nD50k/b10/nZ200/n2_all_cte_nRr1_nD50k_b10_nZ200-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'},  
        {'tag':'P+$\Lambda_3$—NN','checkpoint':'./MODELS/nD50k/b10/nZ200/n3_all_cte_nRr1_nD50k_b10_nZ200-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'}, 
        {'tag':'P+$\Lambda_4$—NN','checkpoint':'./MODELS/nD50k/b10/nZ200/n4_all_cte_nRr1_nD50k_b10_nZ200-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'},
        {'tag':'P+$\emptyset$—NN','checkpoint':'./MODELS/nD50k/b10/nZ200/n0_all_cte_nRr1_nD50k_b10_nZ200-nPx128_ns_all_single/routine_0/train_0/Checkpoint/checkpoint_best-v.pth'},
        ]
    




# ROUTINE
color_dpwfs = [
    # Original colors (with 'orange' replaced by '#00bfff')
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#228B22',  # forest green
    '#d62728',  # brick red
    '#9467bd',  # purple
    '#8c564b',  # chestnut brown
    '#d62728',  # red
    '#1f77b4',  # blue
    '#00bfff',  # deep sky blue (replaces orange)
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
    '#ff9f1c',  # orange-yellow
    '#2ec4b6',  # turquoise
    '#011627',  # very dark blue
    '#ff6f61',  # coral
    '#6a0572',  # purple violet
]
color_pwfs = color_dpwfs
lstyle_pwfs = ['--','--','--','--','--','--']
lstyle_dpwfs = ['-','-','-','-','-','-','-','-','-','-','-','-']


# DEVICE
wfs.device = torch.device(wfs.device if torch.cuda.is_available() else "cpu")
wfs.precision = get_precision(type=wfs.precision_name)# define the dtype for the experiment

if wfs.Dr0v == 'zoomed':
    wfs.Dr0v = [20,15,10,8,6,4,2,1,.1,.01]
    xticks = wfs.Dr0v
    ylim = .3
if wfs.Dr0v == 'extzoomed':
    wfs.Dr0v = [10,9,8,7,6,5,4,3,2,1,.1]
    ylim = .05
elif wfs.Dr0v == 'normal':
    #wfs.Dr0v = np.arange(60,0,-1)
    wfs.Dr0v = [60,55,50,45,40,35,30,25,20,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1]
    xticks = [60,50,40,30,20,10,5,1]
    ylim = .3

wfs.k = ( (2*np.pi)/wfs.wvl )
wfs.dx = wfs.D/wfs.nPx
wfs.fovPx = 2*wfs.samp*wfs.nPx
#
wfs.pupil = CreateTelescopePupil(wfs.nPx)
wfs.pupilLogical = wfs.pupil!=0
pupil = torch.tensor(wfs.pupilLogical).to(wfs.device)
wfs.jModes = torch.arange(wfs.zModes[0], wfs.zModes[1]+1)
wfs.modes = torch.tensor(CreateZernikePolynomials1(wfs)).to(wfs.device).to(wfs.precision.real)

# TESTING PARAMETERS
wfs.Dr0v = np.array(wfs.Dr0v)
wfs.r0v = np.round(wfs.D/wfs.Dr0v,4)
if wfs.T%wfs.batch!=0:
    raise KeyError('Please select a batch that divide the T in integer parts')

# cost function
cost = COST_modal(mode=wfs.cost, dim=1).to(wfs.device)# T[b,z]

## create folders
wfs.expName = wfs.expName + f'_{wfs.precision_name}'
main_path = f'./tests/OL/'
os.makedirs(main_path, exist_ok=True)
# EXPERIMENT PATH
vNs = ''.join(map(str, wfs.vNoise))
exp_path = main_path + f'/EXP/{wfs.expName}_R{wfs.routine}_M{mods}_Mo{len(routine)}_T{wfs.T}_Dro{wfs.Dr0v[0]}-{wfs.Dr0v[-1]}_nZ{wfs.zModes[-1]}_vN{vNs}'
os.makedirs(exp_path, exist_ok=True)

# OPEN LOOP TEST 
print( f'RMSE SWEEP: nH={wfs.nHead} | dvc={wfs.device} | T={wfs.T} | b={wfs.batch} | ms={len(routine)} | Dr0={wfs.Dr0v} | M={wfs.mod}')

## Generating ATM
atm_name = f'/atm_Dro{wfs.Dr0v[0]}-{wfs.Dr0v[-1]}_Z{wfs.zModes[-1]}_T{wfs.T}_{wfs.precision_name}'
atm_path = exp_path + atm_name + '.npz'
if not os.path.exists(atm_path):
    PHI,ZGT = ATM(wfs,r0v=wfs.r0v,nData=wfs.T)
    np.savez(atm_path, Phi=PHI.cpu().numpy(), Zgt=ZGT.cpu().numpy())
else:
    print(f'Importing {atm_name} ATM')
    dataset = np.load(atm_path)
    PHI = torch.from_numpy(dataset['Phi'])
    ZGT = torch.from_numpy(dataset['Zgt'])   
torch.cuda.empty_cache()

# model to use
model = FWFS(wfs ,device=wfs.device).eval()


## PWFS
if len(wfs.mod)!=0 and len(wfs.nHeads)!=0:
    nhs = ''.join(map(str, wfs.nHeads))
    pwfs_name =  f'/pwfs_nh{nhs}_M{mods}_Dro{wfs.Dr0v[0]}-{wfs.Dr0v[-1]}_Z{wfs.zModes[-1]}_T{wfs.T}_{wfs.precision_name}'
    pwfs_path = exp_path + pwfs_name + '.npy'
    atm_std_path = exp_path +  f'/atm_std.npy'
    if not os.path.exists(pwfs_path):
        print(f'PWFS used: alpha={model.pwfs.alpha:.4f} | nHead={model.pwfs.nHead}')
        raw_pwfs = torch.empty((len(wfs.mod),len(wfs.nHeads),len(wfs.r0v),wfs.T), dtype=wfs.precision.real).to('cpu')
        total_pwfs = len(wfs.nHeads)*len(wfs.mod)*len(wfs.r0v)*(wfs.T//wfs.batch)
        raw_atm = torch.empty((len(wfs.r0v),wfs.T), dtype=wfs.precision.real).to('cpu')
        with tqdm(total=total_pwfs, desc='', bar_format=bar_format) as pbar_pwfs: # Barra
            for i,m in enumerate(wfs.mod):
                model.pwfs.modulation = m
                for id_nh,nh in enumerate(wfs.nHeads):
                    model.pwfs.nHead = nh
                    _,piston = model(model.pwfs.Piston, vNoise=wfs.vNoise)
                    pos = np.sqrt(piston.shape[-1]).astype(np.int32)
                    piston = piston.reshape( pos,pos ).detach().cpu()
                    plt.imshow(piston.cpu().squeeze())
                    #plt.colorbar()
                    plt.axis('off')
                    plt.savefig(exp_path+f'/I0_pwfs_m{m}_nh{nh}.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    I0_v,PIMat = model.pwfs.Calibration(mInorm=wfs.mInorm)
                    for r in range(len(wfs.r0v)):
                        for t in range(wfs.T//wfs.batch):
                            pbar_pwfs.set_description(f'Testing PWFS: m={m} | nh={nh}')
                            batch = np.arange(wfs.batch*t, wfs.batch*(t+1))
                            iwfs = UNZ(PHI[r,batch,:,:],1).to(wfs.precision.real).to(wfs.device)
                            zgt = ZGT[r,batch,:].to(wfs.precision.real).to(wfs.device)
                            # forward classic
                            I = model.pwfs.forward(iwfs, vNoise=wfs.vNoise)
                            mI_v = model.pwfs.mI(I[:,0,model.pwfs.idx],I0_v,mInorm=wfs.mInorm)
                            zest = ( PIMat @ mI_v.t() ).t()
                            raw_atm[r,batch] = torch.std(iwfs[...,pupil].cpu(),dim=(-2,-1))# T[T,1,NM] -> T[r,T]
                            #zest,_ = model(iwfs, fourier_mask=fourier_mask)# this resume the above three lines but slower
                            # error model
                            error = cost(zgt,zest).detach().cpu()
                            raw_pwfs[i,id_nh,r,batch] = error.squeeze()
                            pbar_pwfs.update(1)
        np.save(pwfs_path, raw_pwfs)
        np.save(atm_std_path, raw_atm)
    else:   
        print(f'Importing {pwfs_name} PWFS')
        raw_pwfs = torch.from_numpy( np.load(pwfs_path,allow_pickle=True) )
        raw_atm = torch.from_numpy( np.load(atm_std_path,allow_pickle=True) )# T[r,T]
    #pwfs
    mean_pwfs = torch.mean(raw_pwfs, dim=-1)#.view(raw_pwfs.shape[0]*raw_pwfs.shape[1],raw_pwfs.shape[2])
    std_pwfs = torch.std(raw_pwfs, dim=-1)#.view(raw_pwfs.shape[0]*raw_pwfs.shape[1],raw_pwfs.shape[2])
else:
    print("Running without PWFS")
    mean_pwfs = []
    std_pwfs = []
model.pwfs.modulation = 0

## DPWFS
if len(routine)!=0:
    dpwfs_name = f'/dpwfs_Mo{len(routine)}_Z{wfs.zModes[-1]}_M{mods}_Dro{wfs.Dr0v[0]}-{wfs.Dr0v[-1]}_T{wfs.T}'
    dpwfs_path = exp_path + dpwfs_name + '.npy'
    # test
    if not os.path.exists(dpwfs_path):
        raw_dpwfs = torch.empty((len(routine),1,len(wfs.r0v),wfs.T), dtype=wfs.precision.real).to('cpu')
        total_dpwfs= len(routine)*len(wfs.r0v)*(wfs.T//wfs.batch)
        with tqdm(total=total_dpwfs, desc='', bar_format=bar_format) as pbar_dpwfs: # Barra
            for id_mo,mo in enumerate(routine):
                model = torch.load(mo['checkpoint'], map_location=wfs.device).to(wfs.device).eval()
                print(model)
                mo_tag = mo['tag']
                #
                _,piston = model(model.pwfs.Piston, vNoise=wfs.vNoise)
                pos = np.sqrt(piston.shape[-1]).astype(np.int32)
                piston = piston.reshape( pos,pos ).detach().cpu()
                plt.imshow( piston.cpu().squeeze())
                plt.axis('off')
                #plt.colorbar()
                plt.savefig(exp_path+f'/I0_mo{id_mo}.png', dpi=300, bbox_inches='tight')
                plt.close()
                for r in range(len(wfs.r0v)):
                    for t in range(wfs.T//wfs.batch):
                        pbar_dpwfs.set_description(f'Testing DPWFS: model={mo_tag}')
                        batch = np.arange(wfs.batch*t,wfs.batch*(t+1))
                        iwfs = UNZ(PHI[r,batch,:,:],1).to(wfs.precision.real).to(wfs.device)# T[batch,1,N,M]
                        zgt = ZGT[r,batch,:].to(wfs.precision.real).to(wfs.device)
                        zest,_ = model(iwfs, vNoise=wfs.vNoise)
                        error = cost(zgt,zest[:,:int(wfs.zModes[-1])-1]).detach().cpu()
                        raw_dpwfs[id_mo,0,r,batch] = error
                        pbar_dpwfs.update(1)
        np.save(dpwfs_path, raw_dpwfs)
    else:
        print(f'Importing {dpwfs_name} DPWFS')
        raw_dpwfs = torch.from_numpy( np.load(dpwfs_path,allow_pickle=True) )        
    # dpwfs
    mean_dpwfs = torch.mean(raw_dpwfs, dim=-1).view(raw_dpwfs.shape[0]*raw_dpwfs.shape[1],raw_dpwfs.shape[2])
    std_dpwfs = torch.std(raw_dpwfs, dim=-1).view(raw_dpwfs.shape[0]*raw_dpwfs.shape[1],raw_dpwfs.shape[2])
else:
    print('Running without DPWFS')
    mean_dpwfs = []#torch.empty_like(mean_pwfs)
    std_dpwfs = []#torch.empty_like(std_pwfs)


fs = 18
cont = 0
lw = 2.5
j = 0


#
header = ['Models']+[f'{x}' for x in wfs.Dr0v]
rows = []


# PWFS
plt.figure(figsize=(8,3))
for id_m,m in enumerate(wfs.mod):
    for id_nh,nh in enumerate(wfs.nHeads):
        lbl_pwfs =  '' #fr'$\Lambda_{nh}$-IM' 
        plt.plot(wfs.Dr0v, (mean_pwfs[id_m,id_nh,:]), label=lbl_pwfs, color=color_pwfs[cont], linestyle=lstyle_pwfs[cont], linewidth=lw)
        rows.append( [lbl_pwfs]+[f'{x:.3f}' for x in mean_pwfs[id_m,id_nh,:]] )
        cont += 1
        plt.rc('font', size=fs)
        plt.tick_params(axis="x",labelsize=fs)
        plt.tick_params(axis="y",labelsize=fs)
        plt.gca().invert_xaxis()  
        plt.gca().spines['bottom'].set_color('gray')  # Eje x
        plt.gca().spines['left'].set_color('gray')    # Eje y
        plt.legend(fontsize=fs-2, loc='upper right')
        plt.xlabel(r'$D/r_0$',fontsize=fs)
        plt.ylabel('RMSE (rad)',fontsize=fs) if wfs.cost=='rmse' else plt.ylabel('MAE (rad)',fontsize=fs)
        plt.grid(True)
        plt.ylim(0,ylim)
        plt.xlim(wfs.Dr0v[0],wfs.Dr0v[-1])
        plt.xticks( xticks,[ str(xticks[i]) for i in range(len(xticks)) ]) 
        #plt.xticks(wfs.Dr0v,[f'{int(t)}' if t.is_integer() else f'{t:.1f}' for t in wfs.Dr0v])
        # plot
        plt.savefig(exp_path + f'/plot_{wfs.expName}_Z{wfs.zModes[-1]}_T{wfs.T}_M{mods}_Mo{len(routine)}_Dro{wfs.Dr0v[0]:.3f}-{wfs.Dr0v[-1]:.3f}_cum{cont}.png', dpi=300,format='png', bbox_inches='tight')
        plt.savefig(exp_path + f'/plot_{wfs.expName}_Z{wfs.zModes[-1]}_T{wfs.T}_M{mods}_Mo{len(routine)}_Dro{wfs.Dr0v[0]:.3f}-{wfs.Dr0v[-1]:.3f}_cum{cont}.pdf', dpi=300,format='pdf', bbox_inches='tight')

# DPWFS
for id_mo,mo in enumerate(routine):
    lbl_dpwfs = mo['tag']
    plt.plot((wfs.Dr0v), (mean_dpwfs[id_mo,:]), label=fr'{lbl_dpwfs}',linestyle=lstyle_dpwfs[id_mo], color=color_dpwfs[id_mo], linewidth=lw)
    rows.append( [lbl_dpwfs]+[f'{x:.3f}' for x in mean_dpwfs[id_mo,:]] )
    cont += 1
    #
    plt.rc('font', size=fs)
    plt.tick_params(axis="x",labelsize=fs)
    plt.tick_params(axis="y",labelsize=fs)
    #plt.gca().invert_xaxis()  
    plt.gca().spines['bottom'].set_color('gray')  # Eje x
    plt.gca().spines['left'].set_color('gray')    # Eje y
    plt.legend(loc='upper right',fontsize=fs-2)
    plt.xlabel(r'$D/r_0$',fontsize=fs)
    plt.ylabel('RMSE  (rad)',fontsize=fs)
    plt.grid(True)
    plt.ylim(0,ylim)
    plt.xlim(wfs.Dr0v[0],wfs.Dr0v[-1])
    #plt.yticks([0,0.05,0.1,0.15,0.2],[])
    plt.xticks( xticks,[ str(xticks[i]) for i in range(len(xticks)) ])
    #
    plt.savefig(exp_path + f'/plot_{wfs.expName}_Z{wfs.zModes[-1]}_T{wfs.T}_M{mods}_Mo{len(routine)}_Dro{wfs.Dr0v[0]:.3f}-{wfs.Dr0v[-1]:.3f}_cum{cont}.png', dpi=300,format='png', bbox_inches='tight')
    plt.savefig(exp_path + f'/plot_{wfs.expName}_Z{wfs.zModes[-1]}_T{wfs.T}_M{mods}_Mo{len(routine)}_Dro{wfs.Dr0v[0]:.3f}-{wfs.Dr0v[-1]:.3f}_cum{cont}.pdf',format='pdf', dpi=300, bbox_inches='tight')
plt.close()
# results data
exp_name = f'/results_Z{wfs.zModes[-1]}_M{mods}_Dro{wfs.Dr0v[0]}-{wfs.Dr0v[-1]}_T{wfs.T}.mat'
print(f'{exp_name} saved!')

# table
fig,ax = plt.subplots()
ax.axis('off')
table = ax.table(
    cellText=rows,                # 2-D list of cell values
    colLabels=header,            # header row
    loc='center',                 # centre inside the Axes
    cellLoc='center'              # centre-align all text
)
fig.savefig(exp_path+f'/summary_table.png',dpi=300,bbox_inches='tight')
plt.close()
# log
pars_log = {'D':wfs.D, 'nPx':wfs.nPx, 'samp':wfs.samp, 'rooftop':wfs.rooftop,
            'fovPx':wfs.fovPx, 'wvl':wfs.wvl, 'ps':wfs.ps,
            'expName':wfs.expName, 'Device':wfs.device,'precision name':wfs.precision_name,'zModes':wfs.zModes,'crop':wfs.crop}
pars_test = {'mod':wfs.mod, 'Dr0v':wfs.Dr0v, 'T':wfs.T, 'Batch':wfs.batch, 'cost':cost,'command':command}
pars = pars_log | pars_test# combine dictionary
Log(pars, [routine], path=exp_path, name=f'log_{datetime.date.today()}' )