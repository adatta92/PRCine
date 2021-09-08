import sys
sys.path.append("./extreme_mri")
import sigpy.plot as pl

import numpy as np
import scipy as sp
import cupy as cup
import scipy.io as spio
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as ppl
import sigpy as sgp
import sigpy.mri as mr
import math

from gridding_recon import gridding_recon
from multi_scale_low_rank_recon import MultiScaleLowRankRecon

%matplotlib notebook


if 'PRCine' in dir():
    del PRCine
    
class PRCine:
    device = sgp.Device(0)
    prog_flag = True

    out_size = 256
    over_rat = 1.25
    default_exp = -7.0
    
    def __init__(self, data, k, downsample=False):
        self.data = np.transpose(data, (2, 3, 4, 0, 1))
        self.k = k
        
        if downsample:
            self.downsample()
            
        self.num_pc, self.num_cp, self.num_c, self.num_ro, self.num_angles = self.data.shape
        
        self.out_shape = np.array(self.data.shape)
        self.out_shape[-2:] = self.out_size
        
    def downsample(self):
        # FIX sqrt(2) SNR LOSS!!
        self.k = self.k[:, ::2]
        self.data = self.data[:, :, :, ::2, :]
        
        self.num_ro = self.data.shape[3]
        
    def calc_coords(self, k_shift=0):
        rotations = np.concatenate((np.arange(0, self.num_angles/2)+0.5, 
                                    np.arange(self.num_angles/2, self.num_angles)+1))/self.num_angles 
        angles = 2*math.pi*rotations
        
        proj = np.squeeze(self.out_size*(self.k+k_shift)) 
        self.coords = np.outer(proj, np.exp(1j*angles))
        self.coords = np.stack((self.coords.real, self.coords.imag), axis=2)
                          
        pts = self.coords.reshape((self.coords.shape[0]*self.coords.shape[1], 2), order='F')
        self.dcf = self.__voronoi_volumes(pts).reshape(self.coords.shape[:2], order='F')                  
    
    def __voronoi_volumes(self, points):
        v = sp.spatial.Voronoi(points)
        vol = np.zeros(v.npoints)

        for i, reg_num in enumerate(v.point_region):
            indices = v.regions[reg_num]
            vol[i] = sp.spatial.ConvexHull(v.vertices[indices]).volume

        vol = np.minimum(vol, 5)

        return vol                    
    
    def fix_dcf(self, num_beg, num_end, disp=True): 
        dcf = self.dcf
        
        if num_beg>0:
            dcf[:num_beg,] = dcf[num_beg+1,]             
        if num_end>0:
#             self.dcf[-num_end:0,] = self.dcf[-(num_end+1),]  DOESN'T WORK
            dcf[self.num_ro-num_end:self.num_ro,] = dcf[-(num_end+1),]
        
        if disp:
            self.disp_dcf(dcf)
            
        return dcf    
        
    def disp_dcf(self, dcf=None):
        if dcf is None:
            dcf = self.dcf
            
        ppl.figure()
        ppl.contourf(self.coords[:, :, 0], self.coords[:, :, 1], dcf, 100, cmap = 'Greys')
        ppl.show()
    
#     def to_gpu(self):
#         self.dcf_gpu = sgp.to_device(self.dcf, device=self.device)
#         self.data_gpu = sgp.to_device(self.data, device=self.device)
#         self.coords_gpu = sgp.to_device(self.coords, device=self.device)

    def mslr_recon(self, exp = None, blk_widths = [11, 22, 44, 88], save=True, save_name='',
                   disp=True, disp_cp=None, start_cp=3, vmax=6):
        data_mslr, dcf_mslr, coords_mslr = self.__remove_unsampled_spokes()
        
        data_mslr = self.__stack_phases(data_mslr)
        data_mslr = data_mslr.transpose(2, 0, 1)
        dcf_mslr = self.__stack_phases(dcf_mslr)
        coords_mslr = self.__stack_phases(coords_mslr)
        
        T = self.num_pc*self.num_cp
        
        if exp is None:
            exp = [self.default_exp]
            
        lamda = 10**np.asarray(exp)
        
        for n in range(len(lamda)):
            self.img_mslr = MultiScaleLowRankRecon(data_mslr, coords_mslr, dcf_mslr, self.rcvr_maps, 
                                                   T, lamda[n], device=self.device, blk_widths=blk_widths).run()
            img_mslr_np = self.img_mslr[:]
            img_mslr_np = img_mslr_np.reshape(self.num_pc, self.num_cp, self.out_size, self.out_size)

            if disp:
                self.img_mslr_norm = self.__disp(img_mslr_np, lamda[n], 'MSLR', disp_cp, start_cp, vmax)
        
            if save:
                ksp_MSLR = sgp.fft(self.img_mslr_norm, axes = (-2, -1))
                dict_to_save = {'img_MSLR': self.img_mslr_norm, 'ksp_MSLR': ksp_MSLR, 
                                'lambda_MSLR': lamda[n], 'block_w': blk_widths}
                spio.savemat(save_name + '_MSLR_' + str(n) + '.mat', dict_to_save)
        
    def __remove_unsampled_spokes(self, unsamp_rat=12):
        self.num_sampled_spokes = self.num_angles//unsamp_rat
        
        order = (0, 1, 4, 3, 2)
        
        data_reshaped = self.data.transpose(order)
        mask_reshaped = self.mask.transpose(order)
        mask_reshaped = cup.squeeze(mask_reshaped[:, :, :, 0]) # Don't need readout direction

        coords_reshaped = self.coords.transpose(1, 0, 2)
        coords_reshaped = self.__create_phase_dims(coords_reshaped)
        dcf_reshaped = self.dcf.transpose(1, 0)
        dcf_reshaped = self.__create_phase_dims(dcf_reshaped)
        
        data_mslr = np.reshape(data_reshaped[mask_reshaped], 
                               (self.num_pc, self.num_cp, self.num_sampled_spokes, self.num_ro, self.num_c))
        dcf_mslr = np.reshape(dcf_reshaped[mask_reshaped], 
                              (self.num_pc, self.num_cp, self.num_sampled_spokes, self.num_ro))
        coords_mslr = np.reshape(coords_reshaped[mask_reshaped], 
                                 (self.num_pc, self.num_cp, self.num_sampled_spokes, self.num_ro, 2))

        return data_mslr, dcf_mslr, coords_mslr
        
    def __create_phase_dims(self, array):        
        arrays = [array for _ in range(self.num_cp)]
        array_reshaped = np.stack(arrays, 0)
        arrays = [array_reshaped for _ in range(self.num_pc)]
        array_reshaped = np.stack(arrays, 0)
    
        return array_reshaped
    
    def __stack_phases(self, array): #, axes=(1, 0)):
#         slc = [slice(None)]*len(array.shape)
#         slc[axes[0]] = i
#         arrays = [np.squeeze(array[slc]) for i in range(self.num_cp)]
#         stacked = np.concatenate(arrays, axes[0])
        
#         slc = [slice(None)]*len(stacked.shape)
#         slc[axes[1]] = i
#         arrays = [np.squeeze(stacked[slc]) for i in range(self.num_pc)]
#         stacked = np.concatenate(arrays, axes[1])

        arrays = [np.squeeze(array[:, i, ]) for i in range(self.num_cp)]
        stacked = np.concatenate(arrays, 1)
        arrays = [np.squeeze(stacked[i, ]) for i in range(self.num_pc)]
        stacked = np.concatenate(arrays, 0)

        return stacked

    def calib_espirit(self, disp=True):
        with self.device:
            ksp_avg = cup.mean(self.data, axis = (0, 1))           
            avg_img = sgp.nufft_adjoint(ksp_avg * self.dcf, self.coords, oversamp = self.over_rat, 
                                        oshape = self.out_shape[2:])
            ksp_avg_cart = sgp.fft(avg_img, axes = (-2, -1))

            self.rcvr_maps = mr.app.EspiritCalib(ksp_avg_cart, thresh = 0.005, crop = 0.95, show_pbar=self.prog_flag).run()
            self.mask = cup.sum(abs(self.data), axis=2, keepdims = True) > 0
                       
        if disp:
            pl.ImagePlot(self.rcvr_maps, z = 0, title='Sensitivity Maps Estimated by ESPIRiT')

    def espirit_recon(self, lamda=0.01, num_iter=50, save=True, save_name='', disp=True, disp_cp=None, start_cp=3, vmax=6):
        img_espirit = np.zeros((self.num_pc, self.num_cp, 1, self.out_size, self.out_size), np.complex)

        parts = 3
        num_cp_part = (self.num_cp+parts-1)//parts # Ceiling division

        for part in range(parts):
            start_part = max(0, part*num_cp_part)
            end_part = min((part+1)*num_cp_part, self.num_cp)
            
            img_espirit[:, start_part:end_part,] = self.espirit_partial(slice(start_part, end_part), lamda, num_iter)

        if disp:
            self.img_espirit_norm = self.__disp(np.squeeze(img_espirit), lamda, 'ESPIRiT', disp_cp, start_cp, vmax)
    
        if save:
            ksp_ESPIRiT = sgp.fft(self.img_espirit_norm, axes = (-2, -1))
            dict_to_save = {'rcvr_maps': self.rcvr_maps, 'img_ESPIRiT': self.img_espirit_norm, 'ksp_ESPIRiT': ksp_ESPIRiT,
                            'lambda_ESPIRiT': lamda, 'max_iter': num_iter}
            spio.savemat(save_name + '_ESPIRiT.mat', dict_to_save)
    
    def espirit_partial(self, cps=None, lam=0.01, num_iter=50):
        if cps is None:
            cps = slice(self.num_cp)
            
        data = sgp.to_device(self.data[:, cps,], device=self.device)  
        mask = self.mask[:, cps,]    
        
        num_cp = data.shape[1]
        out_shape = self.out_shape
        out_shape[1] = num_cp
            
        with self.device: 
            D = sgp.linop.Multiply(data.shape, self.dcf)
            F_nu = sgp.linop.NUFFT(out_shape, self.coords, oversamp = self.over_rat)
            C_r = sgp.linop.Multiply((self.num_pc, num_cp, 1, F_nu.ishape[3], F_nu.ishape[4]), self.rcvr_maps)

            S = sgp.linop.Multiply(data.shape, mask)

            img_espirit_gpu = cup.zeros((self.num_pc, num_cp, 1, self.out_size, self.out_size), cup.complex)
            sgp.app.LinearLeastSquares(S*F_nu*C_r, data, x=img_espirit_gpu, lamda=lam, max_iter=num_iter, 
                                       show_pbar=self.prog_flag).run()

        return sgp.to_device(img_espirit_gpu, device=-1)
        
    def __disp(self, img, lamda, rec_type='', disp_cp=None, start_cp=0, v_m=6, spacing=4):
        img_norm = img/(np.mean(np.abs(img)))
        img_comb = np.sqrt(np.mean(np.square(np.abs(img_norm)), axis=0))

        if disp_cp is None:
            disp_cp = self.num_cp//2

        pl.ImagePlot(img_norm[:, disp_cp,], z=0, vmax=v_m,
                     title=rec_type+r' Recon, Phase Cycles: $\lambda$ = '+str(lamda))
        pl.ImagePlot(img_comb[start_cp::spacing,], z=0, vmax=v_m, 
                     title=rec_type+r' Recon, Cardiac Phases: $\lambda$ = '+str(lamda))

        return img_norm
    
    def grid_times(self, thresh=0.0015, disp=False): 
        t_max = self.coords.shape[0]
        times = math.pi*np.arange(t_max)/t_max - math.pi/2
        times = np.exp(1j*times)
        times = np.tile(times[:, np.newaxis], (1, self.coords.shape[1]))

        times_grid = sgp.fft(sgp.nufft_adjoint(times, self.coords, oversamp = self.over_rat, 
                                               oshape = (self.out_size, self.out_size)), axes = (0, 1))
        times_grid = sp.signal.fftconvolve(np.pad(times_grid, 1, mode = 'edge'), np.ones([3, 3]), 'valid')
        mask = np.abs(times_grid) > thresh*np.max(np.abs(times_grid[:]))
        
        if disp:
            pl.ImagePlot(mask)

        times_grid = (np.angle(times_grid) + math.pi/2)/math.pi * t_max
        times_grid = np.minimum(t_max, np.maximum(0, times_grid))
        times_grid[~mask] = -999
        
        if disp:
            pl.ImagePlot(times_grid, vmin = 0)

        return times_grid

		
traj_data = spio.loadmat('traj.mat')
k_acq = traj_data['k_acq']
k_rew = traj_data['k_rew']

print(k_acq.shape)
start_rew = k_acq.shape[1]-1

pnums = [46080, 46592, 47616, 48128]

for f in range(len(pnums)): 
    pnum = pnums[f]
    
    data = spio.loadmat('%d_part1.mat' % pnum)
    data2 = spio.loadmat('%d_part2.mat' % pnum)
    kdata = np.concatenate((data['kdata'], data2['kdata']), axis = -2)

    print(kdata.shape)

    # kdata = kdata[:, :, :, :5, :] # FEWER CARDIAC PHASES WHILE DEBUGGING TO RUN FASTER

    kdata_acq = kdata[:start_rew, ] # Data point 494 is last even point... last one kept by 2x downsampling
    kdata_rew = kdata[start_rew:, ] # Data point 495 starts rewind data

    AcqCine = PRCine(kdata_acq, k_acq, downsample=True)
    print(AcqCine.data.shape)

    if f == 0:
        AcqCine.calc_coords()
        AcqCine.disp_dcf()
        AcqCine.dcf = AcqCine.fix_dcf(1, 1, disp=True)

        coords = AcqCine.coords
        dcf = AcqCine.dcf
        t_cart = 2*AcqCine.grid_times(thresh=0.0015)

    else:
        AcqCine = PRCine(kdata_acq, k_acq, downsample=True)
        AcqCine.coords = coords
        AcqCine.dcf = dcf

    AcqCine.calib_espirit()
    rcvr_maps = AcqCine.rcvr_maps
    mask = AcqCine.mask
    AcqCine.espirit_recon(save_name=str(pnum))
    AcqCine.mslr_recon(save_name=str(pnum))

    del AcqCine
    
    if f == 0:
        RewCine = PRCine(kdata_rew, k_rew, downsample=False)
        print(RewCine.data.shape)

        RewCine.calc_coords(k_shift=-0.1/RewCine.out_size)
        RewCine.disp_dcf()
        RewCine.dcf = RewCine.fix_dcf(2, 0, disp=True)

        coords_rew = RewCine.coords
        dcf_rew = RewCine.dcf
        t_rew_cart = start_rew + RewCine.grid_times(thresh=0.0005)

        dict_to_save = {'t_cart': t_cart, 't_rew_cart': t_rew_cart}
        spio.savemat('gridded_times.mat', dict_to_save)

    else:
        RewCine = PRCine(kdata_rew, k_rew, downsample=False)
        RewCine.coords = coords_rew
        RewCine.dcf = dcf_rew

    RewCine.rcvr_maps = rcvr_maps
    RewCine.mask = mask[:, :, :, :RewCine.num_ro, :]
    RewCine.espirit_recon(save_name=str(pnum)+'_rew')
    RewCine.mslr_recon(save_name=str(pnum)+'_rew')
    
    del RewCine
    
    print(pnum)
    input("Press Enter to continue...")