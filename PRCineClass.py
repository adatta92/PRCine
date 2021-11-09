#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

import os
import skimage
from skimage import registration

get_ipython().run_line_magic('matplotlib', 'notebook')




# In[2]:


if 'PRCine' in dir():
    del PRCine, FreeBreathing
    
class PRCine:
    device = sgp.Device(0)
    prog_flag = True

    out_size = 256
    over_rat = 1.25
    default_exp = -8.0
    
    def __init__(self, data, k, downsample=False, acquired_views=None):
        self.data = np.transpose(data, (2, 3, 4, 0, 1))
        self.k = k
        
        if downsample:
            self.downsample()
            
        self.num_pc, self.num_cp, self.num_c, self.num_ro = self.data.shape[:4]
                
        if acquired_views is None:
            self.sampled_spokes_only = False
            self.num_angles = self.data.shape[-1]
        else:
            self.sampled_spokes_only = True
            self.acquired_views = acquired_views.transpose((1, 2, 0)) 
            self.num_sampled_spokes = self.data.shape[-1]
            
        self.out_shape = np.array((self.num_pc, self.num_cp, self.num_c, self.out_size, self.out_size))
        
    def downsample(self):
        mid_ind = len(self.k)//2
        del_k = self.k[mid_ind+1] - self.k[mid_ind]
        unif_k = np.arange(-0.5, 0.5, del_k)
        
        interp_fn = sp.interpolate.interp1d(self.k, self.data, axis=3, fill_value='extrapolate')
        unif_data = interp_fn(unif_k)
        
        (self.data, self.k) = sp.signal.resample(unif_data, len(unif_k)//2, unif_k, axis=3)       
        
#         # FIX sqrt(2) SNR LOSS!!
#         self.k = self.k[::2]
#         self.data = self.data[:, :, :, ::2, :]
        
        self.num_ro = self.data.shape[3]
#         print(self.num_ro)
        
    def calc_coords(self, k_shift=0, rotations=None):
        if rotations is None:
            rotations = np.concatenate((np.arange(0, self.num_angles/2)+0.5, 
                                    np.arange(self.num_angles/2, self.num_angles)+1))/self.num_angles 
        
        angles = 2*math.pi*rotations
        
        proj = self.out_size*(self.k+k_shift) 
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
    
    def fix_dcf(self, num_beg, num_end, disp=True, dcf=None): 
        if dcf is None:
            dcf = self.dcf
        
        if num_beg>0:
            dcf[:num_beg,] = dcf[num_beg+1,]             
        if num_end>0:
#             self.dcf[-num_end:0,] = self.dcf[-(num_end+1),]  DOESN'T WORK
            dcf[self.num_ro-num_end:self.num_ro,] = dcf[-(num_end+1),]
        
        if disp:
            self.disp_dcf(dcf)
            
        return dcf    
        
    def disp_dcf(self, dcf=None, coords=None):
        if dcf is None:
            dcf = self.dcf
        if coords is None:
            coords = self.coords
            
        ppl.figure()
        ppl.contourf(coords[:, :, 0], coords[:, :, 1], dcf, 100, cmap = 'Greys')
        ppl.show()
    
#     def to_gpu(self):
#         self.dcf_gpu = sgp.to_device(self.dcf, device=self.device)
#         self.data_gpu = sgp.to_device(self.data, device=self.device)
#         self.coords_gpu = sgp.to_device(self.coords, device=self.device)

    def mslr_recon(self, exp=None, blk_widths=[11, 22, 44, 88], save=True, save_name='',
                   disp=True, disp_cp=None, start_cp=3, vmax=6, spacing=4, cp_window_length=1, cps=None):
        if cps is None:
            num_cp = (self.num_cp//cp_window_length) * cp_window_length
            cps = [*range(num_cp)]
        else:
            num_cp = len(cps)
            # If provided, cps must be a list who's length is an integer multiple of cp_window_length 
        
        data_mslr, dcf_mslr, coords_mslr = self.__to_mslr_format(cps)
        
        print('data_mslr is ' + str(data_mslr.shape))
        print('dcf_mslr is ' + str(dcf_mslr.shape))
        print('coords_mslr is ' + str(coords_mslr.shape))
        
        T = self.num_pc*num_cp//cp_window_length
                
        if exp is None:
            exp = [self.default_exp]
            
        lamda = 10**np.asarray(exp)
        
        for n in range(len(lamda)):
            self.img_mslr = MultiScaleLowRankRecon(data_mslr, coords_mslr, dcf_mslr, self.rcvr_maps, 
                                                   T, lamda[n], device=self.device, blk_widths=blk_widths).run()
            img_mslr_np = self.img_mslr[:]
            img_mslr_np = img_mslr_np.reshape(self.num_pc, num_cp//cp_window_length, self.out_size, self.out_size)

            if disp:
                self.img_mslr_norm = self.__disp(img_mslr_np, lamda[n], 'MSLR', disp_cp, start_cp, vmax, spacing)
        
            if save:
                ksp_MSLR = sgp.fft(self.img_mslr_norm, axes = (-2, -1))
                dict_to_save = {'img_MSLR': self.img_mslr_norm, 'ksp_MSLR': ksp_MSLR, 
                                'lambda_MSLR': lamda[n], 'block_w': blk_widths}
                spio.savemat(save_name + '_MSLR_' + str(n) + '.mat', dict_to_save)
    
    def __to_mslr_format(self, cps=None):      
        order = (0, 1, 4, 3, 2)
        
        if self.sampled_spokes_only:
            dcf_mslr = np.moveaxis(self.dcf[:, self.acquired_views-1], 0, 3)
            coords_mslr = np.moveaxis(self.coords[:, self.acquired_views-1, :], 0, 3)
            data_mslr = self.data.transpose(order)
        else:
            data_mslr, dcf_mslr, coords_mslr = self.__remove_unsampled_spokes(order)
        
        if cps is None:
            cps = [*range(self.num_cp)]
        
        data_mslr = data_mslr[:, cps,]
        dcf_mslr = dcf_mslr[:, cps,]
        coords_mslr = coords_mslr[:, cps,]
        
        data_mslr = self.__stack_phases(data_mslr)
        data_mslr = data_mslr.transpose(2, 0, 1)
        dcf_mslr = self.__stack_phases(dcf_mslr)
        coords_mslr = self.__stack_phases(coords_mslr)
        
        return data_mslr, dcf_mslr, coords_mslr
    
    def __remove_unsampled_spokes(self, order = (0, 1, 4, 3, 2), unsamp_rat=12):
        self.num_sampled_spokes = self.num_angles//unsamp_rat
        
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
    
    def __stack_phases(self, array): #, axes=(1, 0)):  moveaxis and moveaxis back to stack other dimensions
        arrays = [np.squeeze(array[:, i, ]) for i in range(array.shape[1])] # self.num_cp)]
        stacked = np.concatenate(arrays, 1)
        arrays = [np.squeeze(stacked[i, ]) for i in range(array.shape[0])] # self.num_pc)]
        stacked = np.concatenate(arrays, 0)

        return stacked

    def calib_espirit(self, calib_thresh = 0.01, map_thresh = 0.95, disp=True):
        with self.device:
            if self.sampled_spokes_only:
                ksp_avg, dcf, coords = self.__to_mslr_format()
            else:
                ksp_avg = cup.mean(self.data, axis = (0, 1))                
                dcf = self.dcf
                coords = self.coords
                
                self.mask = cup.sum(abs(self.data), axis=2, keepdims = True) > 0
            
            avg_img = sgp.nufft_adjoint(ksp_avg * dcf, coords, oversamp = self.over_rat, oshape = self.out_shape[2:])
            ksp_avg_cart = sgp.fft(avg_img, axes = (-2, -1))

            self.rcvr_maps = mr.app.EspiritCalib(ksp_avg_cart, thresh = calib_thresh, crop = map_thresh, show_pbar=self.prog_flag).run()
                    # thresh = 0.01, crop = 0.95 
            self.rcvr_maps[abs(self.rcvr_maps)<sys.float_info.epsilon] = sys.float_info.epsilon   
            
        if disp:
            pl.ImagePlot(self.rcvr_maps, z = 0, title='Sensitivity Maps Estimated by ESPIRiT')

    def espirit_recon(self, lamda=0.01, num_iter=50, save=True, save_name='', disp=True, disp_cp=None, 
                      start_cp=3, vmax=6, spacing=4):
        img_espirit = np.zeros((self.num_pc, self.num_cp, 1, self.out_size, self.out_size), np.complex)

        parts = 5
        num_cp_part = (self.num_cp+parts-1)//parts # Ceiling division

        for part in range(parts):
            start_part = max(0, part*num_cp_part)
            end_part = min((part+1)*num_cp_part, self.num_cp)
            
            if start_part < end_part:
                img_espirit[:, start_part:end_part,] = self.espirit_partial(slice(start_part, end_part), lamda, num_iter)

        if disp:
            self.img_espirit_norm = self.__disp(np.squeeze(img_espirit), lamda, 'ESPIRiT', disp_cp, start_cp, vmax, spacing)
    
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
#             D = sgp.linop.Multiply(data.shape, self.dcf)
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
            disp_cp = img_norm.shape[1]//2 # self.num_cp//2

        pl.ImagePlot(img_norm[:, disp_cp,], z=0, vmax=v_m,
                     title=rec_type+r' Recon, Phase Cycles: $\lambda$ = '+str(lamda))
        pl.ImagePlot(img_comb[start_cp::spacing,], z=0, vmax=v_m, 
                     title=rec_type+r' Recon, Cardiac Phases: $\lambda$ = '+str(lamda))

        return img_norm
    
    def grid_times(self, thresh=0.0015, disp=False, coords=None): 
        if coords is None:
            coords = self.coords
        
        t_max = coords.shape[0]
        times = math.pi*np.arange(t_max)/t_max - math.pi/2
        times = np.exp(1j*times)
        times = np.tile(times[:, np.newaxis], (1, coords.shape[1]))

        times_grid = sgp.fft(sgp.nufft_adjoint(times, coords, oversamp = self.over_rat, 
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

class FreeBreathing(PRCine):
    psi_N = 15.4932/360
    num_angles = 1673
    
    def __init__(self, data, k, downsample=False, acquired_views=None):
        self.num_rs = data.shape[4]
        
        arrays = [np.squeeze(data[:, :, :, :, i,]) for i in range(self.num_rs)]
        data = np.concatenate(arrays, 3)
      
        if acquired_views is not None:
            arrays = [np.squeeze(acquired_views[:, :, :, i,]) for i in range(self.num_rs)]
            acquired_views = np.concatenate(arrays, 2)
        
        super().__init__(data, k, downsample, acquired_views)
        
    def calc_coords(self, k_shift=0):
        rotations = (np.arange(self.num_angles)+1)*self.psi_N % 1.0
        super().calc_coords(k_shift, rotations)
        
    def mslr_recon(self, exp=None, blk_widths=[11, 22, 44, 88], save=True, save_name='',
                   disp=True, disp_cp=None, start_cp=3, vmax=6, spacing=4, cp_window_length=1, cps=None):
        if cp_window_length > 1 and cps is None:
            total_num_cp_per_rs = self.num_cp//self.num_rs
            des_num_cp_per_rs = (self.num_cp//self.num_rs//cp_window_length)*cp_window_length
            cps = [i*total_num_cp_per_rs + np.arange(des_num_cp_per_rs) for i in range(self.num_rs)]
            cps = np.concatenate(cps)
            
        super().mslr_recon(exp, blk_widths, save, save_name,
                   disp, disp_cp, start_cp, vmax, spacing, cp_window_length, cps)
            
    def estimate_resp_motion(self, mot_est_window_length=4, mot_img=None, pname='', lr_inds=None, ap_inds=None):
        self.mot_est_window_length = mot_est_window_length
        
        if mot_img is None:
            self.mslr_recon(save_name=pname+'_estimateMotion', cp_window_length=mot_est_window_length)
            mot_img = self.img_mslr_norm
            
        if lr_inds is None:
            lr_inds = slice(0, AcqCine.out_size)
        if ap_inds is None:
            ap_inds = slice(AcqCine.out_size//2, AcqCine.out_size)

        # im_num = 0

        # for frame in range(img_norm.shape[1]):
        #     pl.ImagePlot(img_norm[:, frame, lr_inds, ap_inds], z=0, vmax=9, title='Frame '+str(frame))
        #     ppl.savefig('wall_im%03d.png' % im_num, bbox_inches='tight')
        #     im_num = im_num+1;

        # os.system("ffmpeg -r 5 -i wall_im%03d.png -vcodec mpeg4 -y wallMotion.mp4")

        pl.ImagePlot(mot_img[:, 0, lr_inds, ap_inds], z=0, vmax=9)

        self.shifts = np.zeros((mot_img.shape[0], mot_img.shape[1], 2))

        for pc in range(mot_img.shape[0]):
            ref_img = mot_img[pc, 0, lr_inds, ap_inds]

            for frame in range(mot_img.shape[1]):
                self.shifts[pc, frame,], temp, temp2 = skimage.registration.phase_cross_correlation(ref_img, mot_img[pc, frame, lr_inds, ap_inds], upsample_factor=100)

            self.shifts[pc, :, 1] = self.shifts[pc, :, 1] - np.median(self.shifts[pc, :, 1])    

            ppl.figure()
            ppl.plot(self.shifts[pc, :, 1])
            ppl.show()
        
    def bin_motion_states(self, des_num_rs=4):
        num_RRperPC = self.num_rs

#         print('self.shifts is ' + str(self.shifts.shape))
        
        arrays = np.split(self.shifts[:, :, 1], num_RRperPC, axis=1)
        resp_shift = np.stack(arrays, axis=2)
#         print('resp_shift is ' + str(resp_shift.shape))
        
        arrays = np.split(self.data, num_RRperPC, axis=1)
        data = np.stack(arrays, axis=2)
        arrays = np.split(self.acquired_views, num_RRperPC, axis=1)
        acquired_views = np.stack(arrays, axis=2)
        
        sz = list(data.shape)

        sz[2] = des_num_rs
        num_RRperRS = num_RRperPC//des_num_rs
        sz[-1] *= num_RRperRS

        data_rs = np.zeros(sz, np.complex)
        acquired_views_rs = np.zeros((sz[0], sz[1], sz[2], sz[-1]), np.uint)

        for pc in range(resp_shift.shape[0]):
            for cp_win in range(resp_shift.shape[1]):
                resp_order = np.argsort(resp_shift[pc, cp_win,])

                cp_start = self.mot_est_window_length*cp_win
                
                if cp_win == resp_shift.shape[1]: 
                    cp_end = sz[1]
                else:
                    cp_end = cp_start + self.mot_est_window_length

                if sz[1]-cp_end < self.mot_est_window_length:
                    cp_end = sz[1]

                for rs in range(des_num_rs):    
                    rsRRs = resp_order[rs*num_RRperRS:(rs+1)*num_RRperRS]

                    arrays = [np.squeeze(data[pc, cp_start:cp_end, RR, ]) for RR in rsRRs]
                    data_rs[pc, cp_start:cp_end, rs, ] = np.concatenate(arrays, axis=-1)            

                    arrays = [np.squeeze(acquired_views[pc, cp_start:cp_end, RR, ]) for RR in rsRRs]
                    acquired_views_rs[pc, cp_start:cp_end, rs, ] = np.concatenate(arrays, axis=-1)
                    
        self.num_rs = des_num_rs
        
        arrays = [np.squeeze(data_rs[:, :, i,]) for i in range(self.num_rs)]
        self.data = np.concatenate(arrays, 1)
#         print('self.data is ' + str(self.data.shape))
    
        arrays = [np.squeeze(acquired_views_rs[:, :, i,]) for i in range(self.num_rs)]
        self.acquired_views = np.concatenate(arrays, 1)
#         print('self.acquired_views is ' + str(self.data.shape))
        
        self.num_cp = self.data.shape[1]
        self.num_sampled_spokes = self.data.shape[-1]




# In[3]:


date = "2021_09_19"

free_breathing = True
traj_data = spio.loadmat('traj.mat')

# if free_breathing:
#     traj_data = spio.loadmat('./%s/traj_SR120.mat' % date)
# else:
#     traj_data = spio.loadmat('traj.mat')
    
k_acq = np.squeeze(traj_data['k_acq'])
k_rew = np.squeeze(traj_data['k_rew'])

start_rew = len(k_acq)

pnums = [58368]

for f in range(len(pnums)): 
    pnum = pnums[f]
    
    print(pnum)
#     input("Press Enter to continue...")    
        
    if free_breathing:
        data = spio.loadmat('./%s/FB%d_PC1.mat' % (date, pnum))
        data2 = spio.loadmat('./%s/FB%d_PC2.mat' % (date, pnum))
        data3 = spio.loadmat('./%s/FB%d_PC3.mat' % (date, pnum))
        
        kdata = np.stack((data['k_data_resp'], data2['k_data_resp'], data3['k_data_resp']), axis=2) 
        acquired_views = np.stack((data['acquired_views_resp'], data2['acquired_views_resp'],
                                   data3['acquired_views_resp']), axis=1) 
    else:
        data = spio.loadmat('%s/%d_part1.mat' % (date, pnum))
        data2 = spio.loadmat('%s/%d_part2.mat' % (date, pnum))
        kdata = np.concatenate((data['kdata'], data2['kdata']), axis = -2)
                
    num_cp = kdata.shape[3]

    kdata_acq = kdata[:start_rew, :, :, :num_cp,]
    kdata_rew = kdata[start_rew-1:, :, :, :num_cp,]

    if free_breathing:
        acquired_views = acquired_views[:, :, :num_cp, :]
        
    del kdata

    if free_breathing:
        AcqCine = FreeBreathing(kdata_acq, k_acq, acquired_views=acquired_views)
    else:
        AcqCine = PRCine(kdata_acq, k_acq) #, downsample=True)

    if f == 0:
        print(AcqCine.data.shape)
    
        AcqCine.calc_coords()
        AcqCine.disp_dcf()
        AcqCine.dcf = AcqCine.fix_dcf(2, 2, disp=True) # 1, 1, disp=True)

        coords = AcqCine.coords
        dcf = AcqCine.dcf
        t_cart = 2*AcqCine.grid_times(thresh=0.0015)

    else:
        AcqCine.coords = coords
        AcqCine.dcf = dcf

    AcqCine.calib_espirit()
    rcvr_maps = AcqCine.rcvr_maps
    
    if not free_breathing:
        mask = AcqCine.mask
        AcqCine.espirit_recon(save_name=str(pnum))
    
    if free_breathing:
#         AcqCine.mslr_recon(cp_window_length=4, save_name=str(pnum)+'_estimateMotion')
        AcqCine.estimate_resp_motion(pname=str(pnum))
        AcqCine.bin_motion_states()
        mot_est_window_length = AcqCine.mot_est_window_length
        shifts = AcqCine.shifts
        
    AcqCine.mslr_recon(save_name=str(pnum))

    del AcqCine
    
    if free_breathing:
        RewCine = FreeBreathing(kdata_rew, k_rew, downsample=False, acquired_views=acquired_views)
    else:
        RewCine = PRCine(kdata_rew, k_rew, downsample=False)
    
    if f == 0:        
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
        RewCine.coords = coords_rew
        RewCine.dcf = dcf_rew

    RewCine.rcvr_maps = rcvr_maps
    
    if not free_breathing:
        RewCine.mask = mask[:, :, :, :RewCine.num_ro, :]
        RewCine.espirit_recon(save_name=str(pnum)+'_rew')
    
    if free_breathing:
        RewCine.mot_est_window_length = mot_est_window_length
        RewCine.shifts = shifts
        RewCine.bin_motion_states()
    
    RewCine.mslr_recon(exp=[-7.0], save_name=str(pnum)+'_rew')

    del RewCine