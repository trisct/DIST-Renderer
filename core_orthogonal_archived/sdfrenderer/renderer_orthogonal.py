import os, sys
import torch
import torch.nn as nn
import numpy as np
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.utils.decoder_utils_plaintest import decode_sdf, decode_sdf_gradient
from core.visualize.profiler import Profiler
from core.utils.render_utils import depth2normal
import copy
import time
import matplotlib.pyplot as plt

class SDFRenderer(object):
    def __init__(self, decoder, img_hw=None, march_step=50, buffer_size=5, ray_marching_ratio=1.5, max_sample_dist=0.2, radius=1.0, threshold=5e-5, use_gpu=True, is_eval=True):
        self.decoder = decoder
        print(f'[In renderer] Setting device to {next(self.decoder.parameters()).device}')
        self.device = next(self.decoder.parameters()).device
        if is_eval:
            self.decoder.eval()
        self.march_step = march_step
        self.buffer_size = buffer_size
        self.max_sample_dist = max_sample_dist
        self.ray_marching_ratio = ray_marching_ratio
        self.radius = radius
        self.threshold = threshold

        self.img_hw = img_hw

        print(f'[In renderer] showing candidate devices at line 40...')
        self.imgmap_init = self.init_imgmap(self.img_hw)
        self.imgmap_init.requires_grad=False


        if use_gpu:
            if torch.cuda.device_count() == 0:
                raise ValueError('No GPU device found.')
            self.imgmap_init = self.imgmap_init.to(self.device) # (H*W)

    def get_grid_and_rays(self):
        print(f'Getting camera pos and rays')
        x = torch.linspace(-1, 1, self.img_hw[1], device=self.device)
        y = torch.linspace(-1, 1, self.img_hw[0], device=self.device)
        grid_y, grid_x = torch.meshgrid(y, x)
        grid_z = torch.ones_like(grid_x, device=self.device) * -1
        grid = torch.stack([grid_x, grid_y, grid_z], dim=0).reshape(3,-1) # [3, H*W]
        #print(f'[In render_depth] grid = {grid.shape}')
        
        cam_rays = torch.zeros(3, 1, device=self.device)
        cam_rays[2] = 1.
        cam_rays = cam_rays.expand(3, self.img_hw[0]*self.img_hw[1])
        #print(f'| New cam rays = {cam_rays.shape}')

        maxbound_zdepth = torch.ones(self.img_hw[0]*self.img_hw[1], device=self.device) * 2 # always 2
        init_zdepth = torch.zeros(self.img_hw[0]*self.img_hw[1], device=self.device) # always 0

        return grid, cam_rays, init_zdepth, maxbound_zdepth


    def init_imgmap(self, img_hw):
        h, w = img_hw
        imgmap_init = torch.zeros(h, w)
        return imgmap_init

    def normalize_vectors(self, x):
        '''
        normalize the vector by the first dim
        '''
        norm = torch.norm(x, p=2, dim=0).expand_as(x)
        eps = 1e-12
        x = x.div(norm + eps)
        return x


    def generate_point_samples(self, cam_pos, cam_rays, Zdepth, has_zdepth_grad=False):
        '''
        Input:
        - cam_pos	type torch.Tensor (3, N)
        - cam_rays	type torch.Tensor (3)
        - Zdepth	type torch.Tensor (N)
        Return:
        - points	type torch.Tensor (3, N)
        '''
        if not has_zdepth_grad:
            Zdepth = Zdepth.detach()
        N = Zdepth.shape[0]
        if N == 0:
            raise ValueError('No valid depth.')
        cam_pos_pad = cam_pos # (3, N)
        
        Zdepth_pad = Zdepth[None,:].repeat(3,1) # (3, N)
        points = cam_rays * Zdepth_pad + cam_pos_pad # (3, N)
        
        if not points.requires_grad:
            points.requires_grad=True
        return points

    def get_distance_from_origin(self, cam_pos, cam_rays):
        '''
        get_distance_from_origin
        Input:
        - cam_pos	type torch.FloatTensor (3, H*W)
        - cam_rays	type torch.FloatTensor (3)
        '''
        print(f'[In get_distance_from_origin] cam_rays.shape = {cam_rays.shape}')
        N = cam_pos.shape[1]
        cam_pos_pad = cam_pos # (3, N)
        
        p, q = cam_pos_pad, cam_rays # (3, N), (3, N)
        ptq = (p * q).sum(0) # (N)
        dist = p - ptq[None,:].repeat(3,1) * q # (3, N)
        dist = torch.norm(dist, p=2, dim=0) # (N)
        return dist


    def copy_index(self, inputs, mask, src):
        '''
        out-of-place copy index.
        Input:
        - inputs:	torch.Tensor (H*W) / (H, W) / (H, W, k)
        - mask:		torch.Tensor (H*W)
        - src:		torch.Tensor (N) / (N, k)
        '''
        inputs_shape = inputs.shape
        if len(inputs_shape) <= 2:
            inputs, mask = inputs.reshape(-1), mask.reshape(-1)
        elif len(inputs_shape) == 3:
            inputs, mask = inputs.reshape(-1, inputs_shape[-1]), mask.reshape(-1)
        else:
            raise NotImplementedError
        index = torch.nonzero(mask).reshape(-1).long()
        outputs = inputs.index_copy(0, index, src)
        outputs = outputs.reshape(inputs_shape)
        return outputs

    def get_index_from_sdf_list(self, sdf_list, index_size, index_type='min', clamp_dist=0.1):
        '''
        get index with certain method.
        Input:
        - sdf_list:		type: torch.Tensor (self.march_step, N)
        Return:
        - sdf:			type: torch.Tensor (N, index_size)
        - index:		type: torch.Tensor (N, index_size). Note: the first dimension (index[0]) is always the min index.
        '''
        if index_type == 'min':
            sdf, index = torch.topk(-sdf_list.transpose(1,0), index_size, dim=1)
            sdf = -sdf
        elif index_type == 'min_abs':
            sdf_list_new = torch.abs(sdf_list)
            _, index = torch.topk(-sdf_list_new.transpose(1,0), index_size, dim=1)
            sdf = self.collect_data_from_index(sdf_list, index)
        elif index_type == 'max_neg':
            sdf_list_new = sdf_list.clone()
            sdf_list_pos = (sdf_list_new >= 0)
            sdf_list_new[sdf_list_pos] = sdf_list_new[sdf_list_pos].clone() * (-1) - 2
            sdf, index = torch.topk(sdf_list_new.transpose(1,0), index_size, dim=1) # (N, index_size)
            sdf_pos = (sdf <= -2)
            sdf[sdf_pos] = sdf[sdf_pos].clone() * (-1) - 2
        elif index_type == 'last_valid':
            march_step, N = sdf_list.shape[0], sdf_list.shape[1]
            valid = (torch.abs(sdf_list) < clamp_dist)
            idx_list = torch.arange(0, march_step)[:,None].repeat(1,N).to(sdf_list.device)
            idx_list = idx_list.float() * valid.float()
            _, index = torch.topk(idx_list.transpose(1,0), index_size, dim=1) # (N, index_size)
            sdf = self.collect_data_from_index(sdf_list, index)[0].transpose(1,0)
        elif index_type == 'last':
            march_step, N = sdf_list.shape[0], sdf_list.shape[1]
            sdf = sdf_list[-index_size:, :].transpose(1,0)
            index = torch.arange(march_step - index_size, march_step)[None,:].repeat(N, 1)
            index = index.to(sdf.device)
        else:
            raise NotImplementedError
        return sdf, index

    def collect_data_from_index(self, data, index):
        '''
        Input:
        - data:		type: torch.Tensor (self.march_step, N) / (self.march_step, N, k)
        - index:	type: torch.Tensor (N, index_size)
        Return:
        - data_sampled:	type: torch.Tensor (index_size, N) / (index_size, N, k)
        '''
        index_size = index.shape[1]
        count_index = torch.arange(index.shape[0]).repeat(index_size).to(index.device)
        point_index = index.transpose(1,0).reshape(-1) * data.shape[1] + count_index

        if len(data.shape) == 3:
            data_shape = data.shape
            data_sampled = data.reshape(-1, data_shape[-1])[point_index].reshape(index_size, -1, data_shape[-1]).clone() # (index_size, N, 3)
        elif len(data.shape) == 2:
            data_sampled = data.reshape(-1)[point_index].reshape(index_size, -1).clone() # (index_size, N)
        else:
            raise NotImplementedError
        return data_sampled

    def sample_points_uniform(self, points, cam_rays, num_samples=None):
        '''
        Input:
        points:		type: torch.Tensor (N, 3)
        cam_rays:	type: torch.Tensor (3, N)
        Return:
        points_sampled:	type: torch.Tensor (num_samples, N, 3)
        '''
        if num_samples == None:
            num_samples = self.buffer_size
        N = points.shape[0]
        points = points[None,:,:].repeat(num_samples, 1, 1) # (num_samples, N, 3)
        cam_rays = cam_rays.transpose(1, 0)[None,:,:].repeat(num_samples, 1, 1) # (num_samples, N, 3)
        delta_depth = torch.linspace(0, -self.max_sample_dist, num_samples).to(points.device) # (num_samples)
        delta_depth = delta_depth[:,None,None].repeat(1, N, 3) # (num_samples, N, 3)
        points_sampled = delta_depth * cam_rays + points # (num_smaples, N, 3)
        return points_sampled

    def get_min_sdf_sample(self, sdf_list, points_list, latent, index_type='min_abs', clamp_dist=0.1, profile=False, no_grad=False):
        profiler = Profiler(silent = not profile)
        _, index = self.get_index_from_sdf_list(sdf_list, 1, index_type=index_type)
        points = self.collect_data_from_index(points_list, index)[0] # (N, 3)
        min_sdf_sample = decode_sdf(self.decoder, latent, points, clamp_dist=None, no_grad=no_grad).squeeze(-1)
        profiler.report_process('[DEPTH] [SAMPLING] sample min sdf time\t')
        if no_grad:
            min_sdf_sample = min_sdf_sample.detach()
        return min_sdf_sample

    def get_sample_on_marching_zdepth_along_ray(self, marching_zdepth_list, sdf_list, points_list, cam_rays, latent, index_type='min_abs', use_uniform_sample=False, clamp_dist=0.1, profile=False, no_grad=False):
        # initialization
        profiler = Profiler(silent = not profile)

        # collect points
        if use_uniform_sample:
            sdf_selected, index_selected = self.get_index_from_sdf_list(sdf_list, 1, index_type=index_type, clamp_dist=clamp_dist)
            points = self.collect_data_from_index(points_list, index_selected)[0] # (N, 3)
            points_sampled = self.sample_points_uniform(points, cam_rays)
        else:
            sdf_selected, index_selected = self.get_index_from_sdf_list(sdf_list, self.buffer_size, index_type=index_type, clamp_dist=clamp_dist)
            points_sampled = self.collect_data_from_index(points_list, index_selected)
        profiler.report_process('[DEPTH] [SAMPLING] collect points time\t')

        # generate new marching zdepth
        marching_zdepth = self.collect_data_from_index(marching_zdepth_list, index_selected[:,[0]])[0] # (N)
        marching_zdepth = marching_zdepth + (1 - self.ray_marching_ratio) * torch.clamp(sdf_selected[0,:], -clamp_dist, clamp_dist) # (N)

        if no_grad:
            marching_zdepth_final = marching_zdepth
        else:
            marching_zdepth_new = marching_zdepth
            for i in range(self.buffer_size):
                sdf = decode_sdf(self.decoder, latent, points_sampled[i], clamp_dist=clamp_dist, no_grad=no_grad).squeeze(-1)
                marching_zdepth_new = marching_zdepth_new - sdf.detach() * self.ray_marching_ratio
                marching_zdepth_new = marching_zdepth_new + sdf * self.ray_marching_ratio
            profiler.report_process('[DEPTH] [SAMPLING] re-ray marching time')
            marching_zdepth_final = marching_zdepth_new
        return marching_zdepth_final


    def ray_marching_recursive(self, cam_pos, cam_rays, init_zdepth, valid_mask, latent, march_step=None, stop_threshold=None, clamp_dist=0.1, no_grad=False, use_first_query_check=True):
        print(f'[In ray_marching_recursive]')
        print(f'| | cam_rays = {cam_rays}')
        if stop_threshold is None:
            stop_threshold = self.threshold

        print(f'[In ray_marching_recursive] valid_mask.shape = {valid_mask.shape}')

        valid_cam_rays = cam_rays[:, valid_mask]
        cam_pos = cam_pos[:, valid_mask]
        
        init_zdepth = init_zdepth[valid_mask]
        
        print(f'[In ray_marching_recursive] init_zdepth.shape = {init_zdepth.shape}')
        print(f'[In ray_marching_recursive] init_zdepth = {init_zdepth}')
        
        if march_step is None:
            march_step = self.march_step

        
        #self.get_maxbound_zdepth(cam_pos, valid_cam_rays)
        _, _, _, maxbound_zdepth = self.get_grid_and_rays()
        print(f'[In ray_marching_recursive] maxbound_zdepth.shape = {maxbound_zdepth.shape}')

        marching_zdepth_list, sdf_list, points_list = [], [], []
        marching_zdepth = torch.zeros_like(init_zdepth)

        valid_mask_max_marching_zdepth = (marching_zdepth + init_zdepth < maxbound_zdepth)
        unfinished_mask = valid_mask_max_marching_zdepth # (N)
        print(f'[In ray_marching_recursive] marching_zdepth = {marching_zdepth}')
        print(f'[In ray_marching_recursive] init_zdepth = {init_zdepth}')
        print(f'[In ray_marching_recursive] maxbound_zdepth = {maxbound_zdepth}')
        print(f'[In ray_marching_recursive] unfinished_mask.shape = {unfinished_mask.shape}')
        print(f'[In ray_marching_recursive] unfinished_mask nonzero terms = {(unfinished_mask).sum()}')
        for i in range(march_step):
            # get unfinished
            cam_rays_now = valid_cam_rays[:, unfinished_mask] # (3, K)
            cam_pos_now = cam_pos[:, unfinished_mask]

            init_zdepth_now = init_zdepth[unfinished_mask] # (K)
            print(f'[In ray_marching_recursive] init_zdepth_now.shape = {init_zdepth_now.shape}')
            marching_zdepth_now = marching_zdepth[unfinished_mask] # (K)

            # get corresponding sdf value
            points_now = self.generate_point_samples(cam_pos_now, cam_rays_now, init_zdepth_now + marching_zdepth_now) # (3, K)
            if no_grad:
                points_now = points_now.detach()
            sdf_now = decode_sdf(self.decoder, latent, points_now.transpose(1,0), clamp_dist=None, no_grad=no_grad).squeeze(-1) # (K)
            points = torch.zeros_like(marching_zdepth)[:,None].repeat(1,3)
            points[unfinished_mask,:] = points_now.transpose(1,0)
            if no_grad:
                points = points.detach()
            points_list.append(points[None,:])

            # clamp sdf from below if the flag is invalid, which means that it has not meet any sdf < 0
            sdf = torch.zeros_like(marching_zdepth)
            sdf[unfinished_mask] = sdf_now.detach()
            sdf_marching = torch.clamp(sdf, -clamp_dist, clamp_dist)

            # aggressive ray marching
            marching_zdepth = marching_zdepth + sdf_marching * self.ray_marching_ratio
            marching_zdepth_list.append(marching_zdepth[None,:])

            # update sdf list
            sdf[~unfinished_mask] = 1.0
            sdf_list.append(sdf[None,:])

            # update unfinised mask
            valid_mask_max_marching_zdepth = (marching_zdepth + init_zdepth < maxbound_zdepth)
            unstop_mask = torch.abs(sdf) >= stop_threshold
            unfinished_mask = unfinished_mask & valid_mask_max_marching_zdepth & unstop_mask
            if torch.nonzero(unfinished_mask).shape[0] == 0:
                while(len(marching_zdepth_list) < self.buffer_size):
                    marching_zdepth_list.append(marching_zdepth[None,:])
                    sdf_list.append(sdf[None,:])
                    points_list.append(points[None,:])
                break
        # concat ray marching info
        marching_zdepth_list = torch.cat(marching_zdepth_list, 0) # (self.march_step, N)
        sdf_list = torch.cat(sdf_list, 0)
        points_list = torch.cat(points_list, 0)

        # get valid mask
        valid_mask_max_marching_zdepth = (marching_zdepth_list[-1] + init_zdepth < maxbound_zdepth)
        min_sdf, _ = torch.abs(sdf_list).min(0)
        valid_mask_ray_marching = (min_sdf <= self.threshold)

        # get corner case: the first query is lower than threshold.
        valid_mask_render = valid_mask_max_marching_zdepth & valid_mask_ray_marching # (N)
        if use_first_query_check:
            valid_mask_first_query = sdf_list[0] > self.threshold
            valid_mask_render = valid_mask_render & valid_mask_first_query
        return sdf_list, marching_zdepth_list, points_list, valid_mask_render


    def ray_marching(self, cam_pos, init_zdepth, valid_mask, latent, march_step=None, clamp_dist=0.1, no_grad=False, ray_marching_type='recursive', split_type='raydepth'):
        '''
        ray marching function
        Input:
        - init_zdepth			type: torch.Tensor (H*W)
        - valid_mask			type: torch.Tensor (H*W) with N valid entries
        - split_type                    ['depth', 'raydepth'], which is the spliting strategy for pyramid recursive marching
        Return:
        - sdf_list			type: torch.Tensor (march_step, N)
        - marching_zdepth_list		type: torch.Tensor (march_step, N)
        - points_list			type: torch.Tensor (march_step, N, 3)
        - valid_mask_render		type: torch.Tensor (N)
        '''
        if not (split_type in ['depth', 'raydepth']):
            raise NotImplementedError
        elif ray_marching_type == 'recursive':
            _, cam_rays, _, _ = self.get_grid_and_rays()
            return self.ray_marching_recursive(cam_pos, cam_rays, init_zdepth, valid_mask, latent, march_step=None, clamp_dist=clamp_dist, no_grad=no_grad)
        else:
            raise ValueError('Error! Invalid type of ray marching: {}.'.format(ray_marching_type))

    def render_depth(self, latent, clamp_dist=0.1, sample_index_type='min_abs', profile=False, no_grad=False, no_grad_depth=False, no_grad_mask=False, no_grad_camera=False, ray_marching_type='recursive'):
        if no_grad:
            no_grad_depth, no_grad_mask, no_grad_camera = True, True, True
        
        # Getting camera pos and rays
        cam_pos, cam_rays, init_zdepth, _ = self.get_grid_and_rays()
        

        dist = self.get_distance_from_origin(cam_pos, cam_rays)
        print(f'| cam_pos = {cam_pos.shape}, {cam_pos}')
        print(f'| cam_rays = {cam_rays.shape}, ray viewing direction. For our orthogonal case, we only need a single direction!')
        dbgtmpvar_camraylen = (cam_rays**2).sum(0)
        print(f'| cam_rays lengths: min = {dbgtmpvar_camraylen.min()}, max = {dbgtmpvar_camraylen.max()}, so these are unit vectors.')
        print(f'| dist = {dist.shape}')


        profiler = Profiler(silent = not profile)
        # initialization on the unit sphere
        h, w = self.img_hw
        print(f'Getting initial zdepth and valid mask')
        
        valid_mask = torch.ones_like(init_zdepth, device=self.device).bool()
        #init_zdepth, valid_mask = self.get_intersections_with_unit_spheres(cam_pos, cam_rays)
        print(f'[In render_depth] init_zdepth = {init_zdepth.shape}')
        print(f'| valid_mask = {valid_mask.shape}')
        profiler.report_process('[DEPTH] initialization time')

        # ray marching
        print(f'Marching rays. Clearly the most important marching step happens here.')
        sdf_list, marching_zdepth_list, points_list, valid_mask_render = self.ray_marching(cam_pos, init_zdepth, valid_mask, latent, clamp_dist=clamp_dist, no_grad=no_grad_camera, ray_marching_type=ray_marching_type)
        print(f'| sdf_list = {sdf_list.shape}, the sdfs at all 50 marching steps')
        print(f'| marching_zdepth_list = {marching_zdepth_list.shape}, the depth at all 50 marching steps')
        print(f'| points_list = {points_list.shape}, the points at all 50 marching steps')
        print(f'| valid_mask_render = {valid_mask_render.shape}, only a single image')
        profiler.report_process('[DEPTH] ray marching time')

        # get differnetiable samples
        min_sdf_sample = self.get_min_sdf_sample(sdf_list, points_list, latent, index_type='min_abs', clamp_dist=clamp_dist, profile=profile, no_grad=no_grad_mask)
        marching_zdepth = self.get_sample_on_marching_zdepth_along_ray(marching_zdepth_list, sdf_list, points_list, cam_rays[:, valid_mask], latent, use_uniform_sample=False, index_type=sample_index_type, clamp_dist=clamp_dist, profile=profile, no_grad=no_grad_depth)
        profiler.report_process('[DEPTH] re-sampling time')

        # generate output
        min_sdf_sample_new = torch.zeros_like(valid_mask).float() # (H, W)
        min_sdf_sample_new.requires_grad = True
        min_sdf_sample_new = self.copy_index(min_sdf_sample_new, valid_mask, min_sdf_sample)
        min_sdf_sample_new = self.copy_index(min_sdf_sample_new, ~valid_mask, dist[~valid_mask] + self.threshold - self.radius) # help handle camera gradient

        ## get zdepth
        Zdepth = torch.ones_like(self.imgmap_init) * 1e11 # (H, W)
        Zdepth.requires_grad = True
        src_zdepth = init_zdepth[valid_mask] + marching_zdepth # (N)
        Zdepth = self.copy_index(Zdepth, valid_mask, src_zdepth)
        Zdepth = Zdepth.reshape(-1) # (H*W)

        ## update valid_mask
        valid_mask = valid_mask.clone()
        valid_mask[valid_mask] = valid_mask_render
        profiler.report_process('[DEPTH] finalize time\t')
        if no_grad_depth:
            Zdepth = Zdepth.detach()
        return Zdepth, valid_mask, min_sdf_sample_new # (H*W), (H*W), (H*W)

    def render_normal(self, latent, Zdepth, valid_mask, clamp_dist=0.1, MAX_POINTS=100000, no_grad=False, normalize=True):
        
        cam_pos, cam_rays, init_zdepth, _ = self.get_grid_and_rays()

        ### Getting normals
        h, w = self.img_hw
        Znormal = torch.zeros_like(self.imgmap_init)[None,:,:].repeat(3, 1, 1) # (3, H, W)
        Znormal.requires_grad = True

        # initialization
        valid_cam_pos = cam_pos[:, valid_mask]
        valid_cam_rays = cam_rays[:, valid_mask]
        valid_zdepth = Zdepth[valid_mask]
        if valid_zdepth.shape[0] == 0:
            return Znormal.reshape(3, -1) # (3, H*W)

        # compute normal
        points = self.generate_point_samples(valid_cam_pos, valid_cam_rays, valid_zdepth, has_zdepth_grad=False)
        gradient = decode_sdf_gradient(self.decoder, latent, points.transpose(1,0), clamp_dist=clamp_dist, no_grad=no_grad, MAX_POINTS=MAX_POINTS) # (N, 3)
        gradient = gradient.transpose(1,0) # (3, N)
        if normalize:
            valid_normal = self.normalize_vectors(gradient) # (3, N)
        else:
            valid_normal = gradient
        
        # generate output
        Znormal = self.copy_index(Znormal.permute(1,2,0), valid_mask, valid_normal.transpose(1,0)) # (H, W, 3)
        Znormal = Znormal.reshape(-1, 3).transpose(1,0)

        if no_grad:
            Znormal = Znormal.detach()
        return Znormal # (3, H*W)

        
if __name__ == '__main__':
    pass

