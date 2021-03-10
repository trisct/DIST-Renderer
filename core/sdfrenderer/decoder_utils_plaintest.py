import torch
import json
import os, sys
import numpy as np


def decode_sdf(decoder, latent_vector, points, clamp_dist=0.1, MAX_POINTS=100000, no_grad=False):
    """
    decoder: None
    latent_vector: None
    points: [H*W, 3]
    MAX_POINTS: max number of points to process in a single batch

    -----------------

    This function ignores the decoder and the latent code. It always returns the sdf value by

    SDF(x,y,z) = x+y-z
    """
    
    print(f'[In decode_sdf] points.shape = {points.shape}') # [H*W, 3]
    start, num_all = 0, points.shape[0]
    output_list = []
    while True:
        end = min(start + MAX_POINTS, num_all)
        
        points_batch = points[start:end] # [N_batch, 3]
        sdf_batch = points_batch[:, 0] + points_batch[:, 1] - points_batch[:, 2] # [N_batch]
        sdf_batch = sdf_batch[:, None] # [N_batch, 1]
        start = end
        if no_grad:
            sdf_batch = sdf_batch.detach()
        output_list.append(sdf_batch)
        if end == num_all:
            break
    sdf = torch.cat(output_list, 0)

    if clamp_dist != None:
        sdf = torch.clamp(sdf, -clamp_dist, clamp_dist)
    return sdf

def decode_sdf_gradient(decoder, latent_vector, points, clamp_dist=0.1, MAX_POINTS=100000, no_grad=False):
    """
    This is to get the sdf gradient with merely the torch create_graph and retain_graph functionality.
    
    If any change is being made, you only need to modify 'decode_sdf' above.
    """
    print(f'[In decode_sdf_gradient] Entering...')
    start, num_all = 0, points.shape[0]
    output_list = []
    while True:
        end = min(start + MAX_POINTS, num_all)
        points_batch = points[start:end]
        sdf = decode_sdf(decoder, latent_vector, points_batch, clamp_dist=clamp_dist)
        start = end

        print(f'[In decode_sdf_gradient] In loop. sdf.shape = {sdf.shape}')
        print(f'[In decode_sdf_gradient] In loop, points_batch.shape = {points_batch.shape}')
        
        
        grad_tensor = torch.autograd.grad(outputs=sdf, inputs=points_batch, grad_outputs=torch.ones_like(sdf), create_graph=True, retain_graph=True) # A tuple of length 1
        grad_tensor = grad_tensor[0]
        print(f'[In decode_sdf_gradient] In loop, grad_tensor.shape = {grad_tensor.shape}')

        if no_grad:
            grad_tensor = grad_tensor.detach()
        output_list.append(grad_tensor)
        if end == num_all:
            break
    grad_tensor = torch.cat(output_list, 0)
    return grad_tensor

def decode_color(decoder, color_code, shape_code, points, MAX_POINTS=100000, no_grad=False):
    start, num_all = 0, points.shape[0]
    output_list = []
    while True:
        end = min(start + MAX_POINTS, num_all)

        color_code_batch = color_code.expand(end - start, -1)
        shape_code_batch = shape_code.expand(end - start, -1)
        inputs = torch.cat([shape_code_batch, color_code_batch, points[start:end]], 1)

        color_batch = decoder.inference(inputs)
        start = end
        if no_grad:
            color_batch = color_batch.detach()
        output_list.append(color_batch)
        if end == num_all:
            break
    color = torch.cat(output_list, 0)
    return color

