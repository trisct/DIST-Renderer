import torch
import json
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

def load_decoder(experiment_directory, checkpoint_num=None, color_size=None, experiment_directory_color=None, parallel=True):
    specs_filename = os.path.join(experiment_directory, "specs.json")

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    specs = json.load(open(specs_filename))
    if color_size is not None:
        specs['NetworkSpecs']['dims'][3] = specs['NetworkSpecs']['dims'][3] + color_size

    basedir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    arch = __import__("core.graph." + specs["NetworkArch"], fromlist=["Decoder"])

    if color_size is not None:
        latent_size = specs["CodeLength"] + color_size
        decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"], last_dim=3)
    else:
        latent_size = specs["CodeLength"]
        decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])

    if parallel:
        decoder = torch.nn.DataParallel(decoder)

    if checkpoint_num != None:
        if color_size is not None:
            saved_model_state = torch.load(
            os.path.join(experiment_directory_color, "ModelParameters", checkpoint_num + ".pth")
        )
            # since there is no prefix "module" in the saved decoder_color model, we add them
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in saved_model_state["model_state_dict"].items():
                name = 'module.' + k
                new_state_dict[name] = v
            saved_model_state["model_state_dict"] = new_state_dict
        else:
            saved_model_state = torch.load(
            os.path.join(experiment_directory, "ModelParameters", checkpoint_num + ".pth")
        )

        saved_model_epoch = saved_model_state["epoch"]
        decoder.load_state_dict(saved_model_state["model_state_dict"])
    return decoder

def decode_sdf(decoder, latent_vector, points, clamp_dist=0.1, MAX_POINTS=100000, no_grad=False):
    """
    decoder: DeepSDF decoder
    latent_vector: [256]
    points: [H*W, 3]
    MAX_POINTS: max number of points to process in a single batch

    The points are processed in batches in the following loop. 
    """
    
    print(f'[In decode_sdf] latent_vector.shape = {latent_vector.shape}') # [256]
    print(f'[In decode_sdf] points.shape = {points.shape}') # [H*W, 3]
    start, num_all = 0, points.shape[0]
    output_list = []
    while True:
        end = min(start + MAX_POINTS, num_all)
        if latent_vector is None:
            inputs = points[start:end]
        else:
            ### DeepSDF specific decoding scheme
            latent_repeat = latent_vector.expand(end - start, -1)
            inputs = torch.cat([latent_repeat, points[start:end]], 1)
        sdf_batch = decoder.inference(inputs)
        start = end
        if no_grad:
            sdf_batch = sdf_batch.detach()
        output_list.append(sdf_batch)
        if end == num_all:
            break
    sdf = torch.cat(output_list, 0) # [N_queries, 1], the number of queries may change according to pruning. 
    print(f'[In decode_sdf] sdf.shape = {sdf.shape}')

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

