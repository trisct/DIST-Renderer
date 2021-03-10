import os
import matplotlib
import matplotlib.pyplot as plt

import torch
import numpy as np
from core.sdfrenderer import SDFRenderer


def load_decoder(checkpoint_path=None, parallel=True):
    

    arch = __import__("core.graph." + "deep_sdf_decoder", fromlist=["Decoder"])

    latent_size = 256
    decoder = arch.Decoder(latent_size, **{
        "dims" : [ 512, 512, 512, 512, 512, 512, 512, 512 ],
        "dropout" : [0, 1, 2, 3, 4, 5, 6, 7],
        "dropout_prob" : 0.2,
        "norm_layers" : [0, 1, 2, 3, 4, 5, 6, 7],
        "latent_in" : [4],
        "xyz_in_all" : False,
        "use_tanh" : True,
        "latent_dropout" : False,
        "weight_norm" : True
    })

    if parallel:
        decoder = torch.nn.DataParallel(decoder)

    if checkpoint_path != None:
        saved_model_state = torch.load(os.path.join(checkpoint_path))

        decoder.load_state_dict(saved_model_state["model_state_dict"])
    
    return decoder

decoder = load_decoder('trained_models/99000.pth')
decoder = decoder.module.cuda()

intrinsics = torch.tensor([
    [100.,   0., 150.],
    [  0., 100., 150.],
    [  0.,   0.,   1.]
]).cuda()

renderer = SDFRenderer(decoder, intrinsics)

latent_codes = torch.load('trained_models/lcode/99000.pth')['latent_codes']['weight'].cuda()

R = torch.eye(3).cuda()
T = torch.zeros(3).cuda()
T[2] = .8

res = renderer.render_depth(latent_codes[0], R, T, use_transform=False)
res_reshaped = []

for r in res:
    res_reshaped.append(r.reshape(300,300).detach().cpu().numpy())

#matplotlib.use('GTK3Agg')

plt.imshow(res_reshaped[0])
plt.savefig('zdepth.png')

masked_0 = res_reshaped[0].copy()
masked_0[ ~res_reshaped[1] ] =0.

plt.imshow(masked_0)
plt.savefig('zdepth_masked.png')

plt.imshow(res_reshaped[1])
plt.savefig('valid_mask.png')

plt.imshow(res_reshaped[2])
plt.savefig('min_sdf_sample_new.png')

masked_2 = res_reshaped[2].copy()
masked_2[ ~res_reshaped[1] ] = 0.

plt.imshow(masked_2)
plt.savefig('min_sdf_sample_new_masked.png')
