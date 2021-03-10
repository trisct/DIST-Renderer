import os
import matplotlib
import matplotlib.pyplot as plt

import torch
import numpy as np
from core.sdfrenderer.renderer_orthogonal import SDFRenderer



decoder = None
latent_codes = None

hw = (256, 300)
renderer = SDFRenderer(decoder, img_hw=hw)

latent_codes = torch.load('trained_models/lcode/99000.pth')['latent_codes']['weight'].cuda()

res = renderer.render_depth(latent_codes[0])
res_reshaped = []

for r in res:
    res_reshaped.append(r.reshape(hw[0], hw[1]).detach().cpu().numpy())

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

normal_valid_mask = torch.ones_like(res[0], device=res[0].device).bool()
res_normal = renderer.render_normal(latent_codes[0], res[0], normal_valid_mask, normalize=False)
normal_map = res_normal.reshape(3, hw[0], hw[1]).detach().cpu().permute(1,2,0)

plt.imshow(normal_map)
plt.savefig('normal.png')

normal_masked = normal_map.clone()
normal_masked[ ~torch.tensor(res_reshaped[1].reshape(hw[0], hw[1], 1)).bool().expand(hw[0], hw[1], 3) ] = 0.
plt.imshow(normal_masked)
plt.savefig('normal_masked.png')

#matplotlib.use('GTK3Agg')