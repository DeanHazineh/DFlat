from dflat.metasurface import load_trainer, load_optical_model
import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt

# Call trainer on model
# config_path = "metasurface/ckpt/Nanofins_TiO2_U350H600_Medium/config.yaml"
# config_path = 'metasurface/ckpt/Nanocylinders_TiO2_U180H600_Medium/config.yaml'
# trainer = load_trainer(config_path)
# trainer.train()

## Test out the model
config_path = "metasurface/ckpt/Nanofins_TiO2_U350H600_Medium/config.yaml"
model = load_optical_model(config_path)
model = model.to("cuda")

lam = np.linspace(0, 1, 31)
lx = np.linspace(0, 1, 100)
lx, ly = np.meshgrid(lx, lx)
p = np.stack([lx.flatten(), ly.flatten()]).T
p = p[None, None]

amp, phase = model(p, lam, pre_normalized=True)
print(amp.shape, phase.shape)
amp = amp.view(31, 100, 100, 2).cpu().numpy()
phase = phase.view(31, 100, 100, 2).cpu().numpy()
print(amp.shape, phase.shape)

fig, ax = plt.subplots(2, 2)
for i in range(2):
    ax[i, 0].imshow(amp[16, :, :, i])
    ax[i, 1].imshow(phase[16, :, :, i])
plt.show()
