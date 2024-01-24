from dflat.metasurface import load_trainer

config_path = 'metasurface/ckpt/Nanofins_TiO2_U350H600_Medium/config.yaml'
trainer = load_trainer(config_path)
trainer.train()