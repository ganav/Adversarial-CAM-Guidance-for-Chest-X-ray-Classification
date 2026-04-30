from train_proposed_framing import train_proposed


import os
import warnings

backbone_names=["mobilenet","efficientnetb0","lightweight_fe_mf","densenet121"]
model_name="baseline_keras.h5"

for i in backbone_names:
    train_proposed(
        data_root=r"G:\projects\paper 25\source\data_proposed",
        backbone_name=i,
        img_size=299,
        chan=1,
        batch_size=4,
        epochs=30,
        lambda_adv=1.0,
        warmup_epochs=1,
        lr_cls=1e-4,
        lr_disc=1e-4,
        out_dir="artifacts_proposed"
    )

print('main py is complete...')