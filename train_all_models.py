from train_baseline import train_baseline_keras
import os
import warnings
# 1) Hide Python warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", FutureWarning)

def train_all(backbone_names,size,chan,epochs,batch_size,lr,out_dir,model_name,folders):
    for backbone in backbone_names:
        train_baseline_keras(backbone_name = backbone,img_size=size, chan = chan, epochs = epochs, batch_size = batch_size,lr = lr, out_dir = out_dir, model_name = model_name,folders = folders)
