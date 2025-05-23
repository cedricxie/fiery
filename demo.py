import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from fiery.trainer import TrainingModule
from visualise import download_example_data, plot_prediction

trainer = TrainingModule.load_from_checkpoint('fiery.ckpt', strict=True)
device = torch.device('cuda:0')
trainer = trainer.to(device)
trainer.eval()

# Download data
download_example_data()

# Load data
predictions = []
for data_path in sorted(glob('./example_data/*.npz')):
    data = np.load(data_path)
    image = torch.from_numpy(data['image']).to(device)
    intrinsics = torch.from_numpy(data['intrinsics']).to(device)
    extrinsics = torch.from_numpy(data['extrinsics']).to(device)
    future_egomotions = torch.from_numpy(data['future_egomotion']).to(device)

    # Forward pass
    with torch.no_grad():
        output = trainer.model(image, intrinsics, extrinsics, future_egomotions)

    print("output: ")
    for k, v in output.items():
        print(k, v.shape if hasattr(v, "shape") else v)

    figure_numpy = plot_prediction(image, output, trainer.cfg)
    os.makedirs('./output_vis', exist_ok=True)
    output_filename = os.path.join('./output_vis', os.path.basename(data_path).split('.')[0]) + '.png'
    Image.fromarray(figure_numpy).save(output_filename)
    print(f'Saved output in {output_filename}')
