import numpy as np
import torch
import visiontransformer
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from data_preprocessing import img_comparison
from pathlib import Path
import data_preprocessing
import json

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    return total_params, trainable_params
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_list = np.load("68_months_npz_dataset_normalized.npz")["data"]
    static_mask = visiontransformer.build_static_mask(data_list, img_size=(420, 312), patch_size=(1, 1))

    model = visiontransformer.OceanModel(static_mask=static_mask).to(device)
    count_parameters(model)

    # input_seq = data[k:k+6]
    input_seq = np.expand_dims(data_list[12], axis=0)
    input_seq = torch.tensor(input_seq, dtype=torch.float32)

    input_seq = input_seq.unsqueeze(0)
    input_seq = input_seq.to(device)
    output = model.predict('checkpoints/2025_11_07_11_59_local_model.pth', input_seq)
    # [4, 5, 312, 420]
    # actual_value = data[k+6]
    actual_value = data_list[13]
    print('output value shape: ', output.shape)
    print('actual value shape:' , actual_value.shape)
    with open('norm_stats.json', 'r') as f:
        data = json.load(f)
        mean = np.array(data['mean'])
        std  = np.array(data['std'])
        stats = {
            'mean': mean,
            'std': std,
        }
    vars = ['temperature', 'salinity', 'u velocity', 'v velocity']
    levels = ['0.49 m', '2.64 m', '5.07 m', '7.92 m', '11.4 m']

    # array([4.940250e-01, 1.541375e+00, 2.645669e+00, 3.819495e+00, 5.078224e+00,
    #    6.440614e+00, 7.929560e+00, 9.572997e+00, 1.140500e+01, 1.346714e+01,
    #    1.581007e+01, 1.849556e+01, 2.159882e+01, 2.521141e+01, 2.944473e+01,
    #    3.443415e+01, 4.034405e+01, 4.737369e+01, 5.576429e+01, 6.580727e+01,
    #    7.785385e+01, 9.232607e+01, 1.097293e+02, 1.306660e+02, 1.558507e+02,
    #    1.861256e+02, 2.224752e+02, 2.660403e+02, 3.181274e+02, 3.802130e+02,
    #    4.539377e+02, 5.410889e+02, 6.435668e+02, 7.633331e+02, 9.023393e+02,
    #    1.062440e+03, 1.245291e+03, 1.452251e+03, 1.684284e+03, 1.941893e+03,
    #    2.225078e+03, 2.533336e+03, 2.865703e+03, 3.220820e+03, 3.597032e+03,
    #    3.992484e+03, 4.405224e+03, 4.833291e+03, 5.274784e+03, 5.727917e+03],
    #   dtype=float32)

    # for i in range(0, 4):
    #     output[0] = output[0] * std[i] + mean[i]
    #     actual_value = actual_value * std[i] + mean[i]
    #     output[0] = output[0] * static_mask.numpy()
    #     actual_value = actual_value * static_mask.numpy()
    #     for j in range(0, 5):
    #         img_comparison(actual_value, output[0], i, j, vars[i] + " level " + str(j) + " comparison 68 months")
    
    criterion = visiontransformer.MaskedWeightedMAEMSELoss(mask=torch.tensor(static_mask))
    print(output.shape)
    print(actual_value.shape)
    print("Loss: ", criterion(actual_value, output))
    actual_value = data_preprocessing.denormalize(torch.tensor(actual_value).unsqueeze(0), stats, static_mask).squeeze(0)
    output = data_preprocessing.denormalize(torch.tensor(output).unsqueeze(0), stats, static_mask).squeeze(0)
    

    # output = output * std[:, None, None]  + mean[:, None, None]
    # actual_value = actual_value * std[:, None, None]  + mean[:, None, None]

    for i in range(0, 20):
        img_comparison(actual_value, output, i, vars[i//5] + " level " + levels[i%5] + " comparison 68 months")
    img_comparison(actual_value, output, 20, "Sea Surface Height Comparison 68 months")
    
