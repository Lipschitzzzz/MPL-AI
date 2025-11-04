import numpy as np
import torch
import visiontransformer
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from data_preprocessing import img_comparison
from pathlib import Path
import json

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    return total_params, trainable_params
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "NpzDataset-thetaosouovo"
    file_paths = list(Path(data_dir).glob("*.npz"))
    data_list = visiontransformer.load_data(file_paths)
    patch_size = (4, 4)
    static_mask = visiontransformer.build_static_mask(data_list, img_size=(420, 312), patch_size=(1, 1))
    # print(static_mask.shape)
    model = visiontransformer.OceanForecastNet(
        img_size=(420, 312),
        patch_size=patch_size,
        in_chans=4,
        out_chans=4,
        levels=5,
        t_in=1,
        t_out=1,
        embed_dim=384,
        depth=2,
        num_heads=2,
        static_mask=static_mask
    ).to(device)
    # count_parameters(model)
    # print(static_mask.shape)
    # mask = static_mask.bool()
    # mask = mask.flip(dims=[0])
    # mask_np = mask.numpy()
    # mask_large = np.kron(mask_np, np.ones((2,2)))
    # print(mask_large.shape)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(mask_large, cmap='gray', interpolation='none')
    # plt.savefig("Land Mask.jpg")
    # plt.show()

    data = np.load("npz_dataset_normalized.npz")["data"]
    input_seq = np.expand_dims(data[17], axis=0)
    # input_seq = np.expand_dims(data[2], axis=0)
    input_seq = torch.tensor(input_seq, dtype=torch.float32)

    input_seq = input_seq.unsqueeze(0)
    input_seq = input_seq.to(device)
    output = model.predict('checkpoints/11.03-17.40 model.pth', input_seq)
    # [4, 5, 312, 420]
    actual_value = data[18]
    print(output[0].shape)
    print(actual_value.shape)
    # print("Loss:", np.mean(np.abs(actual_value - output[0])))
    # print(np.max(actual_value[0, 0]))
    # print(np.max(actual_value[1, 0]))
    # print(np.max(actual_value[2, 0]))
    # print(np.max(actual_value[3, 0]))
    # print(np.min(actual_value[0, 0]))
    # print(np.min(actual_value[1, 0]))
    # print(np.min(actual_value[2, 0]))
    # print(np.min(actual_value[3, 0]))
    with open('norm_stats.json', 'r') as f:
        data = json.load(f)
        mean = data['mean']
        std  = data['std']
    print(mean, std)
    vars = ['temperature', 'salinity', 'u velocity', 'v velocity']
    for i in range(0, 4):
        output[0] = output[0] * std[i] + mean[i]
        actual_value = actual_value * std[i] + mean[i]
        output[0] = output[0] * static_mask.numpy()
        actual_value = actual_value * static_mask.numpy()
        for j in range(0, 5):
            img_comparison(actual_value, output[0], i, j, vars[i] + " level " + str(j) + " comparison 1 month")
