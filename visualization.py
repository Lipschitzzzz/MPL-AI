import numpy as np
import torch
import visiontransformer
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from data_preprocessing import img_comparison

if __name__ == "__main__":
    model = visiontransformer.OceanForecastNet(
        img_size=(420, 312),
        patch_size=(4, 4),
        in_chans=4,
        out_chans=4,
        levels=5,
        T_in=6,
        embed_dim=192,
        depth=2,
        num_heads=2
    )
    data = np.load("npz_dataset.npz")["data"]
    input_seq = np.expand_dims(data[2], axis=0)
    input_seq = torch.tensor(input_seq, dtype=torch.float32)

    input_seq = input_seq.unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_seq = input_seq.to(device)
    output = model.predict('checkpoints/2 months 10.30 model.pth', input_seq)
    # [4, 5, 312, 420]
    actual_value = data[3]
    print(output.shape)
    print(actual_value.shape)
    print("MAE:", np.mean(np.abs(actual_value - output)))
    img_comparison(actual_value, output, 0, 0)
