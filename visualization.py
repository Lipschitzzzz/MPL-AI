import numpy as np
import torch
import visiontransformer
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from data_preprocessing import img_comparison
from pathlib import Path

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "NpzDataset-thetaosouovo"
    file_paths = list(Path(data_dir).glob("*.npz"))
    data_list = visiontransformer.load_data(file_paths)
    patch_size = (4, 4)
    # static_mask = visiontransformer.build_static_mask(data_list, img_size=(420, 312), patch_size=patch_size)
    # print(static_mask.shape)
    model = visiontransformer.OceanForecastNet(
        img_size=(420, 312),
        patch_size=patch_size,
        in_chans=4,
        out_chans=4,
        levels=5,
        T_in=1,
        T_out=1,
        embed_dim=384,
        depth=2,
        num_heads=2
        # static_mask=static_mask
    ).to(device)

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

    data = np.load("npz_dataset.npz")["data"]
    input_seq = np.expand_dims(data[3], axis=0)
    # input_seq = np.expand_dims(data[2], axis=0)
    input_seq = torch.tensor(input_seq, dtype=torch.float32)
    print(input_seq.shape)

    input_seq = input_seq.unsqueeze(0)
    input_seq = input_seq.to(device)
    output = model.predict('checkpoints/10.31--17.38 model.pth', input_seq)
    # [4, 5, 312, 420]
    actual_value = data[4]
    print(output[0].shape)
    print(actual_value.shape)
    print("Loss:", np.mean(np.abs(actual_value - output[0])))
    img_comparison(actual_value, output[0], 0, 0)
