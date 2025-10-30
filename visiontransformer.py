import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from pathlib import Path

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=(420, 312), patch_size=(14, 12), in_chans=4, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_h = img_size[0] // patch_size[0]
        self.grid_w = img_size[1] // patch_size[1]
        self.n_patches = self.grid_h * self.grid_w
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans * 5, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, T, C, L, H, W = x.shape
        assert (H == self.img_size[0]) and (W == self.img_size[1])

        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * T, C * L, H, W)

        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class SpatioTemporalTransformer(nn.Module):
    def __init__(self, img_size=(420, 312), patch_size=(14, 12), in_chans=4,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        self.n_patches = self.patch_embed.n_patches
        self.embed_dim = embed_dim

        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_layer_weights)

    def _init_layer_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, T, C, L, H, W = x.shape

        x = self.patch_embed(x)


        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = x.reshape(B, T * self.n_patches, self.embed_dim)

        x = self.transformer(x)
        x = self.norm(x)

        return x

class Decoder(nn.Module):
    def __init__(self, 
                 embed_dim=768, 
                 out_chans=4, 
                 levels=5, 
                 patch_size=(14, 12),
                 img_size=(420, 312)):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.out_chans = out_chans
        self.levels = levels
        self.grid_h = img_size[0] // patch_size[0]
        self.grid_w = img_size[1] // patch_size[1]
        self.embed_dim = embed_dim
        self.upscale_h, self.upscale_w = patch_size

        self.expanded_channels = out_chans * levels * self.upscale_h * self.upscale_w

        self.proj = nn.Linear(embed_dim, self.expanded_channels)

        self.post_conv = nn.Sequential(
            nn.Conv2d(out_chans * levels, out_chans * levels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_chans * levels, out_chans * levels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        B, L_total, E = x.shape
        n_patches = self.grid_h * self.grid_w

        x = x[:, -n_patches:, :]

        x = self.proj(x)

        x = x.reshape(B, self.grid_h, self.grid_w, self.expanded_channels)
        x = x.permute(0, 3, 1, 2)

        x = pixel_shuffle_2d(x, self.upscale_h, self.upscale_w)

        x = self.post_conv(x)

        x = x.reshape(B, self.out_chans, self.levels, self.img_size[0], self.img_size[1])
        return x

class OceanForecastNet(nn.Module):
    def __init__(self, img_size=(420, 312), patch_size=(14, 12), in_chans=4,
                 out_chans=4, levels=5, T_in=6, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.encoder = SpatioTemporalTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        self.decoder = Decoder(
            embed_dim=embed_dim,
            out_chans=out_chans,
            levels=levels,
            patch_size=patch_size,
            img_size=img_size
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    def predict(self, checkpoint_name, input_data):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_name, map_location=device)

        self.load_state_dict(checkpoint['model_state_dict'])

        self.eval()
        with torch.no_grad():
            output = self(input_data)

        prediction = output.squeeze(0).cpu().numpy()
        
        # [4, 312, 420]
        return prediction


class OceanForecastDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
        self.T = len(data_list)
        self.seq_len = 6

    def __len__(self):
        return self.T - self.seq_len

    def __getitem__(self, idx):
        input_seq = self.data[idx:idx + self.seq_len]
        input_seq = np.stack(input_seq, axis=0)

        target = self.data[idx + self.seq_len]

        input_seq = torch.from_numpy(input_seq).float()
        target = torch.from_numpy(target).float()

        return input_seq, target

def load_data(file_paths):
    data_list = []
    for fp in sorted(file_paths):
        with np.load(fp) as data:
            arr = data['data']
            assert arr.shape == (4, 5, 420, 312), f"Wrong shape: {arr.shape}"
            data_list.append(arr)
    return data_list

def create_tensor(data_list, indices, seq_len=1):
    input_seq_list = []
    target_list = []
    for start_idx in indices:
        input_frames = data_list[start_idx : start_idx + seq_len]
        input_seq = np.stack(input_frames, axis=0)

        target = data_list[start_idx + seq_len]
        target_seq = np.expand_dims(target, axis=0)

        input_seq_list.append(input_seq)
        target_list.append(target_seq)

    inputs_tensor = torch.from_numpy(np.stack(input_seq_list, axis=0)).float()
    targets_tensor = torch.from_numpy(np.stack(target_list, axis=0)).float()

    return inputs_tensor, targets_tensor

def create_input_dataset():

    data = np.load("npz_dataset.npz")["data"]
    input_seq = np.expand_dims(data[2], axis=0)
    input_seq = torch.tensor(input_seq, dtype=torch.float32)

    input_seq = input_seq.unsqueeze(0)
    input_seq = input_seq.to(device)
    return input_seq

def pixel_shuffle_2d(x, upscale_h, upscale_w):
    B, C_r2, H, W = x.shape
    C = C_r2 // (upscale_h * upscale_w)
    x = x.reshape(B, C, upscale_h, upscale_w, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3)
    x = x.reshape(B, C, H * upscale_h, W * upscale_w)
    return x

class SubsetByIndices(Dataset):
    def __init__(self, full_data_paths, indices):
        self.full_data_paths = full_data_paths
        self.indices = list(indices)
        self.seq_len = 1

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        
        input_seq = []
        for i in range(start_idx, start_idx + self.seq_len):
            input_seq.append(np.load(self.full_data_paths[i])['data'])
        input_seq = np.stack(input_seq, axis=0)

        target = np.load(self.full_data_paths[start_idx + self.seq_len])['data']

        return torch.from_numpy(input_seq).float(), torch.from_numpy(target).float()

def tv_loss(x, beta=2.0):
    dh = torch.abs(x[..., 1:, :] - x[..., :-1, :])
    dw = torch.abs(x[..., :, 1:] - x[..., :, :-1])
    return (dh ** beta).mean() + (dw ** beta).mean()

if __name__ == "__main__":
    data_dir = "NpzDataset-thetaosouovo"
    save_pth_name = "10.30 best_model 200 1 months.pth"
    file_paths = list(Path(data_dir).glob("*.npz"))[:4]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    assert len(file_paths) == 4, f"Expected 4 files, got {len(file_paths)}"

    file_paths = sorted(file_paths, key=lambda x: int(x.stem.split('_')[-1]))

    data_list = load_data(file_paths)

    model = OceanForecastNet(
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
    input_dataset = create_input_dataset()
    output = model.predict('checkpoints/' + save_pth_name, input_dataset)

    print(output.shape)