import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import data_preprocessing
from pathlib import Path

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    Generate 2D sin-cos position embedding.
    
    Args:
        embed_dim (int): Dimension of the output embedding (must be even).
        grid_size (tuple or int): Height and width of the grid, e.g., (14, 14) or 14.
        cls_token (bool): Whether to add a [CLS] token at the beginning.

    Returns:
        torch.Tensor: Position embedding of shape [grid_h*grid_w (+1), embed_dim]
    """
    if isinstance(grid_size, int):
        grid_h = grid_w = grid_size
    else:
        grid_h, grid_w = grid_size

    yy, xx = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing='ij')
    coords = np.stack([yy, xx], axis=-1)
    coords = coords.reshape(-1, 2)

    pos_embed = get_1d_sincos_pos_embed_from_coords(embed_dim, coords, cls_token=cls_token)

    return pos_embed


def get_1d_sincos_pos_embed_from_coords(embed_dim, coords, cls_token=False):
    """
    Generate 2D sin-cos embedding based on 2D coordinates.
    
    Args:
        embed_dim (int): Dimension of the embedding (must be even).
        coords (np.ndarray): Coordinates of shape [n, 2], in format (row, col).
        cls_token (bool): Whether to prepend a zero embedding for [CLS].

    Returns:
        torch.Tensor: [n (+1), embed_dim]
    """
    assert embed_dim % 2 == 0, "Embed dimension must be even"

    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= (embed_dim // 2) - 1
    omega = 1.0 / 10000**omega

    out = np.einsum('ij,k->ikj', coords, omega)
    out = out.reshape(coords.shape[0], -1)

    emb = np.zeros((coords.shape[0], embed_dim), dtype=np.float64)
    emb[:, 0::2] = np.sin(out[:, 0::2])
    emb[:, 1::2] = np.cos(out[:, 1::2])

    if cls_token:
        emb = np.concatenate([np.zeros([1, embed_dim]), emb], axis=0)

    return emb

def build_static_mask(data_list, img_size=(420, 312), patch_size=(14, 12)):
    H, W = img_size
    pH, pW = patch_size
    h_num, w_num = H // pH, W // pW

    stacked = np.stack(data_list, axis=0)
    reshaped = stacked.reshape(-1, H, W)

    mask = np.zeros((h_num, w_num), dtype=bool)

    for i in range(h_num):
        for j in range(w_num):
            patch = reshaped[:, i*pH:(i+1)*pH, j*pW:(j+1)*pW]
            if not np.allclose(patch, 0, atol=1e-6):
                mask[i, j] = True

    return torch.from_numpy(mask)

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=(420, 312), patch_size=(14, 12), t_in = 6, in_chans=4, embed_dim=768, static_mask=None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_chans = in_chans
        self.t_in = t_in
        self.grid_h = img_size[0] // patch_size[0]
        self.grid_w = img_size[1] // patch_size[1]
        self.n_patches_total = self.grid_h * self.grid_w
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # self.proj = nn.Conv3d(in_chans * in_level * t_in,
        #                       embed_dim,
        #                       kernel_size=(t_in, self.grid_h, self.grid_w),
        #                       stride = (t_in, self.grid_h, self.grid_w))

        if static_mask is not None:
            assert static_mask.shape == (self.img_size[0], self.img_size[1]), \
                f"Mask shape {static_mask.shape} != img_size {self.img_size}"
            
            patch_mask = static_mask.view(
                self.grid_h, self.patch_size[0],
                self.grid_w, self.patch_size[1]
            )
            # print(patch_mask.shape)

            patch_mask = patch_mask.any(dim=(1, 3))
            # print(patch_mask.shape)

            self.register_buffer('patch_mask', patch_mask)

            valid_indices = torch.nonzero(patch_mask.flatten(), as_tuple=False).squeeze(1)
            self.register_buffer('valid_indices', valid_indices)
            # print(valid_indices)
            self.n_valid_patches = len(valid_indices)
            # print('self.n_valid_patches ', self.n_valid_patches)
        else:
            self.patch_mask = None
            self.valid_indices = None
            self.n_valid_patches = self.grid_h * self.grid_w

    def forward(self, x):
        B, T, C, H, W = x.shape
        assert (H == self.img_size[0]) and (W == self.img_size[1]), "Input size mismatch"

        x = x.reshape(B * T, C, H, W)
        x = self.proj(x)

        x = x.flatten(2).transpose(1, 2)
        x = x.reshape(B, T, self.n_patches_total, self.embed_dim)

        if self.valid_indices is not None:
            x = x[:, :, self.valid_indices, :]

        return x

class SpatioTemporalTransformer(nn.Module):
    def __init__(self, img_size=(420, 312), patch_size=(14, 12), in_chans=4, embed_dim=768,
                 depth=12, t_in=6, num_heads=12, mlp_ratio=4., dropout=0.1, static_mask=None):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, t_in, in_chans, embed_dim, static_mask)
        self.grid_h, self.grid_w = self.patch_embed.grid_h, self.patch_embed.grid_w
        self.n_patches = self.patch_embed.n_valid_patches
        self.embed_dim = embed_dim
        # print('self.n_patches ', self.n_patches)

        # if static_mask is not None:
        #     valid_count = static_mask.sum().item()
        #     pos_embed_full = get_2d_sincos_pos_embed(
        #         embed_dim, (self.grid_h, self.grid_w), cls_token=False
        #     )
        #     pos_embed_full = torch.from_numpy(pos_embed_full).float()
        #     pos_embed_valid = pos_embed_full[static_mask.flatten()]
        #     self.register_buffer('pos_embed', pos_embed_valid)
        # else:
        #     valid_count = self.grid_h * self.grid_w
        #     self.pos_embed = nn.Parameter(torch.zeros(1, valid_count, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(t_in, self.n_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True
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
        B, T, C, H, W = x.shape
        x = self.patch_embed(x)
        # print(x.shape, ' ', self.pos_embed.shape)
        x = x + self.pos_embed[:T]
        x = self.pos_drop(x)
        x = x.reshape(B, T * self.n_patches, self.embed_dim)
        x = self.transformer(x)
        x = self.norm(x)
        return x
class Decoder(nn.Module):
    def __init__(self, 
                 embed_dim=768, 
                 out_chans=4,
                 t_out=1,
                 patch_size=(14, 12),
                 img_size=(420, 312),
                 static_mask=None):
        super().__init__()
        self.patch_size = patch_size
        self.t_out = t_out
        self.img_size = img_size
        self.out_chans = out_chans
        self.grid_h = img_size[0] // patch_size[0]
        self.grid_w = img_size[1] // patch_size[1]
        self.embed_dim = embed_dim
        self.upscale_h, self.upscale_w = patch_size

        if static_mask is not None:
            assert static_mask.shape == img_size, f"Mask shape {static_mask.shape} != img_size {img_size}"

            patch_mask = static_mask.view(self.grid_h, self.upscale_h, self.grid_w, self.upscale_w)
            patch_mask = patch_mask.any(dim=(1, 3))
            self.register_buffer('patch_mask', patch_mask)

            valid_indices = torch.nonzero(patch_mask.flatten(), as_tuple=False).squeeze(1)
            self.register_buffer('valid_indices', valid_indices)
            self.n_valid_patches = len(valid_indices)
        else:
            self.patch_mask = None
            self.valid_indices = None
            self.n_valid_patches = self.grid_h * self.grid_w

        self.expanded_channels = out_chans * self.upscale_h * self.upscale_w
        self.proj = nn.Linear(embed_dim, self.expanded_channels)

        self.post_conv = nn.Sequential(
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1)
        )

    def forward(self, x):
        B, L_total, D = x.shape

        x = x[:, -self.n_valid_patches:, :]

        x = self.proj(x)

        C, ph, pw = self.out_chans, self.upscale_h, self.upscale_w
        x = x.reshape(B, self.n_valid_patches, C, ph, pw)

        device = x.device
        feat_map = torch.zeros(B, self.grid_h * self.grid_w, C, ph, pw, device=device)

        if self.valid_indices is not None:
            for i, idx in enumerate(self.valid_indices):
                feat_map[:, idx] = x[:, i]
        else:
            feat_map = x

        feat_map = feat_map.reshape(B, self.grid_h, self.grid_w, C, ph, pw)
        feat_map = feat_map.permute(0, 3, 1, 4, 2, 5)
        feat_map = feat_map.reshape(B, C, self.img_size[0], self.img_size[1])

        out = self.post_conv(feat_map)

        out = out.reshape(B, self.t_out, self.out_chans, self.img_size[0], self.img_size[1])

        return out

class OceanForecastNet(nn.Module):
    def __init__(self, img_size=(420, 312), patch_size=(14, 12), in_chans=4, out_chans=4,
                 t_in=6, t_out=1, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 dropout=0.1, static_mask=None):
        super().__init__()
        self.t_in = t_in
        self.t_out = t_out
        self.encoder = SpatioTemporalTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            t_in = t_in,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            static_mask=static_mask
        )
        self.decoder = Decoder(
            embed_dim=embed_dim,
            out_chans=out_chans,
            patch_size=patch_size,
            t_out=t_out,
            img_size=img_size,
            static_mask=static_mask
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
        
        # [21, 312, 420]
        return prediction

import torch
import torch.nn as nn

class MaskedWeightedMAEMSELoss(nn.Module):
    def __init__(self, mask):
        super().__init__()
        self.register_buffer('mask', mask.float())

    def forward(self, pred, target):
        B, T, C = pred.shape[:3]
        mask = self.mask
        diff = (pred - target) ** 2
        normal_loss = diff
        total_valid = mask.sum() * B * T * C
        normal_loss = normal_loss.sum() / (total_valid + 1e-8)
        return normal_loss
        # masked_diff = diff * mask
        # return masked_diff.sum() / (mask.sum() + 1e-8) / pred.shape[1] / pred.shape[2]


    # def __init__(self, mask=None, var_weights=None, tv_weight=0.0):
    #     super().__init__()
    #     if mask is not None:
    #         if mask.ndim == 2:
    #             mask = mask.unsqueeze(0).unsqueeze(0)
    #         mask = mask.unsqueeze(0).unsqueeze(0)
    #     self.register_buffer('mask', mask)

    #     if var_weights is None:
    #         var_weights = [2.0, 2.0, 1.0, 1.0]
    #     var_weights = torch.tensor(var_weights, dtype=torch.float32)
    #     self.register_buffer('var_weights', var_weights.view(1, 1, -1, 1, 1, 1))

    #     self.tv_weight = tv_weight

    # def tv_loss(self, x, mask=None):
    #     dx = torch.abs(x[:, :, :, :, 1:, :] - x[:, :, :, :, :-1, :])  # [B, T, C, L, H-1, W]
    #     dy = torch.abs(x[:, :, :, :, :, 1:] - x[:, :, :, :, :, :-1])  # [B, T, C, L, H, W-1]

    #     if mask is not None:
    #         mask_dx = mask[:, :, :, :, :-1, :]
    #         mask_dy = mask[:, :, :, :, :, :-1]
    #         dx = dx * mask_dx
    #         dy = dy * mask_dy

    #     tv = dx.sum() + dy.sum()
    #     return tv

    # def forward(self, pred, target):
    #     weight1 = 0.0
    #     weight2 = 1.0

    #     diff = pred - target
    #     l1_loss = torch.abs(diff)
    #     l2_loss = diff ** 2
    #     normal_loss = weight1 * l1_loss + weight2 * l2_loss

    #     normal_loss = normal_loss * self.var_weights

    #     if self.mask is not None:
    #         normal_loss = normal_loss * self.mask
    #         B, T, C, L = pred.shape[:4]
    #         total_valid = self.mask.sum() * B * T * C * L
    #         normal_loss = normal_loss.sum() / (total_valid + 1e-8)
    #     else:
    #         normal_loss = normal_loss.mean()

        # total_loss = recon_loss
        # if self.tv_weight > 0.0:
        #     tv = self.tv_loss(pred, self.mask)
        #     if self.mask is not None:
        #         H, W = pred.shape[-2:]
        #         valid_grads = self.mask.sum() * pred.size(0) * pred.size(1) * pred.size(2) * pred.size(3)
        #         valid_grads = valid_grads * ( (H - 1) / H + (W - 1) / W ) / 2.0
        #         tv = tv / (valid_grads + 1e-8)
        #     else:
        #         tv = tv / pred.numel()
        #     total_loss = total_loss + self.tv_weight * tv

        return normal_loss

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
    file_paths = list(Path(data_dir).glob("*.npz"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    assert len(file_paths) == 20, f"Expected 20 files, got {len(file_paths)}"

    file_paths = sorted(file_paths, key=lambda x: int(x.stem.split('_')[-1]))

    data_list = load_data(file_paths)

    # model = OceanForecastNet(
    #     img_size=(420, 312),
    #     patch_size=(4, 4),
    #     in_chans=4,
    #     out_chans=4,
    #     levels=5,
    #     t_in=1,
    #     embed_dim=7,
    #     depth=2,
    #     num_heads=2
    # )
    # input_dataset = create_input_dataset()
    mask_matrix = build_static_mask(data_list, (420, 312), (4, 4))
    print(mask_matrix.shape)
    # data_preprocessing.mask_visualization(mask_matrix)
    # output = model.predict('checkpoints/' + save_pth_name, input_dataset)
    # print(output.shape)