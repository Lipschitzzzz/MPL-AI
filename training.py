import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import visiontransformer
import torch.nn.functional as F
import torch.nn as nn
from piqa import SSIM
class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        # Sobel 算子
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, bias=False, padding=1)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, bias=False, padding=1)
        sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_x.weight = nn.Parameter(sobel_kernel_x)
        self.sobel_y.weight = nn.Parameter(sobel_kernel_y)
        for param in self.sobel_x.parameters():
            param.requires_grad = False
        for param in self.sobel_y.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        print(pred.shape)
        B, T, C, L, H, W = pred.shape
        total_loss = 0

        for c in range(C):
            p = pred[:, c:c+1]  # [B,1,H,W]
            t = target[:, c:c+1]

            # 计算梯度
            p_x = self.sobel_x(p)
            p_y = self.sobel_y(p)
            t_x = self.sobel_x(t)
            t_y = self.sobel_y(t)

            # L1 损失 on gradient
            loss_x = F.l1_loss(p_x, t_x)
            loss_y = F.l1_loss(p_y, t_y)
            total_loss += loss_x + loss_y

        return total_loss / C
def train_zero_epoch(model, train_loader, val_loader, num_epochs, checkpoint_name_out):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # criterion = torch.nn.L1Loss()
    best_loss = float('inf')
    tv_weight = 0.3
    early_stop_cnt= 0
    best_epoch = 0
    model.train()
    gradient_loss = GradientLoss()
    
    for epoch in range(num_epochs):
        train_loss = 0.0
        for inp, target in train_loader:
            # print(f"inp device: {inp.device}, tgt device: {tgt.device}")
            # print(f"model device: {next(model.parameters()).device}")  # 应该是 'cuda'

            pred = model(inp)
            # l1loss = criterion(pred, tgt)
            print(pred.shape)
            print(target.shape)
            tv_loss_val = visiontransformer.tv_loss(pred)
            pred = model(inp)
            # l1loss = criterion(output, target)
            # grad = gradient_loss(pred, target)
            loss = 5.0 * F.l1_loss(pred, target) + 1.0 * F.mse_loss(pred, target) + tv_weight * tv_loss_val

            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inp, target in val_loader:

                pred = model(inp)
                # l1loss = criterion(output, target)
                tv_loss_val = visiontransformer.tv_loss(pred)
                # loss = l1loss + tv_weight * tv_loss_val
                # grad = gradient_loss(pred, target)
                # loss = 1.0 * F.l1_loss(pred, target) + 1.0 * grad + tv_weight * tv_loss_val
                # loss = 1.0 * F.l1_loss(pred, target) + 1.0 * F.mse_loss(pred, target) + tv_weight * tv_loss_val
                loss = 5.0 * F.l1_loss(pred, target) + 1.0 * F.mse_loss(pred, target) + tv_weight * tv_loss_val

                val_loss += loss.item()

        val_loss /= len(val_loader)

        model.train()

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Best: {best_loss:.6f}")

        if val_loss < best_loss:

            best_loss = val_loss
            best_epoch = epoch + 1
            early_stop_cnt = 0

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'loss': loss,
            }, checkpoint_name_out)

            print(f"Model saved at epoch {epoch+1} with val_loss: {val_loss:.6f}")
        else:
            early_stop_cnt += 1
            print(f"No improvement. Current val_loss: {val_loss:.6f}, Best so far: {best_loss:.6f}, Best epoch {best_epoch}")
        if early_stop_cnt > 10:
            break

def train_non_zero_epoch(model, train_loader, val_loader, num_epochs, checkpoint_name_in, checkpoint_name_out):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.L1Loss()
    best_loss = float('inf')
    model.to(device)
    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)
    model.train()
    try:
        checkpoint = torch.load(checkpoint_name_in, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    except FileNotFoundError as e:
        print(f"error {e}")
    except KeyError as e:
        print(f"error {e}")

    for epoch in range(num_epochs):
        train_loss = 0.0
        for inp, tgt in train_loader:
            inp, tgt = inp.to(device), tgt.to(device)
            tgt = tgt

            pred = model(inp)
            l1loss = criterion(pred, tgt)
            tv_loss_val = visiontransformer.tv_loss(pred)
            loss = l1loss + 0.2 * tv_loss_val

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inp, target = batch
                inp = inp.to(device)
                target = target.to(device)

                output = model(inp)
                target = target
                l1loss = criterion(output, target)
                tv_loss_val = visiontransformer.tv_loss(output)
                loss = l1loss + 0.1 * tv_loss_val

                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Best: {best_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1
            early_stop_cnt = 0

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'loss': loss,
            }, checkpoint_name_out)

            print(f"Model saved at epoch {epoch+1} with val_loss: {val_loss:.6f}")
        else:
            early_stop_cnt += 1
            print(f"No improvement. Current val_loss: {val_loss:.6f}, Best so far: {best_loss:.6f}, Best epoch {best_epoch}")
        if early_stop_cnt > 10:
            break


if __name__ == "__main__":
    data_dir = "NpzDataset-thetaosouovo"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # checkpoint_name = "10.30 best_model 200 1 months.pth"
    file_paths = list(Path(data_dir).glob("*.npz"))
    # # print(file_paths)
    
    # assert len(file_paths) == 4, f"Expected 4 files, got {len(file_paths)}"

    file_paths = sorted(file_paths, key=lambda x: int(x.stem.split('_')[-1]))

    data_list = visiontransformer.load_data(file_paths)[:4]
    # print(f"Loaded {len(data_list)} frames.")

    seq_len = 1

    # max_start_idx = len(data_list) - seq_len - 1
    # assert max_start_idx >= 0, "Not enough frames to form tensor."

    train_indices = range(0, 2)
    val_indices   = range(2, 3)

    print("Creating training tensor...")
    x_train, y_train = visiontransformer.create_tensor(data_list, train_indices, seq_len=seq_len)

    print("Creating validation tensor...")
    x_val, y_val = visiontransformer.create_tensor(data_list, val_indices, seq_len=seq_len)

    print(f"X_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_val shape:   {x_val.shape}")
    print(f"y_val shape:   {y_val.shape}")

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_val = x_val.to(device)
    y_val = y_val.to(device)

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset   = TensorDataset(x_val, y_val)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size:   {len(val_dataset)}")

    batch_size = 1
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
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
    print("device:", device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # criterion = torch.nn.L1Loss()

    train_zero_epoch(model, train_loader, val_loader, 200, "checkpoints/10.31--17.38 model.pth")
    # train_non_zero_epoch(model, train_loader, val_loader, 200, "checkpoints/new 10.30 model.pth", "checkpoints/" + str(patch_size) + "2 10.30 model.pth")


    