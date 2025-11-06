import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import visiontransformer
import torch.nn.functional as F
import time

def train_zero_epoch(model, optimizer, criterion, train_loader, val_loader, num_epochs, checkpoint_name_out):
    # criterion = torch.nn.L1Loss()
    best_loss = float('inf')
    tv_weight = 0.1
    early_stop_cnt= 0
    best_epoch = 0
    model.train()
    
    for epoch in range(num_epochs):
        train_loss = 0.0
        for inp, target in train_loader:

            pred = model(inp)
            # l1loss = criterion(pred, tgt)
            print(pred.shape)
            print(target.shape)
            # tv_loss_val = visiontransformer.tv_loss(pred)
            pred = model(inp)
            # l1loss = criterion(output, target)
            loss = criterion(pred, target)

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
                # tv_loss_val = visiontransformer.tv_loss(pred)
                loss = criterion(pred, target)

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
        if early_stop_cnt > 20:
            break

def train_non_zero_epoch(model, train_loader, val_loader, num_epochs, checkpoint_name_in, checkpoint_name_out):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.L1Loss()
    best_loss = float('inf')
    tv_weight = 0.3
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
            loss = 1.0 * F.l1_loss(pred, target) + 0.2 * F.mse_loss(pred, target) + tv_weight * tv_loss_val
            # loss = l1loss + 0.2 * tv_loss_val

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
                # loss = l1loss + 0.1 * tv_loss_val
                loss = 1.0 * F.l1_loss(pred, target) + 0.2 * F.mse_loss(pred, target) + tv_weight * tv_loss_val

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_list = np.load("68_months_npz_dataset_normalized.npz")["data"]
    seq_len = 1
    train_indices = range(0, 3)
    val_indices   = range(3, 5)
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
    patch_size = (6, 6)
    static_mask = visiontransformer.build_static_mask(data_list, img_size=(420, 312), patch_size=(1, 1))
    print(static_mask.shape)
    
    model = visiontransformer.OceanForecastNet(
        img_size=(420, 312),
        patch_size=patch_size,
        in_chans=21,
        out_chans=21,
        t_in=1,
        t_out=1,
        embed_dim=384,
        depth=2,
        num_heads=2,
        static_mask=static_mask
    ).to(device)
    print("device:", device)
    # criterion = visiontransformer.MaskedWeightedMAEMSELoss(mask=static_mask).cuda()
    # criterion = visiontransformer.MaskedWeightedMAEMSELoss(mask=static_mask).cuda()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.95))
    start_time = time.time()
    train_zero_epoch(model, optimizer, criterion, train_loader, val_loader, 200, "checkpoints/local 11.06 model.pth")
    total_time = time.time() - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Total training time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")
    # train_non_zero_epoch(model, train_loader, val_loader, 200, "checkpoints/new 10.30 model.pth", "checkpoints/" + str(patch_size) + "2 10.30 model.pth")

