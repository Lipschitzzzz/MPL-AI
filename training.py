import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import visiontransformer

def train_zero_epoch(model, train_loader, val_loader, num_epochs, checkpoint_name_out):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.L1Loss()
    best_loss = float('inf')
    early_stop_cnt= 0
    best_epoch = 0
    model.train()

    for epoch in range(num_epochs):
        train_loss = 0.0
        for inp, tgt in train_loader:
            inp, tgt = inp.to(device), tgt.to(device)
            tgt = tgt[0]

            pred = model(inp)
            l1loss = criterion(pred, tgt)
            tv_loss_val = visiontransformer.tv_loss(pred[0])
            loss = l1loss + 0.5 * tv_loss_val

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
                target = target[0]
                l1loss = criterion(output, target)
                tv_loss_val = visiontransformer.tv_loss(output[0])
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
            tgt = tgt[0]

            pred = model(inp)
            l1loss = criterion(pred, tgt)
            tv_loss_val = visiontransformer.tv_loss(pred[0])
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
                target = target[0]
                l1loss = criterion(output, target)
                tv_loss_val = visiontransformer.tv_loss(output[0])
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
    # checkpoint_name = "10.30 best_model 200 1 months.pth"
    file_paths = list(Path(data_dir).glob("*.npz"))[:6]
    # # print(file_paths)
    
    # assert len(file_paths) == 4, f"Expected 4 files, got {len(file_paths)}"

    file_paths = sorted(file_paths, key=lambda x: int(x.stem.split('_')[-1]))

    data_list = visiontransformer.load_data(file_paths)
    # print(f"Loaded {len(data_list)} frames.")

    seq_len = 2

    # max_start_idx = len(data_list) - seq_len - 1
    # assert max_start_idx >= 0, "Not enough frames to form tensor."

    train_indices = range(0, 3)
    val_indices   = range(3, 4)

    print("Creating training tensor...")
    X_train, y_train = visiontransformer.create_tensor(data_list, train_indices, seq_len=seq_len)

    print("Creating validation tensor...")
    X_val, y_val = visiontransformer.create_tensor(data_list, val_indices, seq_len=seq_len)

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_val shape:   {X_val.shape}")
    print(f"y_val shape:   {y_val.shape}")

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset   = TensorDataset(X_val, y_val)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size:   {len(val_dataset)}")

    batch_size = 1
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    patch_size = (4, 4)
    model = visiontransformer.OceanForecastNet(
        img_size=(420, 312),
        patch_size=patch_size,
        in_chans=4,
        out_chans=4,
        levels=5,
        T_in=6,
        embed_dim=192,
        depth=2,
        num_heads=2
    )
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # criterion = torch.nn.L1Loss()

    train_zero_epoch(model, train_loader, val_loader, 200, "checkpoints/2 months 10.30 model.pth")
    # train_non_zero_epoch(model, train_loader, val_loader, 200, "checkpoints/new 10.30 model.pth", "checkpoints/" + str(patch_size) + "2 10.30 model.pth")


    