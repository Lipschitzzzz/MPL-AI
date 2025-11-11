import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import visiontransformer
import torch.nn.functional as F
import time

def train_zero_epoch(model, optimizer, criterion, train_loader, val_loader, num_epochs, checkpoint_name_out):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()
    best_loss = float('inf')
    tv_weight = 0.1
    early_stop_cnt= 0
    best_epoch = 0
    model.train()
    
    for epoch in range(num_epochs):
        train_loss = 0.0
        for inp, target in train_loader:
            inp = inp.to(device)
            target = target.to(device)

            # l1loss = criterion(pred, tgt)

            # tv_loss_val = visiontransformer.tv_loss(pred)
            pred = model(inp)
            print("epoch: ", epoch+1, " pred:   ", pred.shape)
            print("epoch: ", epoch+1, " target: ", target.shape)
            # l1loss = criterion(output, target)
            loss = criterion(pred[0], target[0][0])

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
                inp = inp.to(device)
                target = target.to(device)

                pred = model(inp)
                # l1loss = criterion(output, target)
                # tv_loss_val = visiontransformer.tv_loss(pred)
                loss = criterion(pred[0], target[0][0])
                # loss = criterion(pred, target)

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
    total_time = time.time() - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Total training time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")

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

def initialization(seq_len=5, data_list="609_days_npz_dataset_normalized.npz", train_indices=range(0, 10),
                   val_indices=range(10, 12), patch_size=(6, 6)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_list = np.load(data_list)["data"][:17]
    seq_len = seq_len
    train_indices = train_indices
    val_indices   = val_indices
    print("Creating training tensor...")
    x_train, y_train = visiontransformer.create_tensor(data_list, train_indices, seq_len=seq_len)
    print("Creating validation tensor...")
    x_val, y_val = visiontransformer.create_tensor(data_list, val_indices, seq_len=seq_len)
    print(f"X_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_val shape:   {x_val.shape}")
    print(f"y_val shape:   {y_val.shape}")
    np.savez('train_data.npz', x=x_train, y=y_train)
    np.savez('val_data.npz', x=x_val, y=y_val)

    # x_train = x_train.to(device)
    # y_train = y_train.to(device)
    # x_val = x_val.to(device)
    # y_val = y_val.to(device)
    # train_dataset = TensorDataset(x_train, y_train)
    # val_dataset   = TensorDataset(x_val, y_val)
    train_dataset = visiontransformer.OceanDataSet('train_data.npz')
    val_dataset   = visiontransformer.OceanDataSet('val_data.npz')

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size:   {len(val_dataset)}")
    batch_size = 1
    patch_size = patch_size
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
    static_mask = visiontransformer.build_static_mask(data_list, img_size=(420, 312), patch_size=(1, 1))
    model = visiontransformer.OceanModel(t_in=seq_len, static_mask=static_mask).to(device)
    # print("device:", device)
    channel_weights = (
    [1.0] * 5 +    # 0-4
    [1.0] * 5 +    # 5-9
    [1.0] * 5 +    # 10-14
    [1.0] * 5 +    # 15-19
    [5.0] * 1      # 20
    )
    criterion = visiontransformer.MaskedWeightedMAEMSELoss(mask=static_mask, channel_weights=channel_weights).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95))
    return model, optimizer, criterion, train_loader, val_loader

if __name__ == "__main__":
    start_time = time.time()
    timestamp_str = time.strftime("%Y_%m_%d_%H_%M", time.localtime(start_time))
    best_loss = float('inf')
    model, optimizer, criterion, train_loader, val_loader = initialization()
    train_zero_epoch(model, optimizer, criterion, train_loader, val_loader, 200, "checkpoints/" + timestamp_str + "_local_model.pth")
    
    # train_non_zero_epoch(model, train_loader, val_loader, 200, "checkpoints/new 10.30 model.pth", "checkpoints/" + str(patch_size) + "2 10.30 model.pth")

