import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
import time
import os
import numpy as np
import visiontransformer


def train_zero_epoch_ddp(checkpoint_name_out):
    start_time = time.time()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)


    data_list = np.load("609_days_npz_dataset_normalized.npz")["data"][:17]
    
    static_mask = visiontransformer.build_static_mask(data_list, img_size=(420, 312), patch_size=(1, 1))
    model = visiontransformer.OceanModel(t_in=7, static_mask=static_mask).to(device)
    model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95))
    channel_weights = (
    [1.0] * 5 +    # 0-4
    [1.0] * 5 +    # 5-9
    [1.0] * 5 +    # 10-14
    [1.0] * 5 +    # 15-19
    [5.0] * 1      # 20
        )
    criterion = visiontransformer.MaskedWeightedMAEMSELoss(mask=static_mask, channel_weights=channel_weights)

    full_dataset = visiontransformer.TimeSeriesDataset(
        data_dir="time_steps",
        total_timesteps=609,
        seq_len=30
    )

    total_samples = len(full_dataset)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size

    train_indices = list(range(train_size))
    val_indices = list(range(train_size, total_samples))

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, sampler=train_sampler,
        num_workers=2, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, sampler=val_sampler,
        num_workers=2, pin_memory=True
    )

    best_loss = float('inf')
    num_epochs = 500
    early_stop_cnt = 0
    best_epoch = 0
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)

        model.train()
        train_loss = 0.0
        for inp, target in train_loader:
            inp = inp.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            pred = model(inp)
            loss = criterion(pred[0], target[0][0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inp, target in val_loader:
                inp = inp.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                pred = model(inp)
                loss = criterion(pred[0], target[0][0])
                val_loss += loss.item()

        val_loss /= len(val_loader)

        val_loss_tensor = torch.tensor(val_loss, device=device)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        val_loss = val_loss_tensor.item() / world_size

        if local_rank == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Best: {best_loss:.6f}")

            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch + 1
                early_stop_cnt = 0

                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                }, checkpoint_name_out)

                print(f"Model saved at epoch {epoch+1} with val_loss: {val_loss:.6f}")
            else:
                early_stop_cnt += 1
                print(f"No improvement. Current val_loss: {val_loss:.6f}, Best so far: {best_loss:.6f}, Best epoch {best_epoch}")

            if early_stop_cnt > 25:
                print("Early stopped.")
                break

    total_time = time.time() - start_time
    if local_rank == 0:
        hours, rem = divmod(total_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"Total training time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")

    dist.destroy_process_group()


def main():
    world_size = torch.cuda.device_count()
    print(world_size, " GPU found")
    assert world_size > 0, "No GPUs available"
    start_time = time.time()
    timestamp_str = time.strftime("%Y_%m_%d_%H_%M", time.localtime(start_time))
    train_zero_epoch_ddp(timestamp_str + str(world_size) + " GPU_model.pth")

if __name__ == "__main__":
    main()