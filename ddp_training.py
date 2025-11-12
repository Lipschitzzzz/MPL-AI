import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import visiontransformer

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1)
args = parser.parse_args()

torch.cuda.set_device(args.local_rank)
device = torch.device('cuda', args.local_rank)
torch.distributed.init_process_group(backend='nccl')

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

batch_size = 1

data_list = np.load("609_days_npz_dataset_normalized.npz")["data"][:17]
static_mask = visiontransformer.build_static_mask(data_list, img_size=(420, 312), patch_size=(1, 1))
model = visiontransformer.OceanModel(t_in=7, static_mask=static_mask).to(device)

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

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)


model.to(device)
channel_weights = (
[1.0] * 5 +    # 0-4
[1.0] * 5 +    # 5-9
[1.0] * 5 +    # 10-14
[1.0] * 5 +    # 15-19
[5.0] * 1      # 20
    )
criterion = visiontransformer.MaskedWeightedMAEMSELoss(mask=static_mask, channel_weights=channel_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95))
# training!
if args.local_rank == 0:
    tb_writer = SummaryWriter(comment='ddp-3')

train_sampler = DistributedSampler(train_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

for i, (inputs, labels) in enumerate(train_loader):
    # forward
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # log
    if args.local_rank == 0 and i % 5 == 0:
        tb_writer.add_scalar('loss', loss.item(), i)

if args.local_rank == 0:
    tb_writer.close()