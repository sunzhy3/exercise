import numpy as np
import torch
import torch.nn as nn

import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class ToyModel(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(ToyModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.LayerNorm(n_hidden),
            nn.ReLU()
        )
        self.predict = nn.Linear(n_hidden, n_output)

    def forward(self, input):
        out = self.layer1(input)
        out = self.predict(out)
        return out

def demo_basic(rank, world_size):
    setup(rank, world_size)

    train_dataset = torch.utils.data.TensorDataset(
        torch.randn(100, 10).to(rank), 
        torch.randn(100, 1).to(rank)
    )
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=2,
        shuffle=True,
        num_workers=2
    )

    # create model and move it to GPU with id rank
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    for data in train_loader:
        x, y = data

        outputs = ddp_model(x)
        loss = loss_fn(outputs, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)
