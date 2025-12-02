import os
import torch
from torch import nn, optim
import json, time
import utils
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

if __name__ == "__main__":

    N_EPOCHS = 4 # 60

    model_path = "models/LorentzEQGNN/"
    log_path = "logs/LorentzEQGNN/"
    utils.args_init(args)

    ### set random seed
    torch.manual_seed(42)
    np.random.seed(42)

    ### initialize cpu
    dist.init_process_group(backend='nccl')
    device = torch.device("cuda")
    dtype = torch.float32

    ### load data
    dataloaders = retrieve_dataloaders( batch_size,
                                        num_data=100000, # use all data
                                        cache_dir="data/qg/datasets/",
                                        num_workers=0,
                                        use_one_hot=True)

    model = LorentzEQGNN(n_scalar = 1, n_hidden = 4, n_class = 2,\
                       dropout = 0.2, n_layers = 1,\
                       c_weight = 1e-3)

    model = model.to(device)

    ### print model and dataset information
    # if (args.local_rank == 0):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Model Size:", pytorch_total_params)
    for (split, dataloader) in dataloaders.items():
        print(f" {split} samples: {len(dataloader.dataset)}")

    ### optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    ### lr scheduler
    base_scheduler = CosineAnnealingWarmRestarts(optimizer, 4, 2, verbose = False)
    lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=1,\
                                                warmup_epoch=1,\
                                                after_scheduler=base_scheduler) ## warmup

    ### loss function
    loss_fn = nn.CrossEntropyLoss()

    ### initialize logs
    res = {'epochs': [], 'lr' : [],\
           'train_time': [], 'val_time': [],  'train_loss': [], 'val_loss': [],\
           'train_acc': [], 'val_acc': [], 'best_val': 0, 'best_epoch': 0}

    ### training and testing
    print("Training...")
    train(model, res, N_EPOCHS, model_path, log_path)
    test(model, res, model_path, log_path)