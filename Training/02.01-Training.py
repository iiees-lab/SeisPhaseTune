from seisbench.data import WaveformDataset

import seisbench.generate as sbg
import seisbench.util as sbu
import seisbench.models as sbm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import logging
from pathlib import Path


##########################################################################
import sys
lib_path = [
    r'C:\Users\ikahbasi\OneDrive\Applications\GitHub\SeisRoutine',
    r'C:\Users\ikahb\OneDrive\Applications\GitHub\SeisRoutine',
]
for path in lib_path:
    sys.path.append(path)

import SeisRoutine.config as srconf
import SeisRoutine.seisbench as srsb
##########################################################################
import warnings
warnings.simplefilter('ignore', DeprecationWarning)
##########################################################################
def build_augmentations(config):
    augmentations = []
    for aug in config:
        print(aug)
        aug = aug.to_dict()
        aug_name = aug['cls']
        kwargs = aug.copy()
        kwargs.pop("cls", None)
        augmentations.append(eval(aug_name)(**kwargs))

    return augmentations

def make_generator(split_data, augmentations):
    gen = sbg.GenericGenerator(split_data)
    gen.add_augmentations(augmentations)
    return gen

def loss_fn(y_pred, y_true, eps=1e-5):
    # vector cross entropy loss
    h = y_true * torch.log(y_pred + eps)
    h = h.mean(-1).sum(-1)  # Mean along sample dimension and sum along pick dimension
    h = h.mean()  # Mean over batch axis
    return -h

def train_loop(model, dataloader, optimizer):
    model.train()
    lst_loss = []
    size = len(dataloader.dataset)
    part = 0
    for batch_id, batch in enumerate(dataloader, start=1):
        # Compute prediction and loss
        X = batch["X"].to(model.device)
        y = batch["y"].to(model.device)
        
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #
        part += X.shape[0]
        srconf.ProgressMsg.print(
            part=part,
            total=size,
            step=1,
            subject=f"Training loss: {loss.item():>7f}",
        )
        lst_loss.append(loss.item())
    return lst_loss

def test_loop(dataloader, model):
    num_batches = len(dataloader)
    test_loss = 0

    model.eval()  # close the model for evaluation

    with torch.no_grad():
        for index, batch in enumerate(dataloader):
            # print(index, batch)
            pred = model(batch["X"].to(model.device))
            test_loss += loss_fn(pred, batch["y"].to(model.device)).item()

    model.train()  # re-open model for training stage

    test_loss /= num_batches
    logging.info(f"Test avg loss: {test_loss:>8f} \n")
    return test_loss



##########################################################################

cfg_projects = srconf.Config.load('./Configs/Projects.yml')

for cfg_project in cfg_projects.projects:
    timestamp = srconf.timestamp()
    cfg = srconf.Config.load(
        file_path=cfg_project.parameters_config_path,
        resolve=True,
    )
    context={
        "timestamp": timestamp,
        "project": cfg_project,
    }
    cfg.resolve(context=context)
    
    srconf.configure_logging(**cfg.log.to_dict())

    running_file_info = srconf.RuntimeLocation.get_caller_info()
    msg = f"Running Code | {running_file_info['full_path']}"
    logging.info(msg)
    
    # List all installed packages and their versions
    msg = srconf.EnvironmentInfo().report(include_freeze=True)
    logging.info(msg)
    
    msg = cfg.__str__()
    logging.info(f'Configuration File:\n{msg}')
    
    
    data_format = cfg.to_dict()['dataset']['data_format']
    data_format_tmp = data_format.copy()
    data_format_tmp.pop('dimension_order')
    dataset = WaveformDataset(
        path=cfg.path.dataset,
        **data_format_tmp
    )
    
    ps_pair = srsb.dataset.find_ps_pairs(
        metadata=dataset.metadata
    )
    dataset.metadata['PS-Pairs'] = ps_pair
    

    dataset.metadata['Usable'] = ps_pair
    dataset.metadata.loc[cfg.dataset.MANUAL_EXCLUSIONS, 'Usable'] = False
    
    phase_dict = srsb.dataset.build_phase_mapper(
        dataset.metadata.columns
    )
    context = {
        "phase_dict_keys": list(phase_dict.keys()),
        "phase_dict": phase_dict,
        "np": np
    }
    cfg.resolve(context=context)
    # cfg_augmentation = cfg.augmentation.to_dict()
    augmentations = build_augmentations(cfg.augmentation)
    
    dataset.metadata['split'] = srsb.dataset.build_split_column(
        df=dataset.metadata,
        mask='Usable',
        split_ratios=cfg.dataset.split_ratios.to_dict(),
        shuffle=True,
        random_state=42,
    )

    train, dev, test = dataset.train_dev_test()
    
    generators = {
        name: make_generator(data, augmentations)
        for name, data
        in zip(['train', 'dev', 'test'],
               [train, dev, test])
    }
    
    dataloader = {
        name: DataLoader(gen,
                         worker_init_fn=sbu.worker_seeding,
                         **getattr(cfg.dataloader, name).to_dict())
        for name, gen
        in generators.items()
    }
    
    torch.manual_seed(
        cfg.train.hyperparameters.manual_seed
    )
    
    model_cls = getattr(
        sbm,
        cfg.model.name,
    )
    model = model_cls(
        **cfg.model.hyperparameters.to_dict(),
    )
    
    if torch.cuda.is_available():
        model.cuda()
        msg = "CUDA is available. Training on GPU."
    else:
        msg = "CUDA is NOT available. Training on CPU."
    
    logging.info(msg)
    
    ###
    log_learning = []
    for stage_idx, (base_lr, epochs) in enumerate(zip(
            cfg.train.hyperparameters.learning_rates,
            cfg.train.hyperparameters.epochs_for_each_learning_rate)):
        
        msg = f"Main Learning-Rate: {base_lr}"
        logging.info(msg)
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=base_lr,
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            **cfg.train.hyperparameters.lr_scheduler.ReduceLROnPlateau.to_dict(),
        )
        for epoch in range(epochs):
            epoch += 1
            current_lr  = scheduler.get_last_lr()[0]
            msg = f"Learning-Rate: {current_lr} Epoch: {epoch}"
            logging.info(msg)
            train_loss = train_loop(
                model=model,
                dataloader=dataloader['train'],
                optimizer=optimizer,
            )
            test_loss = test_loop(
                dataloader=dataloader['dev'],
                model=model)
            scheduler.step(test_loss)
            #
            for batch, loss in enumerate(train_loss, start=1):
                log_entry = {
                    'stage': stage_idx,
                    'epoch': epoch,
                    'batch': batch,
                    'lr': current_lr,
                    'loss_train': loss,
                    'loss_test': test_loss,
                }
                log_learning.append(log_entry)
                
    df_loss = pd.DataFrame(log_learning)
    
    path_model = Path(cfg.path.model)
    path_model.resolve().mkdir(
        parents=True,
        exist_ok=True,
    )
    
    fname = f'loss_{cfg.model.version_str}.csv'
    df_loss.to_csv(
        path_model / fname
    )
    
    fname = f"{cfg.model.name}_{cfg.model.version_str}"
    model.save(
        path_model / fname,
        weights_docstring=cfg.__str__(),
        version_str=cfg.model.version_str,
    )