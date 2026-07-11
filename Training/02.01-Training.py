from seisbench.data import WaveformDataset

import seisbench.generate as sbg
import seisbench.util as sbu
import seisbench.models as sbm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import logging
import os
import re
from scipy import signal
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
# from SeisRoutine.seisbench.waveform import Tapering
##########################################################################
import warnings
warnings.simplefilter('ignore', DeprecationWarning)
##########################################################################
def build_augmentations(config):
    augmentations = []

    for aug_name, kwargs in config.items():
        augmentations.append(eval(aug_name)(**kwargs))

    return augmentations

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
    for batch_id, batch in enumerate(dataloader):
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
        srconf.ProgressMsg.print(
            part=batch_id * batch["X"].shape[0],
            total=size,
            step=5,
            subject=f"Training loss: {loss.item():>7f}",
        )
        lst_loss.append(loss)
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
    project = srconf.dict_to_object(cfg_project)
    timestamp = srconf.timestamp()
    cfg = srconf.Config.load(
        file_path=project.parameters_config_path,
        resolve=True
    )
    context={
        "timestamp": timestamp,
        "project": project,
    }
    cfg.resolve(context=context)
    
    srconf.configure_logging(**cfg.to_dict()['log'])

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
    dataset.metadata.loc[23972, 'PS-Pairs'] = False
    


    phase_dict = srsb.dataset.build_phase_mapper(
        dataset.metadata.columns
    )
     
    cfg_augmentation = cfg.augmentation.to_dict()
    cfg_augmentation["sbg.WindowAroundSample"]["metadata_keys"] = list(
        phase_dict.keys()
    )
    cfg_augmentation["sbg.ProbabilisticLabeller"]["label_columns"] = phase_dict
    cfg_augmentation["sbg.ChangeDtype"]["dtype"] = eval(
        cfg_augmentation["sbg.ChangeDtype"]["dtype"]
    )
    augmentations = build_augmentations(cfg_augmentation)
    
    dataset.metadata['split'] = srsb.dataset.build_split_column(
        df=dataset.metadata,
        mask='PS-Pairs',
        split_ratios=cfg.dataset.split_ratios.to_dict(),
        shuffle=True,
        random_state=42,
    )

    train, dev, test = dataset.train_dev_test()
    # print(train, dev, test, sep='\n')
    
    train_generator = sbg.GenericGenerator(train)
    dev_generator = sbg.GenericGenerator(dev)
    test_generator = sbg.GenericGenerator(test)

    train_generator.add_augmentations(augmentations)
    dev_generator.add_augmentations(augmentations)
    test_generator.add_augmentations(augmentations)
    
    
    train_loader = DataLoader(
        train_generator,
        worker_init_fn=sbu.worker_seeding,
        **cfg.dataloader.train.to_dict(),
    )
    dev_loader = DataLoader(
        dev_generator,
        worker_init_fn=sbu.worker_seeding,
        **cfg.dataloader.dev.to_dict(),
    )
    test_loader = DataLoader(
        test_generator,
        worker_init_fn=sbu.worker_seeding,
        **cfg.dataloader.test.to_dict(),
    )
    
    torch.manual_seed(
        cfg.train.hyperparameters.manual_seed
    )
    
    model = getattr(
        sbm,
        cfg.model.name,
    )
    model = model(
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
    for learning_rate, epochs in zip(
            cfg.train.hyperparameters.learning_rates,
            cfg.train.hyperparameters.epochs_for_each_learning_rate):
        
        msg = f"Main Learning-Rate: {learning_rate}"
        logging.info(msg)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            **cfg.train.hyperparameters.lr_scheduler.ReduceLROnPlateau.to_dict(),
        )
        for epoch in range(epochs):
            learning_rate = scheduler.get_last_lr()[0]
            msg = f"Learning-Rate: {learning_rate} Epoch: {epoch+1}"
            logging.info(msg)
            train_loss = train_loop(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
            )
            test_loss = test_loop(
                dataloader=dev_loader,
                model=model)
            scheduler.step(test_loss)
            #
            for batch, loss in enumerate(train_loss):
                dict_tmp = {
                    'epoch': epoch,
                    'batch': batch+1,
                    'loss_train': loss.item(),
                    'loss_test': test_loss,
                }
                log_learning.append(dict_tmp)
                
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