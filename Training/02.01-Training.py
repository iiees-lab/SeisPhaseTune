from seisbench.data import WaveformDataset

import seisbench.util as sbu
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


def loss_func(y_pred, y_true, eps=1e-5):
    # vector cross entropy loss
    h = y_true * torch.log(y_pred + eps)
    # Mean along sample dimension and sum along pick dimension
    h = h.mean(-1).sum(-1)
    # Mean over batch axis
    h = h.mean()
    return -h


def train_loop(
        model,
        dataloader,
        optimizer,
        loss_function,
    ):
    model.train()
    lst_loss = []
    size = len(dataloader.dataset)
    part = 0
    for batch_id, batch in enumerate(dataloader, start=1):
        # Compute prediction and loss
        X = batch["X"].to(model.device)
        y = batch["y"].to(model.device)
        
        pred = model(X)
        loss = loss_function(pred, y)
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

def validation_loop(dataloader, model):
    num_batches = len(dataloader)
    test_loss = 0

    model.eval()  # close the model for evaluation

    with torch.no_grad():
        for index, batch in enumerate(dataloader):
            # print(index, batch)
            pred = model(batch["X"].to(model.device))
            test_loss += loss_func(pred, batch["y"].to(model.device)).item()

    model.train()  # re-open model for training stage

    test_loss /= num_batches
    logging.info(f"Test avg loss: {test_loss:>8f}")
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
    dataset = WaveformDataset(
        path=cfg.path.dataset,
        **data_format_tmp
    )
    
    ps_pair = srsb.dataset.find_ps_pairs(
        metadata=dataset.metadata
    )
    dataset.metadata['PS-Pairs'] = ps_pair
    

    dataset.metadata['Usable'] = ps_pair
    dataset.metadata.loc[cfg.dataset.manual_excluded_samples, 'Usable'] = False
    
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
    augmentations = srsb.dataset.build_augmentations(cfg.augmentation)
    
    dataset.metadata['split'] = srsb.dataset.build_split_column(
        df=dataset.metadata,
        mask='Usable',
        split_ratios=cfg.dataset.split_ratios.to_dict(),
        shuffle=True,
        random_state=42,
    )

    train, dev, test = dataset.train_dev_test()
    
    generators = {
        name: srsb.dataset.make_generator(dataset, augmentations)
        for name, dataset
        in zip(['train', 'dev', 'test'],
               [train, dev, test])
    }
    
    dataloader = {
        name: DataLoader(
            gen,
            worker_init_fn=sbu.worker_seeding,
            **getattr(cfg.dataloader, name).to_dict()
        )
        for name, gen
        in generators.items()
    }
    
    torch.manual_seed(
        cfg.seed
    )
    
    model = srconf.ObjectFactory.create(
        obj_str=cfg.model.cls,
        **cfg.model.hyperparameters.to_dict(),
    )
    if cfg.model.weights_type != "scratch":
        msg = (
            "Training model "
            f"using {cfg.model.weights_type} weight initialization"
        )
        logging.info(msg)
        model = model.from_pretrained(cfg.model.weights_type)
    
    if torch.cuda.is_available():
        model.cuda()
        msg = "CUDA is available. Training on GPU."
    else:
        msg = "CUDA is NOT available. Training on CPU."
    
    logging.info(msg)
    
    model_path = Path(cfg.path.model)
    best_checkpoint_path = model_path / "best.ckpt"
    last_checkpoint_path = model_path / "last.ckpt"
    ###
    log_learning = []
    best_val_loss = np.inf
    for stage_idx, stage_cfg in enumerate(cfg.train_stages):
        msg = f"Training starts on {stage_cfg.name}"
        logging.info(msg)
        
        object_params = stage_cfg.optimizer.to_dict()
        object_name = object_params.pop("cls")
        optimizer = srconf.ObjectFactory.create(
            obj_str=object_name,
            params=model.parameters(),
            **object_params,
        )
        
        object_params = stage_cfg.lr_scheduler.to_dict()
        object_name = object_params.pop("cls")
        scheduler = srconf.ObjectFactory.create(
            obj_str=object_name,
            optimizer=optimizer,
            **object_params,
        )
        
        object_params = stage_cfg.early_stopping.to_dict()
        object_name = object_params.pop("cls")
        early_stopping = srconf.ObjectFactory.create(
            obj_str=object_name,
            **object_params,
        )
        
        for epoch in range(1, stage_cfg.epochs+1):
            current_lr = scheduler.get_last_lr()[0]
            msg = f"Learning-Rate: {current_lr} Epoch: {epoch}"
            logging.info(msg)
            train_losses = train_loop(
                model=model,
                dataloader=dataloader['train'],
                optimizer=optimizer,
                loss_function=loss_func,
            )
            
            avg_train_loss = np.mean(train_losses)
            msg = f"Training loss (Average) {avg_train_loss:>7f}"
            logging.info(msg)
            
            test_loss = validation_loop(
                dataloader=dataloader['dev'],
                model=model,
            )
            scheduler.step(test_loss)

            ####################################
            ###### Early Stop Block: start
            ####################################
            if early_stopping.check(loss=test_loss):
                logging.info(
                    f"Early stopping at Stage {stage_idx}, Epoch {epoch}"
                )
                break
            ####################################
            ###### Early Stop Block: end
            ####################################
            
            ####################################
            ###### Check Point Block: start
            ####################################
            srsb.training.save_checkpoint(
                path=last_checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                stage=stage_idx,
                epoch=epoch,
                val_loss=test_loss,
            )
            
            # Save best checkpoint
            if test_loss < best_val_loss:
                best_val_loss = test_loss
            
                srsb.training.save_checkpoint(
                    path=best_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    stage=stage_idx,
                    epoch=epoch,
                    val_loss=best_val_loss,
                )
            
                logging.info(
                    f"New best model saved "
                    f"(Validation Loss = {best_val_loss:.6f})"
                )
            #
            ####################################
            ###### Check Point Block: End
            ####################################
            for batch, train_loss in enumerate(train_losses, start=1):
                log_entry = {
                    'stage': stage_idx,
                    'epoch': epoch,
                    'batch': batch,
                    'lr': current_lr,
                    'train_loss': train_loss,
                    'test_loss': (test_loss
                                  if batch == len(train_losses)
                                  else np.nan),
                }
                log_learning.append(log_entry)
                
    df_loss = pd.DataFrame(log_learning)
    
    model_path.resolve().mkdir(
        parents=True,
        exist_ok=True,
    )
    
    fname = f'loss_{cfg.model.version_str}.csv'
    df_loss.to_csv(
        model_path / fname
    )
    
    fname = f"{cfg.model.cls}_{cfg.model.version_str}"
    model.save(
        model_path / fname,
        weights_docstring=cfg.__str__(),
        version_str=cfg.model.version_str,
    )