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


##########################################################################
import sys
lib_path = [
    r'C:\Users\ikahbasi\OneDrive\Applications\GitHub\SeisRoutine',
    r'C:\Users\ikahb\OneDrive\Applications\GitHub\SeisRoutine',
]
for path in lib_path:
    sys.path.append(path)

import SeisRoutine.config as srconf
##########################################################################
import warnings
warnings.simplefilter('ignore', DeprecationWarning)
##########################################################################
class Tapering:
    def __init__(self, alpha=0.3, key='X'):
        self.alpha = alpha  # ضریب تیپرینگ
        if isinstance(key, str):
            self.key = (key, key)
        else:
            self.key = key

    def __call__(self, state_dict):
        x, metadata = state_dict[self.key[0]]
        taper = signal.windows.tukey(x.shape[-1], self.alpha)
        x = x * taper
        state_dict[self.key[1]] = (x, metadata)

def build_phase_mapper(
        columns,
        families={"P", "S"},
    ):
    """
    Map arrival columns to their phase family.

    Example:
        trace_Pg_arrival_sample  -> P
        trace_Pn_arrival_sample  -> P
        trace_Sg_arrival_sample  -> S
        trace_Sg 2_arrival_sample -> S
    """
    pattern = re.compile(r"^trace_(.+?)_arrival_sample$")

    mapper = {}

    for col in columns:
        match = pattern.match(col)
        if not match:
            continue

        phase = match.group(1).strip()
        family = phase[0].upper()

        if family in families:
            mapper[col] = family

    return mapper


def build_split_column(
    df: pd.DataFrame,
    mask: pd.Series | None = None,
    split_ratios: dict = None,
    shuffle: bool = True,
    random_state: int | None = None,
    ) -> pd.Series:
    """
    Create a train/dev/test split column for a dataset without modifying it in-place.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    mask : pd.Series or None
        Boolean mask to filter rows before splitting.
        If None, all rows are used.

    split_ratios : dict
        Dictionary with keys: 'train', 'dev', 'test'.
        Values must sum to 1.0.

    shuffle : bool
        Whether to shuffle rows before splitting.

    random_state : int or None
        Seed for reproducible shuffling.

    column_name : str
        Name of the output column.

    Returns
    -------
    pd.Series
        A Series aligned with df.index containing split labels.
    """

    if split_ratios is None:
        split_ratios = {
            "train": 0.9,
            "dev": 0.05,
            "test": 0.05,
        }

    if not np.isclose(sum(split_ratios.values()), 1.0):
        raise ValueError("split_ratios must sum to 1.0")

    if mask is None:
        selected_mask = pd.Series(True, index=df.index)
    elif isinstance(mask, str):
        if mask not in df.columns:
            raise KeyError(f"Column '{mask}' not found in dataframe.")
        selected_mask = df[mask]
    else:
        selected_mask = pd.Series(mask, index=df.index)

    if selected_mask.dtype != bool:
        raise TypeError(
            "mask must be a boolean Series or the name of a boolean column."
        )

    selected_idx = df.index[selected_mask].to_numpy()

    selected_idx = np.array(selected_idx)

    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(selected_idx)

    n_total = len(selected_idx)
    n_train = int(n_total * split_ratios["train"])
    n_dev = int(n_total * split_ratios["dev"])

    split = pd.Series(
        "undefined",
        index=df.index,
        name="split"
    )

    split.loc[selected_idx[: n_train]] = "train"
    split.loc[selected_idx[n_train: n_train + n_dev]] = "dev"
    split.loc[selected_idx[n_train + n_dev:]] = "test"

    return split


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
            subject=f"Training loss: {loss:>7f}",
        )
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

def find_ps_pairs(
        metadata
    ):
    """
    Return a boolean mask indicating rows that contain
    both P and S arrival picks.
    """
    mapper = build_phase_mapper(metadata.columns)

    p_cols = [col for col, phase in mapper.items() if phase == "P"]
    s_cols = [col for col, phase in mapper.items() if phase == "S"]

    p_exists = metadata[p_cols].notna().any(axis=1) if p_cols else False
    s_exists = metadata[s_cols].notna().any(axis=1) if s_cols else False

    ps_pairs = p_exists & s_exists

    return ps_pairs

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
    
    ps_pair = find_ps_pairs(
        metadata=dataset.metadata
        )
    dataset.metadata['PS-Pairs'] = ps_pair
    dataset.metadata.loc[23972, 'PS-Pairs'] = False
    


    phase_dict = build_phase_mapper(
        dataset.metadata.columns
    )
    
    sps = 100
    augmentations = [
        Tapering(),
        sbg.Normalize(
            demean_axis=-1,
            amp_norm_axis=-1,
            amp_norm_type="peak",
        ),
        sbg.FixedWindow(
            p0=-15*sps,
            windowlen=1*60*sps,
            strategy="pad",
            key='X',
        ),
        sbg.WindowAroundSample(
            metadata_keys=list(phase_dict.keys()),
            samples_before=2000,
            windowlen=5000,
            selection="random",
            strategy="variable",
        ),
        sbg.GaussianNoise(
            scale=(0, 0.02),
            key='X',
        ),
        sbg.RandomWindow(
            windowlen=3001,
        ),
        sbg.ChangeDtype(
            np.float32
        ),
        sbg.ProbabilisticLabeller(
            label_columns=phase_dict,
            model_labels=cfg.model.hyperparameters.phases,
            sigma=30,
            dim=0,
        ),
    ]
    
    split_ratios = {
        'train': cfg.dataset.split_ratios.train,
        'dev':   cfg.dataset.split_ratios.dev,
        'test':  cfg.dataset.split_ratios.test
    }
    
    dataset.metadata['split'] = build_split_column(
        df=dataset.metadata,
        mask='PS-Pairs',
        split_ratios=split_ratios,
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
        batch_size=cfg.dataloader.train.batch_size,
        shuffle=cfg.dataloader.train.shuffle,
        num_workers=cfg.dataloader.train.num_workers,
        worker_init_fn=sbu.worker_seeding,
    )
    dev_loader = DataLoader(
        dev_generator,
        batch_size=cfg.dataloader.validation.batch_size,
        shuffle=cfg.dataloader.validation.shuffle,
        num_workers=cfg.dataloader.validation.num_workers,
        worker_init_fn=sbu.worker_seeding,
    )
    test_loader = DataLoader(
        test_generator,
        batch_size=cfg.dataloader.test.batch_size,
        shuffle=cfg.dataloader.test.shuffle,
        num_workers=cfg.dataloader.test.num_workers,
        worker_init_fn=sbu.worker_seeding,
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
    for learning_rate, epochs in zip(cfg.train.hyperparameters.learning_rates,
                                     cfg.train.hyperparameters.epochs_for_each_learning_rate):
        logging.info(f"Main Learning-Rate: {learning_rate}\n" + "-"*70)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(learning_rate),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode='min',
            factor=cfg.train.hyperparameters.lr_scheduler.ReduceLROnPlateau.factor,
            patience=cfg.train.hyperparameters.lr_scheduler.ReduceLROnPlateau.patience,
            threshold=0.0001,
            threshold_mode='rel',
            cooldown=0,
            min_lr=0,
            eps=1e-08,
        )
        for epoch in range(epochs):
            learning_rate = scheduler.get_last_lr()[0]
            logging.info(f"Learning-Rate: {learning_rate} Epoch: {epoch+1}\n" + "-"*70)
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
            for batch, loss in train_loss:
                dict_tmp = {
                    'epoch': epoch,
                    'batch': batch,
                    'loss_train': loss,
                    'loss_test': test_loss,
                }
                log_learning.append(dict_tmp)
    df_loss = pd.DataFrame(log_learning)
    os.makedirs(
        os.path.abspath(cfg.path.model),
        exist_ok=True
    )
    
    df_loss.to_csv(
        os.path.join(
            cfg.path.model,
            f'loss_{cfg.model.version_str}.csv'
        )
    )
    
    model.save(
        path=os.path.join(
            cfg.path.model,
            f"{cfg.model.name}_{cfg.model.version_str}"
        ),
        weights_docstring=cfg.__str__(),
        version_str=cfg.model.version_str,
    )