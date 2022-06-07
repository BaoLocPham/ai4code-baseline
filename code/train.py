import json
from pathlib import Path
from dataset import *
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from model import *
from tqdm import tqdm
import sys, os
from metrics import *
import torch
import argparse
from config import CFG
import wandb

parser = argparse.ArgumentParser(description='Process some arguments')
parser.add_argument('--model_name_or_path', type=str, default='microsoft/codebert-base')
parser.add_argument("--start_epoch", type=int, default=0)
parser.add_argument('--md_max_len', type=int, default=64)
parser.add_argument('--total_max_len', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--accumulation_steps', type=int, default=4)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--n_workers', type=int, default=8)

args = parser.parse_args()

train_df_mark = pd.read_csv(f"{CFG.LOAD_DATA_PATH}/train_mark.csv").drop("parent_id", axis=1).dropna().reset_index(drop=True)
train_fts = json.load(open(f"{CFG.LOAD_DATA_PATH}/train_fts.json"))
val_df_mark = pd.read_csv(f"{CFG.LOAD_DATA_PATH}/val_mark.csv").drop("parent_id", axis=1).dropna().reset_index(drop=True)
val_fts = json.load(open(f"{CFG.LOAD_DATA_PATH}/val_fts.json"))
val_df = pd.read_csv(f"{CFG.LOAD_DATA_PATH}/val.csv")

order_df = pd.read_csv(f"{CFG.LOAD_DATA_PATH}/train_orders.csv").set_index("id")
df_orders = pd.read_csv(
   f'{CFG.LOAD_DATA_PATH}/train_orders.csv',
    index_col='id',
    squeeze=True,
).str.split()

train_ds = MarkdownDataset(train_df_mark, model_name_or_path=CFG.BERT_PATH, md_max_len=CFG.MAX_LEN,
                           total_max_len=CFG.TOTAL_MAX_LEN, fts=train_fts)
val_ds = MarkdownDataset(val_df_mark, model_name_or_path=CFG.BERT_PATH, md_max_len=CFG.MAX_LEN,
                         total_max_len=CFG.TOTAL_MAX_LEN, fts=val_fts)
train_loader = DataLoader(train_ds, batch_size=CFG.BS, shuffle=True, num_workers=CFG.NW,
                          pin_memory=False, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=CFG.BS, shuffle=False, num_workers=CFG.NW,
                        pin_memory=False, drop_last=False)

def read_data(data):
    return tuple(d.cuda() for d in data[:-1]), data[-1].cuda()


def validate(model, val_loader):
    model.eval()

    tbar = tqdm(val_loader, file=sys.stdout)

    preds = []
    labels = []

    with torch.no_grad():
        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)

            with torch.cuda.amp.autocast():
                pred = model(*inputs)

            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())

    return np.concatenate(labels), np.concatenate(preds)

def train(model, train_loader, val_loader, start, epochs):
    # Creating optimizer and lr schedulers
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    num_train_optimization_steps = int(CFG.EPOCHS * len(train_loader) / CFG.ACCUMULATION_STEPS)
    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5,
                      correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.05 * num_train_optimization_steps,
                                                num_training_steps=num_train_optimization_steps)  # PyTorch scheduler

    criterion = torch.nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler()
    wandb.watch(model, log="all")
    for e in range(start, epochs):
        model.train()
        tbar = tqdm(train_loader, file=sys.stdout)
        loss_list = []
        preds = []
        labels = []

        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)

            with torch.cuda.amp.autocast():
                pred = model(*inputs)
                loss = criterion(pred, target)
            scaler.scale(loss).backward()
            if idx % CFG.ACCUMULATION_STEPS == 0 or idx == len(tbar) - 1:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                torch.save(model.state_dict(), f"{CFG.SAVE_PATH}/model_steps_{idx}.bin")
                wandb.save(f'model_steps_{idx}.bin')
            
            loss_list.append(loss.detach().cpu().item())
            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())

            avg_loss = np.round(np.mean(loss_list), 4)

            tbar.set_description(f"Epoch {e + 1} Loss: {avg_loss} lr: {scheduler.get_last_lr()}")
            wandb.log({"MAE":avg_loss})
            
        y_val, y_pred = validate(model, val_loader)
        val_df["pred"] = val_df.groupby(["id", "cell_type"])["rank"].rank(pct=True)
        val_df.loc[val_df["cell_type"] == "markdown", "pred"] = y_pred
        y_dummy = val_df.sort_values("pred").groupby('id')['cell_id'].apply(list)
        score = kendall_tau(df_orders.loc[y_dummy.index], y_dummy)
        print("Preds score", score)
        wandb.log({"Preds score":score})
        torch.save(model.state_dict(), f"{CFG.SAVE_PATH}/model_{e}.bin")

    return model, y_pred

# WandB – Initialize a new run
WANDB_CONFIG = {
    'TRAIN_BS': CFG.BS,
    # 'VALID_BS': CFG.VALID_BS,
    'N_EPOCHS': CFG.EPOCHS,
    'ARCH': CFG.BERT_PATH,
    'MAX_LEN': CFG.MAX_LEN,
    'TOTAL_MAX_LEN': CFG.TOTAL_MAX_LEN,
    'ACCUMULATION_STEPS': CFG.ACCUMULATION_STEPS,
    'LR': CFG.LR,
    'NUM_WORKERS': CFG.NW,
    'OPTIM': "AdamW",
    'LOSS': "MAE",
    'DEVICE': "cuda",
    'competition': 'ai4code',
}
run = wandb.init(name="CodeBert-Baseline", project="AI4Code")
wandb.watch_called = False 
np.random.seed(0)

def wandb_log(**kwargs):
    """
    Logs a key-value pair to W&B
    """
    for k, v in kwargs.items():
        wandb.log({k: v})

os.mkdir(CFG.SAVE_PATH)

model = MarkdownModel(args.model_name_or_path)
model = model.cuda()
model, y_pred = train(model, train_loader, val_loader, start=0, epochs=CFG.EPOCHS)
