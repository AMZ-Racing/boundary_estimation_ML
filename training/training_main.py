# train_set_transformer.py
import os
import glob
import argparse
import time
from typing import List, Tuple

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from set_transformer import SetTransformerClassifier
from val import validate

from models.wrapper import ModelWrapper
from dataset.pandas_tracks_set import get_dataloader

from utils.logging_utils import TensorboardLogger

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    logger = TensorboardLogger(log_dir=args.checkpoint_dir)
    train_loader = get_dataloader()
    val_loader = get_dataloader()

    model = ModelWrapper(
        model_name=args.model,     # e.g. "set_transformer"
        input_dim=2,
        d_model=128,
        nhead=8,
        num_layers=4,
        num_classes=5
    ).to(device)


    criterion = nn.CrossEntropyLoss(ignore_index=-100)  # pads have label -100
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    best_val = -1.0
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, (points, labels, mask, lengths) in enumerate(train_loader, start=1):
            points = points.to(device)            # (B, N, 2)
            labels = labels.to(device)            # (B, N)
            mask = mask.to(device)                # (B, N)

            optimizer.zero_grad()
            logits = model(points, mask)          # (B, N, C)

            # reshape for loss: (B*N, C) vs (B*N,)
            B, N, C = logits.shape
            loss = criterion(logits.view(-1, C), labels.view(-1))
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            epoch_loss += float(loss.item())
            global_step += 1

            if global_step % args.print_every == 0:
                # TODO: add tensorboard logging
                logger.log_train_loss(loss.item(), global_step)
                print(f"[Epoch {epoch} Step {global_step}] loss={loss.item():.4f}")

        scheduler.step()
        t1 = time.time()
        avg_loss = epoch_loss / (batch_idx + 1)
        # TODO: add tensorboard logging
        print(f"Epoch {epoch} finished — avg loss {avg_loss:.4f}, time {t1 - t0:.1f}s")

        # va;
        val_res = validate(model, val_loader, device, id_left=args.id_left, id_right=args.id_right)
        val_score = val_res["macro_f1"]
        print("val f1:", val_res)

        # save ckpt
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(args.checkpoint_dir, f"ckpt_epoch{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_res": val_res,
            "args": vars(args)
        }, ckpt_path)

        if val_score > best_val:
            best_val = val_score
            best_path = os.path.join(args.checkpoint_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_res": val_res,
                "args": vars(args)
            }, best_path)
            print(f"Saved new best model (macro_f1={best_val:.4f}) -> {best_path}")
    
    logger.close()
    print("Training complete. Best macro_f1:", best_val)


# ------------------------------
#  CLI / main
# ------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", required=True)
    p.add_argument("--val_dir", required=True)
    p.add_argument("--checkpoint_dir", default="checkpoints")

    # model                                                 # TODO: all values in one column to make it easier. In the future this should be done via a config
    p.add_argument("--d_model", type=int,                   default=128)
    p.add_argument("--nhead", type=int,                     default=8)
    p.add_argument("--num_layers", type=int,                default=4)
    p.add_argument("--dim_feedforward", type=int,           default=256)
    p.add_argument("--dropout", type=float,                 default=0.1)

    # training
    p.add_argument("--epochs", type=int,                    default=40)
    p.add_argument("--batch_size", type=int,                default=16)
    p.add_argument("--val_batch_size", type=int,            default=8)
    p.add_argument("--max_points", type=int,                default=1024, help="max points per sample (will truncate longer)")
    p.add_argument("--lr", type=float,                      default=1e-3)
    p.add_argument("--weight_decay", type=float,            default=1e-6)
    p.add_argument("--lr_step", type=int,                   default=10)
    p.add_argument("--lr_gamma", type=float,                default=0.5)
    p.add_argument("--grad_clip", type=float,               default=1.0)
    p.add_argument("--print_every", type=int,               default=50)

    # labels
    p.add_argument("--num_classes", type=int,               default=5)
    p.add_argument("--id_left", type=int,                   default=0)
    p.add_argument("--id_right", type=int,                  default=1)
    
    
    
    p.add_argument("--model", type=str,                     default="mlp",
                    choices=["set_transformer", "mlp", "transformer"])
    
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
