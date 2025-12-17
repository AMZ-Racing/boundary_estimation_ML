from torch.utils.tensorboard import SummaryWriter
import os

class TensorboardLogger:
    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

    def log_train_loss(self, loss, step):
        self.writer.add_scalar("train/loss", loss, step)

    def log_val_metrics(self, metrics: dict, epoch: int):
        for k, v in metrics.items():
            self.writer.add_scalar(f"val/{k}", v, epoch)

    def log_lr(self, optimizer, step):
        lr = optimizer.param_groups[0]["lr"]
        self.writer.add_scalar("train/lr", lr, step)

    def close(self):
        self.writer.close()
