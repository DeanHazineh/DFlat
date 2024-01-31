import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from .load_utils import instantiate_from_config


class Trainer_v1:
    def __init__(
        self,
        config_model,
        ckpt_path,
        dataset,
        test_split=0.10,
        learning_rate=1e-3,
        epochs=1000,
        batch_size=256,
        checkpoint_every_n=100,
        update_figure_every_epoch=True,
        gradient_accumulation_steps=1,
        cosine_anneal_warm_restart=False,
        cosine_anneal_minLR=1e-6,
        cosine_anneal_T_0=400,
        cosine_anneal_T_mult=2,
        **kwargs,
    ):
        config_model.params.trainable_model = True
        self.model = instantiate_from_config(config_model)

        locals_dict = locals()
        for name, value in locals_dict.items():
            if name != "self":
                setattr(self, name, value)

        for key, value in kwargs.items():
            setattr(self, key, value)

        assert test_split < 1.0 and test_split > 0.0
        total_size = len(dataset)
        test_size = int(total_size * test_split)
        train_size = total_size - test_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        self.train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_dl = DataLoader(test_dataset, batch_size=batch_size)

    def train(self):
        model = self.model
        test_dl = self.test_dl
        train_dl = self.train_dl
        lr = self.learning_rate
        gradient_accumulation_steps = self.gradient_accumulation_steps
        cosine_anneal_warm_restart = self.cosine_anneal_warm_restart
        ckpt_path = self.ckpt_path
        ckpt_dir = os.path.dirname(ckpt_path)
        epochs = self.epochs
        checkpoint_every_n = self.checkpoint_every_n

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        # Fix grad accumulation step number
        num_batches_in_epoch = len(train_dl)
        if gradient_accumulation_steps > num_batches_in_epoch:
            print(
                f"gradient_accumulation_steps changed to epoch batch num {num_batches_in_epoch}"
            )
            gradient_accumulation_steps = num_batches_in_epoch

        # Set up optimizer
        optimizer = AdamW(model.parameters(), lr=lr)
        if cosine_anneal_warm_restart:
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.cosine_anneal_T_0,
                T_mult=self.cosine_anneal_T_mult,
                eta_min=self.cosine_anneal_minLR,
            )

        # Load the last checkpoint
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            model.load_state_dict(checkpoint["state_dict"])
            model.to("cuda")  # Need to move to cuda before loading optimizer state

            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if cosine_anneal_warm_restart:
                scheduler.load_state_dict(checkpoint["scheduler"])

            start_epoch = checkpoint["epoch"]
            train_stats = pd.read_pickle(os.path.join(ckpt_dir, "training_stats.pkl"))
            print(f"Loaded checkpoint from epoch {start_epoch}")
        else:
            model.to("cuda")
            train_stats = pd.DataFrame()
            start_epoch = 0

        # Run Training with visualization
        for epoch in range(start_epoch, epochs + 1):
            epoch_pbar = tqdm(
                enumerate(train_dl), total=len(train_dl), desc=f"Epoch {epoch}"
            )
            epoch_lr = optimizer.param_groups[0]["lr"]
            step_losses = []
            optimizer.zero_grad()

            # Train Loop
            for step, (x, y) in epoch_pbar:
                x = x.to(dtype=torch.float32, device="cuda")
                y = y.to(dtype=torch.float32, device="cuda")
                loss = model.training_step(x, y)
                loss = loss / gradient_accumulation_steps
                loss.backward()
                if np.mod(step + 1, gradient_accumulation_steps) == 0 or (
                    step + 1
                ) == len(train_dl):
                    optimizer.step()
                    optimizer.zero_grad()

                step_losses.append(loss.item() * gradient_accumulation_steps)
                epoch_pbar.set_postfix(
                    {"loss": loss.item() * gradient_accumulation_steps, "lr": epoch_lr}
                )

            # Evaluation Loop
            test_losses = []
            model.eval()  # Set the model to evaluation mode
            with torch.no_grad():  # Disable gradient computation
                for x, y in test_dl:
                    x = x.to(dtype=torch.float32, device="cuda")
                    y = y.to(dtype=torch.float32, device="cuda")
                    loss = model.training_step(x, y)
                    test_losses.append(loss.item())
            avg_test_loss = np.mean(test_losses)

            # Update the progress bar description with the current loss
            epoch_loss = np.mean(step_losses)
            desc = pd.Series(step_losses).describe()
            desc["lr"] = epoch_lr
            desc["test_loss"] = avg_test_loss
            print(f"loss {epoch_loss} lr {epoch_lr} test_loss: {avg_test_loss}")
            if train_stats.empty:
                train_stats = pd.DataFrame(columns=desc.index)
            train_stats.loc[epoch] = desc

            # step the lr scheduler if used
            if cosine_anneal_warm_restart:
                scheduler.step()

            # Save a snapshot of the model at the current checkpoint
            if self.update_figure_every_epoch or np.mod(epoch, checkpoint_every_n) == 0:
                self.plot_training_loss(train_stats, ckpt_dir)

            if np.mod(epoch, checkpoint_every_n) == 0 and epoch > 0:
                state = {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                if cosine_anneal_warm_restart:
                    state["scheduler"] = scheduler.state_dict()

                torch.save(state, ckpt_path)
                train_stats.to_pickle(os.path.join(ckpt_dir, "training_stats.pkl"))

        return

    def plot_training_loss(self, train_stats, log_folder):
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        ax.plot(train_stats["mean"], "ro-", label="Mean", linewidth=4, alpha=0.3)
        ax.plot(
            train_stats["test_loss"], "go-", label="Test Loss", linewidth=4, alpha=0.3
        )

        ax.scatter(
            train_stats.index,
            train_stats["max"],
            color="grey",
            marker=".",
            label="Max Value",
            alpha=0.2,
        )
        ax.scatter(
            train_stats.index,
            train_stats["min"],
            color="grey",
            marker=".",
            label="Min Value",
            alpha=0.2,
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Statistics")
        ax.set_yscale("log")

        # Twin the y-axis for lr display
        ax2 = ax.twinx()
        ax2.plot(train_stats["lr"], color="blue", alpha=0.4, label="Learning Rate (LR)")
        ax2.set_ylabel("Learning Rate")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc=0)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(log_folder, "training_loss.png"))
        plt.close()

        return
