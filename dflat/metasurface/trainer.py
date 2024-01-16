import os
import shutil
from torch.optim import AdamW
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from .load_utils import instantiate_from_config


def empty_directory(directory_path):
    for root, dirs, files in os.walk(directory_path):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))


class Trainer_v1:
    def __init__(
        self,
        config_model,
        ckpt_path,
        learning_rate=1e-3,
        epochs=1000,
        batch_size=256,
        checkpoint_every_n=100,
        test_split=0.10,
        start_clean=True,
        gradient_accumulation_steps=1,
        cosine_anneal_warm_restart=False,
        cosine_anneal_minLR=1e-6,
        cosine_anneal_T_0=400,
        cosine_anneal_T_mult=2,
        update_figure_every_epoch=True,
        **kwargs,
    ):
        config_model.params.trainable_model = True
        self.model = instantiate_from_config(config_model)
        print(ckpt_path)

        locals_dict = locals()
        for name, value in locals_dict.items():
            if name != "self":
                setattr(self, name, value)

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.__init_dir()

    def train(self):
        model = self.model
        base_lib = model.base_library
        train_dl, test_dl = base_lib.dataloader(self.test_split, self.batch_size)
        if self.start_clean:
            self.__init_dir(overwrite=True)

        # Fix grad accumulation step number
        gradient_accumulation_steps = self.gradient_accumulation_steps
        num_batches_in_epoch = len(train_dl)
        if gradient_accumulation_steps > num_batches_in_epoch:
            print(
                f"gradient_accumulation_steps changed to epoch batch num {num_batches_in_epoch}"
            )
            gradient_accumulation_steps = num_batches_in_epoch

        # Set up optimizer
        optimizer = AdamW(model.parameters(), lr=self.learning_rate)
        cosine_anneal_warm_restart = self.cosine_anneal_warm_restart
        if cosine_anneal_warm_restart:
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.cosine_anneal_T_0,
                T_mult=self.cosine_anneal_T_mult,
                eta_min=self.cosine_anneal_minLR,
            )

        # Load the last checkpoint
        ckpt_path = self.ckpt_path
        last_ckpt_path = ckpt_path + "training_ckpt.ckpt"
        if os.path.exists(last_ckpt_path):
            checkpoint = torch.load(last_ckpt_path)
            model.load_state_dict(checkpoint["state_dict"])
            model.to("cuda")  # Need to move to cuda before loading optimizer state
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"]
            train_stats = pd.read_pickle(ckpt_path + "training_stats.pkl")
            print(f"Loaded checkpoint from epoch {start_epoch}")
        else:
            model.to("cuda")
            train_stats = pd.DataFrame()
            start_epoch = 0

        # Run Training with visualization
        epochs = self.epochs
        for epoch in range(start_epoch, epochs):
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
                    optimizer.step()  # Update weights
                    optimizer.zero_grad()  # Zero out gradients for the next set of accumulation

                step_losses.append(loss.item() * gradient_accumulation_steps)
                epoch_pbar.set_postfix(
                    {"loss": loss.item() * gradient_accumulation_steps, "lr": epoch_lr}
                )

            # Evaluation Loop
            avg_test_loss = 0.0
            if test_dl is not None:
                test_losses = []
                model.eval()  # Set the model to evaluation mode
                with torch.no_grad():  # Disable gradient computation
                    for x, y in test_dl:
                        x = x.to(dtype=torch.float32, device="cuda")
                        y = y.to(dtype=torch.float32, device="cuda")
                        loss = model.training_step(x, y)
                        test_losses.append(loss.item())
                avg_test_loss = np.mean(test_losses)
            else:
                avg_test_loss = 0.0

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
            checkpoint_every_n = self.checkpoint_every_n
            if self.update_figure_every_epoch or np.mod(epoch, checkpoint_every_n) == 0:
                self.plot_training_loss(train_stats, ckpt_path)

            if np.mod(epoch, checkpoint_every_n) == 0 and epoch > 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    ckpt_path + "training_ckpt.ckpt",
                )
                train_stats.to_pickle(ckpt_path + "training_stats.pkl")

        return

    def plot_training_loss(self, train_stats, log_folder):
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        ax.plot(train_stats["mean"], "ro-", label="Mean", linewidth=4, alpha=0.3)
        ax.plot(
            train_stats["test_loss"], "go-", label="Test Loss", linewidth=4, alpha=0.3
        )
        # ax.plot(train_stats["25%"], label="Lower Quartile", color="grey")
        # ax.plot(train_stats["75%"], label="Upper Quartile", color="grey")
        # ax.fill_between(
        #     train_stats.index,
        #     train_stats["25%"],
        #     train_stats["75%"],
        #     color="grey",
        #     alpha=0.2,
        # )

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
        plt.savefig(log_folder + "training_loss.png")
        plt.close()

        return

    def __init_dir(self, overwrite=False):
        ckpt_path = self.ckpt_path

        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        if overwrite:
            empty_directory(ckpt_path)

        return
