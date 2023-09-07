import torch
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
import dflat.plot_utilities.graphFunc as df_plt


#  BASE CLASS FOR PIPELINES TO USE TRAIN HELPERS
class dflat_pipeline(nn.Module):
    """Baseclass for DFlat custom pipelines, inherits nn.Module structure."""

    def __init__(self, savepath, saveAtEpochs=None):
        """Initialization for base class

        Args:
            `savepath` (str): Pipeline savepath to store model checkpoints, data, and figures.
            `saveAtEpochs` (int): Number of training epochs between intermediate saves.
        """
        super().__init__()

        # Define class variables
        self.train_loss_vector = []
        self.test_loss_vector = []
        self.savepath = savepath
        self.saveAtEpochs = saveAtEpochs

        # create the savepath folder if it does not exist
        self.__checkModelPath()

    def forward(self):
        # It is expected that this function is overloaded by child class
        return

    def __call__(self):
        return self.forward()

    def customSaveCheckpoint(self, train_loss_vector=None, test_loss_vector=None, optimizer_state=None, checkpoint_dictionary=None, verbose=True):
        # Store model weights in the checkpoint
        model_state = self.state_dict()
        if checkpoint_dictionary == None:
            checkpoint_dictionary = {"model_state_dict": model_state}
        else:
            checkpoint_dictionary["model_state_dict"] = model_state

        if optimizer_state is not None:
            checkpoint_dictionary["optimizer_state_dict"] = optimizer_state

        # Add losses to the checkpoint file
        fig = plt.figure(figsize=(10, 5))
        ax = df_plt.addAxis(fig, 1, 2)

        if train_loss_vector is not None:
            self.train_loss_vector = np.concatenate((self.train_loss_vector, train_loss_vector))
            ax[0].plot(self.train_loss_vector, "b-.", label="training loss")
            ax[1].plot(np.log10(self.train_loss_vector), "b-.")

        if test_loss_vector is not None:
            self.test_loss_vector = np.concatenate((self.test_loss_vector, test_loss_vector))
            ax[0].plot(self.test_loss_vector, "r-.", label="test loss")
            ax[1].plot(np.log10(self.test_loss_vector), "r-.")

        checkpoint_dictionary["loss"] = self.train_loss_vector
        checkpoint_dictionary["test_loss"] = self.test_loss_vector

        df_plt.formatPlots(fig, ax[0], None, "epoch", "Loss", "Traning Loss", addLegend=True)
        df_plt.formatPlots(fig, ax[1], None, "epoch", "Log10(Loss)", "Traning Log(Loss)")
        plt.savefig(self.savepath + "/png_images/trainingLog_traininghistory.png")
        plt.savefig(self.savepath + "/pdf_images/trainingLog_traininghistory.pdf")
        plt.close()

        model_path = self.savepath + "checkpoint.pth"
        torch.save(checkpoint_dictionary, model_path)
        if verbose:
            print(f"Model checkpoint saved to: ", model_path)

    def customLoad(self, optimizer=None, scheduler=None):
        checkpoint_path = self.savepath + "checkpoint.pth"
        print("Checking for model checkpoint at: ", checkpoint_path)

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.load_state_dict(checkpoint["model_state_dict"])
            print("\n Model Checkpoint Loaded \n")
            self.train_loss_vector = checkpoint["loss"]
            self.test_loss_vector = checkpoint["test_loss"]

            if optimizer is not None and "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                print("\n Optimizer State Reloaded")

            if scheduler is not None and "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        else:
            print("\n no model checkpoint found at\n", checkpoint_path)

        return len(self.train_loss_vector)

    def visualizeTrainingCheckpoint(self, epoch_num):
        # It is expected that this function is overloaded by child class
        return

    def __checkModelPath(self):
        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)

        # Make folders for images too
        if not os.path.exists(self.savepath + "/png_images/"):
            os.makedirs(self.savepath + "/png_images/")
        if not os.path.exists(self.savepath + "/pdf_images/"):
            os.makedirs(self.savepath + "/pdf_images/")

        return

    def get_trainable_variables(self):
        return [(name, param) for name, param in self.named_parameters() if param.requires_grad]
