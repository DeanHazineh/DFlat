import numpy as np
from sklearn.model_selection import train_test_split
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, TensorDataset
from torch_utils import print_model_summary

import dflat.neural_optical_layer.core.models_DNN as MLP_models
from trainer_visualize_ckpt import visualize_nanofins, visualize_nanocylinder, save_training_imgs_as_gifs

torch.manual_seed(0)


def run_training_neural_model(model, epochs, miniEpoch=1000, batch_size=None, lr=1e-4, verbose=True, train=True, visFun=None):
    ### Get training and testing data:
    inputData, outputData = model.returnLibraryAsTrainingData()
    xtrain, xtest, ytrain, ytest = train_test_split(inputData, outputData, test_size=0.10, random_state=13, shuffle=True)

    # Convert to PyTorch tensors
    xtrain, ytrain = torch.tensor(xtrain, dtype=torch.float32), torch.tensor(ytrain, dtype=torch.float32)
    xtest, ytest = torch.tensor(xtest, dtype=torch.float32), torch.tensor(ytest, dtype=torch.float32)

    # If no batch size is specified then use the full training dataset size
    if batch_size == None:
        batch_size = xtrain.shape[0]
        print(batch_size)

    train_dataset = TensorDataset(xtrain, ytrain)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print_model_summary(model)
    if train:
        device = torch.device("cuda:0")
        print("Training on device: ", device)
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        # scheduler = ExponentialLR(optimizer, gamma=0.99)
        # prev_epochs = model.customLoadCheckpoint(optimizer, scheduler)  # Restore checkpoint
        prev_epochs = model.customLoadCheckpoint(optimizer)  # Restore checkpoint
        criterion = nn.MSELoss()
        for split_sess in range(np.ceil(epochs / miniEpoch).astype("int")):
            hold_train_loss = []
            hold_test_loss = []
            for epoch in range(miniEpoch):
                total_epoch = prev_epochs + split_sess * miniEpoch + epoch
                if total_epoch > epochs:
                    break

                model.train()
                for batch in train_loader:
                    inputs, targets = batch
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    xtest, ytest = xtest.to(device), ytest.to(device)
                    test_outputs = model(xtest)
                    test_complex_error = test_outputs - ytest
                    mae_test_set = torch.mean(torch.abs(test_complex_error)).cpu().numpy()

                    # Edit: I want to save a visualization plot at each step to make a gif
                    if visFun is not None and np.mod(epoch, 5) == 0:
                        visFun(model, total_epoch, device)

                # Step the scheduler
                # scheduler.step()
                hold_train_loss.append(loss.item())
                hold_test_loss.append(mae_test_set)
                if verbose:
                    print("Epoch {} [{}/{}], Loss: {:.4f} TestLoss: {:.4f}".format(total_epoch, epoch + 1, miniEpoch, loss.item(), mae_test_set))

            if total_epoch > epochs:
                break
            else:
                save_dict = {
                    "optimizer_state_dict": optimizer.state_dict(),
                    # "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": total_epoch,
                    "loss": loss,
                }
                model.customSaveCheckpoint(test_loss=hold_test_loss, training_loss=hold_train_loss, checkpoint_dictionary=save_dict, verbose=True)

    return


if __name__ == "__main__":
    # Always train neural models with float32 because we do not need float64 and that is not standard
    # save_training_imgs_as_gifs(model_tag="Nanocylinders=")

    run_training_neural_model(
        model=MLP_models.MLP_Nanocylinders_Dense256_U180_H600_SIREN100(dtype=torch.float32),
        epochs=15000,
        miniEpoch=1000,
        batch_size=None,
        lr=1e-3,
        visFun=visualize_nanocylinder,
    )

    # run_training_neural_model(
    #     model=MLP_models.MLP_Nanofins_Dense1024_U350_H600_SIREN100(dtype=torch.float32),
    #     epochs=15000,
    #     miniEpoch=200,
    #     batch_size=int(8e5),
    #     lr=1e-3,
    #     visFun=visualize_nanofins,
    # )
    # run_training_neural_model(
    #     model=MLP_models.MLP_Nanofins_Dense1024_U350_H600(dtype=torch.float32),
    #     epochs=15000,
    #     miniEpoch=200,
    #     batch_size=int(8e5),
    #     lr=1e-3,
    #     visFun=visualize_nanofins,
    # )
    # run_training_neural_model(
    #     model=MLP_models.MLP_Nanofins_Dense1024_U350_H600_SIREN50(dtype=torch.float32),
    #     epochs=15000,
    #     miniEpoch=200,
    #     batch_size=int(8e5),
    #     lr=1e-3,
    #     visFun=visualize_nanofins,
    # )
