import torch
import time
import numpy as np


def run_pipeline_optimization(pipeline, num_epochs, optimizer_type="Adam", lr=1e-3, loss_fn=None, load_previous_ckpt=False):
    """Runs the training for DFlat's custom pipelines."""

    # If no new loss function is provided then define the loss as the pipeline output
    if loss_fn is None:

        def loss_fn(pipeline_output):
            return pipeline_output

    # Get the optimizer
    train_tensors = [param for _, param in pipeline.get_trainable_variables()]
    if optimizer_type == "Adam":
        optimizer = torch.optim.Adam(train_tensors, lr=lr)
    elif optimizer_type == "SGD":
        optimizer = torch.optim.SGD([train_tensors], lr=0.01)
    else:
        raise ValueError("Unrecognized Optimizer string. See torch.optim optimizer names")

    # Run training
    train_loop(pipeline, optimizer, loss_fn, num_epochs, load_previous_ckpt)

    return


def train_loop(pipeline, optimizer, loss_fn, num_epochs, load_previous_ckpt):
    prev_epoch = pipeline.customLoad(optimizer) if load_previous_ckpt else 0
    mini_ckpt = pipeline.saveAtEpochs
    start_iter = prev_epoch
    pipeline.visualizeTrainingCheckpoint(prev_epoch)

    lossVec = []
    for epoch in np.arange(start_iter + 1, num_epochs + 1, 1):
        pipeline.train()
        optimizer.zero_grad()
        start = time.time()
        loss = loss_fn(pipeline())
        loss.backward()
        optimizer.step()
        end = time.time()

        # Logging
        current_loss = loss.item()
        print(f"Training Log | (Step {epoch}, time {end - start:.2f}, loss {current_loss:.5f}, lr {optimizer.param_groups[0]['lr']})")
        lossVec.append(current_loss)
        train_tensors = [param for _, param in pipeline.get_trainable_variables()]

        if mini_ckpt is not None:
            if np.mod(epoch, mini_ckpt) == 0:
                print(f"Log Visualization at step: {epoch}")
                pipeline.visualizeTrainingCheckpoint(epoch)
                print("Save Checkpoint Model:")
                pipeline.customSaveCheckpoint(train_loss_vector=lossVec, optimizer_state=optimizer.state_dict(), verbose=True)
                lossVec = []

    # Save post-training
    pipeline.customSaveCheckpoint(train_loss_vector=lossVec, optimizer_state=optimizer.state_dict(), verbose=True)

    return
