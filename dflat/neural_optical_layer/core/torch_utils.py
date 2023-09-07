def print_model_summary(model):
    """
    Prints a summary of the PyTorch model with layer names and number of parameters.

    Args:
        model (nn.Module): PyTorch model instance
    """
    print("----------------------------------------------------------------")
    print("Layer (type)               Param #     ")
    print("================================================================")
    total_params = 0

    for name, layer in model.named_children():
        num_params = sum(p.numel() for p in layer.parameters())
        total_params += num_params

        print(f"{name} ({layer.__class__.__name__.ljust(20)}) ", end="")
        print(f"{num_params}")

    print("================================================================")
    print(f"Total params: {total_params}")
    print("----------------------------------------------------------------")
    return
