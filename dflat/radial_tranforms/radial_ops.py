import torch
import torch.nn.functional as F


def resize_with_crop_or_pad(input_tensor, target_height, target_width, radial_flag):
    """This function is analogous to tf.image.resize_with_crop_or_pad but with a radial option as well. When radial_flag is true, the input radial vector is only cropped
    or padded asymettrically (instead of centrally)

    Args:
        input_tensor (float): Input tensor of shape [..., H, W]
        target_height (int): _description_
        target_width (int): _description_
        radial_flag (boolean): _decription_

    Returns:
        float: resized tensor
    """

    tensor_shape = input_tensor.shape
    diffH = int(tensor_shape[-2] - target_height)
    diffW = int(tensor_shape[-1] - target_width)

    # Exit early if nothing is needed
    if diffH == 0 and diffW == 0:
        return input_tensor
    else:
        # Run central padding if needed
        if (diffH < 0) or (diffW < 0):
            padding = [0 for _ in range(2 * input_tensor.dim())]
            if diffW < 0:
                padby = abs(diffW)
                if radial_flag:
                    padding[0] = 0
                    padding[1] = padby
                else:
                    padding[0] = padby // 2
                    padding[1] = padby // 2 if padby % 2 == 0 else (padby // 2 + 1)
            if diffH < 0 and not radial_flag:
                padby = abs(diffH)
                padding[2] = padby // 2
                padding[3] = padby // 2 if padby % 2 == 0 else (padby // 2 + 1)

            input_tensor = F.pad(input_tensor, padding, mode="constant", value=0)

            if (diffH < 0) and (diffW < 0):
                return input_tensor

        # Run the central cropping if needed
        else:
            crop_height = min(target_height, tensor_shape[-2])
            crop_width = min(target_width, tensor_shape[-1])
            if radial_flag:
                input_tensor = input_tensor[..., :, :crop_width]
            else:
                start_H = (tensor_shape[-2] - crop_height) // 2
                start_W = (tensor_shape[-1] - crop_width) // 2
                input_tensor = input_tensor[
                    ..., start_H : start_H + crop_height, start_W : start_W + crop_width
                ]

    return input_tensor


def radial_2d_transform(r_array):
    """Transform a radial, real array (...,N) to a 2D profile (,2N-1, 2N-1).

    Args:
        r_array (float): Input radial vector/tensor of shape (..., N).

    Returns:
        float: 2D-converted data of shape (..., 2N-1, 2N-1).
    """
    is_tensor_flag = torch.is_tensor(r_array)
    if not is_tensor_flag:
        r_array = torch.tensor(r_array)

    # Flatten and appropriately reshape the input array for processing
    N = int(r_array.shape[-1])
    batch_shape = r_array.shape[:-1]
    x_r = r_array.view(-1, N)[:, None, None, :]  # (N, C, Hin, Win)
    flat_batch_dim = x_r.shape[0]

    # Define the new output coordinates (normalized to [-1,1])
    xx, yy = torch.meshgrid(
        torch.arange(1 - N, N), torch.arange(1 - N, N), indexing="ij"
    )
    rq = torch.sqrt(xx**2 + yy**2).view(1, -1).to(dtype=x_r.dtype)
    rq = ((rq / (N - 1)) - 0.5) * 2
    yq = torch.zeros(*rq.size(), dtype=rq.dtype)

    coord_list = torch.transpose(torch.cat((rq, yq), dim=0), 0, 1)[None, None]
    coord_list = coord_list.expand([flat_batch_dim, -1, -1, -1])  # (N, 1, Wout, 2)
    coord_list = coord_list.to(r_array.device)

    # Compute the transformation
    r_out = F.grid_sample(
        x_r, coord_list, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    r_out = r_out.view(flat_batch_dim, 2 * N - 1, 2 * N - 1)
    r_out = r_out.view(*batch_shape, *r_out.shape[-2:])

    return r_out if is_tensor_flag else r_out.cpu().numpy()


def radial_2d_transform_wrapped_phase(r_array):
    """Transform a radial, real array of phase values in radians (,N) to a 2D phase profile (,2N-1, 2N-1).
    This function is analogous to radial_2d_transform but properly interpolates the phase-wrapping discontinuity.

    Args:
        r_array (float): Input radial vector/tensor of shape (batch_shape, N).

    Returns:
        float: 2D-converted phase data via tensor of shape (batch_shape, 2N-1, 2N-1).
    """

    is_tensor_flag = torch.is_tensor(r_array)
    if not is_tensor_flag:
        r_array = torch.tensor(r_array)

    r_array = torch.atan2(
        radial_2d_transform(torch.sin(r_array)), radial_2d_transform(torch.cos(r_array))
    )
    return r_array if is_tensor_flag else r_array.cpu().numpy()


def general_interp_regular_1d_grid(x, xi, y):
    """Computes the 1D interpolation of a complex tensor defined on a regular grid.

    Args:
        x (float): tensor specifying the reference grid coordinates.
        xi (float): grid values to interpolate on (may be non-uniform).
        y (float): reference data corresponding to x to interpolate over, of size (..., Nx).
    Returns:
        float: New complex output interpolated on new grid points
    """
    # IMPORTANT NOTE: Using this implementation to snap back to a uniform grid after the qdht calls is is not correct because the qdht returns non-uniform grids
    # However, we will use this to reduce cost and since we find the errors to have negligible effect thus far
    # As of now, there is no auto-differentiable pytorch or tensorflow implementation to take non-uniform grids and interpolate a new non-uniform grid

    # Cast to tensor if not passed in as a torch tensor
    if not torch.is_tensor(y):
        y = torch.tensor(y)
    dtype = y.dtype
    device = y.device

    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=dtype).to(device)
    if not torch.is_tensor(xi):
        xi = torch.tensor(xi, dtype=dtype).to(device)

    if dtype == torch.complex64 or dtype == torch.complex128:
        interpFr = helper_interp_complex(torch.min(x), torch.max(x), xi, y)
    else:
        interpFr = helper_interp_real(torch.min(x), torch.max(x), xi, y)

    return interpFr


def helper_interp_complex(x_min, x_max, xi, y):
    # Get the real signal components
    f_transform_abs = interp_regular_1d_grid(x_min, x_max, torch.abs(y), xi)
    f_transform_real = interp_regular_1d_grid(
        x_min, x_max, torch.cos(torch.angle(y)), xi
    )
    f_transform_imag = interp_regular_1d_grid(
        x_min, x_max, torch.sin(torch.angle(y)), xi
    )

    torch_zero = torch.tensor(0.0, dtype=f_transform_abs.dtype).to(y.device)
    return torch.complex(f_transform_abs, torch_zero) * torch.exp(
        torch.complex(torch_zero, torch.atan2(f_transform_imag, f_transform_real))
    )


def helper_interp_real(x_min, x_max, xi, y):
    f_transform = interp_regular_1d_grid(x_min, x_max, y, xi)
    return f_transform


def interp_regular_1d_grid(x_min, x_max, y_ref, xi):
    """Pytorch related port of tfp.math.interp_regular_1d_grid. Given a batch of input 1D data y_ref of shape (..., N) and
    defined on the regurlar grid of points x_ref, returns reinterpolated values at the new set of 1D coordinates specified by xi.

    Args:
        x_min (float): Minimum bound of the uniform x-grid coordinates x_ref.
        x_max (float): Maximum bound of the uniform x-grid coordinates x_ref.
        y_ref (float): Input data tensor of shape (..., Nx).
        xi (float): List of interpolation points (may be non-uniformly spaced).
    """
    input_rank = y_ref.dim()
    batch_shape = y_ref.shape[:-1]
    N = y_ref.shape[-1]

    if input_rank == 1:
        y_ref = y_ref.unsqueeze(0)
    elif input_rank > 2:
        y_ref = y_ref.view(-1, N)

    flat_batch_dim = y_ref.shape[0]
    y_ref = y_ref[:, None, None, :]

    # Define query points
    xi = ((xi - x_min) / x_max - 0.5) * 2.0  # normalized range [-1, 1]
    yq = torch.zeros(*xi.shape, dtype=y_ref.dtype).to(xi.device)
    grid = torch.transpose(torch.cat((xi[None], yq[None]), dim=0), 0, 1)[
        None, None
    ].contiguous()
    grid = grid.expand([flat_batch_dim, -1, -1, -1])

    # Interpolate by grid_sample
    yi = F.grid_sample(
        y_ref, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    yi = yi.view(*batch_shape, -1)

    return yi
