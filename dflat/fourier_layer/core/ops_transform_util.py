import torch
import torch.nn.functional as F


def resize_with_crop_or_pad(input_tensor, target_height, target_width, radial_flag):
    """This function is analogous to tf.image.resize_with_crop_or_pad but with a radial option as well. When radial_flag is true, the input radial vector is only cropped
    or padded asymettrically (instead of centrally)

    Args:
        tensor (float): Input tensor of shape [..., H, W]
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
                input_tensor = input_tensor[..., start_H : start_H + crop_height, start_W : start_W + crop_width]

    return input_tensor


def radial_2d_transform(r_array):
    """Transform a radial, real array (,N) to a 2D profile (,2N-1, 2N-1).

    Args:
        `r_array` (float): Input radial vector/tensor of shape (..., N).

    Returns:
        `tf.float`: 2D-converted data via tensor of shape (..., 2N-1, 2N-1).
    """

    N = int(r_array.shape[-1])
    batch_shape = r_array.shape[:-1]

    # Flatten and appropriately reshape the input array for processing
    x_r = r_array.view(-1, N)[:, None, None, :]
    flat_batch_dim = x_r.shape[0]

    # Define the new output coordinates (normalized to [-1,1])
    xx, yy = torch.meshgrid(torch.arange(1 - N, N).float(), torch.arange(1 - N, N).float(), indexing="ij")
    rq = torch.sqrt(xx**2 + yy**2).view(1, -1).type_as(x_r)
    rq = ((rq / (N - 1)) - 0.5) * 2

    yq = torch.zeros(*rq.size(), dtype=x_r.dtype)
    coord_list = torch.transpose(torch.cat((rq, yq), dim=0), 0, 1)[None, None]
    coord_list = coord_list.expand([x_r.shape[0], -1, -1, -1])

    # Compute the transformation
    r_out = F.grid_sample(x_r, coord_list, mode="bilinear", padding_mode="zeros", align_corners=True)
    r_out = r_out.view(flat_batch_dim, 2 * N - 1, 2 * N - 1)
    r_out = r_out.view(*batch_shape, *r_out.shape[-2:])

    return r_out


def radial_2d_transform_wrapped_phase(r_array):
    """Transform a radial, real array of phase values in radians (,N) to a 2D phase profile (,2N-1, 2N-1).
    This function is analogous to radial_2d_transform but properly interpolates the phase-wrapping discontinuity.

    Args:
        `r_array` (float): Input radial vector/tensor of shape (batch_shape, N).

    Returns:
        `tf.float`: 2D-converted phase data via tensor of shape (batch_shape, 2N-1, 2N-1).
    """

    realTrans = radial_2d_transform(torch.cos(r_array))
    imagTrans = radial_2d_transform(torch.sin(r_array))
    return torch.atan2(imagTrans, realTrans)


def radial_2d_transform_complex(r_array):
    """Transform a radial, complex array (,N) to a 2D profile (, 2N-1, 2N-1).

    This function is analogous to radial_2d_transform but handles complex data.

    Args:
        `r_array` (tensor.complex): Input radial tensor of shape (batch_shape, N).

    Returns:
        `tensor.complex`: 2D-converted data via tensor of shape (batch_shape, 2N-1, 2N-1).
    """
    torch_zero = torch.tensor(0.0, dtype=r_array.dtype)
    radial_trans = radial_2d_transform(torch.abs(r_array))
    radial_phase = radial_2d_transform_wrapped_phase(torch.angle(r_array))

    return torch.complex(radial_trans, torch_zero) * torch.exp(torch.complex(torch_zero, radial_phase))


def interp_regular_1d_grid(x_min, x_max, y_ref, xi):
    """Pytorch related port of tfp.math.interp_regular_1d_grid. Given a batch of input 1D data of (..., N) y_ref values
    defined on the regurlar grid of points x_ref, returns reinterpolated values at the new set of 1D coordinates specified by xi.

    Args:
        x_min,max(float): Minimum and maximum bound of the uniform x-grid coordinates.
        y_ref (float): Input data tensor of shape (..., Nx).
        xi (float): List of interpolation points (may be non-uniformly spaced).
    """
    # Cast to torch tensors if not passed in as a tensor
    if not torch.is_tensor(y_ref):
        y_ref = torch.tensor(y_ref)
    if not torch.is_tensor(xi):
        xi = torch.tensor(xi, dtype=y_ref.dtype)

    # Enforce data shape and then reshape for suitable processing
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
    yq = torch.zeros(*xi.shape, dtype=y_ref.dtype)
    grid = torch.transpose(torch.cat((xi[None], yq[None]), dim=0), 0, 1)[None, None].contiguous()
    grid = grid.expand([flat_batch_dim, -1, -1, -1])

    # Interpolate by grid_sample
    yi = F.grid_sample(y_ref, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    yi = yi.view(*batch_shape, -1)

    return yi


def helper_spline_complex(r_ref_min, r_ref_max, r, fr):
    # Get the real signal components
    f_transform_abs = interp_regular_1d_grid(r_ref_min, r_ref_max, torch.abs(fr), r)
    f_transform_real = interp_regular_1d_grid(r_ref_min, r_ref_max, torch.cos(torch.angle(fr)), r)
    f_transform_imag = interp_regular_1d_grid(r_ref_min, r_ref_max, torch.sin(torch.angle(fr)), r)

    torch_zero = torch.tensor(0.0, dtype=f_transform_abs.dtype)
    return torch.complex(f_transform_abs, torch_zero) * torch.exp(torch.complex(torch_zero, torch.atan2(f_transform_imag, f_transform_real)))


def helper_spline_real(r_ref_min, r_ref_max, r, fr):
    f_transform = interp_regular_1d_grid(r_ref_min, r_ref_max, fr, r)

    return f_transform


def tf_generalSpline_regular1DGrid(r_ref, r, fr):
    """Computes the 1D interpolation of a complex tensor on a regular grid.

    Args:
        `r_ref` (float): tensor specifying the reference grid coordinates.
        `r` (float): r values of the interpolated grid
        `fr` (float): reference data corresponding to r_ref to interpolate over, of size (..., Nx). Interpolation
            is taken over the inner-most dimension, similar to the scipy.interp1d function.

    Returns:
        `tensor.complex`: New complex output interpolated on the regular grid r
    """
    # IMPORTANT NOTE: Using this implementation to snap back to a uniform grid after the qdht calls is is not correct because the qdht returns non-uniform grids
    # However, we will use this to reduce cost and since we find the errors to have negligible effect thus far
    # As of now, there is no auto-differentiable pytorch or tensorflow implementation to take non-uniform grids and interpolate a new non-uniform grid

    # Cast to tensor if not passed in as a torch tensor
    if not torch.is_tensor(r_ref):
        r_ref = torch.tensor(r_ref)
    if not torch.is_tensor(r):
        r = torch.tensor(r)
    if not torch.is_tensor(fr):
        fr = torch.tensor(fr)

    dtype = fr.dtype
    if dtype == torch.complex64 or dtype == torch.complex128:
        interpFr = helper_spline_complex(torch.min(r_ref), torch.max(r_ref), r, fr)
    else:
        interpFr = helper_spline_real(torch.min(r_ref), torch.max(r_ref), r, fr)

    return interpFr
