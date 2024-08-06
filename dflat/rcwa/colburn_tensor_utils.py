import numpy as np
import torch
import torch.autograd as autograd


def diag_batched(
    diagonal, k=(0,), num_rows=None, num_cols=None, padding_value=0, align="RIGHT_LEFT"
):
    if not torch.is_tensor(diagonal):
        diagonal = torch.tensor(diagonal)

    if isinstance(k, int):
        k = (k,)

    ddim = diagonal.dim()
    if ddim == 1:
        diagonal = diagonal[None]

    # infer the dimensions and potential return square matrix enlarged
    M = diagonal.shape[-1]
    if num_rows is None and num_cols is None:
        max_diag_len = M + max(abs(min(k)), abs(max(k)))
        num_rows = num_cols = max_diag_len
    if num_rows is None:
        num_rows = max(M + max(k), 1)
    if num_cols is None:
        num_cols = max(M - min(k), 1)

    # hold output
    num_diagonals = len(k)
    if num_diagonals == 1:
        batch_size = diagonal.shape[:-1]
    else:
        batch_size = diagonal.shape[:-2]

    output_shape = (*batch_size, num_rows, num_cols)
    output = torch.full(
        output_shape, padding_value, dtype=diagonal.dtype, device=diagonal.device
    )

    if diagonal.dim() > 2:
        padby = (0, max(0, num_cols - M), 0, max(0, num_rows - M))
        diagonal = torch.nn.functional.pad(diagonal, padby)
    else:
        padby = (0, max(0, num_rows - M))
        diagonal = torch.nn.functional.pad(diagonal, padby)

    ####
    ## Simplify the bottom if possible ##
    for i, d in enumerate(k):
        diag_len = min(num_cols - max(d, 0), num_rows + min(d, 0))

        if align in {"RIGHT_LEFT", "RIGHT_RIGHT"}:
            offset = max(0, M - diag_len)
        else:
            offset = 0

        if num_diagonals == 1:
            diag_values = diagonal[..., offset : offset + diag_len]
        else:
            diag_values = diagonal[..., i, offset : offset + diag_len]

        indices = torch.arange(diag_len)
        if d >= 0:
            output[..., indices, indices + d] = diag_values[..., :diag_len]
        else:
            row_indices = indices - d
            valid_mask = row_indices < num_rows
            output[..., row_indices[valid_mask], indices[valid_mask]] = diag_values[
                ..., : len(indices[valid_mask])
            ]

    return output


def expand_and_tile_np(array, batchSize, pixelsX, pixelsY):
    """
    Expands and tile a numpy array for a given batchSize and number of pixels.
    Args:
        array: A `np.ndarray` of shape `(Nx, Ny)`.
    Returns:
        A `np.ndarray` of shape `(batchSize, pixelsX, pixelsY, Nx, Ny)` with
        the values from `array` tiled over the new dimensions.
    """
    array = array[np.newaxis, np.newaxis, np.newaxis, :, :]
    return np.tile(array, reps=(batchSize, pixelsX, pixelsY, 1, 1))


def expand_and_tile_tf(tensor, batchSize, pixelsX, pixelsY):
    """
    Expands and tile a `tf.Tensor` for a given batchSize and number of pixels.
    Args:
        tensor: A `tf.Tensor` of shape `(Nx, Ny)`.
    Returns:
        A `tf.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nx, Ny)` with
        the values from `tensor` tiled over the new dimensions.
    """
    tensor = tensor[None, None, None, :, :]
    return torch.tile(tensor, (batchSize, pixelsX, pixelsY, 1, 1))


class EigGeneralFunction(autograd.Function):
    @staticmethod
    def forward(ctx, A, eps=1e-6):
        """
        Computes the eigendecomposition of a batch of matrices, and provides the reverse mode gradient of the
        eigendecomposition for general, complex matrices that do not have to be self-adjoint.
        """
        # Perform the eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eig(A)

        # Save tensors for backward pass
        ctx.save_for_backward(A, eigenvalues, eigenvectors)
        ctx.eps = eps
        return eigenvalues, eigenvectors

    @staticmethod
    def backward(ctx, grad_D, grad_U):
        A, eigenvalues, eigenvectors = ctx.saved_tensors
        eps = ctx.eps

        D = eigenvalues
        U = eigenvectors

        # Convert eigenvalues gradient to a diagonal matrix
        grad_D = torch.diag_embed(grad_D)

        # Extract the tensor dimensions for later use
        batchSize, pixelsX, pixelsY, Nlay, dim, _ = A.shape

        # Calculate intermediate matrices
        I = torch.eye(dim, dtype=torch.complex64, device=A.device)
        D = D.view(batchSize, pixelsX, pixelsY, Nlay, dim, 1)
        shape_di = (batchSize, pixelsX, pixelsY, Nlay, dim, 1)
        shape_dj = (batchSize, pixelsX, pixelsY, Nlay, 1, dim)
        E = torch.ones(
            shape_di, dtype=torch.complex64, device=A.device
        ) * torch.adjoint(D)
        E = E - D * torch.ones(shape_dj, dtype=torch.complex64, device=A.device)
        E = torch.adjoint(D) - D

        # Lorentzian broadening
        F = E / (E**2 + eps)
        F = F - I * F

        # Compute the reverse mode gradient of the eigendecomposition of A
        grad_A = F.conj() * torch.matmul(torch.adjoint(U), grad_U)
        grad_A = grad_D + grad_A
        grad_A = torch.matmul(grad_A, torch.adjoint(U))
        grad_A = torch.matmul(torch.linalg.inv(torch.adjoint(U)), grad_A)
        return grad_A, None


def eig_general(A, eps=1e-6):
    """
    Computes the eigendecomposition of a batch of matrices, the same as
    `tf.eig()` but assumes the input shape also has extra dimensions for pixels
    and layers. This function also provides the reverse mode gradient of the
    eigendecomposition as derived in 10.1109/ICASSP.2017.7952140. This applies
    for general, complex matrices that do not have to be self adjoint. This
    result gives the exact reverse mode gradient for nondegenerate eigenvalue
    problems. To extend to the case of degenerate eigenvalues common in RCWA, we
    approximate the gradient by a Lorentzian broadening technique that
    introduces a small error but stabilizes the calculation. This is based on
    10.1103/PhysRevX.9.031041.
    Args:
        A: A `tf.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayers, Nx,
        Ny)` and dtype `tf.complex64` where the last two dimensions define
        matrices for which we will calculate the eigendecomposition of their
        reverse mode gradients.

        eps: A `float` defining a regularization parameter used in the
        denominator of the Lorentzian broadening calculation to enable reverse
        mode gradients for degenerate eigenvalues.

    Returns:
        A `Tuple(List[tf.Tensor, tf.Tensor], tf.Tensor)`, where the `List`
        specifies the eigendecomposition as computed by `tf.eig()` and the
        second element of the `Tuple` gives the reverse mode gradient of the
        eigendecompostion of the input argument `A`.
    """
    return EigGeneralFunction.apply(A, eps)


# if __name__ == "__main__":
#     # thetas = [1.0 for i in range(5)]
#     # theta = torch.tensor(thetas, dtype=torch.float32)
#     # theta = theta[:, None, None, None, None, None]
#     # pixelsX, pixelsY = 1, 1
#     # theta = torch.tile(theta, dims=(1, pixelsX, pixelsY, 1, 5, 1))
#     # kx_T = torch.permute(theta, [0, 1, 2, 3, 5, 4])
#     # KX = diag_batched(kx_T)

#     # # Test 1: Main diagonal; k=(0,) diagonal- 2x3
#     diagonal = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
#     output = diag_batched(diagonal).cpu().numpy()
#     print(output, output.shape, "\n")

#     # # Test 2: Superdiagonal; k=(1,), diagonal- 2x3
#     diagonal = np.array([[1, 2, 3], [4, 5, 6]])
#     output = diag_batched(diagonal, k=1).cpu().numpy()
#     print(output, "\n")

#     # Test 3: Tridiagonal band; k = (-1, 1), diagonal- 2x3x3
#     diagonals = np.array(
#         [[[7, 8, 9], [4, 5, 6], [1, 2, 3]], [[16, 17, 18], [13, 14, 15], [10, 11, 12]]]
#     )
#     output = diag_batched(diagonals, k=(-1, 0, 1)).cpu().numpy()
#     print(output, "\n")

#     # test rectangular matrix
#     diagonals = np.array([[1, 2]])
#     output = diag_batched(diagonals, k=-1)
#     print(output)
