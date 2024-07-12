import numpy as np
import torch
import torch.autograd as autograd


def diag_batched(
    diagonal, k=0, num_rows=-1, num_cols=-1, padding_value=0, align="RIGHT_LEFT"
):
    batch_shape = diagonal.shape[:-1]
    diag_len = diagonal.shape[-1]

    if num_rows == -1:
        num_rows = diag_len + max(k, 0)
    if num_cols == -1:
        num_cols = diag_len - min(k, 0)

    result_shape = batch_shape + (num_rows, num_cols)
    result = torch.full(
        result_shape, padding_value, dtype=diagonal.dtype, device=diagonal.device
    )

    if k >= 0:
        diag_idx = torch.arange(
            min(diag_len, num_rows, num_cols - k), device=diagonal.device
        )
        if align == "RIGHT_LEFT":
            result[..., diag_idx, diag_idx + k] = diagonal[..., : len(diag_idx)]
        else:
            result[..., -len(diag_idx) :, k : k + len(diag_idx)] = diagonal[
                ..., : len(diag_idx)
            ]
    else:
        diag_idx = torch.arange(
            min(diag_len, num_rows + k, num_cols), device=diagonal.device
        )
        if align == "RIGHT_LEFT":
            result[..., diag_idx - k, diag_idx] = diagonal[..., : len(diag_idx)]
        else:
            result[..., -len(diag_idx) - k :, : len(diag_idx)] = diagonal[
                ..., : len(diag_idx)
            ]

    return result


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
        grad_A = torch.matmul(torch.linalg.pinv(torch.adjoint(U)), grad_A)
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
