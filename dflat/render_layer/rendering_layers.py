import torch
import torch.nn as nn

from .core.assert_layer import check_inputs
from .core.ops_fft_convolve import general_convolve
from .core.ops_measurement import photons_to_ADU, layer_output_to_rgb


class Fronto_Planar_renderer_incoherent(nn.Module):
    def __init__(self, sensor_parameters, wavelength_set_m, device, PSNR=None):
        super(Fronto_Planar_renderer_incoherent, self).__init__()
        self.sensor_parameters = sensor_parameters
        self._wavelength_set_m = wavelength_set_m.cpu().numpy() if torch.is_tensor(wavelength_set_m) else wavelength_set_m
        self.device = device
        self._check_inputs = True

        # If SNR is specified, then define a rescale factor for the input data
        self._rescale_signal = False
        if PSNR is not None:
            if isinstance(PSNR, float) or isinstance(PSNR, int):
                self._peak_photons = sensor_parameters.SNR_to_meanPhotons(PSNR)
                self._rescale_signal = True
            else:
                raise ValueError("PSNR must be either None or a float/int")

    def __call__(self, psf_intensity, AIF_image, rfft=False, return_stack=True, return_collapse=False, return_rgb=False):
        return self.forward(psf_intensity, AIF_image, rfft, return_stack, return_collapse, return_rgb)

    def forward(self, psf_intensity, AIF_image, rfft=True, return_stack=True, return_collapse=True, return_rgb=True):
        """Renders image with appropriate PSF blurr for a fronto-planar scene components.

        Args:
            'psf_intensity' (torch.float): PSF intensity of shape [Num_wl (optional dimension), num_profile, num_point_source, sensor_pix_y, sensor_pix_x]
            'AIF_image' (torch.float): All in focus radiances (pinhole image), of shape, [BatchSize, num_point_sources, Height, Width, Num_wl]
            rfft (bool, optional): Flag to use rfft2 or fft2 for the Fourier Transforms. Defaults to True.
            return_stack (bool, optional): Flag to return the convolved hyperspectral stack. Defaults to True.
            return_collapse (bool, optional): Flag to return the spectrally integrated image. Defaults to True.
            return_rgb (bool, optional): Flag to return the demosaiced RGB Image. Defaults to True.

        Returns:
            list: list of returned tensors [HSI if return_stack==True, Monochrome if return_collapse==True, RGB if return_rgb==True]. Output tensors have the shape
            [BatchSize, num_profile, num_point_sources, Height, Width, channels] where channels is num_wl for the stack, 3 for RGB, or 1 for monochrome
        """

        if self._check_inputs:
            [psf_intensity, AIF_image], self._check_inputs = check_inputs([psf_intensity, AIF_image], self.device)

        # Allow convenient input size. If rank 4, increase dimension to rank 5
        init_rank = psf_intensity.dim()
        if init_rank == 4:
            psf_intensity = psf_intensity.unsqueeze(0)
            AIF_image = AIF_image.unsqueeze(0)

        # If PSNR is specified then apply rescaling before render
        if self._rescale_signal:
            AIF_image = AIF_image / torch.amax(AIF_image, dim=(-3, -2, -1), keepdim=True) * self._peak_photons

        ### Compute the convolved image and Convert to digital, noisy measurement
        # For the general convolve function we need to move wavelength channels to match the psf shape (last two dimensions reservered for spatial)
        psf_intensity = psf_intensity.unsqueeze(0)  # Add image batch dimension
        AIF_image = torch.permute(AIF_image, dims=(0, 4, 1, 2, 3)).contiguous().unsqueeze(2)
        meas = general_convolve(AIF_image, psf_intensity, rfft=rfft)
        meas = photons_to_ADU(meas, self.sensor_parameters)

        # For convenience with post processing, lets make the spectral related channels the last dimension
        meas = torch.permute(meas, (0, 2, 3, 4, 5, 1)).contiguous()

        # Return with or without collapse
        output = []
        if return_stack:
            output.append(meas)
        if return_collapse:
            output.append(torch.sum(meas, dim=-1, keepdim=True))
        if return_rgb:
            output.append(layer_output_to_rgb(meas, self._wavelength_set_m, demosaic=True, gamma=True))

        return output
