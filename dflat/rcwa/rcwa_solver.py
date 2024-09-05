import torch
import torch.nn as nn
import numpy as np
import warnings

from dflat.rcwa.material_utils import MATERIAL_DICT, get_material_index
from dflat.rcwa.colburn_rcwa_utils import *
import dflat.rcwa.colburn_tensor_utils as tensor_utils


class RCWA_Solver(nn.Module):
    def __init__(
        self,
        wavelength_set_m,
        thetas,
        phis,
        pte,
        ptm,
        pixelsX,
        pixelsY,
        PQ,
        lux,
        luy,
        layer_heights,
        layer_embed_mats,
        material_dielectric,
        Nx,
        Ny,
        er1="Vacuum",
        er2="Vacuum",
        ur1=1.0,
        ur2=1.0,
        urd=1.0,
        urs=1.0,
        rcond=1e-15,
    ):
        """RCWA Solver Class. Computes output field given layered structures.

        Args:
            wavelength_set_m (list): Wavelengths to simulate in batch.
            thetas (list): Polar angles (degrees) for incident light in batch.
            phis (list): Azimuth angles (degrees) for incident light in batch.
            pte (list): TE polarization component magnitudes for incident light in batch.
            ptm (list): TM polarization component magnitudes for incident light in batch.
            pixelsX (int): Number of simulation unit cells in the x-direction.
            pixelsY (int): Number of simulation unit cells in the y-direction.
            PQ (list): Length 2 list specifying number of Fourier modes along X and Y-direction.
            lux (float): Physical width (meters) of a unit cell along X.
            luy (float): Physical width (meters) of a unit cell along Y.
            layer_heights (list): Thickness (in meters) of each layer along Z for the unit cells.
            layer_embed_mats (list): The embedding medium in each layer in layer_heights. Material specifier may be the material name as a string or the relative electric permittivity as a complex float.
            material_dielectric (str): Dielectric material to be used within all embedding mediums. Material specifier may be the material name as a string or the relative electric permittivity as a complex float.
            material name or the relative electric permittivity as a complex float.
            Nx (int): Number of grid points along the x-direction when discretizing unit cells.
            Ny (int): Number of grid points along the x-direction when discretizing unit cells.
            er1 (str/float/complex): Electric permittivity in the reflected region of space; Defaults to 1.0.
            er2 (str/float/complex): Electric permittivity in the transmitted region of space; Defaults to 1.0.
            ur1 (float): Magnetic permeability in the reflected region of space; Defaults to 1.0
            ur2 (float: Magnetic permeability in the transmitted region of space; Defaults to 1.0
            urd (float): Magnetic permeability of the dielectric structures; Defaults to 1.0
            urs (float): Magnetic permeability of the substrate; Defaults to 1.0
        """
        super().__init__()
        self.dtype = torch.float32
        self.cdtype = torch.complex64
        self.input_dict = {
            key: value for key, value in locals().items() if key != "self"
        }
        self.__check_material_entry()
        self.__initialize_tensors()
        self.__precompute_constants()
        self.rcond = rcond

        ref_field = self.forward(
            torch.zeros((self.Nlayers, self.Nx, self.Ny)), ref_field=False
        )
        self.ref_field = nn.Parameter(ref_field, requires_grad=False)

    def forward(self, binary, ref_field=True):
        """Computes a zero-order transmitted field given a binary pattern for each layer.

        Args:
            binary (float): Binary pattern for each layer of shape [Layer, H, W]. The binary will be multiplied by the initialzed eps_d (epsilon for the structures) and the embedding medium will be the initialized layer_eps.
            ref_field (bool, optional): If True, returns the output field normalized relative to the incident field. This is required to get accurate results! Defaults to True.

        Returns:
            complex: Output zero-order complex field transmitted through of shape [Polarization=2 (x,y), Lambda, PixelsX, PixelsY]
        """
        # Assemble the cell from the binary
        torch_zero = self.TORCH_ZERO

        binary = torch.complex(binary, torch_zero)
        urd = self.urd
        layer_eps = self.layer_eps
        eps_d = self.eps_d

        UR = self.urd * self.TORCH_BATCHED_ONES
        ER = [
            layer_eps[i] + (eps_d - layer_eps[i]) * binary[i][None, None, None, None]
            for i in range(layer_eps.shape[0])
        ]
        ER = torch.cat(ER, dim=3)

        outputs = self.simulate(ER, UR)
        PQ_zero = np.prod(self.PQ) // 2
        t = torch.stack([outputs["tx"], outputs["ty"]])[:, :, :, :, PQ_zero, 0]

        if ref_field:
            t_ref = self.ref_field
            tt = torch.abs(t) / torch.abs(t_ref)
            tphi = torch.angle(t) - torch.angle(t_ref)
            t = torch.complex(tt, torch_zero) * torch.exp(
                -1 * torch.complex(torch_zero, tphi)
            )

        return t

    def simulate(self, ER, UR):
        """
        Calculates the transmission/reflection coefficients for a unit cell with a
        given permittivity/permeability distribution and the batch of input conditions
        (e.g., wavelengths, wavevectors, polarizations) for a fixed real space grid
        and number of Fourier harmonics.

        Args:
            ER: A `torch.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
            and dtype `torch.complex64` specifying the relative permittivity distribution
            of the unit cell.

            UR: A `torch.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
            and dtype `torch.complex64` specifying the relative permeability distribution
            of the unit cell.
        Returns:
            outputs: A `dict` containing the keys {'rx', 'ry', 'rz', 'R', 'ref',
            'tx', 'ty', 'tz', 'T', 'TRN'} corresponding to the computed reflection/tranmission
            coefficients and powers. tx has shape [lambda, pixelsX, pixelsY, PQ, 1]
        """
        # # Get the precompute tensors
        # k0 = self.k0
        # kinc_z0 = self.kinc_z0
        # KZref = self.KZref
        # KZtrn = self.KZtrn
        # W0 = self.W0
        # V0 = self.V0
        # ur2_red = self.ur2_red
        # ur1_red = self.ur1_red
        # csrc = self.csrc
        # W_ref = self.W_ref
        # W_trn = self.W_trn
        # KZref_inv = self.KZref_inv
        # KZtrn_inv = self.KZtrn_inv
        # pixelsX = self.pixelsX
        # pixelsY = self.pixelsY
        # layer_heights = self.layer_heights

        ### Step 3: Build convolution matrices for the permittivity and permeability ###
        PQ = self.PQ
        ERC = convmat(ER, PQ[0], PQ[1])
        URC = convmat(UR, PQ[0], PQ[1])

        ### Step 7: Calculate eigenmodes ###
        KX = self.KX
        KY = self.KY

        # Build the eigenvalue problem.
        bri_erc = thl.inv(ERC)
        P_00 = torch.matmul(KX, bri_erc)
        P_00 = torch.matmul(P_00, KY)
        P_01 = torch.matmul(KX, bri_erc)
        P_01 = torch.matmul(P_01, KX)
        P_01 = URC - P_01
        P_10 = torch.matmul(KY, bri_erc)
        P_10 = torch.matmul(P_10, KY) - URC
        P_11 = torch.matmul(-KY, bri_erc)
        P_11 = torch.matmul(P_11, KX)
        P_row0 = torch.cat([P_00, P_01], dim=5)
        P_row1 = torch.cat([P_10, P_11], dim=5)
        P = torch.cat([P_row0, P_row1], dim=4)

        bri_urc = thl.inv(URC)
        Q_00 = torch.matmul(KX, bri_urc)
        Q_00 = torch.matmul(Q_00, KY)
        Q_01 = torch.matmul(KX, bri_urc)
        Q_01 = torch.matmul(Q_01, KX)
        Q_01 = ERC - Q_01
        Q_10 = torch.matmul(KY, bri_urc)
        Q_10 = torch.matmul(Q_10, KY) - ERC
        Q_11 = torch.matmul(-KY, bri_urc)
        Q_11 = torch.matmul(Q_11, KX)
        Q_row0 = torch.cat([Q_00, Q_01], dim=5)
        Q_row1 = torch.cat([Q_10, Q_11], dim=5)
        Q = torch.cat([Q_row0, Q_row1], dim=4)

        # Compute eignmodes for the layers in each pixel for the whole batch.
        OMEGA_SQ = torch.matmul(P, Q)
        LAM, W = tensor_utils.eig_general(OMEGA_SQ)
        LAM = torch.sqrt(LAM)
        LAM = tensor_utils.diag_batched(LAM)
        V = torch.matmul(Q, W)
        V = torch.matmul(V, thl.inv(LAM))

        # Scattering matrices for the layers in each pixel for the whole batch.
        W0 = self.W0
        V0 = self.V0
        k0 = self.k0
        layer_heights = self.layer_heights
        W_inv = thl.inv(W)
        V_inv = thl.inv(V)
        A = torch.matmul(W_inv, W0) + torch.matmul(V_inv, V0)
        B = torch.matmul(W_inv, W0) - torch.matmul(V_inv, V0)
        X = torch.matrix_exp(-LAM * k0 * layer_heights)

        S = dict({})
        A_inv = thl.inv(A)
        S11_left = torch.matmul(X, B)
        S11_left = torch.matmul(S11_left, A_inv)
        S11_left = torch.matmul(S11_left, X)
        S11_left = torch.matmul(S11_left, B)
        S11_left = A - S11_left
        S11_left = thl.inv(S11_left)
        S11_right = torch.matmul(X, B)
        S11_right = torch.matmul(S11_right, A_inv)
        S11_right = torch.matmul(S11_right, X)
        S11_right = torch.matmul(S11_right, A)
        S11_right = S11_right - B
        S["S11"] = torch.matmul(S11_left, S11_right)
        S12_right = torch.matmul(B, A_inv)
        S12_right = torch.matmul(S12_right, B)
        S12_right = A - S12_right
        S12_left = torch.matmul(S11_left, X)
        S["S12"] = torch.matmul(S12_left, S12_right)
        S["S21"] = S["S12"]
        S["S22"] = S["S11"]

        # Update the global scattering matrices.
        SG = {
            "S11": self.SG_S11,
            "S12": self.SG_S12,
            "S21": self.SG_S21,
            "S22": self.SG_S22,
        }
        for l in range(self.Nlayers):
            S_layer = dict({})
            S_layer["S11"] = S["S11"][:, :, :, l, :, :]
            S_layer["S12"] = S["S12"][:, :, :, l, :, :]
            S_layer["S21"] = S["S21"][:, :, :, l, :, :]
            S_layer["S22"] = S["S22"][:, :, :, l, :, :]
            SG = redheffer_star_product(SG, S_layer)

        ### Step 8: Reflection side ###
        ### Step 10: Compute global scattering matrix ###
        SR = {
            "S11": self.SR_S11,
            "S12": self.SR_S12,
            "S21": self.SR_S21,
            "S22": self.SR_S22,
        }
        ST = {
            "S11": self.ST_S11,
            "S12": self.ST_S12,
            "S21": self.ST_S21,
            "S22": self.ST_S22,
        }
        SG = redheffer_star_product(SR, SG)
        SG = redheffer_star_product(SG, ST)

        ### Step 12: Compute reflected and transmitted fields ###
        # # Compute tranmission and reflection mode coefficients.
        csrc = self.csrc
        W_ref = self.W_ref
        W_trn = self.W_trn
        cref = torch.matmul(SG["S11"], csrc)
        ctrn = torch.matmul(SG["S21"], csrc)
        eref = torch.matmul(W_ref, cref)
        etrn = torch.matmul(W_trn, ctrn)

        rx = eref[:, :, :, 0 : np.prod(PQ), :]
        ry = eref[:, :, :, np.prod(PQ) : 2 * np.prod(PQ), :]
        tx = etrn[:, :, :, 0 : np.prod(PQ), :]
        ty = etrn[:, :, :, np.prod(PQ) : 2 * np.prod(PQ), :]

        # Compute longitudinal components.
        KZref_inv = self.KZref_inv
        KZtrn_inv = self.KZtrn_inv

        rz = torch.matmul(KX[:, :, :, 0, :, :], rx) + torch.matmul(
            KY[:, :, :, 0, :, :], ry
        )
        rz = torch.matmul(-KZref_inv, rz)
        tz = torch.matmul(KX[:, :, :, 0, :, :], tx) + torch.matmul(
            KY[:, :, :, 0, :, :], ty
        )
        tz = torch.matmul(-KZtrn_inv, tz)

        ### Step 13: Compute diffraction efficiences ###
        rx2 = torch.real(rx) ** 2 + torch.imag(rx) ** 2
        ry2 = torch.real(ry) ** 2 + torch.imag(ry) ** 2
        rz2 = torch.real(rz) ** 2 + torch.imag(rz) ** 2
        R2 = rx2 + ry2 + rz2

        ur1_red = self.ur1_red
        ur2_red = self.ur2_red
        KZref = self.KZref
        kinc_z0 = self.kinc_z0
        pixelsX = self.pixelsX
        pixelsY = self.pixelsY
        KZtrn = self.KZtrn

        R = torch.real(-1 * KZref[:, :, :, 0, :, :] / ur1_red) / torch.real(
            kinc_z0 / ur1_red
        )
        R = torch.matmul(R, R2)
        R = torch.reshape(R, (self.batch_size, pixelsX, pixelsY, PQ[0], PQ[1]))
        REF = torch.sum(R, dim=[3, 4])

        tx2 = torch.real(tx) ** 2 + torch.imag(tx) ** 2
        ty2 = torch.real(ty) ** 2 + torch.imag(ty) ** 2
        tz2 = torch.real(tz) ** 2 + torch.imag(tz) ** 2
        T2 = tx2 + ty2 + tz2
        T = torch.real(KZtrn[:, :, :, 0, :, :] / ur2_red) / torch.real(
            kinc_z0 / ur2_red
        )
        T = torch.matmul(T, T2)
        T = torch.reshape(T, (self.batch_size, pixelsX, pixelsY, PQ[0], PQ[1]))
        TRN = torch.sum(T, dim=[3, 4])

        # Store the transmission/reflection coefficients and powers in a dictionary.
        outputs = dict({})
        outputs["rx"] = rx
        outputs["ry"] = ry
        outputs["rz"] = rz
        outputs["R"] = R
        outputs["REF"] = REF
        outputs["tx"] = tx
        outputs["ty"] = ty
        outputs["tz"] = tz
        outputs["T"] = T
        outputs["TRN"] = TRN

        return outputs

    def __check_material_entry(self):
        layer_embed_mats = self.input_dict["layer_embed_mats"]
        layer_heights = self.input_dict["layer_heights"]
        material_dielectric = self.input_dict["material_dielectric"]
        er1 = self.input_dict["er1"]
        er2 = self.input_dict["er2"]
        valid_mats = MATERIAL_DICT.keys()

        assert isinstance(layer_embed_mats, list), "layer_embed_mats must be a list."
        assert isinstance(layer_heights, list), "layer_heights must be a list."
        assert len(layer_embed_mats) == len(
            layer_heights
        ), "layer_heights must be same length as layer_embed_mats."

        for layer_mat in layer_embed_mats:
            self.__check_eps(layer_mat)
        self.__check_eps(material_dielectric)
        self.__check_eps(er1)
        self.__check_eps(er2)

        return

    def __initialize_tensors(self):
        # Take the inputs originally passed in and create new class attributes for the rcwa calculation (updated shapes and converted to tensors)
        # General Shapes for batched calculation: lam, pixelx, pixely, Nlayer, Nx, Ny
        wavelength_set_m = self.input_dict["wavelength_set_m"]
        layer_embed_mats = self.input_dict["layer_embed_mats"]
        layer_heights = self.input_dict["layer_heights"]
        material_dielectric = self.input_dict["material_dielectric"]
        er1 = self.input_dict["er1"]
        er2 = self.input_dict["er2"]
        ur1 = self.input_dict["ur1"]
        ur2 = self.input_dict["ur2"]
        urd = self.input_dict["urd"]
        urs = self.input_dict["urs"]
        pixelsX = self.input_dict["pixelsX"]
        pixelsY = self.input_dict["pixelsY"]
        thetas = self.input_dict["thetas"]
        phis = self.input_dict["phis"]
        pte = self.input_dict["pte"]
        ptm = self.input_dict["ptm"]
        PQ = self.input_dict["PQ"]
        Nx = self.input_dict["Nx"]
        Ny = self.input_dict["Ny"]
        Ly = self.input_dict["lux"]
        Lx = self.input_dict["lux"]

        # Gather the layer embedding permittivities
        layer_eps_list = [
            self.__get_eps(layer_mat, wavelength_set_m)[:, None, None, None, None, None]
            for layer_mat in layer_embed_mats
        ]
        self.layer_eps = torch.stack(layer_eps_list)

        # Gather the dielectric permittivity
        eps_d = self.__get_eps(material_dielectric, wavelength_set_m)
        self.eps_d = eps_d[:, None, None, None, None, None]

        # Get the permittivity for reflection and transmission media
        self.er1 = self.__get_eps(er1, wavelength_set_m)[
            :, None, None, None, None, None
        ]
        self.er2 = self.__get_eps(er2, wavelength_set_m)[
            :, None, None, None, None, None
        ]

        # Get the permeability for the reflection and transmission media
        assert not isinstance(
            ur1, str
        ), "Magnetic permeability cannot be given as string atm."
        assert not isinstance(
            ur2, str
        ), "Magnetic permeability cannot be given as string atm."
        assert not isinstance(
            urd, str
        ), "Magnetic permeability cannot be given as string atm."
        assert not isinstance(
            urs, str
        ), "Magnetic permeability cannot be given as string atm."
        self.ur1 = self.__get_eps(ur1, wavelength_set_m)[
            :, None, None, None, None, None
        ]
        self.ur2 = self.__get_eps(ur2, wavelength_set_m)[
            :, None, None, None, None, None
        ]
        self.urd = self.__get_eps(urd, wavelength_set_m)[
            :, None, None, None, None, None
        ]
        self.urs = self.__get_eps(urs, wavelength_set_m)[
            :, None, None, None, None, None
        ]

        # Define incident light wavelengths
        lam0 = torch.tensor(wavelength_set_m, dtype=self.dtype)
        lam0 = lam0[:, None, None, None, None, None]
        lam0 = torch.tile(lam0, dims=(1, pixelsX, pixelsY, 1, 1, 1))
        self.lam0 = lam0

        # Define incident light angles
        theta = torch.tensor(thetas, dtype=self.dtype)
        theta = theta[:, None, None, None, None, None]
        theta = torch.tile(theta, dims=(1, pixelsX, pixelsY, 1, 1, 1))
        self.theta = theta

        phi = torch.tensor(phis, dtype=self.dtype)
        phi = phi[:, None, None, None, None, None]
        phi = torch.tile(phi, dims=(1, pixelsX, pixelsY, 1, 1, 1))
        self.phi = phi

        pte = torch.tensor(pte, dtype=self.cdtype)
        pte = pte[:, None, None, None]
        pte = torch.tile(pte, dims=(1, pixelsX, pixelsY, 1))
        self.pte = pte

        ptm = torch.tensor(ptm, dtype=self.cdtype)
        ptm = ptm[:, None, None, None]
        ptm = torch.tile(ptm, dims=(1, pixelsX, pixelsY, 1))
        self.ptm = ptm

        layer_heights = torch.tensor(layer_heights, dtype=self.cdtype)
        layer_heights = layer_heights[None, None, None, :, None, None]
        self.layer_heights = layer_heights

        # Add useful constants
        self.batch_size = len(wavelength_set_m)
        self.Nlayers = len(layer_embed_mats)
        self.pixelsX = pixelsX
        self.pixelsY = pixelsY
        self.PQ = PQ
        self.Nx = Nx
        self.Lx = Lx
        self.Ly = Ly
        self.Ny = Ny  # int(np.round(Nx * Ly / Ly))
        self.materials_shape = (self.batch_size, pixelsX, pixelsY, self.Nlayers, Nx, Ny)

        return

    def __precompute_constants(self):
        # Most of the tensors in the RCWA calculation can be precomputed once
        PQ = self.PQ
        batch_size = self.batch_size
        pixelsX = self.pixelsX
        pixelsY = self.pixelsY
        Nlayers = self.Nlayers
        er1 = self.er1
        er2 = self.er2
        theta = self.theta
        phi = self.phi
        dtype = self.dtype
        cdtype = self.cdtype
        Lx = self.Lx
        Ly = self.Ly
        ur1 = self.ur1
        ur2 = self.ur2
        pte = self.pte
        ptm = self.ptm
        self.TORCH_ZERO = nn.Parameter(
            torch.torch.tensor(0.0, dtype=self.dtype), requires_grad=False
        )
        self.TORCH_BATCHED_ONES = nn.Parameter(
            torch.ones(self.materials_shape, dtype=self.cdtype)
        )
        self.urd = nn.Parameter(self.urd, requires_grad=False)
        self.eps_d = nn.Parameter(self.eps_d, requires_grad=False)
        self.layer_eps = nn.Parameter(self.layer_eps, requires_grad=False)
        self.layer_heights = nn.Parameter(self.layer_heights, requires_grad=False)

        ### Step 4: Precompute wave-vector expansion
        I = np.eye(np.prod(PQ))
        I = torch.tensor(I, dtype=self.cdtype)
        I = I[None, None, None, None, :, :]
        I = torch.tile(I, (batch_size, pixelsX, pixelsY, Nlayers, 1, 1))

        Z = np.zeros((np.prod(PQ), np.prod(PQ)))
        Z = torch.tensor(Z, dtype=self.cdtype)
        Z = Z[None, None, None, None, :, :]
        Z = torch.tile(Z, (batch_size, pixelsX, pixelsY, Nlayers, 1, 1))

        k0 = 2 * np.pi / self.lam0
        k0 = k0.to(dtype=self.cdtype)
        self.k0 = nn.Parameter(k0, requires_grad=False)

        n1 = torch.sqrt(er1)
        n2 = torch.sqrt(er2)
        kinc_x0 = n1 * torch.sin(theta) * torch.cos(phi)
        kinc_y0 = n1 * torch.sin(theta) * torch.sin(phi)
        kinc_z0 = n1 * torch.cos(theta)
        kinc_z0 = kinc_z0[:, :, :, 0, :, :]
        self.kinc_z0 = nn.Parameter(kinc_z0, requires_grad=False)

        # Build Kx and Ky matrices
        p_max = np.floor(PQ[0] / 2.0)
        q_max = np.floor(PQ[1] / 2.0)
        p = np.linspace(-p_max, p_max, PQ[0])
        p = torch.tensor(p, dtype=self.cdtype)  # indices along T1
        p = p[None, None, None, None, :, None]
        p = torch.tile(p, (1, pixelsX, pixelsY, self.Nlayers, 1, 1))

        q = np.linspace(-q_max, q_max, PQ[1])
        q = torch.tensor(q, dtype=self.cdtype)  # indices along T2
        q = q[None, None, None, None, None, :]
        q = torch.tile(q, (1, pixelsX, pixelsY, self.Nlayers, 1, 1))

        kx_zeros = torch.zeros(PQ[1], dtype=cdtype)
        kx_zeros = kx_zeros[None, None, None, None, None, :]
        ky_zeros = torch.zeros(PQ[0], dtype=cdtype)
        ky_zeros = ky_zeros[None, None, None, None, :, None]
        kx = kinc_x0 - 2 * np.pi * p / (k0 * Lx) - kx_zeros
        ky = kinc_y0 - 2 * np.pi * q / (k0 * Ly) - ky_zeros

        kx_T = torch.permute(kx, [0, 1, 2, 3, 5, 4])
        KX = torch.reshape(kx_T, (batch_size, pixelsX, pixelsY, Nlayers, np.prod(PQ)))
        KX = tensor_utils.diag_batched(KX)
        self.KX = nn.Parameter(KX, requires_grad=False)

        ky_T = torch.permute(ky, [0, 1, 2, 3, 5, 4])
        KY = torch.reshape(ky_T, (batch_size, pixelsX, pixelsY, Nlayers, np.prod(PQ)))
        KY = tensor_utils.diag_batched(KY)
        self.KY = nn.Parameter(KY, requires_grad=False)

        KZref = torch.matmul(torch.conj(ur1 * I), torch.conj(er1 * I))
        KZref = KZref - torch.matmul(KX, KX) - torch.matmul(KY, KY)
        KZref = torch.sqrt(KZref)
        KZref = -torch.conj(KZref)
        self.KZref = nn.Parameter(KZref, requires_grad=False)

        KZtrn = torch.matmul(torch.conj(ur2 * I), torch.conj(er2 * I))
        KZtrn = KZtrn - torch.matmul(KX, KX) - torch.matmul(KY, KY)
        KZtrn = torch.sqrt(KZtrn)
        KZtrn = torch.conj(KZtrn)
        self.KZtrn = nn.Parameter(KZtrn, requires_grad=False)

        ### Step 5: Free Space ###
        KZ = I - torch.matmul(KX, KX) - torch.matmul(KY, KY)
        KZ = torch.sqrt(KZ)
        KZ = torch.conj(KZ)

        Q_free_00 = torch.matmul(KX, KY)
        Q_free_01 = I - torch.matmul(KX, KX)
        Q_free_10 = torch.matmul(KY, KY) - I
        Q_free_11 = -torch.matmul(KY, KX)
        Q_free_row0 = torch.cat([Q_free_00, Q_free_01], dim=5)
        Q_free_row1 = torch.cat([Q_free_10, Q_free_11], dim=5)
        Q_free = torch.cat([Q_free_row0, Q_free_row1], dim=4)

        W0_row0 = torch.cat([I, Z], dim=5)
        W0_row1 = torch.cat([Z, I], dim=5)
        W0 = torch.cat([W0_row0, W0_row1], dim=4)
        self.W0 = nn.Parameter(W0, requires_grad=False)

        LAM_free_row0 = torch.cat([1j * KZ, Z], dim=5)
        LAM_free_row1 = torch.cat([Z, 1j * KZ], dim=5)
        LAM_free = torch.cat([LAM_free_row0, LAM_free_row1], dim=4)
        V0 = torch.matmul(Q_free, thl.inv(LAM_free))
        self.V0 = nn.Parameter(V0, requires_grad=False)

        ### Step 6: Initialize Global Scattering Matrix ###
        SG_S11 = torch.zeros((2 * np.prod(PQ), 2 * np.prod(PQ)), dtype=cdtype)
        SG_S11 = tensor_utils.expand_and_tile_tf(SG_S11, batch_size, pixelsX, pixelsY)
        SG_S12 = torch.eye(2 * np.prod(PQ), dtype=cdtype)
        SG_S12 = tensor_utils.expand_and_tile_tf(SG_S12, batch_size, pixelsX, pixelsY)
        SG_S21 = torch.eye(2 * np.prod(PQ), dtype=cdtype)
        SG_S21 = tensor_utils.expand_and_tile_tf(SG_S21, batch_size, pixelsX, pixelsY)
        SG_S22 = torch.zeros((2 * np.prod(PQ), 2 * np.prod(PQ)), dtype=cdtype)
        SG_S22 = tensor_utils.expand_and_tile_tf(SG_S22, batch_size, pixelsX, pixelsY)
        self.SG_S11 = nn.Parameter(SG_S11, requires_grad=False)
        self.SG_S12 = nn.Parameter(SG_S12, requires_grad=False)
        self.SG_S21 = nn.Parameter(SG_S21, requires_grad=False)
        self.SG_S22 = nn.Parameter(SG_S22, requires_grad=False)

        ## Eignemode calcuation in forward

        ### Step 8: Reflection side ###
        # Eliminate layer dimension for tensors as they are unchanging on this dimension.
        KX = KX[:, :, :, 0, :, :]
        KY = KY[:, :, :, 0, :, :]
        KZref = KZref[:, :, :, 0, :, :]
        KZtrn = KZtrn[:, :, :, 0, :, :]
        Z = Z[:, :, :, 0, :, :]
        I = I[:, :, :, 0, :, :]
        W0 = W0[:, :, :, 0, :, :]
        V0 = V0[:, :, :, 0, :, :]
        ur1_red = ur1[:, :, :, 0, :, :]
        ur2_red = ur2[:, :, :, 0, :, :]
        er1_red = er1[:, :, :, 0, :, :]
        er2_red = er2[:, :, :, 0, :, :]
        self.ur1_red = nn.Parameter(ur1_red, requires_grad=False)
        self.ur2_red = nn.Parameter(ur2_red, requires_grad=False)

        Q_ref_00 = torch.matmul(KX, KY)
        Q_ref_01 = ur1_red * er1_red * I - torch.matmul(KX, KX)
        Q_ref_10 = torch.matmul(KY, KY) - ur1_red * er1_red * I
        Q_ref_11 = -torch.matmul(KY, KX)

        Q_ref_row0 = torch.cat([Q_ref_00, Q_ref_01], dim=4)
        Q_ref_row1 = torch.cat([Q_ref_10, Q_ref_11], dim=4)
        Q_ref = torch.cat([Q_ref_row0, Q_ref_row1], dim=3)

        W_ref_row0 = torch.cat([I, Z], dim=4)
        W_ref_row1 = torch.cat([Z, I], dim=4)
        W_ref = torch.cat([W_ref_row0, W_ref_row1], dim=3)
        self.W_ref = nn.Parameter(W_ref, requires_grad=False)

        LAM_ref_row0 = torch.cat([-1j * KZref, Z], dim=4)
        LAM_ref_row1 = torch.cat([Z, -1j * KZref], dim=4)
        LAM_ref = torch.cat([LAM_ref_row0, LAM_ref_row1], dim=3)
        V_ref = torch.matmul(Q_ref, thl.inv(LAM_ref))

        W0_inv = thl.inv(W0)
        V0_inv = thl.inv(V0)
        A_ref = torch.matmul(W0_inv, W_ref) + torch.matmul(V0_inv, V_ref)
        A_ref_inv = thl.inv(A_ref)
        B_ref = torch.matmul(W0_inv, W_ref) - torch.matmul(V0_inv, V_ref)

        SR_S11 = torch.matmul(-A_ref_inv, B_ref)
        SR_S12 = 2 * A_ref_inv
        SR_S21 = torch.matmul(B_ref, A_ref_inv)
        SR_S21 = torch.matmul(SR_S21, B_ref)
        SR_S21 = 0.5 * (A_ref - SR_S21)
        SR_S22 = torch.matmul(B_ref, A_ref_inv)
        self.SR_S11 = nn.Parameter(SR_S11, requires_grad=False)
        self.SR_S12 = nn.Parameter(SR_S12, requires_grad=False)
        self.SR_S21 = nn.Parameter(SR_S21, requires_grad=False)
        self.SR_S22 = nn.Parameter(SR_S22, requires_grad=False)

        ### Step 9: Transmission side ###
        Q_trn_00 = torch.matmul(KX, KY)
        Q_trn_01 = ur2_red * er2_red * I - torch.matmul(KX, KX)
        Q_trn_10 = torch.matmul(KY, KY) - ur2_red * er2_red * I
        Q_trn_11 = -torch.matmul(KY, KX)
        Q_trn_row0 = torch.cat([Q_trn_00, Q_trn_01], dim=4)
        Q_trn_row1 = torch.cat([Q_trn_10, Q_trn_11], dim=4)
        Q_trn = torch.cat([Q_trn_row0, Q_trn_row1], dim=3)

        W_trn_row0 = torch.cat([I, Z], dim=4)
        W_trn_row1 = torch.cat([Z, I], dim=4)
        W_trn = torch.cat([W_trn_row0, W_trn_row1], dim=3)
        self.W_trn = nn.Parameter(W_trn, requires_grad=False)

        LAM_trn_row0 = torch.cat([1j * KZtrn, Z], dim=4)
        LAM_trn_row1 = torch.cat([Z, 1j * KZtrn], dim=4)
        LAM_trn = torch.cat([LAM_trn_row0, LAM_trn_row1], dim=3)

        V_trn = torch.matmul(Q_trn, thl.inv(LAM_trn))

        W0_inv = thl.inv(W0)
        V0_inv = thl.inv(V0)
        A_trn = torch.matmul(W0_inv, W_trn) + torch.matmul(V0_inv, V_trn)
        A_trn_inv = thl.inv(A_trn)
        B_trn = torch.matmul(W0_inv, W_trn) - torch.matmul(V0_inv, V_trn)

        ST_S11 = torch.matmul(B_trn, A_trn_inv)
        ST_S12 = torch.matmul(B_trn, A_trn_inv)
        ST_S12 = torch.matmul(ST_S12, B_trn)
        ST_S12 = 0.5 * (A_trn - ST_S12)
        ST_S21 = 2 * A_trn_inv
        ST_S22 = torch.matmul(-A_trn_inv, B_trn)
        self.ST_S11 = nn.Parameter(ST_S11, requires_grad=False)
        self.ST_S12 = nn.Parameter(ST_S12, requires_grad=False)
        self.ST_S21 = nn.Parameter(ST_S21, requires_grad=False)
        self.ST_S22 = nn.Parameter(ST_S22, requires_grad=False)

        ### Step 11: Compute source parameters ###
        # Compute mode coefficients of the source.
        delta = torch.zeros((batch_size, pixelsX, pixelsY, np.prod(PQ)), dtype=dtype)
        delta[:, :, :, int(np.prod(PQ) / 2.0)] = 1

        # Incident wavevector.
        kinc_x0_pol = torch.real(kinc_x0[:, :, :, 0, 0])
        kinc_y0_pol = torch.real(kinc_y0[:, :, :, 0, 0])
        kinc_z0_pol = torch.real(kinc_z0[:, :, :, 0])
        kinc_pol = torch.cat([kinc_x0_pol, kinc_y0_pol, kinc_z0_pol], dim=3)

        # Calculate TE and TM polarization unit vectors.
        firstPol = True
        for pol in range(self.batch_size):
            if kinc_pol[pol, 0, 0, 0] == 0.0 and kinc_pol[pol, 0, 0, 1] == 0.0:
                ate_pol = np.zeros((1, pixelsX, pixelsY, 3))
                ate_pol[:, :, :, 1] = 1
                ate_pol = torch.tensor(ate_pol, dtype=dtype)
            else:
                # Calculation of `ate` for oblique incidence.
                n_hat = np.zeros((1, pixelsX, pixelsY, 3))
                n_hat[:, :, :, 0] = 1
                n_hat = torch.tensor(n_hat, dtype=dtype)
                kinc_pol_iter = kinc_pol[pol, :, :, :]
                kinc_pol_iter = kinc_pol_iter[None, :, :, :]
                ate_cross = torch.cross(n_hat, kinc_pol_iter)
                ate_pol = ate_cross / torch.norm(ate_cross, dim=3, keepdim=True)

            if firstPol:
                ate = ate_pol
                firstPol = False
            else:
                ate = torch.cat([ate, ate_pol], dim=0)

        atm_cross = torch.cross(kinc_pol, ate)
        atm = atm_cross / torch.norm(atm_cross, dim=3, keepdim=True)
        ate = ate.to(dtype=cdtype)
        atm = atm.to(dtype=cdtype)

        # Decompose the TE and TM polarization into x and y components.
        EP = pte * ate + ptm * atm
        EP_x = EP[:, :, :, 0]
        EP_x = EP_x[:, :, :, None]
        EP_y = EP[:, :, :, 1]
        EP_y = EP_y[:, :, :, None]

        esrc_x = EP_x * delta
        esrc_y = EP_y * delta
        esrc = torch.cat([esrc_x, esrc_y], dim=3)
        esrc = esrc[:, :, :, :, None]
        W_ref_inv = thl.inv(W_ref)

        ### Step 12: Compute reflected and transmitted fields ###
        csrc = torch.matmul(W_ref_inv, esrc)
        self.csrc = nn.Parameter(csrc, requires_grad=False)

        # # # Compute longitudinal components.
        KZref_inv = thl.inv(KZref)
        KZtrn_inv = thl.inv(KZtrn)
        self.KZref_inv = nn.Parameter(KZref_inv, requires_grad=False)
        self.KZtrn_inv = nn.Parameter(KZtrn_inv, requires_grad=False)

        return

    def __check_eps(self, mat):
        valid_mats = MATERIAL_DICT.keys()
        if isinstance(mat, str):
            assert (
                mat in valid_mats
            ), f"Error: layer material {mat} not in {valid_mats}."
        else:
            assert isinstance(
                mat, (complex, float)
            ), "Layer Material entries must be either string containing the material name or a complex/float"

        return

    def __get_eps(self, mat, wavelength_set_m):
        if isinstance(mat, str):
            warnings.warn(
                "Imaginary index is unverified and may cause issues at the moment.",
                UserWarning,
            )
            eps = get_material_index(mat, wavelength_set_m) ** 2
            eps = torch.tensor(eps, dtype=self.cdtype)
        else:
            eps = torch.ones(
                len(wavelength_set_m), dtype=self.cdtype
            ) * self.__to_torch_complex(mat)

        return eps

    def __to_torch_complex(self, input_value):
        if isinstance(input_value, np.ndarray):
            input_tensor = torch.from_numpy(input_value)
        elif isinstance(input_value, torch.Tensor):
            input_tensor = input_value
        else:
            input_tensor = torch.tensor(input_value)

        input_tensor = input_tensor.to(self.cdtype)
        return input_tensor


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    wavelength_set_m = np.arange(450e-9, 700e-9, 5e-9)
    fourier_modes = 5
    this = RCWA_Solver(
        wavelength_set_m,
        thetas=[0.0 for i in wavelength_set_m],
        phis=[0.0 for i in wavelength_set_m],
        pte=[1 for i in wavelength_set_m],
        ptm=[1 for i in wavelength_set_m],
        pixelsX=1,
        pixelsY=1,
        PQ=[fourier_modes, fourier_modes],
        lux=400e-9,
        luy=400e-9,
        layer_heights=[600e-9],
        layer_embed_mats=["Vacuum"],
        material_dielectric="TiO2",
        Nx=400,
        Ny=400,
        er1="Vacuum",
        er2="Vacuum",
        ur1=1.0,
        ur2=1.0,
        urd=1.0,
        urs=1.0,
    )
    this = this.to("cuda")

    binary = torch.randn((1, 400, 400), dtype=torch.float32, device="cuda")
    out = this(binary)
