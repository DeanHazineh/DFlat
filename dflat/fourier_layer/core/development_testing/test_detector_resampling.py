# if __name__ == "__main__":
#     import dflat.data_structure as df_params
#     import matplotlib.pyplot as plt

#     fourier_params = df_params.prop_params(
#         {
#             "wavelength_m": 532e-9,
#             "ms_samplesM": {"x": 100, "y": 100},
#             "ms_dx_m": {"x": 1e-6, "y": 1e-6},
#             "sensor_distance_m": 40e-3,
#             "initial_sensor_dx_m": {"x": 1e-6, "y": 1e-6},
#             "sensor_pixel_size_m": {"x": 2.5e-6, "y": 2.5e-6},
#             "sensor_pixel_number": {"x": 10, "y": 10},
#             "radial_symmetry": False,
#             "diffractionEngine": "ASM_fourier",
#         }
#     )

#     xx, yy = np.meshgrid(np.arange(1, 10), np.arange(1, 10))
#     data = torch.tensor(xx, dtype=torch.float64)  # torch.rand(10, 10)
#     outint, outphase = sensorMeasurement_intensity_phase(
#         data, data, fourier_params, False
#     )
#     print(data.shape, outint.shape)

#     fig = plt.figure()
#     ax = fig.add_subplot(1, 2, 1)
#     ax.imshow(data)
#     ax = fig.add_subplot(1, 2, 2)
#     ax.imshow(outint)
#     plt.show()
