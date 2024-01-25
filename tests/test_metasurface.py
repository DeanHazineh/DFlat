import pytest
import os

from dflat.metasurface.datasets import (
    Nanofins_TiO2_U350nm_H600nm,
    Nanocylinders_TiO2_U180nm_H600nm,
    Nanoellipse_TiO2_U350nm_H600nm,
)


class Test_cell_library:
    @pytest.fixture(
        params=[
            Nanofins_TiO2_U350nm_H600nm,
            Nanocylinders_TiO2_U180nm_H600nm,
            Nanoellipse_TiO2_U350nm_H600nm,
        ]
    )
    def cell_object(self, request):
        return request.param()

    def test_plot(self, cell_object):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(current_dir, "out/")
        cell_object.plot(savepath=output_path)

    def get_item(self, cell_object):
        size_data = len(cell_object)
        x, y = cell_object[0]   
        assert x.shape[0]==y.shape[0]
