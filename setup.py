from setuptools import setup, find_packages


setup(
    python_requires=">=3.9",
    include_package_data=True,
    package_data={
        "": ["rcwa/material_index/*.mat", "*.txt", "*.csv"],
    },
)
