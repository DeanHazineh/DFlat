import os
import shutil
from setuptools import setup, find_packages, Command
from setuptools.command.install import install
from setuptools.command.develop import develop

import subprocess

# Ensure requests is installed before anything else.
try:
    import requests
except ImportError:
    subprocess.check_call(["pip", "install", "requests"])


class CustomInstallCommand(install):
    """Customized setuptools install command - unzips the data files after installing."""

    def run(self):
        install.run(self)
        from setup_execute_get_data import execute_data_management

        execute_data_management()  # Call your custom function


class CustomDevelopCommand(develop):
    """Customized setuptools develop command - unzips the data files after installing in dev mode."""

    def run(self):
        develop.run(self)
        from setup_execute_get_data import execute_data_management

        execute_data_management()  # Call your custom function


if __name__ == "__main__":
    setup(
        python_requires=">=3.9",
        packages=find_packages(),
        include_package_data=True,
        cmdclass={
            "develop": CustomDevelopCommand,
            "install": CustomInstallCommand,
        },
    )
