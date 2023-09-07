import os
import shutil
from setuptools import setup, find_packages, Command
from setuptools.command.install import install
from setuptools.command.develop import develop


class CleanCommand(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        shutil.rmtree("build", ignore_errors=True)
        print("Removed the build directory")


class CustomInstallCommand(install):
    """Customized setuptools install command - unzips the data files after installing."""

    def run(self):
        install.run(self)

        from setup_execute_get_data import execute_data_management

        execute_data_management()  # Call your custom function

        shutil.rmtree("build", ignore_errors=True)
        print("Removed the build directory")


class CustomDevelopCommand(develop):
    """Customized setuptools develop command - unzips the data files after installing in dev mode."""

    def run(self):
        develop.run(self)

        from setup_execute_get_data import execute_data_management

        execute_data_management()  # Call your custom function

        shutil.rmtree("build", ignore_errors=True)
        print("Removed the build directory")


if __name__ == "__main__":
    setup(
        python_requires=">=3.9",
        packages=find_packages(),
        include_package_data=True,
        cmdclass={
            "develop": CustomDevelopCommand,
            "install": CustomInstallCommand,
            "clean": CleanCommand,
        },
    )
