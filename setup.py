import subprocess
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop

try:
    import requests
except ImportError:
    subprocess.check_call(["pip", "install", "requests"])


class CustomInstallCommand(install):
    def run(self):
        install.run(self)
        from setup_data_retrieval import execute_data_management

        execute_data_management()


class CustomDevelopCommand(develop):
    def run(self):
        develop.run(self)
        from setup_data_retrieval import execute_data_management

        execute_data_management()


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
