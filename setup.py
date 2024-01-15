from setuptools import setup, find_packages

setup(
    name="dflat",
    version="3.0",
    author="Dean Hazineh",
    author_email="dhazineh@g.harvard.edu",
    description="Dflat Version 3. Open source pytorch field propagation and rendering",
    # long_description=open("README.md").read(),
    # long_description_content_type="text/markdown",
    # url='https://github.com/yourusername/yourpackagename',  # Link to your project's GitHub repo
    packages=find_packages(),  # Automatically find your package directories
    classifiers=[
        # Classifiers help users find your project by categorizing it
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Assuming your project is MIT Licensed
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",  # Minimum version requirement of the package
    install_requires=[],
)
