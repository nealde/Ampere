import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ampere",
    version="0.2.3",
    author="Neal Dawson-Elli",
    author_email="nealde@uw.edu",
    description="A Python package for working with battery discharge data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nealde/Ampere",
    packages=setuptools.find_packages(),
    install_requires=['matplotlib', 'numpy', 'scipy'],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ),
)
