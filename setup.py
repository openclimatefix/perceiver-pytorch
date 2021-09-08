from setuptools import setup, find_packages
import os

install_folder = os.path.dirname(os.path.realpath(__file__))
requirementPath = install_folder + '/requirements.txt'
with open(requirementPath) as f:
    install_requires = f.read().splitlines()

setup(
    name="perceiver-model",
    packages=find_packages(),
    version="0.7.0",
    license="MIT",
    description="Multimodal Perceiver - Pytorch",
    author="Jacob Bieker, Jack Kelly, Peter Dudfield",
    author_email="jacob@openclimatefix.org",
    company="Open Climate Fix Ltd",
    url="https://github.com/openclimatefix/perceiver-pytorch",
    keywords=[
        "artificial intelligence",
        "deep learning",
        "transformer",
        "attention mechanism",
    ],
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
)
