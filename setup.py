from setuptools import setup, find_packages
from pathlib import Path
this_directory = Path(__file__).parent
requirementPath = this_directory / 'requirements.txt'
with open(requirementPath) as f:
    install_requires = f.read().splitlines()

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


setup(
    name="perceiver-model",
    packages=find_packages(),
    version="0.7.1",
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
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
)
