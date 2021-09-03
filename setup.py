from setuptools import setup, find_packages

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

setup(
    name="perceiver-pytorch",
    packages=find_packages(),
    version="0.4.0",
    license="MIT",
    description="Perceiver - Pytorch",
    author="Phil Wang",
    author_email="lucidrains@gmail.com",
    url="https://github.com/lucidrains/perceiver-pytorch",
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
