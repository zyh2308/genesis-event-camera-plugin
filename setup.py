import pathlib

from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text(encoding="utf-8")

setup(
    name="genesis-event-plugin-v2-physics",
    version="0.1.0a1",
    description="Genesis Event Camera Plugin — radiance and physics-warp prototype",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Event-based WAM Team",
    url="https://github.com/zyh2308/genesis-event-camera-plugin",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21",
        "torch>=2.0",
        "h5py>=3.0",
        "scipy>=1.7",
        "genesis-world>=1.2.2",
    ],
    extras_require={
        "demo": ["pillow", "imageio", "opencv-python"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)
