from setuptools import setup, find_packages

setup(
    name="vscworkflows",
    version="Planning",
    author="Marnik Bercx",
    packages=find_packages(where=".", exclude="docs"),
    install_requires=[
        "pymatgen",
        "fireworks",
        "click"
    ]
)