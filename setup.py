from setuptools import setup, find_packages

setup(
    name="vscworkflows",
    version="pre-alpha",
    author="Marnik Bercx",
    packages=find_packages(where=".", exclude="docs"),
    install_requires=[
        "numpy",
        "pymatgen",
        "custodian",
        "fireworks",
        "atomate",
        "dnspython",
        "click",
        "monty",
        "icet",
        "tabulate"
    ],
    entry_points='''
        [console_scripts]
        vsc=vscworkflows.cli:main
    '''
)