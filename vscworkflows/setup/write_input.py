# coding: utf8
# Copyright (c) Marnik Bercx, University of Antwerp
# Distributed under the terms of the MIT License

import os

from monty.serialization import loadfn

from pymatgen import Structure
from quotas import QSlab

from vscworkflows.setup.sets import BulkStaticSet, BulkOptimizeSet, SlabOptimizeSet, \
    SlabStaticSet

"""
Scripts that write the VASP input files for various calculations.

"""

__author__ = "Marnik Bercx"
__copyright__ = "Copyright 2018, Marnik Bercx, University of Antwerp"
__version__ = "pre-alpha"
__maintainer__ = "Marnik Bercx"
__email__ = "marnik.bercx@uantwerpen.be"
__date__ = "Jul 2019"

MODULE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "set_configs")

DFT_FUNCTIONAL = "PBE_54"


def _load_yaml_config(filename):
    config = loadfn(os.path.join(MODULE_DIR, "%s.yaml" % filename))
    return config


def _set_up_directory(directory, functional, calculation):
    # Set up the calculation directory
    if directory == "":
        directory = os.path.join(os.getcwd(), functional[0])
        if functional[0] == "pbeu":
            directory += "_" + "".join(k + str(functional[1]["LDAUU"][k]) for k
                                       in functional[1]["LDAUU"].keys())
        directory += "_" + calculation
    else:
        directory = os.path.abspath(directory)

    return directory


def _set_up_calculation(calculation_set, functional):
    """
    Set up a DictSet-based calculation set. This basically runs some checks and adjust
    input settings where necessary.

    Args:
        calculation_set:

    Returns:

    """
    # Check if a magnetic moment was provided for the sites. If so, perform a
    # spin-polarized calculation
    if "magmom" in calculation_set.structure.site_properties.keys():
        calculation_set.user_incar_settings.update({"ISPIN": 2, "MAGMOM": True})

    # Adjust the projector-evaluation scheme to Auto for large unit cells (+20 atoms)
    if len(calculation_set.structure) > 20:
        calculation_set.user_incar_settings.update({"LREAL": "Auto"})

    # Set up the functional
    if functional[0] != "pbe":
        functional_config = _load_yaml_config(functional[0] + "Set")
        functional_config["INCAR"].update(functional[1])
        calculation_set.user_incar_settings.update(functional_config["INCAR"])

    return calculation_set


def optimize(structure, directory="", functional=("pbe", {}),
             is_metal=False):
    """
    Set up a standard geometry optimization calculation for a structure. Optimizes
    both the atomic positions as well as the unit cell (ISIF=3).

    Args:
        structure: pymatgen.Structure OR path to structure file for which to set up the
            geometry optimization calculation.
        directory (str): Path to the directory in which to set up the
            geometry optimization.
        functional (tuple): Tuple with the functional choices. The first element
            contains a string that indicates the functional used ("pbe", "hse", ...),
            whereas the second element contains a dictionary that allows the user
            to specify the various functional tags. E.g. ("hse", {"LAEXX": 0.2}).
        is_metal (bool): Flag that indicates the material being studied is a
            metal, which changes the smearing from Gaussian to second order
            Methfessel-Paxton of 0.2 eV.

    Returns:
        str: Path to the directory in which the calculation is set up.

    """
    # Set up the calculation directory
    directory = _set_up_directory(directory, functional, "optimize")
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass

    # In case the structure is given as a string, load it from the specified path
    if isinstance(structure, str):
        structure = Structure.from_file(structure)

    # Store the full Structure as a json file
    structure.to("json", os.path.join(directory, "initial_structure.json"))

    # Set the defaults for the calculation
    user_incar_settings = {}

    # For metals, use Methfessel Paxton smearing
    if is_metal:
        user_incar_settings.update({"ISMEAR": 2, "SIGMA": 0.2})

    # Set up the geometry optimization
    calculation = _set_up_calculation(
        BulkOptimizeSet(structure=structure, user_incar_settings=user_incar_settings,
                        potcar_functional=DFT_FUNCTIONAL),
        functional=functional
    )
    # Write the setup files to the geometry optimization directory
    calculation.write_input(directory)

    return directory


def optics(structure, directory="", functional=("pbe", {}), k_resolution=0.05,
           is_metal=False):
    """
    Set up a standard geometry optimization calculation for a structure. Optimizes
    both the atomic positions as well as the unit cell (ISIF=3).

    Args:
        structure: pymatgen.Structure OR path to structure file for which to set up the
            geometry optimization calculation.
        directory (str): Path to the directory in which to set up the
            geometry optimization.
        functional (tuple): Tuple with the functional choices. The first element
            contains a string that indicates the functional used ("pbe", "hse", ...),
            whereas the second element contains a dictionary that allows the user
            to specify the various functional tags. E.g. ("hse", {"LAEXX": 0.2}).
        k_resolution (float): Resolution of the k-mesh, i.e. distance between two
            k-points along each reciprocal lattice vector.
        is_metal (bool): Flag that indicates the material being studied is a
            metal, which changes the smearing from Gaussian to second order
            Methfessel-Paxton of 0.2 eV.

    Returns:
        str: Path to the directory in which the calculation is set up.

    """
    # Set up the calculation directory
    directory = _set_up_directory(directory, functional, "optics")
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass

    # In case the structure is given as a string, load it from the specified path
    if isinstance(structure, str):
        structure = Structure.from_file(structure)

    # Store the full Structure as a json file
    structure.to("json", os.path.join(directory, "initial_structure.json"))

    # Set the defaults for the calculation
    user_incar_settings = {"LOPTICS": True, "EDIFF": 1.0e-6}

    # For metals, use a good amount of Gaussian smearing
    if is_metal:
        user_incar_settings.update({"ISMEAR": 0, "SIGMA": 0.3})

    # Set up the geometry optimization
    calculation = _set_up_calculation(
        BulkStaticSet(structure=structure, k_resolution=k_resolution,
                      user_incar_settings=user_incar_settings,
                      potcar_functional=DFT_FUNCTIONAL),
        functional=functional
    )

    # Write the setup files to the geometry optimization directory
    calculation.write_input(directory)

    return directory


def slab_optimize(slab, fix_part, fix_thickness, directory="",
                  functional=("pbe", {}), is_metal=False):
    """
    Set up a geometric optimization for a two dimensional slab.

    Args:
        slab (QSlab): Quotas version of a Slab object, or path to the json file that
            contains the details of the QSlab.
        directory (str): Path to the directory in which to set up the
            geometry optimization.
        functional (tuple): Tuple with the functional choices. The first element
            contains a string that indicates the functional used ("pbe", "hse", ...),
            whereas the second element contains a dictionary that allows the user
            to specify the various functional tags. E.g. ("hse", {"LAEXX": 0.2}).
        fix_part (str): Which part of the slab to fix. Currently only allows for
            "center".
        fix_thickness (int): The thickness of the fixed part of the slab, expressed in
            number of layers.
        is_metal (bool): Flag that indicates the material being studied is a
            metal, which changes the smearing from Gaussian to second order
            Methfessel-Paxton of 0.2 eV.

    Returns:
        relax_dir: Full path to the directory where the geometry
        optimization was set up.

    """
    # Set up the calculation directory
    directory = _set_up_directory(directory, functional, "optimize")
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass

    # In case the slab is given as a string, load it from the specified path
    if isinstance(slab, str):
        slab = QSlab.from_file(slab)

    # Store the full QSlab object
    slab.to("json", os.path.join(directory, "initial_slab.json"))

    # Set the defaults for the calculation
    user_incar_settings = {}

    # For metals, use Methfessel Paxton smearing
    if is_metal:
        user_incar_settings.update({"ISMEAR": 2, "SIGMA": 0.2})

    calculation = _set_up_calculation(
        SlabOptimizeSet(structure=slab,
                        user_incar_settings=user_incar_settings,
                        potcar_functional=DFT_FUNCTIONAL),
        functional=functional
    )

    calculation.fix_slab_bulk(thickness=fix_thickness,
                              part=fix_part)

    # Write the setup files to the calculation directory
    calculation.write_input(directory)

    return directory


def slab_dos(slab, directory="", functional=("pbe", {}),
             k_resolution=0.1, calculate_locpot=False, user_incar_settings=None):
    """
    Set up the DOS / work function calculation.

    Args:
        slab: quotas.QSlab OR path to slab structure file for which to run
            the DOS calculation.
        directory (str): Directory in which the geometry optimization should be
            performed.
        functional (tuple): Tuple with the functional choices. The first element
            contains a string that indicates the functional used ("pbe", "hse", ...),
            whereas the second element contains a dictionary that allows the user to
            specify the various functional tags.
        k_resolution (float): Resolution of the k-mesh, i.e. distance between two
            k-points along each reciprocal lattice vector. Note that for a slab
            calculation we always only consider one point in the c-direction.
        calculate_locpot (bool): Whether to calculate the the local potential, e.g. to
            determine the work function.
        user_incar_settings (dict): #TODO


    Returns:

    """
    user_incar_settings = user_incar_settings or {}

    # Set up the calculation directory
    directory = _set_up_directory(directory, functional, "dos")
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass

    # In case the slab is given as a string, load it from the specified path
    if isinstance(slab, str):
        slab = QSlab.from_file(slab)

    # Store the full QSlab object
    slab.to("json", os.path.join(directory, "initial_slab.json"))

    # Set the defaults for the calculation
    user_incar_settings.update({"NEDOS": 2000})

    # Calculate the local potential if requested (e.g. for the work function)
    if calculate_locpot:
        user_incar_settings.update({"LVTOT": True, "LVHAR": True})

    calculation = _set_up_calculation(
        SlabStaticSet(structure=slab, k_resolution=k_resolution,
                      user_incar_settings=user_incar_settings,
                      potcar_functional=DFT_FUNCTIONAL),
        functional=functional
    )

    # Set the number of bands for the calculation
    if "magmom" in slab.site_properties.keys():
        nbands = int(calculation.nelect * 0.6 + len(slab)) * 3
    else:
        nbands = int((calculation.nelect + len(slab)) / 2) * 3

    calculation.user_incar_settings.update({"NBANDS": nbands})

    # Write the setup files to the calculation directory
    calculation.write_input(directory)

    return directory
