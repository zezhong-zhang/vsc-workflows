# coding: utf8
# Copyright (c) Marnik Bercx, University of Antwerp
# Distributed under the terms of the MIT License

from fireworks import Workflow

from vscworkflows.workflow.fireworks import OptimizeFW, OpticsFW, SlabOptimizeFW, \
    SlabDosFW

"""
Definition of all workflows in the package.

"""

__author__ = "Marnik Bercx"
__copyright__ = "Copyright 2019, Marnik Bercx, University of Antwerp"
__version__ = "pre-alpha"
__maintainer__ = "Marnik Bercx"
__email__ = "marnik.bercx@uantwerpen.be"
__date__ = "Jun 2019"


def get_wf_optimize(structure, directory, functional=("pbe", {}),
                    is_metal=False, in_custodian=False, number_nodes=None,
                    fw_action=None):
    """
    Set up a geometry optimization workflow for a bulk structure.

    Args:
        structure: pymatgen.Structure OR path to the structure file.
        directory (str): Directory in which the geometry optimization should be performed.
        functional (tuple): Tuple with the functional details. The first element
            contains a string that indicates the functional used ("pbe", "hse", ...),
            whereas the second element contains a dictionary that allows the user
            to specify the various functional tags.
        is_metal (bool): Flag that indicates the material being studied is a
                metal, which changes the smearing from Gaussian (0.05 eV) to second
                order Methfessel-Paxton of 0.2 eV.
        in_custodian (bool): Flag that indicates whether the calculation should be
            run inside a Custodian.
        number_nodes (int): Number of nodes that should be used for the calculations.
            Is required to add the proper `_category` to the Firework generated, so
            it is picked up by the right Fireworker.
        fw_action (fireworks.FWAction): FWAction to return after the final
                PulayTask is completed.

    """
    # Set up the geometry optimization Firework
    optimize_fw = OptimizeFW(structure=structure,
                             functional=functional,
                             directory=directory,
                             is_metal=is_metal,
                             in_custodian=in_custodian,
                             number_nodes=number_nodes,
                             fw_action=fw_action)

    # Set up a clear name for the workflow
    workflow_name = str(structure.composition.reduced_formula).replace(" ", "")
    workflow_name += " " + str(functional)

    # Create the workflow
    return Workflow(fireworks=[optimize_fw, ],
                    name=workflow_name)


def get_wf_optics(structure, directory, functional=("pbe", {}), k_resolution=80,
                  is_metal=False, in_custodian=False, number_nodes=None):
    """
    Set up a geometry optimization workflow.

    Args:
        structure: pymatgen.Structure OR path to the structure file.
        directory (str): Directory in which the optics calculation should be performed.
        functional (tuple): Tuple with the functional details. The first element
            contains a string that indicates the functional used ("pbe", "hse", ...),
            whereas the second element contains a dictionary that allows the user
            to specify the various functional tags.
        k_resolution (float): Resolution of the k-mesh, i.e. distance between two
            k-points along each reciprocal lattice vector.
        is_metal (bool): Flag that indicates the material being studied is a
                metal. The calculation will then use a broad Gaussian smearing of 0.3
                eV instead of the tetrahedron method.
        in_custodian (bool): Flag that indicates whether the calculation should be
            run inside a Custodian.
        number_nodes (int): Number of nodes that should be used for the calculations.
            Is required to add the proper `_category` to the Firework generated, so
            it is picked up by the right Fireworker.

    """
    # Set up the geometry optimization Firework
    optics_fw = OpticsFW(
        structure=structure,
        directory=directory,
        functional=functional,
        k_resolution=k_resolution,
        is_metal=is_metal,
        in_custodian=in_custodian,
        number_nodes=number_nodes
    )

    # Set up a clear name for the workflow
    workflow_name = str(structure.composition.reduced_formula).replace(" ", "")
    workflow_name += " " + str(functional)

    # Create the workflow
    return Workflow(fireworks=[optics_fw, ], name=workflow_name)


def get_wf_slab_optimize(slab, directory, fix_part, fix_thickness,
                         functional=("pbe", {}), is_metal=False, in_custodian=False,
                         number_nodes=None):
    """
    Set up a geometry optimization workflow.

    Args:
        slab (Qslab): Slab for which to set up the geometry optimization workflow.
        directory (str): Directory in which the geometry optimization should be performed.
        functional (tuple): Tuple with the functional details. The first element
            contains a string that indicates the functional used ("pbe", "hse", ...),
            whereas the second element contains a dictionary that allows the user
            to specify the various functional tags.
        fix_part (str): Which part of the slab to fix. Currently only allows for
                "center".
        fix_thickness (int): The thickness of the fixed part of the slab, expressed in
            number of layers.
        is_metal (bool): Flag that indicates whether the material for which the
            geometry optimization should be performed is metallic. Determines the
            smearing method used.
        in_custodian (bool): Flag that indicates whether the calculation should be
            run inside a Custodian.
        number_nodes (int): Number of nodes that should be used for the calculations.
            Is required to add the proper `_category` to the Firework generated, so
            it is picked up by the right Fireworker.

    """
    # Set up the geometry optimization Firework
    optimize_fw = SlabOptimizeFW(slab=slab,
                                 directory=directory,
                                 fix_part=fix_part,
                                 fix_thickness=fix_thickness,
                                 functional=functional,
                                 is_metal=is_metal,
                                 in_custodian=in_custodian,
                                 number_nodes=number_nodes)

    # Set up a clear name for the workflow
    workflow_name = str(slab.composition.reduced_formula).replace(" ", "")
    workflow_name += " " + str(slab.miller_index)
    workflow_name += " " + str(functional)

    # Create the workflow
    return Workflow(fireworks=[optimize_fw, ],
                    name=workflow_name)


def get_wf_slab_dos(slab, directory, functional=("pbe", {}), k_resolution=0.1,
                    calculate_locpot=False, in_custodian=False, number_nodes=None):
    """
    Set up a geometry optimization workflow.

    Args:
        slab (Qslab): Slab for which to set up the DOS workflow.
        directory (str): Directory in which the geometry optimization should be performed.
        functional (tuple): Tuple with the functional details. The first element
            contains a string that indicates the functional used ("pbe", "hse", ...),
            whereas the second element contains a dictionary that allows the user
            to specify the various functional tags.
        k_resolution (float): Resolution of the k-mesh, i.e. distance between two
            k-points along each reciprocal lattice vector. Note that for a slab
            calculation we always only consider one point in the c-direction.
        calculate_locpot (bool): Whether to calculate the the local potential, e.g. to
            determine the work function.
        in_custodian (bool): Flag that indicates wheter the calculation should be
            run inside a Custodian.
        number_nodes (int): Number of nodes that should be used for the calculations.
            Is required to add the proper `_category` to the Firework generated, so
            it is picked up by the right Fireworker.

    """
    # Set up the geometry optimization Firework
    dos_fw = SlabDosFW(
        slab=slab,
        directory=directory,
        functional=functional,
        k_resolution=k_resolution,
        calculate_locpot=calculate_locpot,
        in_custodian=in_custodian,
        number_nodes=number_nodes
    )

    # Set up a clear name for the workflow
    workflow_name = str(slab.composition.reduced_formula).replace(" ", "")
    workflow_name += " " + str(slab.miller_index)
    workflow_name += " " + str(functional)

    # Create the workflow
    return Workflow(fireworks=[dos_fw, ], name=workflow_name)


def get_wf_quotas(bulk, slab_list, functional=("pbe", {}),
                  base_k_product=50, number_nodes=None):
    """
    Generate a full QUOTAS worfklow, i.e. one that:

    1. Optimizes the bulk and calculates the optical properties.
    2. Optimizes the slabs in 'slab_list' and then calculates the DOS and work function.

    Args:
        bulk:
        slab_list:
        functional:
        base_k_product:
        number_nodes:

    """
    pass
