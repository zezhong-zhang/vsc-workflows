# coding: utf8
# Copyright (c) Marnik Bercx, University of Antwerp
# Distributed under the terms of the MIT License

from fireworks import Workflow

from vscworkflows.workflow.fireworks import OptimizeFW, SlabOptimizeFW, SlabDosFW

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
    Set up a geometry optimization workflow.

    Args:
        structure (pymatgen.Structure): Structure for which to set up the geometry
            optimization workflow.
        directory (str): Directory in which the geometry optimization should be performed.
        functional (tuple): Tuple with the functional details. The first element
            contains a string that indicates the functional used ("pbe", "hse", ...),
            whereas the second element contains a dictionary that allows the user
            to specify the various functional tags.
        is_metal (bool): Flag that indicates whether the material for which the
            geometry optimization should be performed is metallic. Determines the
            smearing method used.
        in_custodian (bool): Flag that indicates wheter the calculation should be
            run inside a Custodian.
        number_nodes (int): Number of nodes that should be used for the calculations.
            Is required to add the proper `_category` to the Firework generated, so
            it is picked up by the right Fireworker.
        fw_action (fireworks.FWAction): FWAction to return after the final
                PulayTask is completed.

    Returns:
        None

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
        is_metal (bool): Flag that indicates whether the material for which the
            geometry optimization should be performed is metallic. Determines the
            smearing method used.
        in_custodian (bool): Flag that indicates wheter the calculation should be
            run inside a Custodian.
        number_nodes (int): Number of nodes that should be used for the calculations.
            Is required to add the proper `_category` to the Firework generated, so
            it is picked up by the right Fireworker.

    Returns:
        None

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


def get_wf_slab_dos(slab, directory, functional=("pbe", {}), k_product=80,
                    calculate_locpot=False, in_custodian=False, number_nodes=None):
    """
    Set up a geometry optimization workflow.

    Args:
        slab (Qslab): Slab for which to set up the geometry optimization workflow.
        directory (str): Directory in which the geometry optimization should be performed.
        functional (tuple): Tuple with the functional details. The first element
            contains a string that indicates the functional used ("pbe", "hse", ...),
            whereas the second element contains a dictionary that allows the user
            to specify the various functional tags.
        in_custodian (bool): Flag that indicates wheter the calculation should be
            run inside a Custodian.
        number_nodes (int): Number of nodes that should be used for the calculations.
            Is required to add the proper `_category` to the Firework generated, so
            it is picked up by the right Fireworker.

    Returns:
        None

    """

    # Set up the geometry optimization Firework
    dos_fw = SlabDosFW(
        slab=slab,
        directory=directory,
        functional=functional,
        k_product=k_product,
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
