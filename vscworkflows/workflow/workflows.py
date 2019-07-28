# coding: utf8
# Copyright (c) Marnik Bercx, University of Antwerp
# Distributed under the terms of the MIT License

import os

from fireworks import Workflow, FWAction
from monty.serialization import loadfn

from vscworkflows.workflow.fireworks import StaticFW, OptimizeFW, OpticsFW, \
    SlabOptimizeFW, SlabDosFW

"""
Definition of all workflows in the package.

"""

__author__ = "Marnik Bercx"
__copyright__ = "Copyright 2019, Marnik Bercx, University of Antwerp"
__version__ = "pre-alpha"
__maintainer__ = "Marnik Bercx"
__email__ = "marnik.bercx@uantwerpen.be"
__date__ = "Jun 2019"

MODULE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../setup/set_configs"
)


def _load_yaml_config(filename):
    config = loadfn(os.path.join(MODULE_DIR, "%s.yaml" % filename))
    return config


def _set_up_relative_directory(directory, functional, calculation):
    # Set up a calculation directory for a specific functional and calculation

    directory = os.path.join(os.path.abspath(directory), functional[0])
    if functional[0] == "pbeu":
        directory += "_" + "".join(k + str(functional[1]["LDAUU"][k]) for k
                                   in functional[1]["LDAUU"].keys())
    directory += "_" + calculation

    return directory


def _set_up_functional_params(functional):
    """
    Set up the vasp_input_params based on the functional and some other conventions.

    Args:
        functional (tuple): Tuple with the functional details. The first element
            contains a string that indicates the functional used ("pbe", "hse", ...),
            whereas the second element contains a dictionary that allows the user
            to specify the various functional tags.

    Returns:
        dict: dictionary with the standard vasp input parameters.

    """
    vasp_input_params = {"user_incar_settings": {}}

    # Set up the functional
    if functional[0] != "pbe":
        functional_config = _load_yaml_config(functional[0] + "Set")
        functional_config["INCAR"].update(functional[1])
        vasp_input_params["user_incar_settings"].update(functional_config["INCAR"])

    return vasp_input_params


def get_wf_optimize(structure, directory, functional=("pbe", {}),
                    is_metal=False, in_custodian=False, number_nodes=None):
    """
    Set up a geometry optimization workflow for a bulk structure.

    Args:
        structure (Structure): Input Geometry.
        directory (str): Directory in which the geometry optimization should be
            performed.
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

    """
    # Add number of nodes to spec, or "none"
    if number_nodes is not None and number_nodes != 0:
        spec = {"_fworker": str(number_nodes) + "nodes"}
    else:
        spec = {}

    # --> Set up the geometry optimization
    vasp_input_params = _set_up_functional_params(functional)
    spec.update(
        {"_launch_dir": _set_up_relative_directory(directory, functional,
                                                   "optimize"),
         "_pass_job_info": True})

    # For metals, use Methfessel Paxton smearing
    if is_metal:
        vasp_input_params["user_incar_settings"].update(
            {"ISMEAR": 2, "SIGMA": 0.2}
        )

    # Set up the geometry optimization Firework
    optimize_fw = OptimizeFW(structure=structure,
                             vasp_input_params=vasp_input_params,
                             in_custodian=in_custodian,
                             spec=spec)

    # Set up a clear name for the workflow
    workflow_name = str(structure.composition.reduced_formula).replace(" ", "")
    workflow_name += " " + str(functional)

    # Create the workflow
    return Workflow(fireworks=[optimize_fw, ],
                    name=workflow_name)


def get_wf_energy(structure, directory, functional=("pbe", {}),
                  is_metal=False, in_custodian=False, number_nodes=None):
    """
    Set up an accurate energy workflow for a bulk structure. Starts by optimizing
    the geometry and then does a static calculation.

    Args:
        structure (Structure): Input geometry.
        directory (str): Directory in which the geometry optimization should be
            performed.
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

    """
    # Add number of nodes to spec, or "none"
    if number_nodes is not None and number_nodes != 0:
        spec = {"_fworker": str(number_nodes) + "nodes"}
    else:
        spec = {}

    # --> Set up the geometry optimization
    vasp_input_params = _set_up_functional_params(functional)
    spec.update(
        {"_launch_dir": _set_up_relative_directory(directory, functional,
                                                   "optimize"),
         "_pass_job_info": True})

    # For metals, use Methfessel Paxton smearing
    if is_metal:
        vasp_input_params["user_incar_settings"].update(
            {"ISMEAR": 2, "SIGMA": 0.2}
        )

    # Set up the geometry optimization Firework
    optimize_fw = OptimizeFW(structure=structure,
                             vasp_input_params=vasp_input_params,
                             in_custodian=in_custodian,
                             spec=spec)

    # -> Set up the static calculation
    vasp_input_params = _set_up_functional_params(functional)
    spec.update({"_launch_dir": _set_up_relative_directory(directory, functional,
                                                           "static")})
    # Set up the static Firework
    static_fw = StaticFW(vasp_input_params=vasp_input_params,
                         parents=optimize_fw,
                         in_custodian=in_custodian,
                         spec=spec)

    # Set up a clear name for the workflow
    workflow_name = str(structure.composition.reduced_formula).replace(" ", "")
    workflow_name += " " + str(functional)

    # Create the workflow
    return Workflow(fireworks=[optimize_fw, static_fw],
                    name=workflow_name)


def get_wf_optics(structure, directory, functional=("pbe", {}), k_resolution=None,
                  is_metal=False, in_custodian=False, number_nodes=None):
    """
    Set up a workflow to calculate the frequency dependent dielectric matrix.
    Starts with a geometry optimization.

    Args:
        structure (Structure): Input geometry.
        directory (str): Directory in which the optics calculation should be
            performed.
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
    # Add number of nodes to spec, or "none"
    if number_nodes is not None and number_nodes != 0:
        spec = {"_fworker": str(number_nodes) + "nodes"}
    else:
        spec = {}

    # 1. Set up the geometry optimization
    vasp_input_params = _set_up_functional_params(functional)
    spec.update(
        {"_launch_dir": _set_up_relative_directory(directory, functional,
                                                   "optimize")})
    # For metals, use Methfessel Paxton smearing
    if is_metal:
        vasp_input_params["user_incar_settings"].update(
            {"ISMEAR": 2, "SIGMA": 0.2}
        )

    # Set up the Firework
    optimize_fw = OptimizeFW(structure=structure,
                             vasp_input_params=vasp_input_params,
                             in_custodian=in_custodian,
                             spec=spec)

    # 2. Set up the optics calculation
    vasp_input_params = _set_up_functional_params(functional)
    spec.update({"_launch_dir": _set_up_relative_directory(directory, functional,
                                                           "optics")})

    # For metals, use a good amount of Gaussian smearing
    if is_metal:
        vasp_input_params["user_incar_settings"].update(
            {"ISMEAR": 0, "SIGMA": 0.3}
        )
        k_resolution = k_resolution or 0.05
    else:
        k_resolution = k_resolution or 0.1

    # Add the requested k-point resolution to the input parameters
    kpt_divisions = [round(l / k_resolution + 0.5) for l in
                     structure.lattice.reciprocal_lattice.lengths]

    vasp_input_params["user_kpoints_settings"] = {"length": kpt_divisions}

    # Set up the geometry optimization Firework
    optics_fw = OpticsFW(
        parents=optimize_fw,
        vasp_input_params=vasp_input_params,
        in_custodian=in_custodian,
        spec=spec
    )

    # Set up a clear name for the workflow
    workflow_name = str(structure.composition.reduced_formula).replace(" ", "")
    workflow_name += " " + str(functional)

    # Create the workflow
    return Workflow(fireworks=[optimize_fw, optics_fw],
                    links_dict={optimize_fw: [optics_fw]},
                    name=workflow_name)


def get_wf_slab_optimize(slab, directory, user_slab_settings,
                         functional=("pbe", {}), is_metal=False,
                         in_custodian=False, number_nodes=None):
    """
    Set up a slab geometry optimization workflow.

    Args:
        slab (Qslab): Slab for which to set up the geometry optimization workflow.
        directory (str): Directory in which the geometry optimization should be
            performed.
        user_slab_settings (dict): Allows the user to specify the selective
                dynamics of the slab geometry optimization. These are passed to
                the SlabOptimizeSet.fix_slab_bulk() commands as kwargs.
        functional (tuple): Tuple with the functional details. The first element
            contains a string that indicates the functional used ("pbe", "hse", ...),
            whereas the second element contains a dictionary that allows the user
            to specify the various functional tags.
        is_metal (bool): Flag that indicates whether the material for which the
            geometry optimization should be performed is metallic. Determines the
            smearing method used.
        in_custodian (bool): Flag that indicates whether the calculation should be
            run inside a Custodian.
        number_nodes (int): Number of nodes that should be used for the calculations.
            Is required to add the proper `_category` to the Firework generated, so
            it is picked up by the right Fireworker.

    """
    vasp_input_params = _set_up_functional_params(functional)
    spec = {"_launch_dir": directory}

    # For metals, use Methfessel Paxton smearing
    if is_metal:
        vasp_input_params["user_incar_settings"].update(
            {"ISMEAR": 2, "SIGMA": 0.2}
        )

    # Add number of nodes to spec, or "none"
    if number_nodes is not None and number_nodes != 0:
        spec.update({"_fworker": str(number_nodes) + "nodes"})

    # Set up the geometry optimization Firework
    optimize_fw = SlabOptimizeFW(slab=slab,
                                 user_slab_settings=user_slab_settings,
                                 vasp_input_params=vasp_input_params,
                                 in_custodian=in_custodian,
                                 spec=spec)

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
    Set up a slab DOS workflow. Starts with a geometry optimization.

    Args:
        slab (Qslab): Slab for which to set up the DOS workflow.
        directory (str): Directory in which the geometry optimization should be
            performed.
        functional (tuple): Tuple with the functional details. The first element
            contains a string that indicates the functional used ("pbe", "hse", ...),
            whereas the second element contains a dictionary that allows the user
            to specify the various functional tags.
        k_resolution (float): Resolution of the k-mesh, i.e. distance between two
            k-points along each reciprocal lattice vector. Note that for a slab
            calculation we always only consider one point in the c-direction.
        calculate_locpot (bool): Whether to calculate the the local potential,
            e.g. to determine the work function.
        in_custodian (bool): Flag that indicates whether the calculation should be
            run inside a Custodian.
        number_nodes (int): Number of nodes that should be used for the calculations.
            Is required to add the proper `_category` to the Firework generated, so
            it is picked up by the right Fireworker.

    """
    vasp_input_params = _set_up_functional_params(functional)

    # Calculate the local potential if requested (e.g. for the work function)
    if calculate_locpot:
        vasp_input_params["user_incar_settings"].update(
            {"LVTOT": True, "LVHAR": True}
        )

    spec = {"_launch_dir": directory}

    # Add number of nodes to spec, or "none"
    if number_nodes is not None and number_nodes != 0:
        spec.update({"_fworker": str(number_nodes) + "nodes"})

    # Add the requested k-point resolution to the input parameters
    kpt_divisions = [round(l / k_resolution + 0.5) for l in
                     slab.lattice.reciprocal_lattice.lengths]

    vasp_input_params["user_kpoints_settings"] = {"length": kpt_divisions}

    # Set up the geometry optimization Firework
    dos_fw = SlabDosFW(
        slab=slab,
        vasp_input_params=vasp_input_params,
        in_custodian=in_custodian,
        spec=spec
    )

    # Set up a clear name for the workflow
    workflow_name = str(slab.composition.reduced_formula).replace(" ", "")
    workflow_name += " " + str(slab.miller_index)
    workflow_name += " " + str(functional)

    # Create the workflow
    return Workflow(fireworks=[dos_fw, ], name=workflow_name)


# def get_wf_quotas(bulk, slab_list, directory, functional=("pbe", {}),
#                   base_k_resolution=0.1, is_metal=False,
#                   in_custodian=False, number_nodes=None):
#     """
#     Generate a full QUOTAS worfklow, i.e. one that:
#
#     1. Optimizes the bulk and calculates the optical properties.
#     2. Optimizes the slabs in 'slab_list' and then calculates the DOS and work
#     function.
#
#     Args:
#         bulk:
#         slab_list:
#         functional:
#         base_k_resolution:
#         number_nodes:
#
#     """
#
#     optics_k_resolution = base_k_resolution / 3 # TODO Improve this
#     dos_k_resolution = base_k_resolution / 2
#
#     # Set up the directory for the bulk calculations
#     bulk_dir = os.path.join(directory, "bulk")
#
#     optics_wf = get_wf_optics(
#         directory=directory, structure=bulk, functional=functional,
#         k_resolution=optics_k_resolution, is_metal=is_metal,
#         in_custodian=in_custodian, number_nodes=number_nodes
#     )
#
#     for slab_dict in slab_list:
#         # Set up the directories for the slab calculations
#
#         slab_dir = str(slab_dict["slab"].miller_index).strip("()").replace(", ", "")
#
#         slab_optimize_dir = _set_up_relative_directory(
#             directory=os.path.join(directory, slab_dir),
#             functional=functional,
#             calculation="optimize"
#         )
#
#         slab_dos_dir = _set_up_relative_directory(
#             directory=os.path.join(directory, slab_dir),
#             functional=functional,
#             calculation="dos"
#         )
#
#         slab_optimize = SlabOptimizeFW(
#             slab=slab_dict["slab"],
#             directory=slab_optimize_dir,
#             fix_part="center",
#             fix_thickness=slab_dict["fix_thickness"],
#             functional=functional,
#             is_metal=is_metal,
#             in_custodian=in_custodian,
#             number_nodes=number_nodes
#         )
#
#         slab_dos = SlabDosFW(
#             slab=os.path.join(slab_optimize_dir, "final_slab.json"),
#             directory=slab_dos_dir,
#             functional=functional,
#             k_resolution=dos_k_resolution,
#             calculate_locpot=True,
#             in_custodian=in_custodian,
#             number_nodes=number_nodes
#         )
#
#         fireworks.extend([slab_optimize, slab_dos])
#         links_dict.update({slab_optimize: [slab_dos]})
#
#     # Set up a clear name for the workflow
#     workflow_name = str(bulk.composition.reduced_formula).replace(" ", "")
#     workflow_name += " - QUOTAS - "
#     workflow_name += " " + str(functional)
#
#     return Workflow(fireworks=fireworks, links_dict=links_dict, name=workflow_name)
