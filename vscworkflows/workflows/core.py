# coding: utf8
# Copyright (c) Marnik Bercx, University of Antwerp
# Distributed under the terms of the MIT License

import os
import numpy as np

from string import ascii_lowercase
from fireworks import Workflow
from monty.serialization import loadfn
from pymatgen.core.surface import Slab, SlabGenerator
from vscworkflows.misc import QSlab

from vscworkflows.fireworks.core import StaticFW, OptimizeFW, OpticsFW, \
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
            contains a string that indicates the functional used ("pbe", "hse06",
            ...),
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
                    is_metal=False, in_custodian=False, number_nodes=None,
                    auto_parallelization=False):
    """
    Set up a geometry optimization workflow for a bulk structure.

    Args:
        structure (Structure): Input Geometry.
        directory (str): Directory in which the geometry optimization should be
            performed.
        functional (tuple): Tuple with the functional details. The first element
            contains a string that indicates the functional used ("pbe", "hse06",
            ...),
            whereas the second element contains a dictionary that allows the user
            to specify additional INCAR tags.
        is_metal (bool): Flag that indicates the material being studied is a
            metal, which changes the smearing from Gaussian (0.05 eV) to second
            order Methfessel-Paxton of 0.2 eV.
        in_custodian (bool): Flag that indicates whether the calculation should be
            run inside a Custodian.
        number_nodes (int): Number of nodes that should be used for the calculations.
            Is required to add the proper `_fworker` to the Firework spec, so
            it is picked up by a Fireworker running in a job with the specified
            number of nodes.
        auto_parallelization (bool): Automatically parallelize the calculation
            using the VaspParallelizationTask.

    """
    # Add number of nodes to spec, or "none"
    if number_nodes is not None and number_nodes != 0:
        spec = {"_fworker": str(number_nodes) + "nodes"}
    else:
        spec = {}

    # --> Set up the geometry optimization
    vasp_input_params = _set_up_functional_params(functional)
    spec.update({"_launch_dir": directory})

    # For metals, use Methfessel Paxton smearing
    if is_metal:
        vasp_input_params["user_incar_settings"].update(
            {"ISMEAR": 2, "SIGMA": 0.2}
        )

    # Set up the geometry optimization Firework
    optimize_fw = OptimizeFW(structure=structure,
                             vasp_input_params=vasp_input_params,
                             custodian=in_custodian,
                             spec=spec,
                             auto_parallelization=auto_parallelization)

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
        directory (str): Directory in which the workflow should be set up.
        functional (tuple): Tuple with the functional details. The first element
            contains a string that indicates the functional used ("pbe", "hse06",
            ...),
            whereas the second element contains a dictionary that allows the user
            to specify the various functional INCAR tags.
        is_metal (bool): Flag that indicates the material being studied is a
            metal, which changes the smearing of the geometry optimization from
            Gaussian ( 0.05 eV) to second order Methfessel-Paxton of 0.2 eV.
        in_custodian (bool): Flag that indicates whether the calculation should be
            run inside a Custodian.
        number_nodes (int): Number of nodes that should be used for the calculations.
            Is required to add the proper `_fworker` to the Firework spec, so
            it is picked up by a Fireworker running in a job with the specified
            number of nodes.

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
                             custodian=in_custodian,
                             spec=spec)

    # -> Set up the static calculation
    vasp_input_params = _set_up_functional_params(functional)
    spec.update({"_launch_dir": _set_up_relative_directory(directory, functional,
                                                           "static")})
    # Set up the static Firework
    static_fw = StaticFW(vasp_input_params=vasp_input_params,
                         parents=optimize_fw,
                         custodian=in_custodian,
                         spec=spec)

    # Set up a clear name for the workflow
    workflow_name = str(structure.composition.reduced_formula).replace(" ", "")
    workflow_name += " " + str(functional)

    # Create the workflow
    return Workflow(fireworks=[optimize_fw, static_fw],
                    name=workflow_name)


def get_wf_optics(structure, directory, functional=("pbe", {}), k_resolution=None,
                  is_metal=False, user_incar_settings=None, in_custodian=False,
                  number_nodes=None, auto_parallelization=False):
    """
    Set up a workflow to calculate the frequency dependent dielectric matrix.
    Starts with a geometry optimization.

    Args:
        structure (Structure): Input geometry.
        directory (str): Directory in which the workflow should be set up.
        functional (tuple): Tuple with the functional details. The first element
            contains a string that indicates the functional used ("pbe", "hse06",
            ...),
            whereas the second element contains a dictionary that allows the user
            to specify the various functional INCAR tags.
        k_resolution (float): Resolution of the k-mesh, i.e. distance between two
            k-points along each reciprocal lattice vector.
        is_metal (bool): Flag that indicates the material being studied is a
            metal, which changes the smearing from Gaussian (0.05 eV) to second
            order Methfessel-Paxton of 0.2 eV; the optics calculation will use a
            generous gaussian smearing of 0.3 eV instead of the tetrahedron method.
        user_incar_settings (dict): User INCAR settings. This allows a user
                to override INCAR settings, e.g., setting a different MAGMOM for
                various elements or species, or specify parallelization settings (
                KPAR, NPAR, ...). Note that the settings specified here will
                override the INCAR settings for ALL fireworks of the workflow.
        in_custodian (bool): Flag that indicates whether the calculation should be
            run inside a Custodian.
        number_nodes (int): Number of nodes that should be used for the calculations.
            Is required to add the proper `_fworker` to the Firework spec, so
            it is picked up by a Fireworker running in a job with the specified
            number of nodes.
        auto_parallelization (bool): Automatically parallelize the calculation
            using the VaspParallelizationTask.

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

    # Override the INCAR settings with the user specifications
    if user_incar_settings is not None:
        vasp_input_params["user_incar_settings"].update(user_incar_settings)

    # Set up the Firework
    optimize_fw = OptimizeFW(structure=structure,
                             vasp_input_params=vasp_input_params,
                             custodian=in_custodian,
                             spec=spec,
                             auto_parallelization=auto_parallelization)

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

    vasp_input_params["user_kpoints_settings"] = {"k_resolution": k_resolution}

    # Override the INCAR settings with the user specifications
    if user_incar_settings is not None:
        vasp_input_params["user_incar_settings"].update(user_incar_settings)

    # Set up the geometry optimization Firework
    optics_fw = OpticsFW(
        parents=optimize_fw,
        vasp_input_params=vasp_input_params,
        custodian=in_custodian,
        spec=spec,
        auto_parallelization=auto_parallelization
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
            contains a string that indicates the functional used
            ("pbe", "hse06", ...), whereas the second element contains a
            dictionary that allows the user to specify the various functional
            INCAR tags.
        is_metal (bool): Flag that indicates the material being studied is a
            metal, which changes the smearing of the geometry optimization from
            Gaussian ( 0.05 eV) to second order Methfessel-Paxton of 0.2 eV.
        in_custodian (bool): Flag that indicates whether the calculation should be
            run inside a Custodian.
        number_nodes (int): Number of nodes that should be used for the calculations.
            Is required to add the proper `_fworker` to the Firework spec, so
            it is picked up by a Fireworker running in a job with the specified
            number of nodes.

    """
    # Add number of nodes to spec, or "none"
    if number_nodes is not None and number_nodes != 0:
        spec = {"_fworker": str(number_nodes) + "nodes"}
    else:
        spec = {}

    # Set up the geometry optimization Firework
    vasp_input_params = _set_up_functional_params(functional)
    spec.update({"_launch_dir": directory})

    # For metals, use Methfessel Paxton smearing
    if is_metal:
        vasp_input_params["user_incar_settings"].update(
            {"ISMEAR": 2, "SIGMA": 0.2}
        )

    optimize_fw = SlabOptimizeFW(slab=slab,
                                 user_slab_settings=user_slab_settings,
                                 vasp_input_params=vasp_input_params,
                                 custodian=in_custodian,
                                 spec=spec)

    # Set up a clear name for the workflow
    workflow_name = str(slab.composition.reduced_formula).replace(" ", "")
    workflow_name += " " + str(slab.miller_index)
    workflow_name += " " + str(functional)

    # Create the workflow
    return Workflow(fireworks=[optimize_fw, ],
                    name=workflow_name)


def get_wf_slab_dos(slab, directory, user_slab_settings=None,
                    functional=("pbe", {}), k_resolution=0.1,
                    calculate_locpot=False, is_metal=False,
                    user_incar_settings=None, in_custodian=False,
                    number_nodes=None, auto_parallelization=False):
    """
    Set up a slab DOS workflow. Starts with a geometry optimization.

    Args:
        slab (Qslab): Slab for which to set up the DOS workflow.
        directory (str): Directory in which the workflow should be set up.
        user_slab_settings (dict): Allows the user to specify the selective
            dynamics of the slab geometry optimization. These are passed to
            the SlabOptimizeSet.fix_slab_bulk() commands as kwargs.
        functional (tuple): Tuple with the functional details. The first element
            contains a string that indicates the functional used
            ("pbe", "hse06", ...), whereas the second element contains a
            dictionary that allows the user to specify the various functional
            INCAR tags.
        k_resolution (float): Resolution of the k-mesh, i.e. distance between two
            k-points along each reciprocal lattice vector. Note that for a slab
            calculation we always only consider one point in the c-direction.
        calculate_locpot (bool): Whether to calculate the the local potential,
            e.g. to determine the work function.
        is_metal (bool): Flag that indicates the material being studied is a
            metal, which changes the smearing of the geometry optimization from
            Gaussian ( 0.05 eV) to second order Methfessel-Paxton of 0.2 eV.
        user_incar_settings (dict): User INCAR settings. This allows a user to
            override INCAR settings, e.g., setting a different MAGMOM for various
            elements or species, or specify parallelization settings
            (KPAR, NPAR, ...). Note that the settings specified here will
            override the INCAR settings for ALL fireworks of the workflow.
        in_custodian (bool): Flag that indicates whether the calculations should be
            run inside a Custodian.
        number_nodes (int): Number of nodes that should be used for the calculations.
            Is required to add the proper `_fworker` to the Firework spec, so
            it is picked up by a Fireworker running in a job with the specified
            number of nodes.
        auto_parallelization (bool): Automatically parallelize the calculation
            using the VaspParallelizationTask.

    """
    # Add number of nodes to spec, or "none"
    if number_nodes is not None and number_nodes != 0:
        spec = {"_fworker": str(number_nodes) + "nodes"}
    else:
        spec = {}

    # --> Set up the geometry optimization Firework
    vasp_input_params = _set_up_functional_params(functional)
    spec.update(
        {"_launch_dir": _set_up_relative_directory(directory, functional,
                                                   "optimize")}
    )

    # For metals, use Methfessel Paxton smearing
    if is_metal:
        vasp_input_params["user_incar_settings"].update(
            {"ISMEAR": 2, "SIGMA": 0.2}
        )
    if user_incar_settings is not None:
        vasp_input_params["user_incar_settings"].update(user_incar_settings)

    optimize_fw = SlabOptimizeFW(slab=slab,
                                 user_slab_settings=user_slab_settings,
                                 vasp_input_params=vasp_input_params,
                                 custodian=in_custodian,
                                 spec=spec,
                                 auto_parallelization=auto_parallelization)

    # --> Set up the DOS Firework
    vasp_input_params = _set_up_functional_params(functional)

    # Calculate the local potential if requested (e.g. for the work function)
    if calculate_locpot:
        vasp_input_params["user_incar_settings"].update(
            {"LVTOT": True, "LVHAR": True}
        )
    spec.update(
        {"_launch_dir": _set_up_relative_directory(directory, functional,
                                                   "dos")}
    )
    vasp_input_params["user_kpoints_settings"] = {"k_resolution": k_resolution}

    if user_incar_settings is not None:
        vasp_input_params["user_incar_settings"].update(user_incar_settings)

    # Set up the geometry optimization Firework
    dos_fw = SlabDosFW(
        vasp_input_params=vasp_input_params,
        parents=optimize_fw,
        custodian=in_custodian,
        spec=spec,
        auto_parallelization=auto_parallelization
    )

    # Set up a clear name for the workflow
    workflow_name = str(slab.composition.reduced_formula).replace(" ", "")
    workflow_name += " " + str(slab.miller_index)
    workflow_name += " " + str(functional)

    # Create the workflow
    return Workflow(fireworks=[optimize_fw, dos_fw], name=workflow_name)


def get_wf_quotas(bulk, slab_list, directory, functional=("pbe", {}),
                  k_resolution=0.05, is_metal=False,
                  in_custodian=False, number_nodes=None):
    """
    Generate a full QUOTAS worfklow, i.e. one that:

    1. Optimizes the bulk and calculates the optical properties.
    2. Optimizes the slabs in 'slab_list' and then calculates the DOS and work
    function.

    Args:
        bulk (Structure): Input bulk geometry.
        slab_list (list): A list of dictionaries that specify the slabs to be
            included, as well as the settings to be used for each slab. Here is
            an overview of the mandatory keys:

                "slab" - Can be either a QSlab or a list/str that specifies the
                    miller indices of the slab surface. In case only the miller
                    indices are provided, the user must also supply the
                    "min_slab_size" and "min_vacuum_size", which specify the
                    minimum thickness of the slab and vacuum layer in angstrom.
                "user_slab_settings" (dict) - Settings that will be passed to the
                    slab optimization firework. The most important key of this
                    dict is "free_layers", which specifies the number of
                    surface layers to optimize.

        directory (str): Directory in which the workflow should be set up.
        functional (tuple): Tuple with the functional details. The first element
            contains a string that indicates the functional used
            ("pbe", "hse06", ...), whereas the second element contains a
            dictionary that allows the user to specify the various functional
            INCAR tags.
        k_resolution (float): Resolution of the k-mesh, i.e. distance
            between two k-points along each reciprocal lattice vector. Note that
            for a slab calculation we always only consider one point in the
            c-direction.
        is_metal (bool): Flag that indicates the material being studied is a
            metal, which changes the smearing from Gaussian (0.05 eV) to second
            order Methfessel-Paxton of 0.2 eV; the optics calculation will use a
            generous gaussian smearing of 0.3 eV instead of the tetrahedron method.
        in_custodian (bool): Flag that indicates whether the calculation should be
            run inside a Custodian.
        number_nodes (int): Number of nodes that should be used for the calculations.
            Is required to add the proper `_fworker` to the Firework spec, so
            it is picked up by a Fireworker running in a job with the specified
            number of nodes.

    """
    fireworks = list()

    # Set up the directory for the bulk calculations
    bulk_dir = os.path.join(directory, "bulk")

    fireworks.extend(get_wf_optics(
        directory=bulk_dir, structure=bulk, functional=functional,
        k_resolution=k_resolution, is_metal=is_metal,
        in_custodian=in_custodian, number_nodes=number_nodes
    ).fws)

    for slab_dict in slab_list:

        if isinstance(slab_dict["slab"], Slab):

            slab = slab_dict["slab"]

            # Set up the directory for the slab calculations
            slab_dir = "".join([str(c) for c in slab.miller_index])
            slab_dir = os.path.join(directory, slab_dir)

            fireworks.extend(get_wf_slab_dos(
                slab=slab, directory=slab_dir, functional=functional,
                k_resolution=k_resolution,
                user_slab_settings=slab_dict["user_slab_settings"],
                calculate_locpot=True, is_metal=is_metal, in_custodian=in_custodian,
                number_nodes=number_nodes
            ).fws)

        elif isinstance(slab_dict["slab"], list) \
                or isinstance(slab_dict["slab"], str):

            miller_index = [int(c) for c in slab_dict["slab"]]

            try:
                slabgen = SlabGenerator(
                    initial_structure=bulk,
                    miller_index=miller_index,
                    min_slab_size=slab_dict["min_slab_size"],
                    min_vacuum_size=slab_dict["min_vacuum_size"]
                )
            except KeyError:
                raise ValueError("Either min_slab_size or min_vacuum_size were not"
                                 "defined in the slab dictionary for " +
                                 slab_dict["slab"] + ".")

            slab_terminations = slabgen.get_slabs()

            if len(slab_terminations) == 1:
                slab = QSlab.from_slab(slab_terminations[0])

                # Set up the directory for the slab calculations
                slab_dir = "".join([str(c) for c in miller_index])
                slab_dir = os.path.join(directory, slab_dir)

                fireworks.extend(get_wf_slab_dos(
                    slab=slab, directory=slab_dir, functional=functional,
                    k_resolution=k_resolution,
                    user_slab_settings=slab_dict["user_slab_settings"],
                    calculate_locpot=True, is_metal=is_metal,
                    in_custodian=in_custodian,
                    number_nodes=number_nodes
                ).fws)

            if len(slab_terminations) > 1:
                print("Multiple slab terminations found. Adding workflow for each "
                      "termination...")

                for slab, letter in zip(slab_terminations, ascii_lowercase):
                    slab = QSlab.from_slab(slab)

                    # Set up the directory for the slab calculations
                    slab_dir = "".join([str(c) for c in miller_index]) + "_" + letter
                    slab_dir = os.path.join(directory, slab_dir)

                    fireworks.extend(get_wf_slab_dos(
                        slab=slab, directory=slab_dir, functional=functional,
                        k_resolution=k_resolution,
                        user_slab_settings=slab_dict["user_slab_settings"],
                        calculate_locpot=True, is_metal=is_metal,
                        in_custodian=in_custodian,
                        number_nodes=number_nodes
                    ).fws)

    # Set up a clear name for the workflow
    workflow_name = str(bulk.composition.reduced_formula).replace(" ", "")
    workflow_name += " - QUOTAS - "
    workflow_name += " " + str(functional)

    return Workflow(fireworks=fireworks, name=workflow_name)


def get_wf_parallel(structure, directory, nodes, nbands=None,
                    functional=("pbe", {}), user_kpoints_settings=None,
                    user_incar_settings=None, handlers=None, cores_per_node=28,
                    kpar_range=None, min_npar=1):
    # Set defaults
    user_kpoints_settings = user_kpoints_settings or {"reciprocal_density": 300}
    user_incar_settings = user_incar_settings or {}
    handlers = handlers or []
    n_cores = int(cores_per_node * nodes)
    kpar_range = kpar_range or [1, n_cores]

    fw_list = []

    suitable_kpars = np.array(
        [i for i in range(kpar_range[0], kpar_range[1] + 1)
         if n_cores % i == 0]
    )

    for kpar in suitable_kpars:

        cores_per_k = n_cores / kpar

        suitable_npars = np.array(
            [i for i in range(min_npar, int(cores_per_k) + 1)
             if cores_per_k % i == 0]
        )

        if nbands is not None:
            suitable_npars = [npar for npar in suitable_npars if nbands % npar == 0]

        for npar in suitable_npars:
            spec = {}
            # Set up the static calculation
            spec.update({"_launch_dir": os.path.join(
                directory, str(nodes) + "nodes", str(kpar) + "kpar",
                (str(npar) + "npar")
            )})
            spec.update({"_fworker": str(nodes) + "nodes"})

            vasp_input_params = _set_up_functional_params(functional)
            vasp_input_params["user_kpoints_settings"] = user_kpoints_settings
            vasp_input_params["user_incar_settings"].update(user_incar_settings)
            vasp_input_params["user_incar_settings"].update(
                {"KPAR": kpar, "NPAR": npar})
            vasp_input_params["force_gamma"] = True

            # Set up the Firework and add it to the list
            fw_list.append(
                StaticFW(structure=structure,
                         vasp_input_params=vasp_input_params,
                         spec=spec,
                         custodian=handlers)
            )

    workflow_name = ("Parallel-Test: " + structure.composition.reduced_formula
                     + " " + str(nodes) + "nodes.")

    # Create the workflow
    return Workflow(fireworks=fw_list, name=workflow_name)
