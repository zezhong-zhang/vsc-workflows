# coding: utf8
# Copyright (c) Marnik Bercx, University of Antwerp
# Distributed under the terms of the MIT License

import os

from fireworks import PyTask, Firework

from atomate.vasp.firetasks import WriteVaspFromIOSet

from vscworkflows.workflow.firetasks import VaspTask, CustodianTask, \
    VaspWriteFinalStructureTask, VaspWriteFinalSlabTask, VaspParallelizationTask, \
    PulayTask, WriteVaspFromIOSet
from vscworkflows.setup.sets import BulkRelaxSet

"""
Package that contains all the fireworks to construct Workflows.

"""

__author__ = "Marnik Bercx"
__copyright__ = "Copyright 2019, Marnik Bercx, University of Antwerp"
__version__ = "pre-alpha"
__maintainer__ = "Marnik Bercx"
__email__ = "marnik.bercx@uantwerpen.be"
__date__ = "Jun 2019"


class StaticFW(Firework):

    def __init__(self, structure=None, name="Static calculation",
                 vasp_input_params=None, parents=None,
                 in_custodian=False, spec=None):
        """
        Create a FireWork for performing a static calculation.

        Args: #TODO
            structure: pymatgen.Structure OR path to structure file for which to run
                the static calculation.
            in_custodian (bool): Flag that indicates whether the calculation should be
                run inside a Custodian.

        Returns:
            Firework: A firework that represents a static calculation.

        """

        tasks = list()

        vasp_input_params = vasp_input_params or {}
        spec = spec if spec is not None else {}

        if structure is not None:
            tasks.append(WriteVaspFromIOSet(
                structure=structure,
                vasp_input_set=BulkRelaxSet(structure, **vasp_input_params)
            ))
        elif parents is not None:  # TODO What if multiple parents?
            tasks.append(WriteVaspFromIOSet(
                parent=parents,
                vasp_input_set="vscworkflows.setup.sets.BulkStaticSet"
            ))
        else:
            raise ValueError("You must provide either an input structure or "
                             "parent firework to StaticFW!")

        # Configure the parallelization settings
        tasks.append(VaspParallelizationTask())

        # Run the calculation
        if in_custodian:
            tasks.append(CustodianTask())
        else:
            tasks.append(VaspTask())

        # Combine the two FireTasks into one FireWork
        super().__init__(
            tasks=tasks, name=name, spec=spec
        )


class OptimizeFW(Firework):

    def __init__(self, structure, name="Geometry Optimization",
                 vasp_input_params=None, parents=None, in_custodian=False,
                 fw_action=None, spec=None):
        """
        Initialize a Firework for a geometry optimization.

        Args:
            structure: structure: pymatgen.Structure OR path to the structure file.
            name (str): # TODO
            vasp_input_params (dict):
            parents (Firework):
            in_custodian (bool): Flag that indicates whether the calculation should be
                run inside a Custodian.
            fw_action (fireworks.FWAction): FWAction to return after the final
                PulayTask is completed.

        """
        tasks = list()

        vasp_input_params = vasp_input_params or {}
        spec = spec if spec is not None else {}

        tasks.append(WriteVaspFromIOSet(
            structure=structure,
            vasp_input_set=BulkRelaxSet(structure, **vasp_input_params)
        ))

        # Configure the parallelization settings
        tasks.append(VaspParallelizationTask())

        # Run the calculation
        if in_custodian:
            tasks.append(CustodianTask())
        else:
            tasks.append(VaspTask())

        # Write the final structure to a json file for subsequent calculations
        # tasks.append(VaspWriteFinalStructureTask()) TODO

        # Check the Pulay stress
        tasks.append(
            PulayTask(in_custodian=in_custodian,
                      fw_action=fw_action,
                      spec=spec)
        )

        # Combine the FireTasks into one FireWork
        super().__init__(tasks=tasks,
                         parents=parents,
                         name=name,
                         spec=spec)


class OpticsFW(StaticFW):

    def __init__(self, structure=None, name="Optics calculation",
                 vasp_input_params=None, parents=None,
                 in_custodian=False, spec=None):
        """
        Initialize a Firework for a geometry optimization.

        Args: # TODO
            structure: pymatgen.Structure OR path to structure file for which to run
                the geometry optimization.

        """
        # Default input parameters
        optics_input_params = {
            "user_incar_settings": {"LOPTICS": True, "NEDOS": 2000, "EDIFF": 1.0e-6},
            "user_kpoint_settings": {"reciprocal_density": 200}
        }
        # Update the defaults with the user specified input parameters
        for k, v in vasp_input_params:
            if k in optics_input_params.keys():
                optics_input_params[k].update(v)
            else:
                optics_input_params[k] = v

        super().__init__(structure=structure, name=name,
                         vasp_input_params=optics_input_params, parents=parents,
                         in_custodian=in_custodian, spec=spec)


class SlabOptimizeFW(Firework):

    def __init__(self, slab, directory, functional, fix_part,
                 fix_thickness, is_metal=False,
                 in_custodian=False, number_nodes=None):
        """
        Initialize a Firework for a geometry optimization.

        Args:
            slab: quotas.QSlab OR path to slab structure file for which to run
                the geometry optimization.
            directory (str): Directory in which the geometry optimization should be
                performed.
            functional (tuple): Tuple with the functional choices. The first element
                contains a string that indicates the functional used ("pbe", "hse", ...),
                whereas the second element contains a dictionary that allows the user
                to specify the various functional tags.
            fix_part (str): Which part of the slab to fix. Currently only allows for
                "center".
            fix_thickness (int): The thickness of the fixed part of the slab, expressed in
                number of layers.
            is_metal (bool): Flag that indicates the material being studied is a
                metal, which changes the smearing from Gaussian to second order
                Methfessel-Paxton of 0.2 eV.
            in_custodian (bool): Flag that indicates whether the calculation should be
                run inside a Custodian.
            number_nodes (int): Number of nodes that should be used for the calculations.
                Is required to add the proper `_category` to the Firework generated, so
                it is picked up by the right Fireworker.

        """
        tasks = list()

        # Set up the input files of the calculation
        tasks.append(
            PyTask(func="vscworkflows.setup.write_input.slab_optimize",
                   kwargs={"slab": slab,
                           "fix_part": fix_part,
                           "fix_thickness": fix_thickness,
                           "directory": directory,
                           "functional": functional,
                           "is_metal": is_metal})
        )

        # Configure the parallelization settings
        tasks.append(VaspParallelizationTask(directory=directory))

        # Run the calculation
        if in_custodian:
            tasks.append(CustodianTask(directory=directory))
        else:
            tasks.append(VaspTask(directory=directory))

        # Write the final slab to a json file for subsequent calculations
        tasks.append(VaspWriteFinalSlabTask(directory=directory))

        # Only add number of nodes to spec if specified
        firework_spec = {}
        if number_nodes is None or number_nodes == 0:
            firework_spec.update({"_category": "none"})
        else:
            firework_spec.update({"_category": str(number_nodes) + "nodes"})

        # Combine the FireTasks into one FireWork
        super().__init__(tasks=tasks,
                         name="Geometry optimization",
                         spec=firework_spec)


class SlabDosFW(Firework):

    def __init__(self, slab, directory, functional, k_resolution=0.1,
                 calculate_locpot=False, in_custodian=False,
                 number_nodes=None):
        """
        Firework for calculating the DOS and work function of a slab.

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
        in_custodian (bool): Flag that indicates whether the calculation should be
            run inside a Custodian.
        number_nodes (int): Number of nodes that should be used for the calculations.
            Is required to add the proper `_category` to the Firework generated, so
            it is picked up by the right Fireworker.

        """
        tasks = list()

        # Set up the DOS calculation, based on the structure found from the
        # geometry optimization.
        tasks.append(
            PyTask(func="vscworkflows.setup.write_input.slab_dos",
                   kwargs={
                       "slab": slab,
                       "directory": directory,
                       "functional": functional,
                       "k_resolution": k_resolution * 3,
                       "calculate_locpot": False,
                       "user_incar_settings": {"LCHARG": True, "EDIFF": 1e-3}
                   })
        )

        # Create the PyTask that runs the calculation
        if in_custodian:
            tasks.append(CustodianTask(directory=directory))
        else:
            tasks.append(VaspTask(directory=directory))

        tasks.append(
            PyTask(func="vscworkflows.setup.write_input.slab_dos",
                   kwargs={
                       "slab": slab,
                       "directory": directory,
                       "functional": functional,
                       "k_resolution": k_resolution,
                       "calculate_locpot": calculate_locpot,
                       "user_incar_settings": {"ICHARG": 1}
                   })
        )
        # Create the PyTask that runs the calculation
        if in_custodian:
            tasks.append(CustodianTask(directory=directory))
        else:
            tasks.append(VaspTask(directory=directory))

        # Write the final slab to a json file for subsequent calculations
        tasks.append(VaspWriteFinalSlabTask(directory=directory))

        # Only add number of nodes to spec if specified
        firework_spec = {}
        if number_nodes is None or number_nodes == 0:
            firework_spec.update({"_category": "none"})
        else:
            firework_spec.update({"_category": str(number_nodes) + "nodes"})

        super().__init__(tasks=tasks,
                         name="DOS Calculation",
                         spec=firework_spec)
