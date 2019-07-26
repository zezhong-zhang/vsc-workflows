# coding: utf8
# Copyright (c) Marnik Bercx, University of Antwerp
# Distributed under the terms of the MIT License

import os

from fireworks import PyTask, Firework

from vscworkflows.workflow.firetasks import VaspTask, CustodianTask, \
    VaspWriteFinalStructureTask, VaspWriteFinalSlabTask, VaspParallelizationTask, \
    PulayTask, WriteVaspFromIOSet
from vscworkflows.setup.sets import BulkStaticSet, BulkOptimizeSet, \
    SlabStaticSet, SlabOptimizeSet

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
                vasp_input_set=BulkStaticSet(structure, **vasp_input_params)
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

        tasks.append(WriteVaspFromIOSet(
            structure=structure,
            vasp_input_set=BulkOptimizeSet(structure, **vasp_input_params)
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
        vasp_input_params = vasp_input_params or {}

        # Default input parameters
        optics_input_params = {
            "user_incar_settings": {"LOPTICS": True, "NEDOS": 2000, "EDIFF": 1.0e-6},
            "user_kpoints_settings": {"reciprocal_density": 200}
        }
        # Update the defaults with the user specified input parameters
        for k, v in vasp_input_params.items():
            if k in optics_input_params.keys():
                optics_input_params[k].update(v)
            else:
                optics_input_params[k] = v

        super().__init__(structure=structure, name=name,
                         vasp_input_params=optics_input_params, parents=parents,
                         in_custodian=in_custodian, spec=spec)


class SlabStaticFW(Firework):

    def __init__(self, slab, name="Slab Static", vasp_input_params=None,
                 parents=None, in_custodian=False, spec=None):
        """
        #TODO

        Args:
            slab: quotas.QSlab OR path to slab structure file for which to run
            the DOS calculation.

        calculate_locpot (bool): Whether to calculate the the local potential, e.g. to
            determine the work function.
        in_custodian (bool): Flag that indicates whether the calculation should be
            run inside a Custodian.


        """
        tasks = list()
        vasp_input_params = vasp_input_params or {}

        if slab is not None:
            tasks.append(WriteVaspFromIOSet(
                structure=slab,
                vasp_input_set=SlabStaticSet(slab, **vasp_input_params)
            ))
        elif parents is not None:  # TODO What if multiple parents?
            tasks.append(WriteVaspFromIOSet(
                parent=parents,
                vasp_input_set="vscworkflows.setup.sets.SlabStaticSet"
            ))
        else:
            raise ValueError("You must provide either an input structure or "
                             "parent firework to SlabStaticFW!")

        # Configure the parallelization settings
        tasks.append(VaspParallelizationTask())

        # Create the PyTask that runs the calculation
        if in_custodian:
            tasks.append(CustodianTask())
        else:
            tasks.append(VaspTask())

        # Write the final slab to a json file for subsequent calculations
        # tasks.append(VaspWriteFinalSlabTask(directory=directory)) # TODO

        super().__init__(tasks=tasks,
                         name=name,
                         spec=spec)


class SlabOptimizeFW(Firework):

    def __init__(self, slab, name="Slab optimize", vasp_input_params=None,
                 user_slab_settings=None, parents=None, in_custodian=False,
                 spec=None):
        """
        Initialize a Firework for a geometry optimization.

        Args:
            slab: quotas.QSlab OR path to slab structure file for which to run
                the geometry optimization.


        """  # TODO Add intuitive way of including kpoint settings
        tasks = list()
        vasp_input_params = vasp_input_params or {}

        # Set up the input files of the calculation
        tasks.append(WriteVaspFromIOSet(
            vasp_input_set=SlabOptimizeSet(structure=slab,
                                           user_slab_settings=user_slab_settings,
                                           **vasp_input_params)
        ))

        # Configure the parallelization settings
        tasks.append(VaspParallelizationTask())

        # Run the calculation
        if in_custodian:
            tasks.append(CustodianTask())
        else:
            tasks.append(VaspTask())

        # Write the final slab to a json file for subsequent calculations
        # tasks.append(VaspWriteFinalSlabTask()) # TODO

        # Combine the FireTasks into one FireWork
        super().__init__(tasks=tasks, name=name, parents=parents, spec=spec)


class SlabDosFW(Firework):

    def __init__(self, slab, name="Slab optimize", vasp_input_params=None,
                 parents=None, in_custodian=False, spec=None):
        """
        Firework for calculating the DOS and work function of a slab.

        Args:
            slab: quotas.QSlab OR path to slab structure file for which to run
            the DOS calculation.

        calculate_locpot (bool): Whether to calculate the the local potential, e.g. to
            determine the work function.
        in_custodian (bool): Flag that indicates whether the calculation should be
            run inside a Custodian.


        """
        tasks = list()

        # Set up the input files of the low precision static calculation
        tasks.append(WriteVaspFromIOSet(
            vasp_input_set=SlabStaticSet(
                structure=slab,
                user_incar_settings={"LCHARG": True}
        )))

        # Create the PyTask that runs the calculation
        if in_custodian:
            tasks.append(CustodianTask())
        else:
            tasks.append(VaspTask())

        # Default input parameters
        dos_input_params = {
            "user_incar_settings": {"LOPTICS": True, "NEDOS": 2000,
                                    "EDIFF": 1.0e-6},
            "user_kpoints_settings": {"reciprocal_density": 200}
        }

        # Update the defaults with the user specified input parameters
        vasp_input_params = vasp_input_params or {}

        for k, v in vasp_input_params.items():
            if k in dos_input_params.keys():
                dos_input_params[k].update(v)
            else:
                dos_input_params[k] = v

        # Set up the input files of the calculation
        tasks.append(WriteVaspFromIOSet(
            vasp_input_set=SlabStaticSet(structure=slab, **dos_input_params)
        ))

        # Create the PyTask that runs the calculation
        if in_custodian:
            tasks.append(CustodianTask())
        else:
            tasks.append(VaspTask())

        # Write the final slab to a json file for subsequent calculations
        # tasks.append(VaspWriteFinalSlabTask()) # TODO

        super().__init__(tasks=tasks, name=name, parents=parents, spec=spec)
