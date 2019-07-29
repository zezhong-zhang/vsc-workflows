# coding: utf8
# Copyright (c) Marnik Bercx, University of Antwerp
# Distributed under the terms of the MIT License

import os, warnings

from fireworks import PyTask, Firework

from vscworkflows.workflow.firetasks import VaspTask, CustodianTask, \
    VaspParallelizationTask, PulayTask, WriteVaspFromIOSet
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

        Args:
            structure (Structure): Input structure for the calculation.
            name (str): Name of the Firework
            vasp_input_params (dict): User defined input parameters for the
                calculation. These are passed as kwargs to the VASP input set.
                Common examples for keys of this dictionary are
                "user_incar_settings", "user_kpoints_settings", etc.
            parents (Firework or List): Firework or list of Fireworks that are the
                parents of this Firework.
            in_custodian (bool): Flag that indicates whether the calculation should
                be run inside a Custodian.
            spec (dict): Firework spec. Can be used to set e.g. the '_launch_dir',
                '_category', etc.

        Returns:
            Firework: A firework that represents a static calculation.

        """

        tasks = list()
        vasp_input_params = vasp_input_params or {}

        if structure is not None:
            tasks.append(WriteVaspFromIOSet(
                vasp_input_set=BulkStaticSet(structure, **vasp_input_params)
            ))
        elif parents is not None:  # TODO What if multiple parents?
            tasks.append(WriteVaspFromIOSet(
                parents=parents,
                vasp_input_set="vscworkflows.setup.sets.BulkStaticSet"
            ))
        else:
            tasks.append(WriteVaspFromIOSet(
                vasp_input_set="vscworkflows.setup.sets.BulkStaticSet"
            ))

        # Configure the parallelization settings
        tasks.append(VaspParallelizationTask())

        # Run the calculation
        if in_custodian:
            tasks.append(CustodianTask())
        else:
            tasks.append(VaspTask())

        # Combine the two FireTasks into one FireWork
        super().__init__(
            tasks=tasks, parents=parents, name=name, spec=spec
        )


class OptimizeFW(Firework):

    def __init__(self, structure, name="Geometry Optimization",
                 vasp_input_params=None, parents=None, in_custodian=False,
                 spec=None):
        """
        Initialize a Firework for a geometry optimization.

        Args:
            structure (Structure): Input structure for the calculation.
            name (str): Name of the Firework
            vasp_input_params (dict): User defined input parameters for the
                calculation. These are passed as kwargs to the VASP input set.
                Common examples for keys of this dictionary are
                "user_incar_settings", "user_kpoints_settings", etc.
            parents (Firework or List): Firework or list of Fireworks that are the
                parents of this Firework.
            in_custodian (bool): Flag that indicates whether the calculation should
                be run inside a Custodian.
            spec (dict): Firework spec. Can be used to set e.g. the '_launch_dir',
                '_category', etc.

        """
        tasks = list()
        vasp_input_params = vasp_input_params or {}

        tasks.append(WriteVaspFromIOSet(
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
        Calculate the dielectric function of a structure.

        Args:
            structure (Structure): Input structure for the calculation.
            name (str): Name of the Firework
            vasp_input_params (dict): User defined input parameters for the
                calculation. These are passed as kwargs to the VASP input set.
                Common examples for keys of this dictionary are
                "user_incar_settings", "user_kpoints_settings", etc.
            parents (Firework or List): Firework or list of Fireworks that are the
                parents of this Firework.
            in_custodian (bool): Flag that indicates whether the calculation should
                be run inside a Custodian.
            spec (dict): Firework spec. Can be used to set e.g. the '_launch_dir',
                '_category', etc.

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
        A static calculation for a slab structure.

        Args:
            slab (Slab): Geometry of the slab.
            name (str): Name of the Firework.
            vasp_input_params (dict): User defined input parameters for the
                calculation. These are passed as kwargs to the VASP input set.
                Common examples for keys of this dictionary are
                "user_incar_settings", "user_kpoints_settings", etc.
            parents (Firework or List): Firework or list of Fireworks that are the
                parents of this Firework.
            in_custodian (bool): Flag that indicates whether the calculation should be
                run inside a Custodian.
            spec (dict): Firework spec. Can be used to set e.g. the '_launch_dir',
                '_category', etc.
        """
        tasks = list()
        vasp_input_params = vasp_input_params or {}

        if slab is not None:
            tasks.append(WriteVaspFromIOSet(
                vasp_input_set=SlabStaticSet(slab, **vasp_input_params)
            ))
        elif parents is not None:  # TODO What if multiple parents?
            tasks.append(WriteVaspFromIOSet(
                parents=parents,
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

    # TODO Add intuitive way of including kpoint settings

    def __init__(self, slab, name="Slab optimize", vasp_input_params=None,
                 user_slab_settings=None, parents=None, in_custodian=False,
                 spec=None):
        """
        Geometry optimization of a slab.
        
        Args:
            slab (Slab): Geometry of the slab.
            name (str): Name of the Firework.
            user_slab_settings (dict): Allows the user to specify the selective 
                dynamics of the slab geometry optimization. These are passed to 
                the SlabOptimizeSet.fix_slab_bulk() commands as kwargs.
                # TODO: improve this description
            vasp_input_params (dict): User defined input parameters for the
                calculation. These are passed as kwargs to the VASP input set.
                Common examples for keys of this dictionary are
                "user_incar_settings", "user_kpoints_settings", etc.
            parents (Firework or List): Firework or list of Fireworks that are the
                parents of this Firework.
            in_custodian (bool): Flag that indicates whether the calculation should be
                run inside a Custodian.
            spec (dict): Firework spec. Can be used to set e.g. the '_launch_dir',
                '_category', etc.
        """
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

    def __init__(self, slab=None, name="Slab DOS", vasp_input_params=None,
                 parents=None, in_custodian=False, spec=None):
        """
        DOS calculation of a slab.

        Args:
            slab (Slab): Geometry of the slab.
            name (str): Name of the Firework.
            vasp_input_params (dict): User defined input parameters for the
                calculation. These are passed as kwargs to the VASP input set.
                Common examples for keys of this dictionary are
                "user_incar_settings", "user_kpoints_settings", etc.
            parents (Firework or List): Firework or list of Fireworks that are the
                parents of this Firework.
            in_custodian (bool): Flag that indicates whether the calculation should be
                run inside a Custodian.
            spec (dict): Firework spec. Can be used to set e.g. the '_launch_dir',
                '_category', etc.
        """
        tasks = list()

        if slab is not None:
            # Set up the input files of the low precision static calculation
            tasks.append(WriteVaspFromIOSet(
                vasp_input_set=SlabStaticSet(
                    structure=slab,
                    user_incar_settings={"LCHARG": True, "EDIFF": 1e-3},
                    user_kpoints_settings={"k_resolution": 0.4}
                )))
        elif parents is not None:  # TODO What if multiple parents?
            tasks.append(WriteVaspFromIOSet(
                parents=parents,
                vasp_input_set="vscworkflows.setup.sets.SlabStaticSet",
                vasp_input_params={
                    "user_incar_settings": {"LCHARG": True, "EDIFF": 1e-3},
                    "user_kpoints_settings": {"k_resolution": 0.4}
                }
            ))
        else:
            raise ValueError("You must provide either an input structure or "
                             "parent firework to StaticFW!")

        # Configure the parallelization settings
        tasks.append(VaspParallelizationTask())

        # Create the PyTask that runs the calculation
        if in_custodian:
            tasks.append(
                CustodianTask(stdout_file="chgrun.out", stderr_file="chgrun.out")
            )
        else:
            tasks.append(VaspTask(stdout_file="chgrun.out",
                                  stderr_file="chgrun.out"))

        # Default input parameters
        dos_input_params = {
            "user_incar_settings": {"NEDOS": 2000, "EDIFF": 1.0e-6},
            "user_kpoints_settings": {"k_resolution": 0.05}
        }

        # Update the defaults with the user specified input parameters
        vasp_input_params = vasp_input_params or {}

        for k, v in vasp_input_params.items():
            if k in dos_input_params.keys():
                dos_input_params[k].update(v)
            else:
                dos_input_params[k] = v

        if slab is not None:
            # Set up the input files of the low precision static calculation
            tasks.append(WriteVaspFromIOSet(
                vasp_input_set=SlabStaticSet(
                    structure=slab,
                    **dos_input_params
                )))
        elif parents is not None:  # TODO What if multiple parents?
            tasks.append(WriteVaspFromIOSet(
                parents=parents,
                vasp_input_set="vscworkflows.setup.sets.SlabStaticSet",
                vasp_input_params=dos_input_params
            ))

        # Configure the parallelization settings
        tasks.append(VaspParallelizationTask())

        # Create the PyTask that runs the calculation
        if in_custodian:
            tasks.append(CustodianTask())
        else:
            tasks.append(VaspTask())

        # Write the final slab to a json file for subsequent calculations
        # tasks.append(VaspWriteFinalSlabTask()) # TODO

        super().__init__(tasks=tasks, name=name, parents=parents, spec=spec)
