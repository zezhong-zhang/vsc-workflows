# coding: utf8
# Copyright (c) Marnik Bercx, University of Antwerp
# Distributed under the terms of the MIT License

from fireworks import Firework

from custodian.custodian import ErrorHandler

from vscworkflows.firetasks.core import VaspTask, VaspCustodianTask, \
    VaspParallelizationTask, IncreaseNumberOfBands, PulayTask, WriteVaspFromIOSet, \
    AddFinalGeometryToSpec
from vscworkflows.setup.sets import BulkStaticSet, BulkOptimizeSet, \
    SlabStaticSet, SlabOptimizeSet
from vscworkflows.utils import vasp_input_update

"""

Module that contains all the fireworks to construct Workflows.

"""

__author__ = "Marnik Bercx"
__copyright__ = "Copyright 2019, Marnik Bercx, University of Antwerp"
__version__ = "pre-alpha"
__maintainer__ = "Marnik Bercx"
__email__ = "marnik.bercx@uantwerpen.be"
__date__ = "Jun 2019"


class StaticFW(Firework):

    def __init__(self, structure=None, name="Static calculation",
                 vasp_input_params=None, parents=None, spec=None,
                 custodian=False, auto_parallelization=False):
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
            custodian (bool): Flag that indicates whether the calculation should
                be run inside a Custodian. #TODO
            spec (dict): Firework spec. Can be used to set e.g. the '_launch_dir',
                '_category', etc.
            auto_parallelization (bool): Automatically parallelize the calculation
                using the VaspParallelizationTask.

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
                vasp_input_set="vscworkflows.setup.sets.BulkStaticSet",
                vasp_input_params=vasp_input_params
            ))
        else:
            tasks.append(WriteVaspFromIOSet(
                vasp_input_set="vscworkflows.setup.sets.BulkStaticSet",
                vasp_input_params=vasp_input_params
            ))

        # Configure the parallelization settings
        if auto_parallelization:
            tasks.append(VaspParallelizationTask())

        # Run the calculation
        if custodian is True:
            tasks.append(VaspCustodianTask())
        elif isinstance(custodian, list):
            assert all([isinstance(h, ErrorHandler) for h in custodian]), \
                "Not all elements in 'custodian' list are instances of " \
                "the ErrorHandler class!"
            tasks.append(VaspCustodianTask(handlers=custodian))
        else:
            tasks.append(VaspTask())

        # Add the final geometry to the fw_spec of this firework and its children
        tasks.append(AddFinalGeometryToSpec())

        # Combine the two FireTasks into one FireWork
        super().__init__(tasks=tasks, parents=parents, name=name, spec=spec)


class OptimizeFW(Firework):

    def __init__(self, structure=None, name="Geometry Optimization",
                 vasp_input_params=None, parents=None, spec=None,
                 custodian=False, auto_parallelization=False,
                 pulay_condition=None, pulay_tolerance=None):
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
            custodian (bool): Flag that indicates whether the calculation should
                be run inside a Custodian.
            spec (dict): Firework spec. Can be used to set e.g. the '_launch_dir',
                '_category', etc.
            auto_parallelization (bool): Automatically parallelize the calculation
                using the VaspParallelizationTask.

        """
        tasks = list()
        vasp_input_params = vasp_input_params or {}

        if structure is not None:
            tasks.append(WriteVaspFromIOSet(
                vasp_input_set=BulkOptimizeSet(structure, **vasp_input_params)
            ))
        elif parents is not None:  # TODO What if multiple parents?
            tasks.append(WriteVaspFromIOSet(
                parents=parents,
                vasp_input_set="vscworkflows.setup.sets.BulkOptimizeSet",
                vasp_input_params=vasp_input_params
            ))
        else:
            raise ValueError("Please provide either an input "
                             "structure or a parent Firework.")

        # Configure the parallelization settings
        if auto_parallelization:
            tasks.append(VaspParallelizationTask())

        # Run the calculation
        if custodian is True:
            tasks.append(VaspCustodianTask())
        elif isinstance(custodian, list):
            assert all([isinstance(h, ErrorHandler) for h in custodian]), \
                "Not all elements in 'custodian' list are instances of " \
                "the ErrorHandler class!"
            tasks.append(VaspCustodianTask(handlers=custodian))
        else:
            tasks.append(VaspTask())

        # Add the final geometry to the fw_spec of this firework and its children
        tasks.append(AddFinalGeometryToSpec())

        # Check the Pulay stress
        tasks.append(
            PulayTask(custodian=custodian,
                      spec=spec,
                      condition=pulay_condition,
                      tolerance=pulay_tolerance)
        )

        # Combine the FireTasks into one FireWork
        super().__init__(tasks=tasks, parents=parents, name=name, spec=spec)


class OpticsFW(Firework):

    def __init__(self, structure=None, name="Optics calculation",
                 vasp_input_params=None, parents=None, spec=None,
                 custodian=False, auto_parallelization=False,
                 bands_multiplier=3):
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
            custodian (bool): Flag that indicates whether the calculation should
                be run inside a Custodian.
            spec (dict): Firework spec. Can be used to set e.g. the '_launch_dir',
                '_category', etc.
            auto_parallelization (bool): Automatically parallelize the calculation
                using the VaspParallelizationTask.
            bands_multiplier (float): Use IncreaseNumberOfBands task to multiply
                set the total number of bands equal to a multiple of the VASP
                default.

        """
        # Default input parameters
        optics_input_params = {
            "user_incar_settings": {
                "LOPTICS": True, "NEDOS": 2000, "EDIFF": 1.0e-6, "CSHIFT": 0.01},
            "user_kpoints_settings": {"reciprocal_density": 200}
        }

        tasks = list()
        vasp_input_params = vasp_input_params or {}

        # Update the defaults with the user specified input parameters
        for k, v in vasp_input_params.items():
            if k in optics_input_params.keys() and k != "user_kpoints_settings":
                optics_input_params[k].update(v)
            else:
                optics_input_params[k] = v

        if structure is not None:
            tasks.append(WriteVaspFromIOSet(
                vasp_input_set=BulkStaticSet(structure, **optics_input_params)
            ))
        elif parents is not None:  # TODO What if multiple parents?
            tasks.append(WriteVaspFromIOSet(
                parents=parents,
                vasp_input_set="vscworkflows.setup.sets.BulkStaticSet",
                vasp_input_params=optics_input_params
            ))
        else:
            tasks.append(WriteVaspFromIOSet(
                vasp_input_set="vscworkflows.setup.sets.BulkStaticSet",
                vasp_input_params=optics_input_params
            ))

        # Configure the parallelization settings
        if auto_parallelization:
            tasks.append(VaspParallelizationTask())

        # Increase the number of bands, unless the user specified NBANDS
        if not "NBANDS" in optics_input_params["user_incar_settings"].keys():
            tasks.append(IncreaseNumberOfBands(multiplier=bands_multiplier))

        # Run the calculation
        if custodian is True:
            tasks.append(VaspCustodianTask())
        elif isinstance(custodian, list):
            assert all([isinstance(h, ErrorHandler) for h in custodian]), \
                "Not all elements in 'custodian' list are instances of " \
                "the ErrorHandler class!"
            tasks.append(VaspCustodianTask(handlers=custodian))
        else:
            tasks.append(VaspTask())

        # Add the final geometry to the fw_spec of this firework and its children
        tasks.append(AddFinalGeometryToSpec())

        # Combine the two FireTasks into one FireWork
        super().__init__(tasks=tasks, parents=parents, name=name, spec=spec)


class SlabStaticFW(Firework):

    def __init__(self, slab=None, name="Slab Static", vasp_input_params=None,
                 parents=None, spec=None, custodian=False,
                 auto_parallelization=False):
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
            custodian (bool): Flag that indicates whether the calculation should be
                run inside a Custodian.
            spec (dict): Firework spec. Can be used to set e.g. the '_launch_dir',
                '_category', etc.
            auto_parallelization (bool): Automatically parallelize the calculation
                using the VaspParallelizationTask.
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
                vasp_input_set="vscworkflows.setup.sets.SlabStaticSet",
                vasp_input_params=vasp_input_params
            ))
        else:
            raise ValueError("You must provide either an input structure or "
                             "parent firework to SlabStaticFW!")

        # Configure the parallelization settings
        if auto_parallelization:
            tasks.append(VaspParallelizationTask())

        # Run the calculation
        if custodian is True:
            tasks.append(VaspCustodianTask())
        elif isinstance(custodian, list):
            assert all([isinstance(h, ErrorHandler) for h in custodian]), \
                "Not all elements in 'custodian' list are instances of " \
                "the ErrorHandler class!"
            tasks.append(VaspCustodianTask(handlers=custodian))
        else:
            tasks.append(VaspTask())

        # Add the final geometry to the fw_spec of this firework and its children
        tasks.append(AddFinalGeometryToSpec())

        super().__init__(tasks=tasks, parents=parents, name=name, spec=spec)


class SlabOptimizeFW(Firework):

    # TODO Add intuitive way of including kpoint settings

    def __init__(self, slab, name="Slab optimize", vasp_input_params=None,
                 user_slab_settings=None, parents=None, spec=None,
                 custodian=False, auto_parallelization=False):
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
            custodian (bool): Flag that indicates whether the calculation should be
                run inside a Custodian.
            spec (dict): Firework spec. Can be used to set e.g. the '_launch_dir',
                '_category', etc.
            auto_parallelization (bool): Automatically parallelize the calculation
                using the VaspParallelizationTask.
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
        if auto_parallelization:
            tasks.append(VaspParallelizationTask())

        # Run the calculation
        if custodian is True:
            tasks.append(VaspCustodianTask())
        elif isinstance(custodian, list):
            assert all([isinstance(h, ErrorHandler) for h in custodian]), \
                "Not all elements in 'custodian' list are instances of " \
                "the ErrorHandler class!"
            tasks.append(VaspCustodianTask(handlers=custodian))
        else:
            tasks.append(VaspTask())

        # Add the final geometry to the fw_spec of this firework and its children
        tasks.append(AddFinalGeometryToSpec())

        # Combine the FireTasks into one FireWork
        super().__init__(tasks=tasks, name=name, parents=parents, spec=spec)


class SlabDosFW(Firework):

    def __init__(self, slab=None, name="Slab DOS", vasp_input_params=None,
                 parents=None, spec=None, custodian=False,
                 auto_parallelization=False, bands_multiplier=3):
        """
        DOS calculation of a slab. Starts with a calculation of the charge density
        using a low density k-point mesh.

        Args:
            slab (Slab): Geometry of the slab.
            name (str): Name of the Firework.
            vasp_input_params (dict): User defined input parameters for the
                calculation. These are passed as kwargs to the VASP input set.
                Common examples for keys of this dictionary are
                "user_incar_settings", "user_kpoints_settings", etc.
            parents (Firework or List): Firework or list of Fireworks that are the
                parents of this Firework.
            spec (dict): Firework spec. Can be used to set e.g. the '_launch_dir',
                '_category', etc.
            custodian (bool): Flag that indicates whether the calculation should be
                run inside a Custodian.
            auto_parallelization (bool): Automatically parallelize the calculation
                using the VaspParallelizationTask.
            bands_multiplier (float): Use IncreaseNumberOfBands task to multiply
                set the total number of bands equal to a multiple of the VASP
                default.

        """
        # Default input parameters for charge density run
        chgrun_input_params = {
            "user_incar_settings": {"LCHARG": True, "EDIFF": 1.0e-6},
            "user_kpoints_settings": {"k_resolution": 0.4}
        }

        # Default input parameters for actual DOS run
        dos_input_params = {
            "user_incar_settings": {"NEDOS": 2000, "EDIFF": 1.0e-6,
                                    "ICHARG": 11},
            "user_kpoints_settings": {"k_resolution": 0.05}
        }

        # Update the defaults with the user specified input parameters
        vasp_input_params = vasp_input_params or {}

        vasp_input_update(chgrun_input_params, vasp_input_params)
        vasp_input_update(dos_input_params, vasp_input_params)

        tasks = list()

        if slab is not None:
            # Set up the input files of the low precision static calculation
            tasks.append(WriteVaspFromIOSet(
                vasp_input_set=SlabStaticSet(
                    structure=slab, **chgrun_input_params
                )))
        elif parents is not None:  # TODO What if multiple parents?
            tasks.append(WriteVaspFromIOSet(
                parents=parents,
                vasp_input_set="vscworkflows.setup.sets.SlabStaticSet",
                vasp_input_params=chgrun_input_params
            ))
        else:
            raise ValueError("You must provide either an input structure or "
                             "parent firework to StaticFW!")

        # Configure the parallelization settings
        if auto_parallelization:
            tasks.append(VaspParallelizationTask())

        # Run the calculation
        if custodian is True:
            tasks.append(VaspCustodianTask(stdout_file="chgrun.out",
                                           stderr_file="chgrun.out"))
        elif isinstance(custodian, list):
            assert all([isinstance(h, ErrorHandler) for h in custodian]), \
                "Not all elements in 'custodian' list are instances of " \
                "the ErrorHandler class!"
            tasks.append(VaspCustodianTask(stdout_file="chgrun.out",
                                           stderr_file="chgrun.out",
                                           handlers=custodian))
        else:
            tasks.append(VaspTask(stdout_file="chgrun.out",
                                  stderr_file="chgrun.out"))

        if slab is not None:
            # Set up the input files of the actual DOS run
            tasks.append(WriteVaspFromIOSet(
                vasp_input_set=SlabStaticSet(
                    structure=slab, **dos_input_params
                )))
        elif parents is not None:  # TODO What if multiple parents?
            tasks.append(WriteVaspFromIOSet(
                parents=parents,
                vasp_input_set="vscworkflows.setup.sets.SlabStaticSet",
                vasp_input_params=dos_input_params
            ))

        # Configure the parallelization settings
        if auto_parallelization:
            tasks.append(VaspParallelizationTask())

        # Increase the number of bands, unless the user specified NBANDS
        if "NBANDS" not in vasp_input_params["user_incar_settings"].keys():
            tasks.append(IncreaseNumberOfBands(multiplier=bands_multiplier))

        # Run the calculation
        if custodian is True:
            tasks.append(VaspCustodianTask())
        elif isinstance(custodian, list):
            assert all([isinstance(h, ErrorHandler) for h in custodian]), \
                "Not all elements in 'custodian' list are instances of " \
                "the ErrorHandler class!"
            tasks.append(VaspCustodianTask(handlers=custodian))
        else:
            tasks.append(VaspTask())

        # Add the final geometry to the fw_spec of this firework and its children
        tasks.append(AddFinalGeometryToSpec())

        super().__init__(tasks=tasks, name=name, parents=parents, spec=spec)
