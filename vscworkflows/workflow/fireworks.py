# coding: utf8
# Copyright (c) Marnik Bercx, University of Antwerp
# Distributed under the terms of the MIT License

import os

from fireworks import PyTask, Firework

from vscworkflows.workflow.firetasks import VaspTask, CustodianTask, PulayTask

"""
Package that contains all the fireworks to construct Workflows.

"""

__author__ = "Marnik Bercx"
__copyright__ = "Copyright 2019, Marnik Bercx, University of Antwerp"
__version__ = "pre-alpha"
__maintainer__ = "Marnik Bercx"
__email__ = "marnik.bercx@uantwerpen.be"
__date__ = "Jun 2019"


class OptimizeFW(Firework):

    def __init__(self, structure, directory, functional, is_metal=False,
                 in_custodian=False, number_nodes=None, fw_action=None):
        """
        Initialize a Firework for a geometry optimization.

        Args:
            structure: structure: pymatgen.Structure OR path to the structure file.
            directory (str): Directory in which the geometry optimization should be
                performed.
            functional (tuple): Tuple with the functional choices. The first element
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
        # Create the PyTask that sets up the calculation
        setup_optimize = PyTask(
            func="vscworkflows.setup.write_input.optimize",
            kwargs={"structure": structure,
                    "directory": directory,
                    "functional": functional,
                    "is_metal": is_metal}
        )

        # Create the PyTask that runs the calculation
        if in_custodian:
            vasprun = CustodianTask(directory=directory)
        else:
            vasprun = VaspTask(directory=directory)

        # Create the PyTask that check the Pulay stresses
        pulay_task = PulayTask(directory=directory,
                               in_custodian=in_custodian,
                               number_nodes=number_nodes,
                               fw_action=fw_action)

        # Only add number of nodes to spec if specified
        firework_spec = {}
        if number_nodes is None or number_nodes == 0:
            firework_spec.update({"_category": "none"})
        else:
            firework_spec.update({"_category": str(number_nodes) + "nodes"})

        # Combine the FireTasks into one FireWork
        super().__init__(
            tasks=[setup_optimize, vasprun, pulay_task],
            name="Geometry optimization", spec=firework_spec
        )


class OpticsFW(Firework):

    def __init__(self, structure, directory, functional, k_resolution=0.05,
                 is_metal=False, in_custodian=False, number_nodes=None):
        """
        Initialize a Firework for a geometry optimization.

        Args:
            structure: pymatgen.Structure OR path to structure file for which to run
                the geometry optimization.
            directory (str): Directory in which the geometry optimization should be
                performed.
            functional (tuple): Tuple with the functional choices. The first element
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
        # Create the PyTask that sets up the calculation
        setup_optics = PyTask(
            func="vscworkflows.setup.write_input.optics",
            kwargs={"structure": structure,
                    "directory": directory,
                    "functional": functional,
                    "k_resolution": k_resolution,
                    "is_metal": is_metal}
        )

        # Create the PyTask that runs the calculation
        if in_custodian:
            vasprun = CustodianTask(directory=directory)
        else:
            vasprun = VaspTask(directory=directory)

        # Only add number of nodes to spec if specified
        firework_spec = {}
        if number_nodes is None or number_nodes == 0:
            firework_spec.update({"_category": "none"})
        else:
            firework_spec.update({"_category": str(number_nodes) + "nodes"})

        # Combine the FireTasks into one FireWork
        super().__init__(
            tasks=[setup_optics, vasprun],
            name="Optics", spec=firework_spec
        )


class SlabOptimizeFW(Firework):

    def __init__(self, slab, directory, functional, fix_part,
                 fix_thickness, is_metal=False,
                 in_custodian=False, number_nodes=None, fw_action=None):
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
            fw_action (fireworks.FWAction): FWAction to return after the geometry
                optimization is completed. # TODO

        """
        # Create the PyTask that sets up the calculation
        setup_optimize = PyTask(
            func="vscworkflows.setup.write_input.slab_optimize",
            kwargs={"slab": slab,
                    "fix_part": fix_part,
                    "fix_thickness": fix_thickness,
                    "directory": directory,
                    "functional": functional,
                    "is_metal": is_metal}
        )

        # Create the PyTask that runs the calculation
        if in_custodian:
            vasprun = CustodianTask(directory=directory)
        else:
            vasprun = VaspTask(directory=directory)

        # Only add number of nodes to spec if specified
        firework_spec = {}
        if number_nodes is None or number_nodes == 0:
            firework_spec.update({"_category": "none"})
        else:
            firework_spec.update({"_category": str(number_nodes) + "nodes"})

        # Combine the FireTasks into one FireWork
        super().__init__(
            tasks=[setup_optimize, vasprun],
            name="Geometry optimization", spec=firework_spec
        )


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
        # Set up the DOS calculation, based on the structure found from the
        # geometry optimization.
        setup_dos = PyTask(func="vscworkflows.setup.write_input.slab_dos",
                           kwargs={
                               "slab": slab,
                               "directory": directory,
                               "functional": functional,
                               "k_resolution": k_resolution,
                               "calculate_locpot": calculate_locpot
                           })

        # Create the PyTask that runs the calculation
        if in_custodian:
            vasprun = CustodianTask(directory=directory)
        else:
            vasprun = VaspTask(directory=directory)

        # Only add number of nodes to spec if specified
        firework_spec = {}
        if number_nodes is None or number_nodes == 0:
            firework_spec.update({"_category": "none"})
        else:
            firework_spec.update({"_category": str(number_nodes) + "nodes"})

        # Combine the FireTasks into one FireWork
        super().__init__(
            tasks=[setup_dos, vasprun],
            name="DOS Calculation", spec=firework_spec
        )
