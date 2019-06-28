# coding: utf8
# Copyright (c) Marnik Bercx, University of Antwerp
# Distributed under the terms of the MIT License

import os
import subprocess

import numpy as np

from pymatgen import Structure
from fireworks import Firework, FWAction, FiretaskBase, ScriptTask, PyTask, \
    explicit_serialize
from custodian import Custodian
from custodian.vasp.jobs import VaspJob
from custodian.vasp.handlers import VaspErrorHandler, UnconvergedErrorHandler

"""
Definition of the FireTasks for the workflows.

"""

__author__ = "Marnik Bercx"
__copyright__ = "Copyright 2019, Marnik Bercx, University of Antwerp"
__version__ = "pre-alpha"
__maintainer__ = "Marnik Bercx"
__email__ = "marnik.bercx@uantwerpen.be"
__date__ = "Jun 2019"


@explicit_serialize
class VaspTask(FiretaskBase):
    """
    Firetask that represents a VASP calculation run.

    Required params:
        directory (str): Directory in which the VASP calculation should be run.

    """
    required_params = ["directory"]
    optional_params = ["stdout_file", "stderr_file"]

    def run_task(self, fw_spec):
        os.chdir(self["directory"])
        stdout_file = self.get("stdout_file", os.path.join(self["directory"], "out"))
        stderr_file = self.get("stderr_file", os.path.join(self["directory"], "out"))
        vasp_cmd = fw_spec["_fw_env"]["vasp_cmd"].split(" ")

        with open(stdout_file, 'w') as f_std, \
                open(stderr_file, "w", buffering=1) as f_err:
            p = subprocess.Popen(vasp_cmd, stdout=f_std, stderr=f_err)
            p.wait()


@explicit_serialize
class CustodianTask(FiretaskBase):
    """
    Firetask that represents a calculation run inside a Custodian.

    Required params:
        directory (str): Directory in which the VASP calculation should be run.

    """
    required_params = ["directory"]
    optional_params = ["stdout_file", "stderr_file"]

    def run_task(self, fw_spec):
        directory = os.path.abspath(self["directory"])
        os.chdir(directory)

        stdout_file = self.get("stdout_file", os.path.join(self["directory"], "out"))
        stderr_file = self.get("stderr_file", os.path.join(self["directory"], "out"))
        vasp_cmd = fw_spec["_fw_env"]["vasp_cmd"].split(" ")

        handlers = [VaspErrorHandler(output_filename=stdout_file),
                    UnconvergedErrorHandler(output_filename=stdout_file)]

        jobs = [VaspJob(vasp_cmd=vasp_cmd,
                        output_file=stdout_file,
                        stderr_file=stderr_file,
                        auto_npar=False)]

        c = Custodian(handlers, jobs, max_errors=10)
        c.run()


@explicit_serialize
class VaspSetupTask(FiretaskBase):
    """
    FireTask used for setting up the setup files of a calculation. The setup scripts
    are defined in the vscworkflows.write_input.py module.

    """

    def run_task(self, fw_spec):
        pass  # TODO


@explicit_serialize
class PulayTask(FiretaskBase):
    """
    Check if the lattice vectors of a structure have changed significantly during
    the geometry optimization, which could indicate that there where Pulay stresses
    present. If so, start a new geometry optimization with the final structure.

    Required params:
        directory (str): Directory in which the geometry optimization calculation
            was run.

    Optional params:
        in_custodian (bool): Flag that indicates whether the calculation should be
            run inside a Custodian.
        number_nodes (int): Number of nodes that should be used for the calculations.
            Is required to add the proper `_category` to the Firework generated, so
            it is picked up by the right Fireworker.
        tolerance (float): Tolerance that indicates the maximum change in norm for the
            matrix defined by the cartesian coordinates of the lattice vectors.
            If the norm changes more than the tolerance, another geometry optimization
            is performed starting from the final geometry.

    """
    required_params = ["directory"]
    option_params = ["in_custodian", "number_nodes", "tolerance", "fw_action"]

    # Standard tolerance for deciding to perform another geometry optimization.
    # Basically, PulayTask calculates the 2-norm of the absolute matrix taken from the
    # difference between the initial and final matrices of the lattice vectors of the
    # structure.
    pulay_tolerance = 5e-2

    def run_task(self, fw_spec):
        """

        Args:
            fw_spec:

        Returns:
            FWAction

        """
        # Extract the parameters into variables; this makes for cleaner code IMO
        directory = self["directory"]
        in_custodian = self.get("in_custodian", False)
        number_nodes = self.get("number_nodes", None)
        tolerance = self.get("tolerance", PulayTask.pulay_tolerance)
        fw_action = self.get('fw_action', {})

        # Check if the lattice vectors have changed significantly
        initial_structure = Structure.from_file(
            os.path.join(directory, "POSCAR")
        )
        final_structure = Structure.from_file(
            os.path.join(directory, "CONTCAR")
        )

        sum_differences = np.linalg.norm(
            initial_structure.lattice.matrix - final_structure.lattice.matrix
        )

        # If the difference is small, return an empty FWAction
        if sum_differences < tolerance:
            if fw_action:
                return FWAction.from_dict(fw_action)
            else:
                return FWAction()

        # Else, set up another geometry optimization
        else:
            print("Lattice vectors have changed significantly during geometry "
                  "optimization. Performing another full geometry optimization to "
                  "make sure there were no Pulay stresses present.\n\n")

            # Create the ScriptTask that copies the CONTCAR to the POSCAR
            copy_contcar = ScriptTask.from_str(
                "cp " + os.path.join(directory, "CONTCAR") +
                " " + os.path.join(directory, "POSCAR")
            )

            # Create the PyTask that runs the calculation
            if in_custodian:
                vasprun = CustodianTask(directory=directory)
            else:
                vasprun = VaspTask(directory=directory)

            # Create the PyTask that check the Pulay stresses again
            pulay_task = PulayTask(
                directory=directory, in_custodian=in_custodian,
                number_nodes=number_nodes, tolerance=tolerance,
                fw_action=fw_action
            )

            # Add number of nodes to spec, or "none"
            firework_spec = {}
            if number_nodes is None or number_nodes == 0:
                firework_spec.update({"_category": "none"})
            else:
                firework_spec.update({"_category": str(number_nodes) + "nodes"})

            # Combine the two FireTasks into one FireWork
            optimize_fw = Firework(tasks=[copy_contcar, vasprun, pulay_task],
                                   name="Pulay Step",
                                   spec=firework_spec)

            return FWAction(additions=optimize_fw)