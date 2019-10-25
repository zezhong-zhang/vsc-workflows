# coding: utf8
# Copyright (c) Marnik Bercx, University of Antwerp
# Distributed under the terms of the MIT License

import subprocess
import numpy as np

from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io.vasp.outputs import Oszicar

from custodian.vasp.interpreter import VaspModder, VaspInput
from custodian.custodian import ErrorHandler
from custodian.utils import backup

"""
Package that contains all the fireworks to construct Workflows.

"""

__author__ = "Marnik Bercx"
__copyright__ = "Copyright 2019, Marnik Bercx, University of Antwerp"
__version__ = "pre-alpha"
__maintainer__ = "Marnik Bercx"
__email__ = "marnik.bercx@uantwerpen.be"
__date__ = "Sep 2019"


VASP_BACKUP_FILES = {"INCAR", "KPOINTS", "POSCAR", "OUTCAR", "CONTCAR",
                     "OSZICAR", "vasprun.xml", "vasp.out", "std_err.txt"}


class MemoryErrorHandler(ErrorHandler):
    """
    Check if the job has run into memory issues.

    """

    is_monitor = False

    error_msgs = {
        "bad_termination" : ["BAD TERMINATION OF ONE OF YOUR APPLICATION PROCESSES"]
    }

    def __init__(self, output_filename="vasp.out", kpoints_multiplier=0.95):
        """
        Initializes the handler with the output file to check.

        Args:
            output_filename (str): Filename for the stdout.
            kpoints_multiplier (float): Number < 1 which to multiply the number
                of k-points with when a memory issue is detected.

        """
        self.output_filename = output_filename
        self.kpoints_multiplier = kpoints_multiplier
        self.errors = set()

    def check(self):

        with open(self.output_filename, "r") as file:
            for line in file:
                for err, msgs in MemoryErrorHandler.error_msgs:
                    if any([line.find(msg) for msg in msgs]):
                        self.errors.add(err)
        return len(self.errors) > 0

    def correct(self):

        kpoints = Kpoints.from_file("KPOINTS")
        kpoints.kpts[0] = [int(self.kpoints_multiplier * k)
                           for k in kpoints.kpts[0]]
        kpoints.write_file("KPOINTS")


class MemoryMonitorHandler(ErrorHandler):
    """
    Monitor the memory of a job and reduce k-points if needed.

    """

    is_monitor = True

    def __init__(self, output_filename="vasp.out",
                 min_free_memory=1e3, kpoints_multiplier=0.95):
        """
        Initializes the handler with the output file to check.

        Args:
            output_filename (str): Filename for the stdout.
            kpoints_multiplier (float): Number < 1 which to multiply the number
                of k-points with when a memory issue is detected.

        """
        self.output_filename = output_filename
        self.kpoints_multiplier = kpoints_multiplier
        self.min_free_memory = min_free_memory
        self.errors = set()

    def check(self):

        result = subprocess.check_output(['bash', '-c', 'free -m'])
        free_memory = result.split('\n')[1].split()[3]

        with open("memory.log", "a+") as file:
            file.write(str(free_memory) + " MB.")

        return free_memory < self.min_free_memory

    def correct(self):

        kpoints = Kpoints.from_file("KPOINTS")
        kpoints.kpts[0] = [int(self.kpoints_multiplier * k)
                           for k in kpoints.kpts[0]]
        kpoints.write_file("KPOINTS")


class ElectronicConvergenceMonitor(ErrorHandler):
    """
    Monitor if the electronic optimization of the current ionic step is converging,
    by looking at the trend of a linear fit to the logarithm of the absolute energy
    differences.

    """
    is_monitor = True

    def __init__(self, min_electronic_steps=40, max_allowed_incline=0.01):
        """
        Initializes the handler with the output file to check.

        Args:

        """
        self.min_electronic_steps = min_electronic_steps
        self.max_allowed_incline = max_allowed_incline

    def check(self):

        vi = VaspInput.from_directory(".")
        nelmdl = vi["INCAR"].get("NELMDL", 5)

        try:
            oszicar = Oszicar("OSZICAR")
            dE_log = [np.log(abs(d['dE'])) for d in oszicar.electronic_steps[-1]]

            if len(dE_log) > self.min_electronic_steps:

                current_incline = np.polyfit(x=range(len(dE_log) - nelmdl),
                                             y=dE_log[nelmdl:],
                                             deg=1)[0]

                if current_incline > self.max_allowed_incline:
                    return True
        except:
            pass
        return False

    def correct(self):
        vi = VaspInput.from_directory(".")
        algo = vi["INCAR"].get("ALGO", "Normal")
        amix = vi["INCAR"].get("AMIX", 0.4)
        bmix = vi["INCAR"].get("BMIX", 1.0)
        amin = vi["INCAR"].get("AMIN", 0.1)
        actions = []

        # Ladder from VeryFast to Fast to Fast to All
        # These progressively switches to more stable but more
        # expensive algorithms
        if algo == "VeryFast":
            actions.append({"dict": "INCAR",
                            "action": {"_set": {"ALGO": "Fast"}}})
        elif algo == "Fast":
            actions.append({"dict": "INCAR",
                            "action": {"_set": {"ALGO": "Normal"}}})
        elif algo == "Normal":
            actions.append({"dict": "INCAR",
                            "action": {"_set": {"ALGO": "All"}}})
        elif amix > 0.1 and bmix > 0.01:
            # Try linear mixing
            actions.append({"dict": "INCAR",
                            "action": {"_set": {"AMIX": 0.1, "BMIX": 0.01,
                                                "ICHARG": 2}}})
        elif bmix < 3.0 and amin > 0.01:
            # Try increasing bmix
            actions.append({"dict": "INCAR",
                            "action": {"_set": {"AMIN": 0.01, "BMIX": 3.0,
                                                "ICHARG": 2}}})

        if actions:
            backup(VASP_BACKUP_FILES)
            VaspModder(vi=vi).apply_actions(actions)
            return {"errors": ["Non-converging job"], "actions": actions}
        # Unfixable error. Just return None for actions.
        else:
            return {"errors": ["Non-converging job"], "actions": None}