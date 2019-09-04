# coding: utf8
# Copyright (c) Marnik Bercx, University of Antwerp
# Distributed under the terms of the MIT License

import subprocess

from pymatgen.io.vasp.inputs import Kpoints

from custodian.custodian import ErrorHandler

"""
Package that contains all the fireworks to construct Workflows.

"""

__author__ = "Marnik Bercx"
__copyright__ = "Copyright 2019, Marnik Bercx, University of Antwerp"
__version__ = "pre-alpha"
__maintainer__ = "Marnik Bercx"
__email__ = "marnik.bercx@uantwerpen.be"
__date__ = "Sep 2019"


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
