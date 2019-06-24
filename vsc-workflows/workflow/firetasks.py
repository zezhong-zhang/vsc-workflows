# coding: utf8
# Copyright (c) Marnik Bercx, University of Antwerp
# Distributed under the terms of the MIT License

import os
import subprocess

import numpy as np
from custodian import Custodian
from custodian.vasp.handlers import VaspErrorHandler, UnconvergedErrorHandler
from custodian.vasp.jobs import VaspJob
from fireworks import FiretaskBase, explicit_serialize
from fireworks import Firework, FWAction, ScriptTask, PyTask
from pymatgen import Structure

from pybat.core import Cathode

"""
Definition of the FireTasks for the workflows.

"""

__author__ = "Marnik Bercx"
__copyright__ = "Copyright 2019, Marnik Bercx, University of Antwerp"
__version__ = "pre-alpha"
__maintainer__ = "Marnik Bercx"
__email__ = "marnik.bercx@uantwerpen.be"
__date__ = "Mar 2019"


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
class VaspSetupTask(FiretaskBase):
    """
    FireTask used for setting up the input files of a calculation. The setup scripts
    are defined in the vsc-workflows.setup.py module.

    """
    pass  # TODO
