# coding: utf8
# Copyright (c) Marnik Bercx, University of Antwerp
# Distributed under the terms of the MIT License

import subprocess, os, time
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from monty.re import regrep

from pymatgen.io.vasp.inputs import Kpoints, Incar

from custodian.vasp.interpreter import VaspModder, VaspInput
from custodian.custodian import ErrorHandler
from custodian.utils import backup

"""
Module that contains all the Custodian ErrorHandlers.

"""

__author__ = "Marnik Bercx"
__copyright__ = "Copyright 2019, Marnik Bercx, University of Antwerp"
__version__ = "pre-alpha"
__maintainer__ = "Marnik Bercx"
__email__ = "marnik.bercx@uantwerpen.be"
__date__ = "Sep 2019"

VASP_BACKUP_FILES = {"INCAR", "KPOINTS", "POSCAR", "OUTCAR", "CONTCAR",
                     "OSZICAR", "vasprun.xml", "vasp.out", "std_err.txt"}


class QuotasErrorHandler(ErrorHandler):
    """
    Quotas VaspErrorHandler class that handles a number of common errors
    that occur during VASP runs.

    Copied and stripped from custodian.vasp.handlers.VaspErrorHandler in order to
    design a specific ErrorHandler that deals with the issues that occur during
    the calculations for the Quotas model.

    """

    is_monitor = True

    error_msgs = {
        "subspacematrix": ["WARNING: Sub-Space-Matrix is not hermitian in "
                           "DAV"],
    }

    def __init__(self, output_filename="vasp.out", natoms_large_cell=100,
                 errors_subset_to_catch=None):
        """
        Initializes the handler with the output file to check.

        Args:
            output_filename (str): This is the file where the stdout for vasp
                is being redirected. The error messages that are checked are
                present in the stdout. Defaults to "vasp.out", which is the
                default redirect used by :class:`custodian.vasp.jobs.VaspJob`.
            natoms_large_cell (int): Number of atoms threshold to treat cell
                as large. Affects the correction of certain errors. Defaults to
                100.
            errors_subset_to_catch (list): A subset of errors to catch. The
                default is None, which means all supported errors are detected.
                Use this to only catch only a subset of supported errors.
                E.g., ["eddrrm", "zheev"] will only catch the eddrmm and zheev
                errors, and not others. If you wish to only excluded one or
                two of the errors, you can create this list by the following
                lines:

                ```
                subset = list(VaspErrorHandler.error_msgs.keys())
                subset.pop("eddrrm")

                handler = VaspErrorHandler(errors_subset_to_catch=subset)
                ```
        """
        self.output_filename = output_filename
        self.errors = set()
        self.error_count = Counter()
        # threshold of number of atoms to treat the cell as large.
        self.natoms_large_cell = natoms_large_cell
        self.errors_subset_to_catch = errors_subset_to_catch or \
                                      list(QuotasErrorHandler.error_msgs.keys())

    def check(self):
        incar = Incar.from_file("INCAR")
        self.errors = set()
        with open(self.output_filename, "r") as f:
            for line in f:
                l = line.strip()
                for err, msgs in QuotasErrorHandler.error_msgs.items():
                    if err in self.errors_subset_to_catch:
                        for msg in msgs:
                            if l.find(msg) != -1:
                                # this checks if we want to run a charged
                                # computation (e.g., defects) if yes we don't
                                # want to kill it because there is a change in
                                # e-density (brmix error)
                                if err == "brmix" and 'NELECT' in incar:
                                    continue
                                self.errors.add(err)
        return len(self.errors) > 0

    def correct(self):
        backup(VASP_BACKUP_FILES | {self.output_filename})
        actions = []
        vi = VaspInput.from_directory(".")

        if "subspacematrix" in self.errors:
            actions.append({"dict": "INCAR",
                            "action": {"_set": {"ALGO": "All"}}})

        VaspModder(vi=vi).apply_actions(actions)
        return {"errors": list(self.errors), "actions": actions}


class MemoryErrorHandler(ErrorHandler):
    """
    Check if the job has run into memory issues.

    """

    is_monitor = False

    error_msgs = {
        "bad_termination": ["BAD TERMINATION OF ONE OF YOUR APPLICATION PROCESSES"]
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
    by looking at the trend of a linear fit to the logarithm of the residual
    charge.

    """
    is_monitor = True

    def __init__(self, min_electronic_steps=30, max_allowed_incline=-0.005,
                 max_fit_range=40, output_data=False):
        """
        Initializes the handler with the output file to check.

        Args:
            min_electronic_steps (int): Minimum number of electronic steps for
                which the charge must have been updated to check for convergence.
            max_allowed_incline (float): Maximum incline of the linear fit to the
                logarithm of the residual charge. If this value is exceeded,
                the check method returns positive and the monitor kills the job
                and corrects.
            max_fit_range (int): Maximum number of steps to consider for the
                linear fit.
            output_data (bool): Output the results of the checks, as well as a
                .png file of the incline per electronic step.

        """
        self.min_electronic_steps = min_electronic_steps
        self.max_allowed_incline = max_allowed_incline
        self.max_fit_range = max_fit_range
        self.output_data = output_data

    def check(self):

        vi = VaspInput.from_directory(".")
        nelmdl = abs(vi["INCAR"].get("NELMDL", -5))

        try:

            lines = []
            with open("OSZICAR", "r") as file:
                for l in file.readlines():
                    lines.append(l)

            residual_charge = [np.log(abs(float(l[-11:-1]))) for l in
                               lines[nelmdl + 2:-2]]

            if len(residual_charge) + nelmdl > self.min_electronic_steps:

                current_incline = np.polyfit(
                    x=range(min([len(residual_charge), self.max_fit_range])),
                    y=residual_charge[-self.max_fit_range:],
                    deg=1
                )[0]

                if self.output_data:

                    with open("convergence.out", "a") as file:
                        file.write(str(current_incline) + "\n")

                    incline_per_step = []

                    for i in range(self.min_electronic_steps, len(residual_charge)):
                        incline_per_step.append(
                            np.polyfit(range(min([i, self.max_fit_range])),
                                       residual_charge[
                                       max([0, i - self.max_fit_range]):i], 1)[0]
                        )

                    ax1 = plt.subplot(2, 1, 1)
                    ax2 = plt.subplot(2, 1, 2)

                    ax1.plot(residual_charge)
                    incline_plot = [0] * self.min_electronic_steps
                    incline_plot.extend(incline_per_step)
                    ax2.plot(incline_plot)

                    ax1.set_ylabel("log(rms (c))")
                    ax2.set_ylabel("Incline")

                    plt.savefig("convergence")

                if current_incline > self.max_allowed_incline:
                    return True

        except Exception as e:
            print(e)
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


class ParallelizationTestMonitor(ErrorHandler):
    """
    Monitor that is designed to run during calculations for scaling tests. Will
    shut down the calculation once a specified number of electronic steps have run,
    or in case the time for one electronic step is larger than some threshold.

    """
    is_monitor = True
    is_terminating = False
    raises_runtime_error = False
    max_num_corrections = 1

    def __init__(self, max_elec_steps=10, max_elec_step_time=3600):
        """
        Initializes the ParallelizationTestMonitor.

        Args:
            max_elec_steps (int): Maximum number self-consistent steps (i.e.
                excluding the first NELMDL steps).
            max_elec_step_time (float): Maximum allowed time per electronic step.

        """
        self.max_elec_steps = max_elec_steps
        self.max_elec_step_time = max_elec_step_time

    def check(self):

        vi = VaspInput.from_directory(".")
        nelmdl = abs(vi["INCAR"].get("NELMDL", -5))

        loop_pattern = r"\s+LOOP:\s+cpu\stime\s+\S+:\sreal\stime\s+(\S+)"
        loop_timing = regrep(
            filename="OUTCAR", patterns={"loop": loop_pattern})["loop"]

        if len(loop_timing) > 0:
            max_loop = np.max([float(e[0][0]) for e in loop_timing])
            if max_loop > self.max_elec_step_time:
                return True

        with open("temp.out", "w") as file:
            file.write("Number of steps: " + str(len(loop_timing)))

        if len(loop_timing) >= self.max_elec_steps + nelmdl - 1:
            return True
        else:
            return False

    def correct(self):

        with open("STOPCAR", "w") as file:
            file.write("LABORT = True")

        return {"errors": ["Parallelization Monitor"],
                "actions": None}


class JobTerminator(ErrorHandler):
    """
    Looks for Errors in stdout and terminates the job without any corrections.

    Mainly designed to shut down calculations that have bad parallelization
    settings during scaling tests, as here the input settings cannot be changed in
    order to allow a fair comparison of the performance.

    """
    is_monitor = True

    error_msgs = {
        "brmix": ["BRMIX: very serious problems"],
        "subspacematrix": ["WARNING: Sub-Space-Matrix is not hermitian in "
                           "DAV"],
        "too_few_bands": ["TOO FEW BANDS"],
        "rot_matrix": ["Found some non-integer element in rotation matrix"],
        "pricel": ["internal error in subroutine PRICEL"],
        "zpotrf": ["LAPACK: Routine ZPOTRF failed"],
        "pssyevx": ["ERROR in subspace rotation PSSYEVX"],
        "eddrmm": ["WARNING in EDDRMM: call to ZHEGV failed"],
        "edddav": ["Error EDDDAV: Call to ZHEGV failed"],
        "zheev": ["ERROR EDDIAG: Call to routine ZHEEV failed!"],
        "intel_mkl": ["Intel MKL ERROR: Parameter 6 was incorrect on entry to "
                      "DGEMV"],
        "edwav": ["EDWAV: internal error, the gradient is not orthogonal"]
    }

    def __init__(self, output_filename="vasp.out", natoms_large_cell=100,
                 errors_subset_to_catch=None, timeout=28800):
        """
        Initializes the handler with the output file to check.

        Args:
            output_filename (str): This is the file where the stdout for vasp
                is being redirected. The error messages that are checked are
                present in the stdout. Defaults to "vasp.out", which is the
                default redirect used by :class:`custodian.vasp.jobs.VaspJob`.
            natoms_large_cell (int): Number of atoms threshold to treat cell
                as large. Affects the correction of certain errors. Defaults to
                100.
            errors_subset_to_catch (list): A subset of errors to catch. The
                default is None, which means all supported errors are detected.
                Use this to only catch only a subset of supported errors.
                E.g., ["eddrrm", "zheev"] will only catch the eddrmm and zheev
                errors, and not others. If you wish to only excluded one or
                two of the errors, you can create this list by the following
                lines:

                ```
                subset = list(JobTerminator.error_msgs.keys())
                subset.pop("eddrrm")

                handler = JobTerminator(errors_subset_to_catch=subset)
                ```
        """
        self.output_filename = output_filename
        self.errors = set()
        self.error_count = Counter()
        # threshold of number of atoms to treat the cell as large.
        self.natoms_large_cell = natoms_large_cell
        self.errors_subset_to_catch = errors_subset_to_catch or \
                                      list(JobTerminator.error_msgs.keys())
        self.timeout = timeout

    def check(self):
        incar = Incar.from_file("INCAR")
        self.errors = set()
        with open(self.output_filename, "r") as f:
            for line in f:
                l = line.strip()
                for err, msgs in JobTerminator.error_msgs.items():
                    if err in self.errors_subset_to_catch:
                        for msg in msgs:
                            if l.find(msg) != -1:
                                # this checks if we want to run a charged
                                # computation (e.g., defects) if yes we don't
                                # want to kill it because there is a change in
                                # e-density (brmix error)
                                if err == "brmix" and 'NELECT' in incar:
                                    continue
                                self.errors.add(err)

        st = os.stat(self.output_filename)
        if time.time() - st.st_mtime > self.timeout:
            return True

        return len(self.errors) > 0

    def correct(self):

        return {"errors": ["Parallelization Monitor"], "actions": None}
