# coding: utf8
# Copyright (c) Marnik Bercx, University of Antwerp
# Distributed under the terms of the MIT License

import os
import signal
import subprocess
import time
import warnings

import numpy as np
from atomate.utils.utils import load_class
from custodian import Custodian
from custodian.custodian import ErrorHandler
from custodian.vasp.handlers import VaspErrorHandler, UnconvergedErrorHandler
from custodian.vasp.jobs import VaspJob
from fireworks import Firework, FWAction, FiretaskBase, ScriptTask, \
    explicit_serialize
from pymatgen import Structure
from pymatgen.io.vasp.inputs import Incar, Kpoints
from pymatgen.io.vasp.outputs import Vasprun, Outcar
from pymatgen.io.vasp.sets import get_vasprun_outcar, get_structure_from_prev_run
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from vscworkflows.misc import QSlab, Cathode

"""
Definition of the FireTasks for the workflows.

"""

__author__ = "Marnik Bercx"
__copyright__ = "Copyright 2019, Marnik Bercx, University of Antwerp"
__version__ = "pre-alpha"
__maintainer__ = "Marnik Bercx"
__email__ = "marnik.bercx@uantwerpen.be"
__date__ = "Nov 2019"


def _find_irr_k_points(directory):
    """
    Determine the number of irreducible k-points based on the VASP input files in a
    directory.

    Args:
        directory (str): Path to the directory that contains the VASP input files.

    Returns:
        int: Number of irreducible k-points.

    """
    # TODO Still fails for many calculations.
    warnings.warn("Currently, the _find_irr_k_points method still fails regularly "
                  "to find the same number of irreducible k-points as VASP. Use "
                  "with care.")

    directory = os.path.abspath(directory)

    structure = Structure.from_file(os.path.join(directory, "POSCAR"))

    incar = Incar.from_file(os.path.join(directory, "INCAR"))
    if incar.get("MAGMOM", None) is not None:
        structure.add_site_property("magmom", incar.get("MAGMOM", None))
        structure.add_oxidation_state_by_site(
            [round(magmom, 3) for magmom in structure.site_properties["magmom"]]
        )

    kpoints = Kpoints.from_file(os.path.join(directory, "KPOINTS"))

    spg = SpacegroupAnalyzer(structure, symprec=1e-5)

    return len(spg.get_ir_reciprocal_mesh(kpoints.kpts))


def _find_fw_structure(firework):
    """
    Look for the final geometry in the spec/tasks of a Firework.

    Args:
        firework (Firework): Firework in which to look for the geometry.

    Returns:
        Last specified geometry in the Firework and its parents.

    """
    structure = None

    if "final_geometry" in firework.spec:
        structure = firework.spec["final_geometry"]

    else:
        for t in firework.tasks:
            if "WriteVaspFromIOSet" in t["_fw_name"]:

                try:
                    structure = t["structure"]
                    break
                except KeyError:
                    pass

                try:
                    structure = t["vasp_input_set"].structure
                    break
                except TypeError:
                    pass
                except AttributeError:
                    pass

                try:
                    structure = _find_fw_structure(
                        Firework.from_dict(t["parents"])
                    )
                except KeyError:
                    pass

    if issubclass(structure.__class__, Structure):
        return structure
    else:
        raise ValueError("Failed to extract structure from Firework.")


def _load_structure_from_dir(directory):
    """
    Find the final geometry from a directory which contains the output of a VASP
    run, preferably performed within a fireworks workflow. For simple Structure
    objects, the full geometry can be derived from the VASP outputs. However,
    more complex subclasses such as Cathode and QSlab require information about
    the original instance. This can be retrieved from the FW.json file, if present.

    Args:
        directory: Directory of the completed VASP run.

    Returns:
        Structure: The output geometry of the calculation. Either a Structure or
            subclass of a Structure.

    """
    if os.path.exists(os.path.join(directory, "FW.json")):

        fw = Firework.from_file(os.path.join(directory, "FW.json"))
        structure = _find_fw_structure(fw)

        if structure.__class__ == Structure:
            vasprun, outcar = get_vasprun_outcar(directory)
            return get_structure_from_prev_run(vasprun, outcar)

        elif issubclass(structure.__class__, QSlab):
            structure.update_sites(directory)
            return structure

        elif issubclass(structure.__class__, Cathode):
            structure.update_sites(directory)
            return structure
        else:
            raise ValueError

    else:
        warnings.warn("No FW.json file in the specified directory. Output geometry "
                      "will be returned as a Structure instance, even though the "
                      "input geometry may have been derived from a more complex "
                      "subclass.")
        vasprun, outcar = get_vasprun_outcar(directory)
        return get_structure_from_prev_run(vasprun, outcar)


@explicit_serialize
class VaspTask(FiretaskBase):
    """
    Perform a VASP calculation run.

    Optional params:
        directory (str): Directory in which the VASP calculation should be run.
        stdout_file (str): File to which to direct the stdout during the run.
        stderr_file (str): File to which to direct the stderr during the run.

    """
    optional_params = ["directory", "stdout_file", "stderr_file"]

    def run_task(self, fw_spec):
        if self.get("directory", None) is not None:
            os.chdir(self["directory"])
            directory = self["directory"]
        else:
            directory = os.getcwd()

        stdout_file = self.get("stdout_file", os.path.join(directory, "vasp.out"))
        stderr_file = self.get("stderr_file", os.path.join(directory, "vasp.out"))

        if os.path.exists(stdout_file):
            subprocess.run("cat " + stdout_file + " >> backup.out", shell=True)

        vasp_cmd = fw_spec["_fw_env"]["vasp_cmd"].split(" ")

        with open(stdout_file, 'w') as f_std, \
                open(stderr_file, "w", buffering=1) as f_err:
            p = subprocess.Popen(vasp_cmd, stdout=f_std, stderr=f_err)
            p.wait()


@explicit_serialize
class VaspCustodianTask(FiretaskBase):
    """
    Run VASP inside a Custodian.

    Optional params:
        directory (str): Directory in which the VASP calculation should be run.
        stdout_file (str): File to which to direct the stdout during the run.
        stderr_file (str): File to which to direct the stderr during the run.
        handlers (list): List of custodian ErrorHandler instances to use in the
            custodian run.
        monitor_freq (int): The number of polling steps before monitoring occurs.
            As the default polling_time_step is 10 seconds, using e.g. a
            monitor_freq of 30 (the default) means that Custodian uses the
            monitors to check for errors every 30 x 10 = 300 seconds, i.e.,
            5 minutes.
    """
    optional_params = ["directory", "stdout_file", "stderr_file", "handlers",
                       "monitor_freq"]

    def run_task(self, fw_spec):
        directory = self.get("directory", os.getcwd())
        os.chdir(directory)

        stdout_file = self.get("stdout_file", os.path.join(directory, "vasp.out"))
        stderr_file = self.get("stderr_file", os.path.join(directory, "vasp.out"))
        vasp_cmd = fw_spec["_fw_env"]["vasp_cmd"].split(" ")

        default_handlers = [VaspErrorHandler(), UnconvergedErrorHandler()]

        handlers = self.get("handlers", default_handlers)
        for handler in handlers:
            handler.output_filename = stdout_file

        jobs = [VaspJob(vasp_cmd=vasp_cmd,
                        output_file=stdout_file,
                        stderr_file=stderr_file,
                        auto_npar=False)]

        c = Custodian(handlers, jobs, max_errors=10,
                      monitor_freq=self.get("monitor_freq", 30))
        c.run()


@explicit_serialize
class VaspParallelizationTask(FiretaskBase):
    """
    Set up the parallelization setting for a VASP calculation. As I do not seem to
    be able to properly determine the number of irreducible kpoints that VASP uses
    based on the input files, this Firetask runs the VASP calculation until the
    IBZKPT file is created, and then reads the number of irreducible kpoints from
    this file.

    The current parallelization scheme first makes a list of KPAR values that:

        1. Is a divisor of the total number of cores.
        2. Do not waste too many resources over a single electronic step.

    Then, it looks for the largest KPAR value in this list for NCORE as close as
    possible to an optimal NCORE value. Based on tests on our machine, 7 is a good
    value to aim for, but this is most likely machine-dependent.

    If the calculation includes some Hartree-Fock mixing (AEXX != 0), NPAR is set
    as close as possible to an optimal value, once again maximizing KPAR without
    introducing too much waste of resources due to inactive cores. Based on our
    tests, 8 is a good NPAR value to aim for using when hybrid functionals.

    Optional params:
        directory (str): Directory of the VASP run. If not specified, the Task
            will run in the current directory.
        opt_band_parallel (int): Optimal value for NCORE (PBE) or NPAR (HSE),
            overriding the defaults.
        NBANDS (int): Set a restriction on the number of bands, i.e. NPAR must be
            a divisor of this number if it is set.
        KPAR (int): Override the KPAR value.
        NCORE (int): Override the NCORE value.

    """
    optional_params = ["directory", "opt_band_parallel", "NBANDS", "KPAR", "NCORE"]
    OPTIMAL_NCORE_DEFAULT_PBE = 7
    OPTIMAL_NPAR_DEFAULT_HSE = 8

    def run_task(self, fw_spec):

        directory = self.get("directory", os.getcwd())
        nbands = self.get("NBANDS", None)
        kpar = self.get("KPAR", None)
        ncore = self.get("NCORE", None)

        os.chdir(directory)

        is_hybrid = Incar.from_file("INCAR").get("AEXX", 0.0) != 0.0

        if is_hybrid:
            opt_band_parallel = self.get(
                "opt_band_parallel",
                VaspParallelizationTask.OPTIMAL_NPAR_DEFAULT_HSE
            )
        else:
            opt_band_parallel = self.get(
                "opt_band_parallel",
                VaspParallelizationTask.OPTIMAL_NCORE_DEFAULT_PBE
            )

        # Get the total number of nodes/cores
        try:
            number_of_nodes = int(os.environ["PBS_NUM_NODES"])
            number_of_cores = int(os.environ["PBS_NP"])
            cores_per_node = number_of_cores / number_of_nodes
        except KeyError:
            try:
                number_of_nodes = int(os.environ["SLURM_NNODES"])
                cores_per_node = int(os.environ["SLURM_CPUS_ON_NODE"])
                number_of_cores = number_of_nodes * cores_per_node
            except KeyError:
                raise NotImplementedError(
                    "The VaspParallelizationTask currently only supports "
                    "PBS and SLURM schedulers.")

        if kpar is None:

            if ncore is not None:
                raise NotImplementedError("Specifying NCORE but not KPAR is "
                                          "currently not possible.")

            stdout_file = "temp.out"
            stderr_file = "temp.out"
            vasp_cmd = fw_spec["_fw_env"]["vasp_cmd"].split(" ")

            try:
                os.remove("IBZKPT")
            except FileNotFoundError:
                pass

            # Get the number of k-points
            with open(stdout_file, 'w') as f_std, \
                    open(stderr_file, "w", buffering=1) as f_err:
                p = subprocess.Popen(vasp_cmd, stdout=f_std, stderr=f_err,
                                     preexec_fn=os.setsid)

                while not os.path.exists("IBZKPT"):
                    time.sleep(1)

                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                time.sleep(3)

            os.remove("temp.out")

            with open("IBZKPT", "r") as file:
                nkpts = int(file.read().split('\n')[1])

            with open("parallel.out", "w") as file:
                file.write("Number_of kpoints = " + str(nkpts) + "\n")

            kpar, ncore = self._optimize_parallelization(
                nkpts, nbands, number_of_cores, cores_per_node,
                opt_band_parallel, is_hybrid
            )

        elif ncore is None:
            optimal_ncore = opt_band_parallel if not is_hybrid else \
                number_of_cores // kpar // opt_band_parallel

            ncore = self._find_ncore(
                cores_per_k=number_of_cores // kpar,
                optimal_ncore=optimal_ncore,
                nbands=nbands
            )

        with open("parallel.out", "a+") as file:
            file.write("Number of cores = " + str(number_of_cores) + "\n")
            file.write("KPAR = " + str(kpar) + "\n")
            file.write("NPAR = " + str(number_of_cores // kpar // ncore) + "\n")
            file.write("NCORE = " + str(ncore) + "\n")

        self._set_incar_parallelization(kpar, ncore)

    def _set_incar_parallelization(self, kpar, ncore=None):

        directory = self.get("directory", os.getcwd())

        incar = Incar.from_file(os.path.join(directory, "INCAR"))
        if incar.get("ALGO", "Normal") == "Fast":
            warnings.warn("Based on our current tests, the VaspParallelizationTask "
                          "does not do a good job of optimizing the "
                          "parallelization settings for the RMM-DIIS algorithm.")
        incar.update({"KPAR": kpar})
        if ncore is not None:
            incar.update({"NCORE": ncore})
        incar.write_file(os.path.join(directory, "INCAR"))

    @staticmethod
    def _optimize_parallelization(nkpts, nbands, number_of_cores, cores_per_node,
                                  opt_band_parallel, is_hybrid):

        kpar_list = VaspParallelizationTask._find_kpar_list(
            nkpts, number_of_cores, cores_per_node
        )

        choice = {"kpar": 0, "band_parallel": 0}

        for k in kpar_list:

            band_parallel = VaspParallelizationTask._find_closest_divisor(
                value=number_of_cores // k,
                optimal_value=opt_band_parallel
            )
            accept = [k > choice["kpar"],
                      abs(band_parallel - opt_band_parallel) <= abs(
                          choice["band_parallel"] - opt_band_parallel)]

            if nbands is not None:
                accept.append(
                    (nbands % band_parallel == 0) if is_hybrid else
                    (nbands % (number_of_cores // k // band_parallel) == 0)
                )
            if all(accept):
                choice = {"kpar": k, "band_parallel": band_parallel}

        kpar = choice["kpar"]
        ncore = (choice["band_parallel"] if not is_hybrid else
                 number_of_cores // choice["kpar"] // choice["band_parallel"])

        return kpar, ncore

    @staticmethod
    def _find_kpar_list(n_kpoints, n_cores, cores_per_node):

        ncores_divisors = np.array(
            [i for i in list(range(1, n_cores + 1)) if n_cores % i == 0]
        )

        kpar_list = []

        # Only consider kpars for which not too much core run time is wasted
        for kpar in ncores_divisors:

            core_waste = VaspParallelizationTask._find_core_waste(
                n_kpoints, kpar, n_cores)

            too_much_waste = core_waste > cores_per_node
            waste_fraction = core_waste / n_cores

            too_much_waste = too_much_waste or waste_fraction >= 1 / 3

            if not too_much_waste:
                kpar_list.append(kpar)

        return kpar_list

    @staticmethod
    def _find_ncore(cores_per_k, optimal_ncore, nbands=None):

        divisors = np.array(
            [i for i in list(range(1, cores_per_k + 1)) if cores_per_k % i == 0]
        )

        if nbands:
            divisors = np.array(
                [i for i in divisors if nbands % (cores_per_k // i) == 0]
            )

        ncore = divisors[(np.abs(divisors - optimal_ncore)).argmin()]

        return ncore

    @staticmethod
    def _find_closest_divisor(value, optimal_value):

        divisors = np.array(
            [i for i in list(range(1, value + 1)) if value % i == 0]
        )
        return divisors[(np.abs(divisors - optimal_value)).argmin()]

    @staticmethod
    def _find_core_waste(n_kpoints, kpar, n_cores):

        if n_cores % kpar != 0:
            raise ValueError("KPAR is not a divisor of the number of available "
                             "cores!")

        kpar_groups = n_kpoints // kpar

        if n_kpoints % kpar != 0.0:
            kpar_groups += 1

        if n_kpoints % kpar == 0:
            lost_cores = 0
        else:
            lost_cores = (kpar - n_kpoints % kpar) * n_cores / kpar

        return lost_cores / kpar_groups


@explicit_serialize
class IncreaseNumberOfBands(FiretaskBase):
    """
    Increase the default number of bands included in a VASP calculation by
    multiplying it by a specified integer. Useful for calculations that require a
    large number of unoccupied bands, e.g. for calculating the dielectric tensor.

    Optional params:
        directory (str): Directory of the VASP run. If not specified, the Task
            will run in the current directory.
        multiplier (int): Multiplier used for increasing the VASP default number
            of bands.

    """
    optional_params = ["directory", "multiplier"]

    # TODO: Remove the need for a testrun by obtaining the number of electrons
    #  from the input files, as is done by the DictSet class.

    def run_task(self, fw_spec):

        directory = self.get("directory", os.getcwd())
        multiplier = self.get("multiplier", 3)

        os.chdir(directory)

        nelect_written = False

        try:
            with open("OUTCAR", "r") as file:
                if "NELECT" in file.read():
                    nelect_written = True
        except FileNotFoundError:
            pass

        if not nelect_written:

            # Do a trial run to figure out the number of standard bands
            stdout_file = "temp.out"
            stderr_file = "temp.out"
            vasp_cmd = fw_spec["_fw_env"]["vasp_cmd"].split(" ")

            with open(stdout_file, 'w') as f_std, \
                    open(stderr_file, "w", buffering=1) as f_err:
                p = subprocess.Popen(vasp_cmd, stdout=f_std, stderr=f_err,
                                     preexec_fn=os.setsid)

                while not nelect_written:
                    try:
                        with open("OUTCAR", "r") as file:
                            if "NELECT" in file.read():
                                nelect_written = True
                    except FileNotFoundError:
                        pass
                    time.sleep(1)

                os.killpg(os.getpgid(p.pid), signal.SIGTERM)

            time.sleep(3)
            os.remove(os.path.join(directory, "temp.out"))

        outcar = Outcar("OUTCAR")
        incar = Incar.from_file("INCAR")
        pattern = r"\s+NELECT\s=\s+(\d+).\d+\s+total\snumber\sof\selectrons"
        outcar.read_pattern({"nelect": pattern})

        nions = len(Structure.from_file("POSCAR"))
        nelect = int(outcar.data["nelect"][0][0])
        ispin = int(incar.get("ISPIN", 1))

        if ispin == 1:
            nbands = int(round(nelect / 2 + nions / 2)) * multiplier
        elif ispin == 2:
            nbands = int(nelect * 3 / 5 + nions) * multiplier
        else:
            raise ValueError("ISPIN Value is not set to 1 or 2!")

        incar.update({"NBANDS": nbands})
        incar.write_file("INCAR")


@explicit_serialize
class WriteVaspFromIOSet(FiretaskBase):
    """
    Create VASP input files using implementations of pymatgen's VaspInputSet.
    An input set can be provided as an object or as a String/parameter combo.

    Notes: 
        - If a full initialized VaspInputSet is passed for the vasp_input_set
        argument, the optional arguments are not necessary and hence ignored.
        Make sure that you've passed the preferred vasp_input_params to the
        VaspInputSet!
        - Even though both 'structure' and 'parent' are optional parameters, 
        at least one of them must be set in case the vasp_input_set is defined as 
        a string. Else the Firetask will not know which geometry to set up the 
        calculation for.

    Required params:
        vasp_input_set (VaspInputSet or str): Either a VaspInputSet instance
            or a string name for the VASP input set. Best practise is to provide
            the full module path (e.g. pymatgen.io.vasp.sets.MPRelaxSet). It's
            also possible to only provide the class name (e.g. MPRelaxSet). In
            this case the Task will look for the set in our list of sets and then
            the pymatgen sets.

    Optional params:
        structure (Structure): Input geometry.
        parent (Firework): A single parent firework from which to extract the
            final geometry.
        vasp_input_params (dict): Only when using a string name for VASP input
            set, use this as a dict to specify kwargs for instantiating the input
            set parameters. For example, if you want to change the
            user_incar_settings, you should provide: {"user_incar_settings": ...}
            This setting is ignored if you provide an instance of a VaspInputSet
            rather than a String.

    """
    required_params = ["vasp_input_set"]
    optional_params = ["structure", "parents", "vasp_input_params"]

    def run_task(self, fw_spec):
        # If a full VaspInputSet object was provided
        if hasattr(self['vasp_input_set'], 'write_input'):
            input_set = self['vasp_input_set']

            # Check if the user has also provided optional params
            if any(i in self.keys() for i in WriteVaspFromIOSet.optional_params):
                warnings.warn("Vasp input set was provided as an instance of a "
                              "VaspInputSet, however optional parameter were also "
                              "specified. These will not be used to overwrite the "
                              "settings specified in the VaspInputSet, and will "
                              "hence be ignored!")  # TODO: fix this

        # If VaspInputSet String + parameters was provided
        else:
            # If the user has provided a full module path to class
            if "." in self["vasp_input_set"]:
                classname = self["vasp_input_set"].split(".")[-1]
                modulepath = ".".join(self["vasp_input_set"].split(".")[:-1])
                input_set_cls = load_class(modulepath, classname)
            else:
                # Try our sets first
                try:
                    input_set_cls = load_class("vscworkflows.setup.sets",
                                               self["vasp_input_set"])
                # Check the pymatgen sets for the requested set
                except ModuleNotFoundError:
                    input_set_cls = load_class("pymatgen.io.vasp.sets",
                                               self["vasp_input_set"])

            if "structure" in self.keys():
                input_set = input_set_cls(self["structure"],
                                          **self.get("vasp_input_params", {}))
            elif "parents" in self.keys():
                try:
                    structure = fw_spec["final_geometry"]
                except KeyError:
                    try:
                        parent_dir = self["parents"]["spec"]["_launch_dir"]
                    except KeyError:
                        # TODO Check this branch for atomate parents
                        parent_dir = self["parents"]["launches"][-1]["launch_dir"]
                    structure = _load_structure_from_dir(parent_dir)

                input_set = input_set_cls(structure,
                                          **self.get("vasp_input_params", {}))
            elif fw_spec["parents"]:
                parent_dir = fw_spec["parents"]["spec"]["_launch_dir"]
                structure = _load_structure_from_dir(parent_dir)
                input_set = input_set_cls(structure,
                                          **self.get("vasp_input_params", {}))
            else:
                raise ValueError("You must provide either an input structure or "
                                 "parent firework to WriteVaspFromIOSet!")

        input_set.write_input(".")


@explicit_serialize
class AddFinalGeometryToSpec(FiretaskBase):
    """
    Add the final geometry in the current or specified directory to the FireTask.
    
    Optional params:
        directory (str): Path to the directory which the geometry should be 
            extracted from.
        
    """  # TODO: remove the Final in the name of the firetask
    required_params = []
    optional_params = ["directory"]

    def run_task(self, fw_spec):
        directory = self.get("directory", os.getcwd())

        structure = _load_structure_from_dir(directory)
        return FWAction(update_spec={"final_geometry": structure})


@explicit_serialize
class PulayTask(FiretaskBase):
    """
    Check a geometry optimization to see if an extra optimization run might be
    necessary to avoid Pulay stresses, based on a specified condition/tolerance.
    If so, start a new geometry optimization with the final structure in the same
    directory.

    Required params:
        None

    Optional params:
        directory (str): Directory in which the geometry optimization calculation
            was run.
        custodian (bool): Flag that indicates whether the calculation should be
            run inside a Custodian.
        condition (str): Condition that determines whether or not an additional
            geometry optimization should be performed. There are several options:

            "ionic_steps" - (default) Maximum number of ionic steps.
            "energy" - Maximum energy difference between the initial and final
                geometry of the optimization, expressed in meV/atom.
            "lattice" - Maximum allowed 2-norm of the matrix defined by taking
                the difference between the initial and final matrices
                constructed from the lattice vectors.
        tolerance (float): Maximum allowed value for the condition. If this value
            is exceeded, an extra geometry optimization is performed.

    """
    option_params = ["directory", "custodian", "condition", "tolerance"]

    # Standard tolerances for deciding to perform another geometry optimization.
    pulay_tolerance_dict = {"ionic_steps": 1, "energy": 1e-3, "lattice": 5e-2}

    def run_task(self, fw_spec):

        directory = self.get("directory", os.getcwd())
        custodian = self.get("custodian", False)
        condition = self.get("condition", "energy") or "energy"
        tolerance = self.get(
            "tolerance", PulayTask.pulay_tolerance_dict[condition]
        ) or PulayTask.pulay_tolerance_dict[condition]

        perform_pulay_step = False

        if condition == "ionic_steps":

            vasprun = Vasprun(os.path.join(directory, "vasprun.xml"))

            if vasprun.nionic_steps > tolerance:
                print("Number of ionic steps of geometry optimization is more "
                      "than specified tolerance (" + str(tolerance) +
                      "). Performing another geometry optimization.")
                perform_pulay_step = True

        elif condition == "energy":

            vasprun = Vasprun(os.path.join(directory, "vasprun.xml"))

            ionic_energies = [step['e_wo_entrp'] for step in vasprun.ionic_steps]
            structure = vasprun.final_structure

            if abs(ionic_energies[-1] - ionic_energies[0]) / len(structure) \
                    > tolerance:
                print("Difference in energy per atom between first ionic step and "
                      "final ionic step is larger than specified tolerance (" +
                      str(tolerance) + "). Performing another geometry "
                                       "optimization.")
                perform_pulay_step = True

        elif condition == "lattice":

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
            if sum_differences > tolerance:
                print("Lattice vectors have changed significantly during geometry "
                      "optimization. Performing another full geometry optimization "
                      "to make sure there were no Pulay stresses present.\n\n")
                perform_pulay_step = True

        if perform_pulay_step:

            tasks = list()

            # Change to quasi-Newton scheme
            incar = Incar.from_file(os.path.join(directory, "INCAR"))
            incar.update({"IBRION": 1})
            incar.write_file(os.path.join(directory, "INCAR"))

            # Create the ScriptTask that copies the CONTCAR to the POSCAR
            tasks.append(ScriptTask.from_str(
                "cp " + os.path.join(directory, "CONTCAR") +
                " " + os.path.join(directory, "POSCAR")
            ))

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
            tasks.append(AddFinalGeometryToSpec(directory=directory))

            # Create the PyTask that check the Pulay stresses again
            tasks.append(PulayTask(
                directory=directory, custodian=custodian,
                condition=condition, tolerance=tolerance
            ))

            # Combine the two FireTasks into one FireWork
            optimize_fw = Firework(tasks=tasks,
                                   name="Pulay Step",
                                   spec=fw_spec)

            return FWAction(detours=optimize_fw)
