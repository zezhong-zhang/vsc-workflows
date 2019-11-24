# coding: utf8
# Copyright (c) Marnik Bercx, University of Antwerp
# Distributed under the terms of the MIT License

import os
import warnings

from monty.serialization import loadfn
from pymatgen.io.vasp.inputs import Poscar, Kpoints
from pymatgen.io.vasp.sets import DictSet

"""
Defines the various input sets used for setting up the calculations.

"""

__author__ = "Marnik Bercx"
__copyright__ = "Copyright 2019, Marnik Bercx, University of Antwerp"
__version__ = "pre-alpha"
__maintainer__ = "Marnik Bercx"
__email__ = "marnik.bercx@uantwerpen.be"
__date__ = "Jul 2019"

MODULE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "set_configs")


def _load_yaml_config(fname):
    config = loadfn(os.path.join(MODULE_DIR, "%s.yaml" % fname))
    return config


def _set_structure_incar_settings(structure, config_dict):
    """
    Set up the default VASP INCAR settings based on the structure.

    Args:
        structure (Structure): Input geometry.

    Returns:
        dict: dictionary with the standard vasp input parameters.

    """
    # Check if a magnetic moment was provided for the sites. If so, perform a
    # spin-polarized calculation
    if "magmom" in structure.site_properties.keys():
        config_dict["INCAR"].update({"ISPIN": 2, "MAGMOM": True})

        # TODO Add non-collinear functionality
        # TODO Consider user_incar_settings MAGMOM -> What if None/True?

    # Adjust the projector-evaluation scheme to Auto for large unit cells (+20 atoms)
    if len(structure) > 20:
        config_dict["INCAR"].update({"LREAL": "Auto"})

    return config_dict


class BulkStaticSet(DictSet):
    """
    VASP input set for a bulk static calculation.

    """

    def __init__(self, structure, **kwargs):

        config_dict = _set_structure_incar_settings(
            structure=structure, config_dict=_load_yaml_config("staticSet")
        )
        super().__init__(structure=structure, config_dict=config_dict, **kwargs)
        self.kwargs = kwargs

    @property
    def kpoints(self):
        """
        Sets up the k-points for the static calculation.

        Returns:
            :class: pymatgen.io.vasp.inputs.Kpoints

        """
        settings = self.user_kpoints_settings or self._config_dict["KPOINTS"]

        if "k_resolution" in settings:
            # Use k_resolution to calculate kpoints
            k_kpoint_resolution = settings["k_resolution"]
            kpt_divisions = [round(l / k_kpoint_resolution + 0.5) for l in
                             self.structure.lattice.reciprocal_lattice.lengths]

            return Kpoints.gamma_automatic(kpts=kpt_divisions)
        else:
            return super().kpoints


class BulkOptimizeSet(DictSet):
    """
    VASP input set for the bulk geometry optimization.

    """

    def __init__(self, structure, **kwargs):
        config_dict = _set_structure_incar_settings(
            structure=structure, config_dict=_load_yaml_config("relaxSet")
        )
        super().__init__(structure=structure, config_dict=config_dict, **kwargs)
        self.kwargs = kwargs

    @property
    def kpoints(self):
        """
        Sets up the k-points for the static calculation.

        Returns:
            :class: pymatgen.io.vasp.inputs.Kpoints

        """
        settings = self.user_kpoints_settings or self._config_dict["KPOINTS"]

        if "k_resolution" in settings:
            # Use k_resolution to calculate kpoints
            kpt_divisions = [round(l / settings["k_resolution"] + 0.5) for l in
                             self.structure.lattice.reciprocal_lattice.lengths]

            return Kpoints.gamma_automatic(kpts=kpt_divisions)
        else:
            return super().kpoints


class SlabStaticSet(DictSet):

    def __init__(self, structure, **kwargs):

        config_dict = _set_structure_incar_settings(
            structure=structure, config_dict=_load_yaml_config("staticSet")
        )
        super().__init__(structure=structure, config_dict=config_dict, **kwargs)

        # Default settings for a static slab calculation
        incar_defaults = {"ISMEAR": 0, "SIGMA": 0.05, "SYMPREC": 1e-8,
                          "LREAL": "Auto", "NELM": 300}

        self._config_dict["INCAR"].update(incar_defaults)
        self._config_dict["KPOINTS"].update({"k_resolution": 0.2})
        self.kwargs = kwargs

    @property
    def kpoints(self):
        """
        Sets up the k-points for the static calculation.

        Returns:
            :class: pymatgen.io.vasp.inputs.Kpoints

        """
        settings = self.user_kpoints_settings or self._config_dict["KPOINTS"]

        if "k_resolution" in settings:

            # Use k_resolution to calculate kpoints
            kpt_divisions = [round(l / settings["k_resolution"] + 0.5) for l in
                             self.structure.lattice.reciprocal_lattice.lengths]
            kpt_divisions[2] = 1  # Only one k-point in c-direction for slab

            return Kpoints.gamma_automatic(kpts=kpt_divisions)
        else:
            kpoints = super().kpoints
            kpoints.kpts[0][2] = 1  # Only one k-point in c-direction for slab
            return kpoints


class SlabOptimizeSet(DictSet):
    """
    A VASP input set that is used to optimize a slab structure.

    """

    def __init__(self, structure, user_slab_settings=None, **kwargs):
        config_dict = _set_structure_incar_settings(
            structure=structure, config_dict=_load_yaml_config("relaxSet")
        )
        super().__init__(structure=structure, config_dict=config_dict, **kwargs)

        # Defaults for a slab optimization
        defaults = {"ISIF": 2, "SYMPREC": 1e-8, "LREAL": "Auto"}

        self._config_dict["INCAR"].update(defaults)
        self._config_dict["KPOINTS"].update({"k_resolution": 0.3})
        self.kwargs = kwargs

        self.user_slab_settings = user_slab_settings
        self.selective_dynamics = None
        if user_slab_settings is not None:
            try:
                self.fix_slab_bulk(**user_slab_settings)
            except TypeError:
                raise ValueError("No 'free_layers' specified in user_slab_settings! "
                                 "As currently the only purpose for this argument "
                                 "is to apply selective dynamics on a slab "
                                 "geometry optimization, this key must be assigned "
                                 "a value.")

    def fix_slab_bulk(self, free_layers, optimize_both_sides=True):
        """
        Fix atoms of the slab to represent the bulk of the material.

        Args:
            free_layers (int): Number of free atomic layers at the surface.
            optimize_both_sides (bool): Optimize both sides of the slab. Defaults
                to True.
        """
        # Set up the selective dynamics property
        selective_dynamics = []

        atomic_layers = self.structure.find_atomic_layers()
        if optimize_both_sides:
            fixed_layers = atomic_layers[free_layers:-free_layers]
        else:
            fixed_layers = atomic_layers[free_layers:]

        if len(fixed_layers) == 0:
            warnings.warn("No layers fixed in 'fix_slab_bulk' method!")
        else:
            # Combine the sites of the fixed layers into one list
            fixed_sites = [site for layer in fixed_layers for site in layer]

            for site in self.structure.sites:
                if site in fixed_sites:
                    selective_dynamics.append([False, False, False])
                else:
                    selective_dynamics.append([True, True, True])

        self.selective_dynamics = selective_dynamics

    @property
    def poscar(self):
        """
        Similar to the standard POSCAR, but with selective dynamics added.

        Returns:
            :class: pymatgen.io.vasp.inputs.Poscar
        """
        return Poscar(self.structure,
                      selective_dynamics=self.selective_dynamics)

    @property
    def kpoints(self):
        """
        Sets up the k-points for the static calculation.

        Returns:
            :class: pymatgen.io.vasp.inputs.Kpoints

        """
        settings = self.user_kpoints_settings or self._config_dict["KPOINTS"]

        if "k_resolution" in settings:

            # Use k_resolution to calculate kpoints
            kpt_divisions = [round(l / settings["k_resolution"] + 0.5) for l in
                             self.structure.lattice.reciprocal_lattice.lengths]
            kpt_divisions[2] = 1  # Only one k-point in c-direction for slab

            return Kpoints.gamma_automatic(kpts=kpt_divisions)
        else:
            kpoints = super().kpoints
            kpoints.kpts[0][2] = 1  # Only one k-point in c-direction for slab
            return kpoints

class MDSet(DictSet):
    """
     Class for writing a vasp md run. This DOES NOT do multiple stage
     runs.
     """

    def __init__(self, structure, ensemble, thermostat,
                 start_temp, end_temp, nsteps, time_step=2,
                 spin_polarized=False, **kwargs):
        """
        Args:
            structure (Structure): Input structure.
            ensemble: ensemble (string) default here choose is 'NVT'
            see https://www.vasp.at/wiki/index.php/Category:Ensembles
            thermostat: thermostat (string) default here choose is 'Nose-Hoover'
            see # see https://www.vasp.at/wiki/index.php/Category:Thermostats
            start_temp (int): Starting temperature.
            end_temp (int): Final temperature.
            nsteps (int): Number of time steps for simulations. NSW parameter.
            time_step (int): The time step for the simulation. The POTIM
                parameter. Defaults to 2fs.
            spin_polarized (bool): Whether to do spin polarized calculations.
                The ISPIN parameter. Defaults to False.
            **kwargs: Other kwargs supported by :class:`DictSet`.
        """
        # load MDSet
        config_dict = _set_structure_incar_settings(
            structure=structure, config_dict=_load_yaml_config("MDSet")
        )

        super().__init__(structure=structure, config_dict=config_dict, **kwargs)

        self.start_temp = start_temp
        self.end_temp = end_temp
        self.nsteps = nsteps
        self.time_step = time_step
        self.spin_polarized = spin_polarized
        self.kwargs = kwargs
        self.ensemble = ensemble
        self.thermostat = thermostat

        # MD default settings
        defaults = {'TEBEG': start_temp, 'TEEND': end_temp, 'NSW': nsteps,
                    'POTIM': time_step,
                    'ISPIN': 2 if spin_polarized else 1}

        # choose ensemble and thermostat
        ensemble_option = ["NVE", "NVT", "NPT", "NPH"]
        thermostat_option = ["Andersen", "Nose-Hoover", "Langevin", "Multiple Andersen"]

        if ensemble is None:
            ensemble = "NVT"
        if ensemble not in ensemble_option:
            raise ValueError('ensemble not in the list of '
                             '["NVE", "NVT", "NPT", "NPH"]')
        if thermostat is None:
            thermostat = "Nose-Hoover"
        if thermostat not in thermostat_option:
            raise ValueError('thermostat not in the list of '
                             '["Andersen", "Nose-Hoover", "Langevin", "Multiple Andersen"]')

        # update according to ensemble and thermostat
        ensemble_thermostat_combo = dict()
        if ensemble == "NVE":
            ensemble_thermostat_combo = {'MDALGO':0, 'SMASS':-3}
        elif ensemble == "NVT":
            if thermostat == "Andersen":
                ensemble_thermostat_combo = {'MDALGO':1, 'ISIF':2}
            if thermostat == "Nose-Hoover":
                ensemble_thermostat_combo = {'MDALGO':2, 'ISIF':2}
            if thermostat == "Langevin":
                ensemble_thermostat_combo = {'MDALGO':3, 'ISIF':2}
            if thermostat == "Multiple Andersen":
                ensemble_thermostat_combo = {'MDALGO':13, 'ISIF':2}
        elif ensemble == "NPT":
            if thermostat == "Langevin":
                ensemble_thermostat_combo ={'MDALGO':13, 'ISIF':3}
            else:
                raise ValueError('NPT is only availiable for Langevin thermostat'
                                 'see https://www.vasp.at/wiki/index.php/Category:Thermostats')
        elif ensemble == "NPH":
            ensemble_thermostat_combo = {'MDALGO':13, 'ISIF':3, 'LANGEVIN_GAMMA_L': 0.0}
        defaults.update(ensemble_thermostat_combo)

        # finally update the INCAR with defaults, remove spin and magnetic moment if not spin_polarized
        if defaults['ISPIN'] == 1:
            self._config_dict["INCAR"].pop('MAGMOM', None)
        self._config_dict["INCAR"].update(defaults)

    @property
    def kpoints(self):
        """
        Sets up the k-points for the static calculation.

        Returns:
            :class: pymatgen.io.vasp.inputs.Kpoints

        """
        settings = self.user_kpoints_settings or self._config_dict["KPOINTS"]

        if "k_resolution" in settings:
            # Use k_resolution to calculate kpoints
            k_kpoint_resolution = settings["k_resolution"]
            kpt_divisions = [round(l / k_kpoint_resolution + 0.5) for l in
                             self.structure.lattice.reciprocal_lattice.lengths]

            return Kpoints.gamma_automatic(kpts=kpt_divisions)
        else:
            return super().kpoints
