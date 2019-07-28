# coding: utf8
# Copyright (c) Marnik Bercx, University of Antwerp
# Distributed under the terms of the MIT License

import os

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


class SlabStaticSet(DictSet):

    def __init__(self, structure, k_resolution=0.1, **kwargs):

        config_dict = _set_structure_incar_settings(
            structure=structure, config_dict=_load_yaml_config("staticSet")
        )
        super().__init__(structure=structure, config_dict=config_dict, **kwargs)

        # Default settings for a static slab calculation
        defaults = {"AMIN": 0.01, "AMIX": 0.2, "BMIX": 0.001, "ISMEAR": 0,
                    "SIGMA": 0.01, "SYMPREC": 1e-8}

        self._config_dict["INCAR"].update(defaults)
        self.k_resolution = k_resolution
        self.kwargs = kwargs

    @property
    def kpoints(self):
        """
        Sets up the k-points for the static calculation.

        Returns:
            :class: pymatgen.io.vasp.inputs.Kpoints

        """
        if self.user_kpoints_settings is not None:
            return super().kpoints
        else:
            # Use k_resolution to calculate kpoints
            kpt_divisions = [int(l / self.k_resolution + 0.5) for l in
                             self.structure.lattice.reciprocal_lattice.lengths]
            kpt_divisions[2] = 1  # Only one k-point in c-direction for slab

            kpoints = Kpoints.gamma_automatic(kpts=kpt_divisions)

            return kpoints


class SlabOptimizeSet(DictSet):
    """
    A VASP input set that is used to optimize a slab structure.

    """

    def __init__(self, structure, k_resolution=0.2, user_slab_settings=None,
                 **kwargs):
        config_dict = _set_structure_incar_settings(
            structure=structure, config_dict=_load_yaml_config("relaxSet")
        )
        super().__init__(structure=structure, config_dict=config_dict, **kwargs)

        # Defaults for a slab optimization
        defaults = {"ISIF": 2, "AMIN": 0.01, "AMIX": 0.2, "BMIX": 0.001,
                    "SYMPREC": 1e-8}

        self._config_dict["INCAR"].update(defaults)
        self.kwargs = kwargs
        self.k_resolution = k_resolution
        self.user_slab_settings = user_slab_settings
        self.selective_dynamics = None
        if user_slab_settings is not None:
            try:
                self.fix_slab_bulk(**user_slab_settings)
            except TypeError:
                raise ValueError("No 'thickness' specified in user_slab_settings. "
                                 "As currently the only purpose for this argument "
                                 "is to apply selective dynamics on a slab "
                                 "geometry optimization, this key must be assigned "
                                 "a value")

    # TODO: Think over these arguments, so they are more intuitive for the user
    def fix_slab_bulk(self, thickness, method="layers", part="center"):
        """
        Fix atoms of the slab to represent the bulk of the material. Which atoms are
        fixed depends on whether the user wants to fix one side or the center, and
        how exactly the part of the slab is defined.

        Args:
            thickness (float): The thickness of the fixed part of the slab,
                expressed in number of layers or Angstroms, depending on the
                method.

            method (string): How to define the thickness of the part of the slab
                that is fixed:

                    "layers" (default): Fix a set amount of layers. The layers are
                    found using the 'find_atomic_layers' method.
                    "angstroms": Fix a part of the slab of a thickness defined in
                    angstroms.

            part (string): Which part of the slab to fix:

                    "center" (default): Fix the atoms at the center of the slab.
        """

        if method == "layers":

            atomic_layers = self.structure.find_atomic_layers()

            if part == "center":

                # Even number of layers
                if len(atomic_layers) % 2 == 0:

                    # Check if the user requested an odd number of layers for the
                    # fixed part of the slab
                    if thickness % 2 == 1:
                        print("Found an even number of layers, but the user " +
                              "requested an odd number of fixed layers. Adding "
                              "one layer to the fixed part of the slab.")
                        thickness += 1

                # Odd number of layers
                if len(atomic_layers) % 2 == 1:

                    # Check if the user requested an even number of layers for the
                    # fixed part of the slab
                    if thickness % 2 == 0:
                        print("Found an odd number of layers, but the user " +
                              "requested an even number of fixed layers. Adding "
                              "one layer to the fixed part of the slab.")
                        thickness += 1

                # Calculate the number of layers to optimize on each site
                n_optimize_layers = int((len(atomic_layers) - thickness) / 2)

                if n_optimize_layers < 5:
                    print("WARNING: Less than 5 layers are optimized on each "
                          "side of the slab.")  # TODO: make proper warning

                # Take the fixed layers from the atomic layers of the slab
                fixed_layers = atomic_layers[n_optimize_layers: n_optimize_layers +
                                                                thickness]

            else:
                raise NotImplementedError("Requested part is not implemented " +
                                          "(yet).")
                # TODO Implement oneside

            # Combine the sites of the fixed layers into one list
            fixed_sites = [site for layer in fixed_layers for site in layer]

            # Set up the selective dynamics property
            selective_dynamics = []

            for site in self.structure.sites:
                if site in fixed_sites:
                    selective_dynamics.append([False, False, False])
                else:
                    selective_dynamics.append([True, True, True])

            self.selective_dynamics = selective_dynamics

        else:
            raise NotImplementedError("Requested method is not implemented (yet).")
            # TODO Implement angstrom

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
        if self.user_kpoints_settings is not None:
            return super().kpoints
        else:
            # Use k_resolution to calculate kpoints
            kpt_divisions = [int(l / self.k_resolution + 0.5) for l in
                             self.structure.lattice.reciprocal_lattice.lengths]
            kpt_divisions[2] = 1  # Only one k-point in c-direction for slab

            kpoints = Kpoints.gamma_automatic(kpts=kpt_divisions)

            return kpoints
