# coding: utf8
# Copyright (c) Marnik Bercx, University of Antwerp
# Distributed under the terms of the MIT License

import os

from quotas import QSlab

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
__date__ = "Jun 2019"

MODULE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "set_configs")


def _load_yaml_config(fname):
    config = loadfn(os.path.join(MODULE_DIR, "%s.yaml" % fname))
    return config


class BulkRelaxSet(DictSet):
    """
    VASP input set for the bulk geometry optimization.

    """
    CONFIG = _load_yaml_config("relaxSet")

    def __init__(self, structure, **kwargs):
        super(BulkRelaxSet, self).__init__(
            structure, BulkRelaxSet.CONFIG, **kwargs)
        self.kwargs = kwargs


class SlabRelaxSet(DictSet):
    """
    A VASP input set that is used to optimize a slab structure.

    """

    CONFIG = _load_yaml_config("relaxSet")

    def __init__(self, structure, k_product=50, **kwargs):
        super(SlabRelaxSet, self).__init__(structure=structure,
                                           config_dict=SlabRelaxSet.CONFIG,
                                           **kwargs)

        # Defaults for a slab optimization
        defaults = {"ISIF": 2, "AMIN": 0.01, "AMIX": 0.2, "BMIX": 0.001}

        self.structure = QSlab.from_slab(self.structure.copy())
        # This step is unfortunately necessary because the `get_sorted_structure()`
        # method of `pymatgen.core.surface.Slab` returns a Slab object. It would be
        # better if this method used self.__class__() instead of using Slab() to
        # initialize the returned structure.
        self._config_dict["INCAR"].update(defaults)
        self.k_product = k_product
        self.selective_dynamics = None
        self.kwargs = kwargs

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
                          "side of the slab.")

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
        Sets up the k-points for the slab relaxation.

        For slabs, the number of
        k-points corresponding to the third reciprocal lattice vector is set
        to 1. The number of k-point divisions in the other two directions is
        determined by the length of the corresponding lattice vector, by making
        sure the product of the number of k-points and the length of the
        corresponding lattice vector is equal to k_product, defined in the
        initialization of the slab calculation.

        Returns:
            :class: pymatgen.io.vasp.inputs.Kpoints

        """
        # If the user requested a specific kpoint setting
        if self.user_kpoints_settings is not None:
            # Use the setting requested
            return super(SlabRelaxSet, self).kpoints()
        else:
            # Use k_product to calculate kpoints
            abc = self.structure.lattice.abc
            kpt_calc = [int(self.k_product / abc[0] + 0.5),
                        int(self.k_product / abc[1] + 0.5), 1]

            kpoints = Kpoints.gamma_automatic(kpts=kpt_calc)

            return kpoints
