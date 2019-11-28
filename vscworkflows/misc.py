# coding: utf8
# Copyright (c) Marnik Bercx, University of Antwerp
# Distributed under the terms of the MIT License

import json
import os
import warnings
from fnmatch import fnmatch

import numpy as np
from monty.io import zopen
from monty.json import MontyDecoder, MontyEncoder
from pymatgen import Lattice, PeriodicSite
from pymatgen.analysis.chemenv.coordination_environments.voronoi \
    import DetailedVoronoiContainer
from pymatgen.core import Structure, Composition, Site, Element
from pymatgen.core.surface import Slab
from pymatgen.io.vasp.outputs import Outcar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from tabulate import tabulate

"""
Miscellaneous classes and methods for which I can't find a better location. 

"""


class QSlab(Slab):
    """
    A Quotas version of the pymatgen.core.surface.Slab object. All methods can be
    inherited, but we need to add some convenience methods of our own.

    """

    def __init__(self, lattice, species, coords, miller_index,
                 oriented_unit_cell, shift, scale_factor, reorient_lattice=True,
                 validate_proximity=False, to_unit_cell=False,
                 reconstruction=None, coords_are_cartesian=False,
                 site_properties=None, energy=None):
        super(QSlab, self).__init__(lattice, species, coords, miller_index,
                                    oriented_unit_cell, shift, scale_factor,
                                    reorient_lattice, validate_proximity,
                                    to_unit_cell,
                                    reconstruction, coords_are_cartesian,
                                    site_properties, energy)

    @classmethod
    def from_file(cls, filename, primitive=False, sort=False, merge_tol=0.0):
        """
        Load the QSlab from a file.

        Args:
            filename (str): Path to file that contains the details of the slab.
            primitive (bool):
            sort (bool):
            merge_tol (float):

        Returns:

        """
        if fnmatch(os.path.basename(filename).lower(), "*.json"):
            with zopen(filename, "r") as file:
                return cls.from_str(file.read())
        else:
            raise NotImplementedError("Only .json files are currently supported.")

    def as_dict(self):
        d = super(Slab, self).as_dict()
        d["@module"] = self.__class__.__module__
        d["@class"] = self.__class__.__name__
        d["oriented_unit_cell"] = self.oriented_unit_cell.as_dict()
        d["miller_index"] = self.miller_index
        d["shift"] = self.shift
        d["scale_factor"] = MontyEncoder().default(self.scale_factor)
        d["reconstruction"] = self.reconstruction
        d["energy"] = self.energy
        return d

    @classmethod
    def from_dict(cls, d):
        lattice = Lattice.from_dict(d["lattice"])
        sites = [PeriodicSite.from_dict(sd, lattice) for sd in d["sites"]]
        s = Structure.from_sites(sites)

        return cls(
            lattice=lattice,
            species=s.species_and_occu, coords=s.frac_coords,
            miller_index=d["miller_index"],
            oriented_unit_cell=Structure.from_dict(d["oriented_unit_cell"]),
            shift=d["shift"],
            scale_factor=MontyDecoder().process_decoded(d["scale_factor"]),
            site_properties=s.site_properties, energy=d["energy"]
        )

    @classmethod
    def from_str(cls, input_string, fmt="json", primitive=False, sort=False,
                 merge_tol=0.0):

        if fmt == "json":
            return QSlab.from_slab(json.loads(input_string, cls=MontyDecoder))
        else:
            raise NotImplementedError("Currently only the json format is supported.")

    def to(self, fmt=None, filename=None, **kwargs):
        """
        Write the QSlab to a file.

        Args:
            fmt:
            filename:

        Returns:

        """
        if fmt == "json":
            with open(filename, "w") as file:
                file.write(self.to_json())
        else:
            super(QSlab, self).to(fmt, filename, **kwargs)

    def get_sorted_structure(self, key=None, reverse=False):
        """
        Get a sorted copy of the structure. The parameters have the same
        meaning as in list.sort. By default, sites are sorted by the
        electronegativity of the species. Note that Slab has to override this
        because of the different __init__ args.

        Args:
            key: Specifies a function of one argument that is used to extract
                a comparison key from each list element: key=str.lower. The
                default value is None (compare the elements directly).
            reverse (bool): If set to True, then the list elements are sorted
                as if each comparison were reversed.
        """
        sites = sorted(self, key=key, reverse=reverse)
        s = Structure.from_sites(sites)
        return self.__class__(s.lattice, s.species_and_occu, s.frac_coords,
                              self.miller_index, self.oriented_unit_cell, self.shift,
                              self.scale_factor, site_properties=s.site_properties,
                              reorient_lattice=self.reorient_lattice)

    def copy(self, site_properties=None, sanitize=False):
        """
        Convenience method to get a copy of the structure, with options to add
        site properties.

        Args:
            site_properties (dict): Properties to add or override. The
                properties are specified in the same way as the constructor,
                i.e., as a dict of the form {property: [values]}. The
                properties should be in the order of the *original* structure
                if you are performing sanitization.
            sanitize (bool): If True, this method will return a sanitized
                structure. Sanitization performs a few things: (i) The sites are
                sorted by electronegativity, (ii) a LLL lattice reduction is
                carried out to obtain a relatively orthogonalized cell,
                (iii) all fractional coords for sites are mapped into the
                unit cell.

        Returns:
            A copy of the Structure, with optionally new site_properties and
            optionally sanitized.
        """
        props = self.site_properties
        if site_properties:
            props.update(site_properties)
        return self.__class__(self.lattice, self.species_and_occu, self.frac_coords,
                              self.miller_index, self.oriented_unit_cell, self.shift,
                              self.scale_factor, site_properties=props,
                              reorient_lattice=self.reorient_lattice)

    @classmethod
    def from_slab(cls, slab):
        return cls(lattice=slab.lattice, species=slab.species,
                   coords=slab.frac_coords,
                   miller_index=slab.miller_index,
                   oriented_unit_cell=slab.oriented_unit_cell,
                   shift=slab.shift,
                   scale_factor=slab.scale_factor,
                   reconstruction=slab.reconstruction,
                   coords_are_cartesian=False,
                   site_properties=slab.site_properties, energy=slab.energy)

    def find_atomic_layers(self, layer_tol=2e-2):
        """
            Determines the atomic layers in the c-direction of the slab. Note that as
            long as a site is "close enough" to ONE other site of a layer (determined
            by the 'layer_tol' variable), it will be added to that layer.

            Another option would be to demand that the distance is smaller than
            'layer_tol' for ALL sites of the layer, but then the division in layers
            could depend on the order of the sites.

            Args:
                layer_tol (float): Tolerance for the maximum distance (in
                    angstrom) between layers for them to still correspond to the same
                    layer.

            Returns:
                (list) List of the atomic layers, sorted by their position in the c-direction.
                Each atomic layer is also represented by a list of sites.
            """

        atomic_layers = []

        # Get a unit vector perpendicular to the layers (i.e. the a and b lattice vector)
        m = self.lattice.matrix
        u = np.cross(m[0, :], m[1, :])
        u /= np.linalg.norm(u)
        c_proj = np.dot(u, m[2, :])

        for site in self.sites:

            is_in_layer = False

            # Check to see if the site is in a layer that is already in our list
            for layer in atomic_layers:

                # Compare the third fractional coordinate of the site with that of
                # the atoms in the considered layer
                for atom_site in layer.copy():

                    distance = abs(atom_site.frac_coords[2] - site.frac_coords[2]) * \
                               c_proj

                    if distance < layer_tol or abs(distance - c_proj) < layer_tol:
                        is_in_layer = True
                        layer.append(site)
                        break  # Break out of the loop, else the site is added
                        # multiple times

            # If the site is not found in any of the atomic layers, create a new
            # atomic layer
            if not is_in_layer:
                atomic_layers.append([site, ])

        # Sort the atomic layers
        atomic_layers.sort(key=lambda layer: layer[0].frac_coords[2])

        return atomic_layers

    def update_sites(self, directory, ignore_magmom=False):
        """
        Based on the CONTCAR and OUTCAR of a VASP calculation, update the
        site coordinates and magnetic moments of the slab.

        Args:
            directory (str): Directory in which the calculation output files (i.e.
                CONTCAR and OUTCAR) are stored.
            ignore_magmom (bool): Flag that indicates that the final magnetic
                moments of the calculation should be ignored.

        """
        new_slab = Structure.from_file(os.path.join(directory, "CONTCAR"))

        if ignore_magmom and "magmom" in self.site_properties.keys():
            new_slab.add_site_property("magmom", self.site_properties["magmom"])
        else:
            out = Outcar(os.path.join(directory, "OUTCAR"))
            if len(out.magnetization) == 0:
                if "magmom" in self.site_properties.keys():
                    warnings.warn("Outcar does not contain any magnetic moments! "
                                  "Keeping magnetic moments from initial slab.")
                    new_slab.add_site_property("magmom",
                                               self.site_properties["magmom"])
            else:
                new_slab.add_site_property("magmom",
                                           [site["tot"] for site in
                                            out.magnetization])

        # Update the lattice
        self._lattice = new_slab.lattice

        # Update the coordinates of the occupied sites.
        for i, site in enumerate(self):
            new_site = new_slab.sites[i]

            # Update the site coordinates
            self.replace(i, species=new_site.species,
                         coords=new_site.frac_coords,
                         properties=new_site.properties)


class Cathode(Structure):
    """
    A class representing a cathode material in a battery.

    The main idea of this class is to keep track of the original sites of removed
    working ions (e.g. Li, Na, ...) by using sites with empty Compositions. This is
    important to make sure that the voronoi decomposition is successful, and hence
    essential if we want to look at coordinations and neighbors. Another advantage
    is that we can consider the empty cation sites for final positions of transition
    metal migrations.

    By designing a new class, we can update the Structure I/O methods that do not deal
    with sites that have an empty composition well. Moreover, we can design and bundle
    new methods which are useful in the context of battery cathode research.

    However, the class has currently not been fully tested yet for all the methods it
    inherits from Structure, so some of these may produce some unintented results.

    """

    # Tuple of standard working ions for typical battery insertion cathodes.
    # Lawrencium is also in there for the enumerate workaround.
    standard_working_ions = ("Li", "Na", "Lr")

    # Tuple of standard anions for typical battery insertion cathodes.
    standard_anions = ("O", "F")

    def __init__(self, lattice, species, coords, charge=None,
                 validate_proximity=False,
                 to_unit_cell=False, coords_are_cartesian=False,
                 site_properties=None):

        super(Cathode, self).__init__(
            lattice=lattice, species=species, coords=coords, charge=charge,
            validate_proximity=validate_proximity, to_unit_cell=to_unit_cell,
            coords_are_cartesian=coords_are_cartesian,
            site_properties=site_properties
        )

        self._voronoi = None

    def __str__(self):
        """
        Overwritten string representation, in order to provide information about the
        vacancy sites, as well as the VESTA index, which can be useful when defining
        structural changes.

        Returns:
            (str) String representation of the Cathode.

        """
        outs = ["Full Formula ({s})".format(s=self.composition.formula),
                "Reduced Formula: {}".format(self.composition.reduced_formula)]
        to_s = lambda x: "%0.6f" % x
        outs.append("abc   : " + " ".join([to_s(i).rjust(10)
                                           for i in self.lattice.abc]))
        outs.append("angles: " + " ".join([to_s(i).rjust(10)
                                           for i in self.lattice.angles]))
        if self._charge:
            if self._charge >= 0:
                outs.append("Overall Charge: +{}".format(self._charge))
            else:
                outs.append("Overall Charge: -{}".format(self._charge))
        outs.append("Sites ({i})".format(i=len(self)))
        data = []
        props = self.site_properties
        keys = sorted(props.keys())
        vesta_index = 1
        for i, site in enumerate(self):
            if site.species.num_atoms == 0:
                row = [str(i), "-", "Vac"]

            else:
                row = [str(i), vesta_index, site.species_string]
                vesta_index += 1

            row.extend([to_s(j) for j in site.frac_coords])
            for k in keys:
                row.append(props[k][i])
            data.append(row)

        outs.append(
            tabulate(data,
                     headers=["#", "#VESTA", "SP", "a", "b", "c"] + keys,
                     ))
        return "\n".join(outs)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    @property
    def working_ion_configuration(self):
        """
        A list of all sites which correspond to working ions in the Cathode.

        Returns:
            (list): A list of pymatgen.Sites that correspond to the working ions
                in the Cathode.

        """
        return [site for site in self.sites
                if site.species_string in Cathode.standard_working_ions]

    @working_ion_configuration.setter
    def working_ion_configuration(self, configuration):
        """

        Args:
            configuration (dict or list): A dictionary mapping or list of pymatgen.Sites
                that describes the configuration of the working ions in the cathode.

        Returns:
            None

        """
        # TODO Add checks

        # Remove all working ions
        for working_ion in [ion for ion in Cathode.standard_working_ions
                            if Element(ion) in set(self.composition.keys())]:
            self.replace_species({working_ion: {working_ion: 0}})

        # Add the working ion sites
        if isinstance(configuration, dict):
            for working_ion in configuration.keys():
                for index in configuration[working_ion]:
                    self.replace(index, working_ion, properties={"magmom": 0})

        elif all([isinstance(item, Site) for item in configuration]):
            for ion_site in configuration:
                for i, site in enumerate(self):
                    if np.linalg.norm(site.distance(ion_site)) < 0.05:
                        self.replace(i, ion_site.specie,
                                     properties={"magmom": 0})

        else:
            raise TypeError("Working ion configurations should be a dictionary "
                            "mapping working ions to site indices, or a list of "
                            "sites.")

    @property
    def concentration(self):
        """
        The working ion concentration of the cathode, defined versus the pristine,
        i.e. fully discharged structure as a percentage.

        Returns:
            (float): The working ion concentration

        """
        working_ion_sites = [site for site in self.sites if
                             site.species_string in self.standard_working_ions
                             or site.species == Composition()]
        return len(self.working_ion_configuration) / len(working_ion_sites)

    @property
    def voronoi(self):
        """
        Pymatgen ChemEnv voronoi decomposition of the cathode structure.

        Returns:
            pymatgen.analysis.chemenv.coordination_environments.voronoi.\
            DetailedVoronoiContainer

        """
        if self._voronoi is None:
            self._voronoi = DetailedVoronoiContainer(self)

        return self._voronoi

    @voronoi.setter
    def voronoi(self, voronoi_container):
        self._voronoi = voronoi_container

    def add_cations(self, sites=None):
        """
        Args:
            sites: A dictionary mapping or list of pymatgen.Sites that describes the
                configuration of the working ions in the cathode.

        Returns:
            None

        """

        # Add the cation sites
        if isinstance(sites, dict):
            for cation in sites.keys():
                for index in sites[cation]:
                    self.replace(index, cation, properties={"magmom": 0})

        elif all([isinstance(item, Site) for item in sites]):
            for catsite in sites:
                for i, site in enumerate(self):
                    if np.linalg.norm(site.distance(catsite)) < 0.05:
                        self.replace(i, catsite.specie,
                                     properties={"magmom": 0})

        else:
            raise TypeError("Cation configurations should be a dictionary "
                            "mapping cations to site indices or a list of "
                            "sites.")

    def remove_working_ions(self, sites=None):
        """
        Remove working ions from the cathode, i.e. delithiate the structure in
        case Li is the working ion of the cathode.

        Note that this does not remove the sites from the pymatgen Structure.
        The occupancy is simply adjusted to an empty Composition object.

        Args:
            sites: List of indices OR
                List of pymatgen.core.Sites which are to be removed.

        Returns:
            None

        """

        # TODO add checks

        # If no sites are given
        if sites is None:
            # Remove all the working ions
            self.working_ion_configuration = []


        # If a List of integers is given
        elif all([isinstance(item, int) for item in sites]):
            for index in sites:
                self.replace(index, Composition(), properties={"magmom": 0})

        # If a List of sites is given
        elif all([isinstance(item, Site) for item in sites]):
            for site in sites:
                # Check if the provided site corresponds to a working ion site
                if site in self.working_ion_configuration:
                    ion_configuration = self.working_ion_configuration.copy()
                    ion_configuration.remove(site)
                    self.working_ion_configuration = ion_configuration
                else:
                    raise Warning("Requested site not found in working ion "
                                  "configuration.")
        else:
            raise IOError("Incorrect site input.")

    def migrate_element(self, site, final_site):
        """
        Migrate an element to an empty site.

        Args:
            site: Site or site index of the migrating element.
            final_site: Site or site index of the site the element is migrating to.

        Returns:

        """
        if isinstance(site, Site):
            site_index = self.index[site]
        elif isinstance(site, int):
            site_index = site
            site = self.sites[site]
        else:
            raise IOError("Input sites must be either an integer or pymatgen.Site!")

        if isinstance(final_site, Site):
            final_site_index = self.index[final_site]
        elif isinstance(final_site, int):
            final_site_index = final_site
            final_site = self.sites[final_site]
        else:
            raise IOError("Input sites must be either an integer or pymatgen.Site!")

        if final_site.species == Composition():

            # Store the initial magnetic moments, if any
            magmom = self.site_properties.get("magmom", None)

            # Adjust the Species of the initial and final site
            self.replace(final_site_index, site.species)
            self.replace(site_index, Composition())

            # Switch the magnetic moments (if any) and update the site properties
            if magmom:
                magmom[site_index], magmom[final_site_index] = \
                    magmom[final_site_index], magmom[site_index]
                self.add_site_property("magmom", magmom)
        else:
            raise ValueError("Final migration site is not empty!")

    def change_site_distance(self, sites, distance):
        """
        Change the coordinates of two sites in a structure in order to adjust
        their distance.

        Args:
            sites (list): List of two site indices or pymatgen.Sites of
                elements whose distance should be changed.
            distance (float): Final distance between the two sites provided.

        Returns:
            None

        """

        if all(isinstance(el, int) for el in sites):
            site_a = self.sites[sites[0]]
            site_b = self.sites[sites[1]]
        elif all(isinstance(el, Site) for el in sites):
            site_a = sites[0]
            site_b = sites[1]
        else:
            raise IOError("Incorrect input provided.")

        # Find the distance between the sites, as well as the image of site B
        # closest to site A
        (original_distance, closest_image_b) = site_a.distance_and_image(site_b)

        image_cart_coords = self.lattice.get_cartesian_coords(
            site_b.frac_coords + closest_image_b
        )

        # Calculate the vector that connects site A with site B
        connection_vector = image_cart_coords - site_a.coords
        connection_vector /= np.linalg.norm(connection_vector)  # Unit vector

        # Calculate the distance the sites need to be moved.
        site_move_distance = (original_distance - distance) / 2

        # Calculate the new cartesian coordinates of the sites
        new_site_a_coords = site_a.coords + site_move_distance * connection_vector
        new_site_b_coords = site_b.coords - site_move_distance * connection_vector

        # Change the sites in the structure
        self.replace(i=sites[0], species=site_a.species_string,
                     coords=new_site_a_coords,
                     coords_are_cartesian=True,
                     properties=site_a.properties)

        self.replace(i=sites[1], species=site_b.species_string,
                     coords=new_site_b_coords,
                     coords_are_cartesian=True,
                     properties=site_b.properties)

    def update_sites(self, directory, ignore_magmom=False):
        """
        Based on the CONTCAR and OUTCAR of a geometry optimization, update the
        site coordinates and magnetic moments that were optimized. Note that
        this method relies on the cation configuration of the cathode not
        having changed.

        Args:
            directory (str): Directory in which the geometry optimization
                output files (i.e. CONTCAR and OUTCAR) are stored.
            ignore_magmom (bool): Flag that indicates that the final magnetic
                moments of the optimized structure should be ignored. This means
                that the magnetic moments of the Cathode structure will
                remain the same.

        Returns:
            None

        """

        new_cathode = Cathode.from_file(os.path.join(directory, "CONTCAR"))

        out = Outcar(os.path.join(directory, "OUTCAR"))

        if ignore_magmom:
            magmom = [site.properties["magmom"] for site in self.sites
                      if site.species != Composition()]
        else:
            magmom = [site["tot"] for site in out.magnetization]

        if len(out.magnetization) != 0:
            new_cathode.add_site_property("magmom", magmom)

        # Update the lattice
        self.lattice = new_cathode.lattice

        # Update the coordinates of the occupied sites.
        new_index = 0
        for i, site in enumerate(self):

            # If the site is not empty
            if site.species != Composition():
                new_site = new_cathode.sites[new_index]
                # Update the site coordinates
                self.replace(i, species=new_site.species,
                             coords=new_site.frac_coords,
                             properties=new_site.properties)
                new_index += 1

    def set_to_high_spin(self):
        """

        :return:
        """
        raise NotImplementedError

    def set_to_low_spin(self):
        """

        :return:
        """
        raise NotImplementedError

    def find_noneq_cations(self, symm_prec=1e-3):
        """
        Find a list of the site indices of all non-equivalent cations.

        Returns:
            (list): List of site indices

        """
        symmops = SpacegroupAnalyzer(
            self, symprec=symm_prec).get_space_group_operations()

        cation_indices = [
            index for index in range(len(self.sites))
            if self.sites[index].species_string not in Cathode.standard_anions
        ]

        # Start with adding the first cation
        inequiv_cations = [cation_indices[0], ]

        for index in cation_indices[1:]:

            s1 = [self.sites[index], ]

            # Check if the site is equivalent with one of the sites in the
            # inequivalent list.
            inequivalent = True

            for inequive_index in inequiv_cations:

                s2 = [self.sites[inequive_index], ]

                if symmops.are_symmetrically_equivalent(
                        s1, s2, symm_prec=symm_prec):
                    inequivalent = False

            if inequivalent:
                inequiv_cations.append(index)

        return inequiv_cations

    # Temporarily removed: icet build takes too long...
    #
    # def get_cation_configurations(self, substitution_sites, cation_list, sizes,
    #                               concentration_restrictions=None,
    #                               max_configurations=None):
    #     """
    #     Get all non-equivalent cation configurations within a specified range of unit
    #     cell sizes and based on certain restrictions.
    #
    #     Based on the icet.tools.structure_enumeration.enumerate_structures() method.
    #     Because of the fact that vacancies can not be inserted in enumerate_structures,
    #     we will introduce a little workaround using Lawrencium.
    #
    #     Currently also returns a list of Cathodes, for easy implementation and usage. It
    #     might be more useful/powerful to design it as a generator later.
    #
    #     Args:
    #         substitution_sites (list): List of site indices or pymatgen.Sites to be
    #             substituted.
    #         cation_list (list): List of string representations of the cation elements
    #             which have to be substituted on the substitution sites. Can also
    #             include "Vac" to introduce vacancy sites.
    #             E.g. ["Li", "Vac"]; ["Mn", "Co", "Ni"]; ...
    #         sizes (list): List of unit supercell sizes to be considered for the
    #             enumeration of the configurations.
    #             E.g. [1, 2]; range(1, 4); ...
    #         concentration_restrictions (dict): Dictionary of allowed concentration
    #             ranges for each element. Note that the concentration is defined
    #             versus the total amount of atoms in the unit cell.
    #             E.g. {"Li": (0.2, 0.3)}; {"Ni": (0.1, 0.2, "Mn": (0.05, 0.1)}; ...
    #         max_configurations (int): Maximum number of configurations to generate.
    #
    #     Returns:
    #         (list): List of Cathodes representing different configurations.
    #
    #     """
    #     # Check substitution_site input
    #     if all(isinstance(site, int) for site in substitution_sites):
    #         substitution_sites = [self.sites[index] for index in substitution_sites]
    #
    #     # Set up the configuration space
    #     configuration_space = []
    #     cation_list = ["Lr" if cat == "Vac" else cat for cat in cation_list]
    #
    #     for site in self.sites:
    #         if site in substitution_sites:
    #             configuration_space.append(cation_list)
    #         else:
    #             configuration_space.append([site.species_string, ])
    #
    #     # Substitute the concentration restriction for "Vac" by "Lr"
    #     if concentration_restrictions and "Vac" in concentration_restrictions.keys():
    #         concentration_restrictions["Lr"] = concentration_restrictions.pop("Vac")
    #
    #     # TODO Currently, the user can't specify the final magnetic moment of the
    #     #  substituted elements....
    #     # Check if the magnetic moment is defined in the Cathode
    #     try:
    #         self.site_properties["magmom"]
    #     except KeyError:
    #         print("No magnetic moments found in structure, setting to zero.")
    #         self.add_site_property("magmom", [0] * len(self))
    #
    #     # Set up the icet configuration generator
    #     configuration_generator = enumerate_structures(
    #         atoms=AseAtomsAdaptor.get_atoms(self.as_ordered_structure()),
    #         sizes=sizes,
    #         chemical_symbols=configuration_space,
    #         concentration_restrictions=concentration_restrictions
    #     )
    #     configuration_list = []
    #
    #     for atoms in configuration_generator:
    #
    #         structure = AseAtomsAdaptor.get_structure(atoms)
    #         # Add the magnetic moment
    #         structure.add_site_property(
    #             "magmom",
    #             self.site_properties["magmom"] * int(len(structure) / len(self))
    #         )
    #         # Sort the structure and redefine it as a cathode
    #         cathode = self.__class__.from_structure(
    #             structure.get_sorted_structure())
    #         cathode.remove_working_ions(
    #             [i for i, site in enumerate(cathode)
    #              if site.species_string == "Lr"]
    #         )
    #         configuration_list.append(cathode)
    #
    #         if len(configuration_list) == max_configurations:
    #             break  # Quit if the number of configurations is obtained
    #
    #     return configuration_list

    def as_ordered_structure(self):
        """
        Return the structure as a pymatgen.core.Structure, removing the
        unoccupied sites. This is because many of the IO methods of pymatgen
        run into issues when empty occupancies are present.

        Returns:
            pymatgen.core.Structure

        """

        return Structure.from_sites(
            [site for site in self.sites
             if site.species != Composition()]
        )

    def to(self, fmt=None, filename=None, **kwargs):
        """
        Structure method override to solve issue with writing the Cathode to a
        POSCAR file.

        # TODO Figure out what exactly was the problem here again... Should
        have written this down immediately! I think it had something to do
        with the order of the Sites changing...

        Args:
            fmt:
            filename:
            **kwargs:

        Returns:

        """

        if fmt == "poscar":
            structure = self.as_ordered_structure()
            return structure.to(fmt, filename, **kwargs)

        else:
            return super(Cathode, self).to(fmt, filename, **kwargs)

    @classmethod
    def from_structure(cls, structure):
        """
        Initializes a Cathode from a pymatgen.core.Structure.

        Args:
            structure (pymatgen.core.Structure): Structure from which to
            initialize the Cathode.

        Returns:
            pybat.core.Structure

        """

        return cls.from_sites(structure.sites)
