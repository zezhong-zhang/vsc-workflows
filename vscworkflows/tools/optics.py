# Encoding: UTF-8
# Copyright (c) Marnik Bercx, University of Antwerp;
#               Kamal Choudhary, National Institute of Standards and Technology
# Distributed under the terms of the GNU License
# Initially forked and (extensively) adjusted from https://github.com/ldwillia/SL3ME

from __future__ import unicode_literals, print_function

import cmath
import json
import math
import os
import warnings
from fnmatch import fnmatch
from xml.etree.ElementTree import ParseError

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as constants
from monty.io import zopen
from monty.json import MSONable, MontyDecoder, MontyEncoder
from pymatgen.io.vasp.outputs import Vasprun, Outcar
from scipy.integrate import simps
from scipy.interpolate import interp1d

"""
Module for calculating the SLME metric, including several classes for representing 
the optical properties of the material or an electromagnetic absorption spectrum.

"""

# Defining constants for tidy equations
c = constants.c  # speed of light, m/s
h = constants.h  # Planck's constant J*s (W)
h_e = constants.h / constants.e  # Planck's constant eV*s
k = constants.k  # Boltzmann's constant J/K
k_e = constants.k / constants.e  # Boltzmann's constant eV/K
e = constants.e  # Coulomb


class Waveder(MSONable):
    """
    Class that represents a VASP WAVEDER file.

    """
    pass


class DielTensor(MSONable):
    """
    Class that represents the energy-dependent dielectric tensor of a solid
    state material.

    """

    def __init__(self, energies, dielectric_tensor):
        """
        Initializes a DielTensor instance from the dielectric data.

        Args:
            energies (numpy.array): (N,) array with the energy grid in eV.
            dielectric_tensor (numpy.array): (N, 3, 3) array with the dielectric
                tensor.

        """
        self._energies = energies
        self._dielectric_tensor = dielectric_tensor

    def check_dielectric_data(self):
        """
        Function that performs some tests on the dielectric data, to make sure
        input satisfies some constrains based on what we know about the dielectric
        tensor.

        Returns:
            None

        """
        pass  # TODO

    @property
    def energies(self):
        """
        Energy grid for which the dielectric tensor is defined in the original data.

        Returns:
            numpy.array: (N,) shaped array with the energies of the grid in eV.

        """
        return self._energies

    @property
    def dielectric_tensor(self):
        """
        Dielectric tensor of the material, calculated for each energy in the energy
        grid.

        Returns:
            numpy.array: (N, 3, 3) shaped array, where N corresponds to the number
                of energy grid points, and 3 to the different directions x,y,z.
        """
        return self._dielectric_tensor

    @property
    def dielectric_function(self):
        """
        The averaged dielectric function, derived from the tensor components by
        averaging the diagonal elements.

        Returns:
            np.array: (N,) shaped array with the dielectric function.
        """
        return np.array([np.mean(tensor.diagonal())
                         for tensor in self.dielectric_tensor])

    @property
    def absorption_coefficient(self):
        """
        Calculate the optical absorption coefficient from the dielectric data.
        For now the script only calculates the averaged absorption coefficient,
        i.e. by first averaging the diagonal elements and then using this
        dielectric function to calculate the absorption coefficient.

        Notes:
            The absorption coefficient is calculated as
            .. math:: \\alpha = \\frac{2 E}{ \hbar c} k(E)
            with $k(E)$ the imaginary part of the square root of the dielectric
            function

        Returns:
            np.array: (N,) shaped array with the energy (eV) dependent absorption
                coefficient in m^{-1}, where the energies correspond to
                self.energies.
        """

        energy = self.energies
        ext_coeff = np.array([cmath.sqrt(v).imag for v in self.dielectric_function])

        return 2.0 * energy * ext_coeff / (
                constants.hbar / constants.e * constants.c)

    def add_intraband_dieltensor(self, plasma_frequency, damping=0.1):
        """
        Add intraband component of the dielectric tensor based on the Drude model.

        Args:
            plasma_frequency:
            damping:

        Returns:

        """
        drude_diel = self.from_drude_model(plasma_frequency=plasma_frequency,
                                           energies=self.energies, damping=damping)

        self._dielectric_tensor += drude_diel.dielectric_tensor - 1

    def get_absorptivity(self, thickness, method="beer-lambert"):
        """
        Calculate the absorptivity for an absorber layer with a specified thickness
        and cell construction.

        Args:
            thickness (float): Thickness of the absorber layer, expressed in meters.
            method (str): Method for calculating the absorptivity.

        Returns:
            np.array: (N,) shaped array with the energy (eV) dependent absorptivity,
                where the energies correspond to self.energies.

        """
        if method == "beer-lambert":
            return 1.0 - np.exp(-2.0 * self.absorption_coefficient * thickness)
        else:
            raise NotImplementedError("Unrecognized method for calculating the "
                                      "absorptivity.")

    def get_loss_function(self, surface=False):
        """
        Calculate the loss function based on the averaged dielectric function of
        the dielectric tensor.

        Args:
            surface:

        Returns:

        """

        er = self.dielectric_function.real
        ei = self.dielectric_function.imag

        if surface:
            loss_function = ei / ((er + 1) ** 2 + ei ** 2)
        else:
            loss_function = ei / (er ** 2 + ei ** 2)

        loss_function[np.isnan(loss_function)] = 0

        return loss_function

    def shift_imaginary(self, shift):
        """
        Shift the imaginary part of the dielectric tensor by a specified value.

        Args:
            shift (float): Requested shift to apply to the imaginary part of the
                dielectric tensor. Positive values shift the onset to higher
                energies, negative shift it to lower energies.

        """
        # Note: Check index N=1 because N=0 is always zero
        if not (self.dielectric_tensor[1, :, :].imag == np.zeros(3)).all():
            warnings.warn("Onset of the imaginary part of the dielectric tensor "
                          "starts at zero. This indicates that the material is "
                          "metallic, and that shifting the imaginary function "
                          "might not make a lot of sense.")

        interp_function = interp1d(self.energies, self.dielectric_tensor.imag,
                                   axis=0, bounds_error=False, fill_value=0)

        imag_diel = interp_function(self._energies - shift)

        if not (imag_diel[1, :, :] == np.zeros(3)).all():
            warnings.warn("The requested shift sets the onset of the imaginary "
                          "part of the dielectric tensor at zero, effectively "
                          "turning the material metallic.")

        real_diel = kkr(np.average(np.diff(self.energies)), imag_diel, 1e-3)

        self._dielectric_tensor = real_diel + imag_diel * 1j

    def plot(self, part="diel", variable_range=None, diel_range=None):
        """
        Plot the real and/or imaginary part of the dielectric function.

        Args:
            part (str): Which part of the dielectric function to plot, i.e. either
                "real", "imag" or "all".
            variable_range (tuple): Lower and upper limits of the variable which
                the requested function is plotted against.
            diel_range (tuple): Lower and upper limits of the range of the requested
                function in the plotted figure.

        Returns:
            None

        """
        if part == "diel":
            f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

            ax1.plot(self.energies, self.dielectric_function.real)
            ax2.plot(self.energies, self.dielectric_function.imag)
            if variable_range:
                ax1.set(xlim=variable_range)
                ax2.set(xlim=variable_range)
            if diel_range:
                ax1.set(ylim=diel_range)
                ax2.set(ylim=diel_range)
            ax1.set(ylabel=r"$\varepsilon_1$")
            ax2.set(xlabel="Energy (eV)", ylabel=r"$\varepsilon_2$")
            f.subplots_adjust(hspace=0.1)
            plt.show()

        elif part == "real":

            plt.plot(self.energies, self.dielectric_function.real)
            plt.xlabel("Energy (eV)")
            if variable_range:
                plt.xlim(variable_range)
            if diel_range:
                plt.ylim(diel_range)
            plt.ylabel(r"$\varepsilon_1$")
            plt.show()

        elif part == "imag":

            plt.plot(self.energies, self.dielectric_function.imag)
            plt.xlabel("Energy (eV)")
            if variable_range:
                plt.xlim(variable_range)
            if diel_range:
                plt.ylim(diel_range)
            plt.ylabel(r"$\varepsilon_2$")
            plt.show()

        elif part == "abs_coeff":

            plt.plot(self.energies, self.absorption_coefficient)
            plt.xlabel("Energy (eV)")
            if variable_range:
                plt.xlim(variable_range)
            if diel_range:
                plt.ylim(diel_range)
            plt.ylabel(r"$\alpha(E)$")
            plt.yscale("log")
            plt.show()

    def as_dict(self):
        """
        Note: stores the real and imaginary part of the dielectric tensor
        separately, due to issues with JSON serializing complex numbers.

        Returns:
            dict: Dictionary representation of the DielTensor instance.
        """
        d = dict()
        d["energies"] = MontyEncoder().default(self.energies)
        d["real_diel"] = MontyEncoder().default(self.dielectric_tensor.real)
        d["imag_diel"] = MontyEncoder().default(self.dielectric_tensor.imag)
        return d

    @classmethod
    def from_dict(cls, d):
        """
        Initializes a DielTensor object from a dictionary.

        Args:
            d (dict): Dictionary from which the DielTensor should be initialized.

        Returns:
            DielTensor

        """
        energies = MontyDecoder().process_decoded(d["energies"])
        real_diel = MontyDecoder().process_decoded(d["real_diel"])
        imag_diel = MontyDecoder().process_decoded(d["imag_diel"])
        return cls(energies, real_diel + 1j * imag_diel)

    def to(self, filename):
        """
        Write the DielTensor to a JSON file.

        Args:
            filename (str): Path to the file in which the DielTensor should
                be written.

        Returns:
            None

        """
        with zopen(filename, "w") as f:
            f.write(self.to_json())

    @classmethod
    def from_file(cls, filename, fmt=None):
        """
        Initialize a DielTensor instance from a file.

        Args:
            filename (str): Path to file from which the dielectric data will be
                loaded. Can (so far) either be a vasprun.xml, OUTCAR or json file.
            fmt (str): Format of the file that contains the dielectric function
                data. Is optional, as the method can also figure out the format
                based on the filename.

        Returns:
            DielTensor: Dielectric tensor object from the dielectric data.

        """
        # Vasprun format: dielectric data is length 3 tuple
        if fmt == "vasprun" or filename.endswith(".xml"):

            dielectric_data = Vasprun(filename, parse_potcar_file=False).dielectric

            energies = np.array(dielectric_data[0])
            dielectric_tensor = np.array(
                [to_matrix(*real_data) + 1j * to_matrix(*imag_data)
                 for real_data, imag_data in zip(dielectric_data[1],
                                                 dielectric_data[2])]
            )
            return cls(energies, dielectric_tensor)

        # OUTCAR format: dielectric data is length 2 tuple
        elif fmt == "outcar" or fnmatch(filename, "*OUTCAR*"):
            outcar = Outcar(filename)
            outcar.read_freq_dielectric()
            return cls(outcar.frequencies, outcar.dielectric_tensor_function)

        # JSON format
        if fmt == "json" or filename.endswith(".json"):
            with zopen(filename, "r") as f:
                return cls.from_dict(json.loads(f.read()))

        else:
            raise IOError("Format of file not recognized. Note: Currently "
                          "only vasprun.xml and OUTCAR files are supported.")

    @classmethod
    def from_drude_model(cls, plasma_frequency, energies, damping=0.05):
        """
        Initialize a DielTensor object based on the Drude model for metals.

        Returns:

        """

        try:
            if not plasma_frequency.shape == (3, 3):
                raise ValueError("Plasma frequency array does not have right shape!")
        except AttributeError:
            if isinstance(plasma_frequency, float):
                plasma_frequency *= np.eye(3)
            else:
                raise TypeError("The plasma frequency must be expressed either as "
                                "a float or a numpy array of shape (3, 3).")

        dieltensor = np.array(
            [1 - omega ** 2 / (energies ** 2 + 1j * energies * damping)
             for omega in plasma_frequency.reshape(9)]
        ).reshape((3, 3, len(energies)))
        dieltensor = dieltensor.swapaxes(0, 2)

        return cls(energies, dieltensor)


class EMRadSpectrum(MSONable):
    """
    Class that represents a electromagnetic radiation spectrum, e.g. the solar
    spectrum or the black body spectrum. The standard form we choose to express
    the spectrum is as the energy-dependent photon flux per square meter,
    where the energy is expressed in electron volts (Units ~ m^{-2} s^{-1} eV^{-1}).

    """

    def __init__(self, energy, photon_flux):
        """
        Initialize the Radiation Spectrum object from the energy grid and the
        photon flux. The input is expected to be provided in the units described
        below.

        Args:
            energy (numpy.array): Energy grid for which the photon flux is given.
                Has to be given in electron volt (eV).
            photon_flux (numpy.array): Number of photons per square meter per
                second per electron volt. (~ m^{-2} s^{-1} eV^{-1})

        """
        self._energy = energy
        self._photon_flux = photon_flux

    @property
    def energy(self):
        """
        Energy grid for which the electromagnetic radiation spectrum is defined.

        Returns:
            numpy.array: (N,) shaped array

        """
        return self._energy

    @property
    def photon_flux(self):
        """
        Electromagnetic radiation spectrum expressed in photon flux.

        Returns:
            numpy.array: (N,) shaped array

        """
        return self._photon_flux

    def get_total_power_density(self):
        """
        Get the total power density in W m^{-2}.

        Returns:
            float: Total calculated power density.

        """
        return simps(self.photon_flux * self.energy * e, self.energy)

    def get_interp_function(self, variable="energy", spectrum_units="flux"):
        """
        Obtain the 1D interpolation function using the scipy.interpolate.interp1d
        method. Linear interpolation is chosen, as it is more robust.

        Still have to implement the option to change the variable and type of
        spectrum.

        Args:
            variable (str): Choice of the variable which the EMRadSpectrum is
                dependent upon. Currently only allows the default: "energy".
            spectrum_units (str): Units in which the spectrum should be expressed.
                Currently only allows the default: "flux", which corresponds to
                the photon flux.

        Returns:
            scipy.interpolate.interpolate.interp1d

        """
        # TODO complete for other variables and spectrum choices
        return interp1d(self._energy, self._photon_flux, kind='linear',
                        fill_value=0.0, bounds_error=False)

    def to(self, filename):
        """
        Write the EMRadSpectrum to a JSON file.

        Args:
            filename (str): Path to the file in which the EMRadSpectrum should
                be written.

        Returns:
            None

        """
        with zopen(filename, "w") as f:
            f.write(self.to_json())

    @classmethod
    def from_file(cls, filename):
        """
        Load the EMRadSpectrum from a file.

        Args:
            filename (str): Path to the file from which the EMRadSpectrum should
                be loaded.

        Returns:
            EMRadSpectrum

        """
        if filename.endswith(".json"):
            with zopen(filename, "r") as f:
                return cls.from_dict(json.loads(f.read()))
        else:
            raise IOError("Filename does not have .json extension.")

    @classmethod
    def from_data(cls, data, variable="energy", spectrum_units="flux"):
        """
        Initialize an EMRadSpectrum from a tuple that contains the data of the
        spectrum.

        Args:
            data (tuple): Tuple with length 2 that contains the data from which to
                construct the radiation spectrum. data[0] must contain a (N,) shaped
                numpy.array with the grid of the variable (e.g. energy, wavelength),
                data[1] should contain the spectral distribution.
            variable (str): Variable which the spectrum is dependent on in the data
                that is provided. Currently allows "energy" or "wavelength".
            spectrum_units (str): Units in which the spectrum is expressed. Currently
                has two options: "flux" or "irradiance".

        Returns:
            EMRadSpectrum

        """
        if variable == "energy":
            energy = data[0]
            spectrum = data[1]

        elif variable == "wavelength":
            energy = np.flip(h_e * c / data[0])
            spectrum = np.flip(data[1]) * h_e * c / energy ** 2

        else:
            raise NotImplementedError

        if spectrum_units == "flux":
            photon_flux = spectrum
            return cls(energy, photon_flux)

        elif spectrum_units == "irradiance":
            photon_flux = spectrum / (e * energy)
            return cls(energy, photon_flux)

        else:
            raise NotImplementedError

    @classmethod
    def get_solar_spectrum(cls, spectrum="am1.5g"):
        """
        Load a solar spectrum based on a data file that is stored in the jarvis
        package.

        Args:
            spectrum: Chosen spectrum to load. Currently, only the Air Mass 1.5
                global tilt spectrum ("am1.5g") is available.

        Returns:
            EMRadSpectrum:

        """

        data_file = os.path.join(os.path.dirname(__file__), spectrum + ".dat")

        wavelength, irradiance = np.loadtxt(
            data_file, usecols=[0, 1], unpack=True, skiprows=2
        )

        # Transfer units to m instead of nm
        wavelength *= 1e-9
        irradiance *= 1e9

        return cls.from_data((wavelength, irradiance), variable="wavelength",
                             spectrum_units="irradiance")

    @classmethod
    def get_blackbody(cls, temperature, grid, variable="energy",
                      spectrum_units="flux"):
        """
        Construct the blackbody spectrum of a specific temperature.

        Args:
            temperature (float): Temperature of the black body.
            grid (numpy.array): Grid of the spectral variable which the black
                body is dependent on.
            variable (str): Spectral variable of the distribution.
            spectrum_units (str): Units in which the spectrum should be expressed.
                Currently only support expressing the black body as a photon flux.

        Returns:
            EMRadSpectrum

        """
        if variable == "energy":
            energy = grid
        else:
            raise NotImplementedError

        # Define a exponential function that does not cause a range overflow
        def exponential(x):
            try:
                return math.exp(x)
            except OverflowError:
                return math.exp(700)  # ~= 1e304

        # Calculation of energy-dependent blackbody spectrum (~ W m^{-2})
        if spectrum_units == "flux":
            photon_flux = 2 * energy ** 2 / (h_e ** 3 * c ** 2) * (
                    1 / (np.array([exponential(energy / (k_e * temperature)) - 1
                                   for energy in energy]))
            )
        else:
            raise NotImplementedError

        return cls(energy, photon_flux)


class SolarCell(MSONable):
    """
    Class that represents a single p-n junction solar cell. Contains several modeling
    techniques for the calculation of the theoretical efficiency using metrics
    based on the optical properties and electronic structure (e.g. band gap)
    of the material being considered as the absorber layer.

    """

    def __init__(self, dieltensor, bandgaps):
        """
        Initialize an instance of the EfficiencyCalculator class.

        Args:
            dieltensor (DielTensor): Dielectric tensor of the absorber material.
            bandgaps (tuple): Tuple that contains the fundamental and direct
                allowed band gap of the absorber material, in that order.

        Returns:
            SolarCell

        """
        self._dieltensor = dieltensor
        self._bandgaps = bandgaps

    @property
    def dieltensor(self):
        return self._dieltensor

    @property
    def bandgaps(self):
        return self._bandgaps

    def slme(self, temperature=298.15, thickness=5e-7, interp_mesh=0.001,
             plot_iv_curve=False, cut_abs_below_bandgap=False):
        """
        Calculate the Spectroscopic Limited Maximum Efficiency.

        Args:
            temperature (float): Temperature of the solar cell.
            thickness (float): Thickness of the absorber layer.
            interp_mesh (float): Distance between two energy points in the grid
                used for the interpolation.
            plot_iv_curve (bool): Defaults to False. If set to true, the I-V curve
                of the solar cell is plotted.
            cut_abs_below_bandgap (bool): Remove any absorption below the band gap,
                i.e. set the absorptivity to zero for all energy values below the
                band gap.

        Returns:
            tuple: efficiency, v_oc, j_sc, j_0

        """
        # Set up the energy grid for the calculation
        energies = self.dieltensor.energies
        abs_coeff = self.dieltensor.absorption_coefficient

        # If the user has requested the onset below the band gap to be removed
        if cut_abs_below_bandgap:
            # Set the absorptivity to zero for energies below the *direct* band gap
            abs_coeff[energies < self.bandgaps[1]] = 0

        bandgap_index = np.where(energies < self.bandgaps[1])[0][-1]
        if abs_coeff[bandgap_index] != 0:
            warnings.warn("Found non-zero absorption below the direct band gap. "
                          "This may be an indication that the imaginary part of "
                          "the dielectric function was smeared by the VASP "
                          "calculation.")
        else:
            energies = np.insert(energies, bandgap_index, self.bandgaps[1])
            abs_coeff = np.insert(abs_coeff, bandgap_index, 0)

        energy = np.linspace(
            np.min(energies) + interp_mesh, np.max(energies),
            int(np.ceil((np.max(energies) - np.min(energies)) / interp_mesh))
        )
        # Interpolation of the absorptivity to the new energy grid
        abs_coeff = interp1d(
            energies, abs_coeff, kind='linear', fill_value=0, bounds_error=False
        )(energy)

        absorptivity = 1.0 - np.exp(-2.0 * abs_coeff * thickness)

        # Load energy-dependent total solar spectrum photon flux
        # (~m^{-2}s^{-1}eV^{-1})
        solar_spectrum = \
            EMRadSpectrum.get_solar_spectrum("am1.5g").get_interp_function()(energy)

        # Calculation of energy-dependent blackbody spectrum (~m^{-2}s^{-1}eV^{-1})
        blackbody_spectrum = EMRadSpectrum.get_blackbody(temperature,
                                                         energy).photon_flux

        # Numerically integrating photon flux over energy grid
        j_0_r = e * np.pi * simps(blackbody_spectrum * absorptivity, energy)

        # Calculate the fraction of radiative recombination
        delta = self._bandgaps[1] - self._bandgaps[0]
        fr = np.exp(-delta / (k_e * temperature))
        j_0 = j_0_r / fr

        # Numerically integrating irradiance over wavelength array ~ A/m**2
        j_sc = e * simps(solar_spectrum * absorptivity, energy)

        # Determine the open circuit voltage.
        v_oc = 0
        voltage_step = 0.001
        while j_sc - j_0 * (np.exp(e * v_oc / (k * temperature)) - 1.0) > 0:
            v_oc += voltage_step

        if plot_iv_curve:  # TODO Add some more details.
            voltage = np.linspace(0, v_oc, 2000)

            current = j_sc - j_0 * (np.exp(e * voltage / (k * temperature)) - 1.0)
            power = current * voltage

            plt.plot(voltage, current)
            plt.plot(voltage, power)
            plt.show()

        # Maximize the power density versus the voltage
        max_power = self.maximize_power(j_sc, j_0, temperature)

        # Calculation of integrated solar spectrum
        power_in = EMRadSpectrum.get_solar_spectrum().get_total_power_density()

        # Calculate the maximized efficiency
        efficiency = max_power / power_in

        return efficiency, v_oc, j_sc, j_0

    def calculate_bandgap_sq(self, temperature=298.15, fr=1.0, interp_mesh=0.001):
        """
        Calculate the Shockley-Queisser limit of the corresponding fundamental
        band gap.

        Args:
            temperature (float): Temperature of the solar cell. Defaults to 25 °C,
                 or 298.15 K.
            fr (float): Fraction of radiative recombination.
            interp_mesh (float): Distance between two energy points in the grid
                used for the interpolation.

        Returns:
            (float): Shockley-Queisser detailed balance limit of the band gap of
                the material.

        """
        return self.sq(self._bandgaps[1], temperature, fr, interp_mesh,
                       np.max(self.dieltensor.energies))

    def plot_slme_vs_thickness(self, temperature=298.15, add_sq_limit=True,
                               cut_abs_below_bandgap=False, add_to_axis=None,
                               **kwargs):
        """
        Make a plot of the calculated SLME for a large range of thickness values,
        for a specific temperature.

        Args:
            temperature (float): Temperature of the solar cell. Defaults to 298.15 K.
            add_sq_limit (bool): Specifies whether the user would like to add a
                line representing the Shockley-Queisser limit that corresponds to
                the band gap of the absorber material at the temperature specified.
            cut_abs_below_bandgap (bool): Remove any absorption below the band gap,
                i.e. set the absorptivity to zero for all energy values below the
                band gap.

        Returns:
            None

        """
        thickness = 10 ** np.linspace(-9, -3, 40)
        efficiency = np.array(
            [self.slme(thickness=d, temperature=temperature,
                       cut_abs_below_bandgap=cut_abs_below_bandgap)[0]
             for d in thickness]
        )

        if add_to_axis is not None:

            add_to_axis.plot(thickness, efficiency, **kwargs)

            if add_sq_limit:
                add_to_axis.axhline(
                    self.calculate_bandgap_sq(temperature=temperature)[0],
                    color="k", linestyle='--'
                )

        else:
            plt.plot(thickness, efficiency)
            if add_sq_limit:
                plt.axhline(self.calculate_bandgap_sq(temperature=temperature)[0],
                            color="k", linestyle='--')
                plt.legend(("SLME", "SQ"))
            else:
                plt.legend(("SLME",))

            plt.xlabel("Thickness (m)")
            plt.ylabel("Efficiency")
            plt.xscale("log")

            plt.show()

    def get_currents(self, temperature):
        pass

    def get_iv_curve(self, temperature=298.15, thickness=5e-7, interp_mesh=0.001,
                     cut_abs_below_bandgap=False, iv_mesh=0.01, add_to_axis=None):

        # TODO: improve modularization!

        v_oc, j_sc, j_0 = self.slme(
            temperature=temperature, thickness=thickness, interp_mesh=interp_mesh,
            cut_abs_below_bandgap=cut_abs_below_bandgap
        )[1:]

        voltage = np.linspace(0, v_oc, int(v_oc / iv_mesh))

        j = j_sc - j_0 * (np.exp(e * voltage / (k * temperature)) - 1.0)
        p = j * voltage

        if add_to_axis:

            add_to_axis.plot(voltage, j, 'k')
            add_to_axis.plot(voltage, p, "k--")

        else:

            plt.plot(voltage, j, 'k')
            plt.plot(voltage, p, "k--")

    def shift_bandgap_to(self, bandgap, is_direct=True):
        """
        Method that shifts the optical/direct (default) or fundamental bandgap to
        a specified value. Useful when the bandgap is determined with a
        different, more accurate, method than the dielectric tensor. Adjusts both
        the imaginary part of the dielectric tensor as well as the band gaps.

        Args:
            bandgap (float): New user-specified band gap.
            is_direct (bool): Whether or not the specified band gap is the direct
                (optical) band gap. If set to False, the fundamental band gap is
                expected.
        """
        if is_direct:
            shift = bandgap - self._bandgaps[1]
            self._bandgaps = [self._bandgaps[0] + shift, bandgap]
        else:
            shift = bandgap - self._bandgaps[0]
            self._bandgaps = [bandgap, self._bandgaps[1] + shift]

        self._dieltensor.shift_imaginary(shift)

    def as_dict(self):
        """
        Note: stores the real and imaginary part of the dielectric tensor
        separately, due to issues with JSON serializing complex numbers.

        Returns:
            dict: Dictionary representation of the DielTensor instance.
        """
        d = dict()
        d["dieltensor"] = self._dieltensor.as_dict()
        d["bandgaps"] = self._bandgaps

        return d

    @classmethod
    def from_dict(cls, d):
        """
        Initializes a DielTensor object from a dictionary.

        Args:
            d (dict): Dictionary from which the DielTensor should be initialized.

        Returns:
            DielTensor

        """
        dieltensor = DielTensor.from_dict(d["dieltensor"])
        bandgaps = d["bandgaps"]

        return cls(dieltensor, bandgaps)

    @classmethod
    def from_file(cls, filename):
        """
        Loads a SolarCell instance from a vasprun.xml. # TODO extend

        Args:
            filename (str): vasprum.xml file from which to load the SolarCell object.

        Returns:
            SolarCell

        """
        try:
            vasprun = Vasprun(filename, parse_potcar_file=False)
            diel_tensor = DielTensor.from_file(filename)
        except ParseError:
            raise IOError("Error while parsing the input file. Currently the "
                          "SolarCell class can only be constructed from "
                          "the vasprun.xml file. If you have provided this "
                          "file, check if the run has completed.")

        # Extract the information on the direct and indirect band gap
        bandstructure = vasprun.get_band_structure()
        bandgaps = (bandstructure.get_band_gap()["energy"],
                    bandstructure.get_direct_band_gap())

        return cls(diel_tensor, bandgaps)

    @staticmethod
    def maximize_power(j_sc, j_0, temperature):
        """
        Maximize the power density based on the short-circuit current j_sc, the
        recombination current j_0 and the temperature of the solar cell.

        Args:
            j_sc (float): Short-Circuit current density.
            j_0 (float): Recombination current density.
            temperature (float): Temperature of the solar cell.

        Returns:
            float: The calculated maximum power.

        """

        # Calculate the current density J for a specified voltage V
        def current_density(voltage):
            j = j_sc - j_0 * (np.exp(e * voltage / (k * temperature)) - 1.0)
            return j

        # Calculate the corresponding power density P
        def power(voltage):
            p = current_density(voltage) * voltage
            return p

        # A somewhat primitive, but perfectly robust way of getting a reasonable
        # estimate for the maximum power.
        test_voltage = 0
        voltage_step = 0.001
        while power(test_voltage + voltage_step) > power(test_voltage):
            test_voltage += voltage_step

        return power(test_voltage)

    @staticmethod
    def calculate_slme_from_vasprun(filename, temperature=298.15, thickness=5e-7):
        """
        Calculate the SLME straight from a vasprun.xml file.

        Args:
            filename (str):
            temperature (float):
            thickness (float):

        Returns:
            float: Calculated SLME.

        """
        return SolarCell.from_file(filename).slme(temperature, thickness)

    @staticmethod
    def sq(bandgap, temperature=298.15, fr=1.0, interp_mesh=0.001, max_energy=20.0):
        """
        Calculate the Shockley-Queisser limit for a specified bandgap, temperature
        and fraction of radiative recombination.

        Args:
            bandgap (float): Band gap of the absorber material in eV.
            temperature (float): Temperature of the solar cell. Defaults to 25 °C,
                 or 298.15 K.
            fr (float): Fraction of radiative recombination.
            interp_mesh (float): Distance between two energy points in the grid
                used for the interpolation.
            max_energy (float): Maximum energy in the energy grid.

        Returns:
            (float): Shockley-Queisser detailed balance limit.

        """
        # Set up the energy grid for the calculation
        energy = np.linspace(
            interp_mesh, max_energy, int(np.ceil(max_energy) / interp_mesh)
        )

        # Set up the absorption coefficient (Step function for SQ)
        absorptivity = (energy >= bandgap).astype(float)

        # Get total solar_spectrum
        solar_spectrum = \
            EMRadSpectrum.get_solar_spectrum("am1.5g").get_interp_function()(energy)

        # Calculation of energy-dependent blackbody spectrum, in units of W / m**2
        blackbody_spectrum = EMRadSpectrum.get_blackbody(temperature,
                                                         energy).photon_flux

        # Numerically integrating irradiance over energy grid ~ A/m**2
        j_0_r = e * np.pi * simps(blackbody_spectrum * absorptivity, energy)
        j_0 = j_0_r / fr

        # Numerically integrating irradiance over wavelength array ~ A/m**2
        j_sc = e * simps(solar_spectrum * absorptivity, energy)

        # Determine the open circuit voltage.
        v_oc = 0
        voltage_step = 0.001
        while j_sc - j_0 * (np.exp(e * v_oc / (k * temperature)) - 1.0) > 0:
            v_oc += voltage_step

        # Maximize the power versus the voltage
        max_power = SolarCell.maximize_power(j_sc, j_0, temperature)

        # Calculation of integrated solar spectrum
        power_in = EMRadSpectrum.get_solar_spectrum().get_total_power_density()

        # Calculate the maximized efficiency
        efficiency = max_power / power_in

        return efficiency, v_oc, j_sc, j_0


# Utility method

def to_matrix(xx, yy, zz, xy, yz, xz):
    """
    Convert a list of matrix components to a symmetric 3x3 matrix.
    Inputs should be in the order xx, yy, zz, xy, yz, xz.

    Args:
        xx (float): xx component of the matrix.
        yy (float): yy component of the matrix.
        zz (float): zz component of the matrix.
        xy (float): xy component of the matrix.
        yz (float): yz component of the matrix.
        xz (float): xz component of the matrix.

    Returns:
        (np.array): The matrix, as a 3x3 numpy array.

    """
    matrix = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    return matrix


# Copied from https://github.com/utf/kramers-kronig/blob/master/kkr.py
# All copyright belongs to Alex Ganose

def kkr(de, eps_imag, cshift=1e-6):
    """Calculate the Kramers-Kronig transformation on imaginary part of dielectric
    Doesn't correct for any artefacts resulting from finite window function.
    Args:
        de (float): Energy grid size at which the imaginary dielectric constant
            is given. The grid is expected to be regularly spaced.
        eps_imag (np.array): A numpy array with dimensions (n, 3, 3), containing
            the imaginary part of the dielectric tensor.
        cshift (float, optional): The implemented method includes a small
            complex shift. A larger value causes a slight smoothing of the
            dielectric function.
    Returns:
        A numpy array with dimensions (n, 3, 3) containing the real part of the
        dielectric function.
    """
    eps_imag = np.array(eps_imag)
    nedos = eps_imag.shape[0]
    cshift = complex(0, cshift)
    w_i = np.arange(0, nedos * de, de, dtype=np.complex_)
    w_i = np.reshape(w_i, (nedos, 1, 1))

    def integration_element(w_r):
        factor = w_i / (w_i ** 2 - w_r ** 2 + cshift)
        total = np.sum(eps_imag * factor, axis=0)
        return total * (2 / math.pi) * de + np.diag([1, 1, 1])

    return np.real([integration_element(w_r) for w_r in w_i[:, 0, 0]])
