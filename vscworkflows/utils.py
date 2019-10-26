# coding: utf8
# Copyright (c) Marnik Bercx, University of Antwerp
# Distributed under the terms of the MIT License

import collections

"""
Utility methods for the vsc-workflows package.

"""

__author__ = "Marnik Bercx"
__copyright__ = "Copyright 2019, Marnik Bercx, University of Antwerp"
__version__ = "pre-alpha"
__maintainer__ = "Marnik Bercx"
__email__ = "marnik.bercx@uantwerpen.be"
__date__ = "Oct 2019"


def vasp_input_update(vasp_input_params, d):

    for k, v in d.items():
        if isinstance(v, dict) and k != "user_kpoints_settings":
            vasp_input_params[k].update(v)
        else:
            vasp_input_params[k] = v
