# coding: utf8
# Copyright (c) Marnik Bercx, University of Antwerp
# Distributed under the terms of the MIT License

import os

import click
from vscworkflows.config import load_config

"""
Command line interface for the vscworkflows package.

"""

__author__ = "Marnik Bercx"
__copyright__ = "Copyright 2018, Marnik Bercx, University of Antwerp"
__version__ = "pre-alpha"
__maintainer__ = "Marnik Bercx"
__email__ = "marnik.bercx@uantwerpen.be"
__date__ = "Jun 2019"

# This is used to make '-h' a shorter way to access the CLI help
CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


def _load_launchpad(name="base"):
    """
    Load the launchpad from the configuration folder in
    $HOME/.workflow_config/launchpad.

    Args:
        name (str): Name of the launchpad. Defaults to "base".

    Returns:
        fireworks.LaunchPad corresponding to the requested name.

    """
    if name != "base":
        try:
            return load_config("launchpad", name)
        except FileNotFoundError:
            raise FileNotFoundError(
                "Could not find requested launchpad in "
                "$HOME/.workflow_config/launchpad.  Use 'vsc config launchpad' to "
                "set up new launchpads."
            )
    else:
        # Try loading the base launchpad
        try:
            return load_config("launchpad", name)
        except FileNotFoundError:
            raise FileNotFoundError("Could not find a base launchpad. Use 'vsc "
                                    "config launchpad' to set up new launchpads.")


# endregion

@click.group(context_settings=CONTEXT_SETTINGS)
def main():
    """
    CLI tools for configuring and launching jobs for workflows.

    """
    pass


@main.command(context_settings=CONTEXT_SETTINGS)
@click.option("-l", "--lpad_name", default="base",
              help="Name of the configured launchpad that contains the details "
                   "of the  mongoDB server you want to run Fireworks from. "
                   "Defaults to 'base'.")
@click.option("-f", "--fworker_name", default="base",
              help="Name of the fireworker which you are submitting the jobs to, "
                   "i.e.  the cluster you are currently logged into. Defaults to "
                   "'base'. If  you have configured the workflows for another "
                   "cluster (e.g. hopper), you can use this option to use that "
                   "configuration.")
@click.option("-n", "--number_nodes", default=1,
              help="Number of nodes to request for the job. This will be change "
                   "the name of the fireworker, so it will pick up Fireworks "
                   "which have _fworker specified.")
@click.option("-t", "--walltime", default=72,
              help="Walltime of the job, expressed in hours. Defaults to 72.")
@click.option("-j", "--number_jobs", default=1,
              help="The number of jobs to submit to the queue.")
@click.option("--hog", is_flag=True,
              help="Hog the nodes for the whole walltime, i.e. do not stop the "
                   "job in case there are no more Fireworks to be launched.")
def qlaunch(lpad_name, fworker_name, number_nodes, walltime, number_jobs, hog):
    """
    Launch jobs to the queue that will accept Fireworks.

    """
    from fireworks.queue.queue_launcher import rapidfire
    from vscworkflows.config import load_config

    try:
        queue_adapter = load_config("qadapter", fworker_name)
    except FileNotFoundError:
        raise FileNotFoundError(
            "Could not find the qadapter of the fireworker in "
            "$HOME/.workflow_config/fworker. Use 'vsc config fworker' to set up new "
            "fireworkers."
        )
    queue_adapter["nnodes"] = number_nodes
    queue_adapter["walltime"] = str(walltime)
    queue_adapter["launchpad_file"] = os.path.join(
        os.path.expanduser("~"), ".workflow_config", "launchpad",
        lpad_name + "_launchpad.yaml"
    )
    queue_adapter["fireworker_file"] = os.path.join(
        os.path.expanduser("~"), ".workflow_config", "fworker",
        fworker_name + "_fworker.yaml"
    )
    if hog:
        queue_adapter["rocket_launch"] = "rapidfire --nlaunches infinite --sleep 10"
    else:
        # This line adds the timeout option to the
        queue_adapter["rocket_launch"] += " --timeout " + str(walltime * 3000)

    rapidfire(launchpad=load_config("launchpad", lpad_name),
              fworker=load_config("fworker", fworker_name), qadapter=queue_adapter,
              launch_dir=queue_adapter["logdir"], nlaunches=number_jobs,
              njobs_queue=0, njobs_block=500,
              sleep_time=0, reserve=False, fill_mode=True)


@main.group(context_settings=CONTEXT_SETTINGS)
def config():
    """
    Configure the Workflows setup.

    In order to submit and run workflows, you need to configure your fireworker and
    launchpad. Examples of configuration files and more information can be found in
    the vscworkflows/examples/config folder:

    https://github.com/mbercx/pybat/tree/master/vscworkflows/examples/config
    TODO: update

    """
    pass


@config.command(context_settings=CONTEXT_SETTINGS)
@click.option("-l", "--launchpad_file", default="")
@click.option("-N", "--name", default="base")
def launchpad(launchpad_file, name):
    """
    Configure a Workflows server or launchpad.

    Although the information can be put in manually when using the command without
    options, it's probably easiest to first set up the launchpad file and then use
    the '-l' option to configure the launchpad based on this file.

    Note that specifying a name for the launchpad allows you to differentiate between
    different database servers when submitting workflows or using 'vsc qlaunch'. If
    no name is specified, the launchpad will be set up as the base launchpad.

    """
    from vscworkflows.config import launchpad

    launchpad(launchpad_file=launchpad_file, lpad_name=name)


@config.command(context_settings=CONTEXT_SETTINGS)
@click.option("-f", "--fworker_file", default="")
@click.option("-N", "--name", default="base")
def fworker(fworker_file, name):
    """
    Configure the basic settings of a fireworker.

    Although the information can be put in manually when using the command without
    options, it's probably easiest to first set up the fireworker file and then use
    the '-f option to configure the fireworker based on this file.

    Note that specifying a name for the fworker allows you to configure multiple
    computational resources or settings.

    """
    from vscworkflows.config import fworker

    fworker(fireworker_file=fworker_file, fworker_name=name)


@config.command(context_settings=CONTEXT_SETTINGS)
@click.option("-q", "--qadapter_file", default="")
@click.option("-N", "--name", default="base")
def qadapter(qadapter_file, name):
    """
    Configure the standard queue adapter of a fireworker.

    """
    from vscworkflows.config import qadapter

    qadapter(qadapter_file=qadapter_file, fworker_name=name)


@config.command(context_settings=CONTEXT_SETTINGS)
@click.argument("template_file", nargs=1)
@click.option("-N", "--name", default="base")
def jobscript(template_file, name):
    """
    Add the job template of a fireworker.

    """
    from vscworkflows.config import jobscript

    jobscript(template_file=template_file, fworker_name=name)


@config.command(context_settings=CONTEXT_SETTINGS)
def check():
    from vscworkflows.config import check
    check()
