import sys
import os

parent = os.path.dirname(os.path.realpath("../"))
sys.path.append(parent)

import numpy as np

from musik import *
from utils import phantom_builder
from utils import geometry

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import open3d as o3d


def build_phantom(voxel_size=0.001, asset_dir=None):
    """Build the cardiac TEE phantom with tissues."""
    if asset_dir is None:
        asset_dir = f"{parent}/assets/cardiac_TEE_phantom"

    surface_mesh = o3d.io.read_triangle_mesh(
        f"{asset_dir}/esophageal_surface.ply"
    )
    body_mask = phantom_builder.voxelize(voxel_size, mesh=surface_mesh)

    test_phantom = phantom.Phantom(
        voxel_dims=(voxel_size, voxel_size, voxel_size),
        matrix_dims=body_mask.shape,
        baseline=(1540, 1000),
        seed=None,
    )

    blood = tissue.Tissue(name="blood", c=1578, rho=1060, sigma=1.3, scale=0.00001, label=1)
    myocardium = tissue.Tissue(
        name="myocardium", c=1592, rho=1081, sigma=20, scale=0.0001, label=2
    )
    esophagus = tissue.Tissue(
        name="esophagus", c=1500, rho=1100, sigma=10, scale=0.0001, label=3
    )
    fat = tissue.Tissue(name="fat", c=1480, rho=970, sigma=15, scale=0.0001, label=4)

    heart_tissue_list = [blood, myocardium, esophagus]

    test_phantom.build_organ_from_mesh(
        surface_mesh, voxel_size, heart_tissue_list, dir_path=asset_dir
    )
    test_phantom.set_default_tissue(fat)

    return test_phantom


def build_transducer_set(baseline_speed, num_transducers=1, seed=8888):
    """Initialize a set of focused transducers."""
    transducers = [
        transducer.Focused(
            max_frequency=1.0e6,
            elements=128,
            width=20e-3,
            height=20e-3,
            sensor_sampling_scheme="not_centroid",
            sweep=np.pi / 2,
            ray_num=32,
            imaging_ndims=2,
            focus_azimuth=100e-3,
            focus_elevation=150e-3,
            cycles=3,
        )
        for _ in range(num_transducers)
    ]

    for t in transducers:
        t.make_sensor_coords(baseline_speed)
        
    test_transducer_set = transducer_set.TransducerSet(transducers, seed=seed)
    
    test_transducer_set.assign_pose(0, geometry.Transform([0,0,0], [0,0,0]))

    return test_transducer_set


def build_sim_properties():
    """Build simulation properties."""
    return simulation.SimProperties(
        grid_size=(200e-3, 25e-3, 25e-3),
        voxel_size=(0.5e-3, 0.5e-3, 0.5e-3),
        PML_size=(16, 16, 16),
        PML_alpha=2,
        t_end=12e-5,
        bona=6,
        alpha_coeff=0.5,
        alpha_power=1.5,
    )


def initialize_simulation(simulation_path="simulate_autoregressive_guidance", save=True):
    """
    Initialize a complete simulation experiment for autoregressive guidance.

    Returns:
        experiment.Experiment: The configured experiment object.
    """
    test_phantom = build_phantom()
    test_transducer_set = build_transducer_set(test_phantom.baseline[0])
    test_sensor = sensor.Sensor(
        transducer_set=test_transducer_set, aperture_type="transmit_as_receive"
    )
    simprops = build_sim_properties()

    test_experiment = experiment.Experiment(
        simulation_path=simulation_path,
        sim_properties=simprops,
        phantom=test_phantom,
        transducer_set=test_transducer_set,
        sensor=test_sensor,
        nodes=1,
        results=None,
        indices=None,
        workers=4,
        additional_keys=[],
    )

    if save:
        test_experiment.save()

    return test_experiment


if __name__ == "__main__":
    test_experiment = initialize_simulation()