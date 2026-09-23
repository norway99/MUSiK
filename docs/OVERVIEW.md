# MUSiK: Multi-transducer Ultrasound Simulations in K-wave

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Module Architecture](#module-architecture)
4. [Core Concepts](#core-concepts)
5. [Typical Workflow](#typical-workflow)
6. [API Quick Reference](#api-quick-reference)
7. [Demo Notebooks Guide](#demo-notebooks-guide)
8. [Advanced Topics](#advanced-topics)

---

## Introduction

MUSiK (Multi-transducer Ultrasound Simulations in K-wave) is a Python framework for simulating multi-transducer ultrasound imaging systems. Built on top of [k-Wave](http://www.k-wave.org/), it provides high-level abstractions for:

- **Phantom creation**: Define tissue volumes with realistic acoustic properties
- **Transducer configuration**: Model focused and planewave transducers with arbitrary positioning
- **Simulation orchestration**: Run large-scale simulations with multi-GPU and SLURM support
- **Image reconstruction**: Delay-and-sum (DAS) beamforming and multi-view compounding

### Target Audience

- Medical imaging researchers
- Ultrasound system designers
- Acoustic simulation developers
- Students learning ultrasound physics

### Key Features

- Multi-transducer support with arbitrary spatial configurations
- Realistic tissue modeling with heterogeneous acoustic properties
- Parallel simulation execution (multi-process, multi-GPU, SLURM)
- Multiple reconstruction algorithms (DAS, compounding)
- Memory-efficient processing for large-scale simulations

---

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (recommended for k-Wave acceleration)
- Git (for cloning with submodules)

### Installation Steps

1. **Clone the repository with submodules:**
   ```bash
   git clone --recursive https://github.com/norway99/MUSiK.git
   cd MUSiK
   ```

2. **Install the package:**
   ```bash
   pip install -e .
   ```

3. **Install optional dependencies:**
   ```bash
   # For development
   pip install -e ".[dev]"

   # For full functionality (mesh processing, DICOM, etc.)
   pip install -e ".[full]"

   # For documentation building
   pip install -e ".[docs]"

   # For Jupyter notebooks
   pip install -e ".[jupyter]"
   ```

4. **Download k-Wave binaries:**

   Follow the instructions in the [k-wave-python repository](https://github.com/waltsims/k-wave-python) to download the required binaries for your platform.

### Troubleshooting

See [troubleshooting.md](../troubleshooting.md) for common issues and solutions, including:
- Submodule initialization problems
- k-Wave binary compatibility
- CUDA/GPU setup

---

## Module Architecture

### Package Structure

```
musik/
├── __init__.py           # Package initialization, version info
├── simulation.py         # SimProperties, Simulation classes
├── experiment.py         # Experiment, Results classes
├── reconstruction.py     # DAS, Compounding, Reconstruction classes
├── phantom.py            # Phantom class for medium definition
├── transducer.py         # Transducer, Focused, Planewave classes
├── transducer_set.py     # TransducerSet for multi-probe setups
├── sensor.py             # Sensor class for signal capture
├── tissue.py             # Tissue dataclass for material properties
├── analytic_wave.py      # Analytical wave field calculations
└── utils/
    ├── utils.py          # I/O utilities, JSON/numpy helpers
    ├── geometry.py       # Transform class, coordinate operations
    ├── diff_geo.py       # Differential geometry utilities
    └── phantom_builder.py # DICOM reading, mesh voxelization
```

### Dependency Graph

```
                    ┌─────────────┐
                    │   Tissue    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   Phantom   │
                    └──────┬──────┘
                           │
┌───────────────┐   ┌──────▼──────┐   ┌─────────────┐
│  Transducer   │──▶│ Experiment  │◀──│   Sensor    │
│ (Focused/PW)  │   └──────┬──────┘   └─────────────┘
└───────┬───────┘          │
        │           ┌──────▼──────┐
        ▼           │ Simulation  │
┌───────────────┐   └──────┬──────┘
│ TransducerSet │          │
└───────────────┘   ┌──────▼──────┐
                    │   Results   │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │Reconstruction│
                    │ (DAS/Compound)│
                    └─────────────┘
```

---

## Core Concepts

### SimProperties

`SimProperties` defines the computational parameters for k-Wave simulations.

**Key Attributes:**
| Attribute | Type | Description |
|-----------|------|-------------|
| `grid_size` | tuple[float, float, float] | Physical dimensions of simulation domain (meters) |
| `voxel_size` | float | Spatial resolution (meters) |
| `PML_size` | int | Perfectly Matched Layer padding for boundary absorption |
| `PML_alpha` | float | PML absorption coefficient |
| `t_end` | float | Simulation end time (seconds) - auto-computed based on grid_size |
| `bona` | float | Nonlinearity coefficient (B/A parameter) |
| `alpha_coeff` | float | Acoustic attenuation coefficient |
| `alpha_power` | float | Frequency power law for attenuation |

**Key Methods:**
- `optimize_simulation_parameters(frequency)`: Auto-tune grid resolution based on acoustic frequency
- `calc_matrix_size()`: Compute FFT-friendly grid dimensions
- `save(path)` / `load(path)`: Serialize/deserialize configuration

### Phantom

`Phantom` represents the 3D tissue volume being imaged.

**Key Attributes:**
| Attribute | Type | Description |
|-----------|------|-------------|
| `voxel_dims` | tuple[float, float, float] | Physical size of each voxel (meters) |
| `matrix_dims` | tuple[int, int, int] | Number of voxels in each dimension |
| `baseline` | tuple[float, float] | Default (speed_of_sound, density) |
| `mask` | np.ndarray | Binary tissue map |
| `tissues` | dict[int, Tissue] | Mapping from mask values to Tissue objects |

**Key Methods:**
- `create_from_image(image, ...)`: Create phantom from CT/MRI data
- `create_from_list(shapes, ...)`: Create phantom from geometric primitives
- `interpolate_phantom(transform, ...)`: Transform phantom for specific ray geometry
- `get_complete()`: Return full (speed_of_sound, density) arrays

### Transducer

`Transducer` models ultrasound transducer hardware. Two subclasses are provided:

**`Focused`**: Phased array with focal point
- `focal_distance`: Distance to focal point
- `steering_angles`: Array of beam steering directions

**`Planewave`**: Flat-faced transducer for plane wave transmission
- `steering_angles`: Plane wave steering directions

**Common Attributes:**
| Attribute | Type | Description |
|-----------|------|-------------|
| `max_frequency` | float | Transducer center frequency (Hz) |
| `elements` | int | Number of transducer elements |
| `width` | float | Transducer width (meters) |
| `height` | float | Transducer height (meters) |
| `ray_num` | int | Number of transmit rays |
| `ray_transforms` | list[Transform] | Spatial transform for each ray |

**Key Methods:**
- `make_pulse(dt, t_end)`: Generate transmit pulse waveform
- `make_notatransducer(kgrid, ...)`: Create k-Wave source object
- `preprocess(signal, dt, ...)`: Signal conditioning (filtering, TGC)
- `make_scan_line(signal, dt, ...)`: Extract beamformed signal for one ray

### TransducerSet

`TransducerSet` manages multiple transducers and their spatial configurations.

**Key Attributes:**
| Attribute | Type | Description |
|-----------|------|-------------|
| `transducers` | list[Transducer] | List of transducer objects |
| `poses` | list[Transform] | Spatial position/orientation of each transducer |
| `transmit` | list[bool] | Which transducers are active for transmission |

**Key Methods:**
- `transmit_transducers()`: Get list of transmitting transducers
- `transmit_poses()`: Get poses of transmitting transducers
- `generate_extrinsics(mode, ...)`: Auto-position transducers (spherical, cylindrical, random)
- `save(path)` / `load(path)`: Serialize/deserialize configuration

### Sensor

`Sensor` configures how acoustic signals are captured.

**Aperture Types:**
| Type | Description |
|------|-------------|
| `transmit_as_receive` | Same elements transmit and receive |
| `extended_aperture` | All transducer elements receive |
| `pressure_field` | Capture full pressure field in volume |
| `microphone` | Point receivers at specified locations |

**Key Attributes:**
- `aperture_type`: One of the above types
- `sensor_coords`: Global 3D coordinates of sensor points
- `element_lookup`: Mapping from sensors to transducer elements

**Key Methods:**
- `make_sensor_mask(kgrid)`: Create k-Wave sensor mask
- `voxel_to_element(data)`: Aggregate voxel data to transducer elements

### Experiment

`Experiment` orchestrates the complete simulation pipeline.

**Key Attributes:**
- `sim_properties`: SimProperties instance
- `phantom`: Phantom instance
- `transducer_set`: TransducerSet instance
- `sensor`: Sensor instance
- `path`: Directory for saving results

**Key Methods:**
- `run(workers=1)`: Execute simulations (single or multi-process)
- `subdivide(total_nodes, node_index)`: Distribute work across compute nodes
- `save()` / `load(path)`: Serialize/deserialize experiment configuration

### Results

`Results` provides indexed access to saved simulation outputs.

**Key Methods:**
- `__getitem__(index)`: Load specific result (returns time array and signal)
- `indices()`: Get list of available simulation indices

### Reconstruction Classes

#### DAS (Delay-and-Sum)

Standard beamforming reconstruction.

**Key Methods:**
- `preprocess_data(workers=1, ...)`: Multi-worker preprocessing with optional batch saving
- `get_image(coords, ...)`: Generate 2D/3D reconstruction at specified coordinates
- `get_signals(coords, ...)`: Get per-transducer interpolated signals

#### Compounding

Advanced reconstruction using all receive elements with multi-view combination.

**Key Methods:**
- `compound(output_coords, ...)`: Full reconstruction with pressure field weighting
- `scanline_reconstruction(index, ...)`: Single beam reconstruction with apodization

---

## Typical Workflow

### Step 1: Define Tissues

```python
from musik import tissue

# Create tissue with acoustic properties
muscle = tissue.Tissue(
    sound_speed=1580,      # m/s
    density=1050,          # kg/m³
    alpha_coeff=0.5,       # attenuation coefficient
    alpha_power=1.1,       # frequency power law
    heterogeneity=0.02,    # random variation
)

fat = tissue.Tissue(
    sound_speed=1450,
    density=950,
    alpha_coeff=0.6,
    alpha_power=1.0,
)
```

### Step 2: Create Phantom

```python
from musik import phantom

# Create phantom with specified dimensions
test_phantom = phantom.Phantom(
    voxel_dims=(0.5e-3, 0.5e-3, 0.5e-3),  # 0.5mm isotropic
    matrix_dims=(128, 128, 64),
    baseline=(1540, 1000),                 # water properties
    seed=42,
)

# Add tissue regions
test_phantom.create_from_list([
    {'shape': 'sphere', 'center': (0, 0, 0.03), 'radius': 0.01, 'tissue': muscle},
    {'shape': 'sphere', 'center': (0.02, 0, 0.04), 'radius': 0.008, 'tissue': fat},
])
```

### Step 3: Configure Transducers

```python
from musik import transducer, transducer_set
from musik.utils import geometry

# Create focused transducer
tx = transducer.Focused(
    max_frequency=2e6,     # 2 MHz
    elements=128,
    width=0.04,            # 40mm aperture
    height=0.01,
    focal_distance=0.05,
    ray_num=64,            # number of scan lines
)

# Create transducer set with positioning
tx_set = transducer_set.TransducerSet(
    transducers=[tx],
    poses=[geometry.Transform()],  # identity transform (at origin)
    transmit=[True],
)
```

### Step 4: Configure Sensor

```python
from musik import sensor

sens = sensor.Sensor(
    transducer_set=tx_set,
    aperture_type='transmit_as_receive',
)
```

### Step 5: Set Simulation Properties

```python
from musik import simulation

sim_props = simulation.SimProperties(
    grid_size=(0.08, 0.08, 0.08),
    voxel_size=0.5e-3,
    PML_size=20,
)
sim_props.optimize_simulation_parameters(tx.max_frequency)
```

### Step 6: Create and Run Experiment

```python
from musik import experiment

exp = experiment.Experiment(
    sim_properties=sim_props,
    phantom=test_phantom,
    transducer_set=tx_set,
    sensor=sens,
    path='./results/my_experiment',
)
exp.save()

# Run simulations (use workers>1 for parallel execution)
exp.run(workers=4)
```

### Step 7: Reconstruct Image

```python
from musik import reconstruction
import numpy as np

# Load experiment and create reconstruction object
exp = experiment.Experiment.load('./results/my_experiment')
das = reconstruction.DAS(exp)

# Preprocess data
das.preprocess_data(workers=4)

# Define output coordinates
x = np.linspace(-0.03, 0.03, 256)
z = np.linspace(0.01, 0.07, 512)
coords = np.stack(np.meshgrid(x, np.zeros(1), z, indexing='ij'), axis=-1)

# Generate image
image = das.get_image(coords)
```

---

## API Quick Reference

### Classes

| Class | Module | Purpose |
|-------|--------|---------|
| `SimProperties` | `simulation` | Simulation grid and timing configuration |
| `Simulation` | `simulation` | Single simulation execution |
| `Phantom` | `phantom` | Tissue volume definition |
| `Tissue` | `tissue` | Material acoustic properties |
| `Transducer` | `transducer` | Base transducer class |
| `Focused` | `transducer` | Focused/phased array transducer |
| `Planewave` | `transducer` | Plane wave transducer |
| `TransducerSet` | `transducer_set` | Multi-transducer management |
| `Sensor` | `sensor` | Signal capture configuration |
| `Experiment` | `experiment` | Simulation orchestration |
| `Results` | `experiment` | Result data access |
| `Reconstruction` | `reconstruction` | Base reconstruction class |
| `DAS` | `reconstruction` | Delay-and-sum beamforming |
| `Compounding` | `reconstruction` | Multi-view compounding |
| `Transform` | `utils.geometry` | 3D rotation and translation |

### Common Parameters

| Parameter | Typical Values | Description |
|-----------|----------------|-------------|
| `voxel_size` | 0.1e-3 to 1e-3 | Spatial resolution in meters |
| `max_frequency` | 1e6 to 10e6 | Transducer frequency in Hz |
| `elements` | 64 to 256 | Number of array elements |
| `PML_size` | 10 to 30 | Boundary layer thickness in voxels |
| `alpha_coeff` | 0.1 to 2.0 | Attenuation in dB/(MHz·cm) |
| `alpha_power` | 1.0 to 1.5 | Frequency dependence exponent |

---

## Demo Notebooks Guide

### Getting Started

| Notebook | Description |
|----------|-------------|
| `demos/start_here/intro_demo.ipynb` | Basic workflow: phantom, transducer, simulation, reconstruction |

### In Vitro Phantoms

| Notebook | Description |
|----------|-------------|
| `demos/invitro/resolution_phantom.ipynb` | Wire grid phantom for resolution testing |
| `demos/invitro/contrast_phantom.ipynb` | Spheres with varying attenuation |
| `demos/invitro/psf_cross_arc.ipynb` | PSF characterization with wire patterns |
| `demos/invitro/psf_testing_spheres.ipynb` | Sphere detectability testing |

### Analytical Wave

| Notebook | Description |
|----------|-------------|
| `demos/analytical_wave/excitation_compensation_single.ipynb` | Single transducer pressure field compensation |
| `demos/analytical_wave/excitation_compensation_multi.ipynb` | Multi-transducer sequential vs synchronous imaging |

### Clinical Anatomy

| Notebook | Description |
|----------|-------------|
| `demos/kidney/kidney_scan_focused.ipynb` | Kidney imaging with focused transducer |
| `demos/kidney/kidney_scan_sa.ipynb` | Kidney with synthetic aperture |
| `demos/breast/breast_synthetic_aperture.ipynb` | 3D breast imaging with many transducers |
| `demos/arm/ct_2_phantom.ipynb` | CT-to-phantom conversion pipeline |
| `demos/arm/invitro_arm_tomography.ipynb` | Arm tomographic imaging |
| `demos/heart/TEE/cardiac_tee.ipynb` | Transesophageal echocardiography |
| `demos/heart/TTE/cardiac_tte.ipynb` | Transthoracic echocardiography |

---

## Advanced Topics

### Parallel Execution

The `workers` parameter controls how many simulations are prepared in parallel
(RAM intensive), while GPU simulations run sequentially:

```python
# Prepare up to 3 simulations in parallel, run on single GPU
exp.run(workers=3)
```

For multi-GPU setups, use SLURM array jobs or launch separate processes
with different `CUDA_VISIBLE_DEVICES` environment variables.

### SLURM Array Jobs

```python
# In your SLURM script, use subdivide to distribute work
import os
node_index = int(os.environ['SLURM_ARRAY_TASK_ID'])
total_nodes = int(os.environ['SLURM_ARRAY_TASK_COUNT'])

exp = experiment.Experiment.load('./results/my_experiment')
exp.subdivide(total_nodes, node_index)
exp.run(workers=1)
```

### Large-Scale Reconstruction

For simulations with many (>100K) rays, use incremental reconstruction preprocessing:

```python
das = reconstruction.DAS(exp)

# Preprocess in batches with incremental saving
das.preprocess_data(
    workers=8,
    batch_size=100,
    save_batches=True,
    resume=True,  # Resume from last saved batch
)
```

### Custom Tissue Properties

```python
# Create tissue from CT Hounsfield units
from musik.utils import phantom_builder

# Load CT slices and convert to acoustic properties
sos, density = phantom_builder.ct_to_acoustic(
    ct_images,
    hu_to_sos_func=my_transfer_function,
    hu_to_density_func=my_density_function,
)
```

### Memory-Efficient Data Loading

```python
# Use lazy loading for large datasets
loader = reconstruction.PreprocessedDataLoader(
    exp,
    batch_dir='./preprocessed_batches',
    cache_size=5,  # LRU cache for batches
)

# Access data without loading all into memory
for i in range(len(loader)):
    coords, intensities = loader[i]
    # Process incrementally
```

---

## References

- [k-Wave Documentation](http://www.k-wave.org/documentation.php)
- [k-wave-python GitHub](https://github.com/waltsims/k-wave-python)
- [MUSiK GitHub Repository](https://github.com/norway99/MUSiK)
