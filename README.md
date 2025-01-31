# Solving the Discretised Shallow Water Equations Using Non-Uniform Grids and Machine-Learning Libraries
This repository contains the implementation of a novel approach for solving the discretised Shallow Water Equations (SWEs) using non-uniform structured grids. The method leverages machine-learning libraries to represent finite element discretisations as convolutional operations, enabling efficient execution on CPUs, GPUs, and AI processors. The model is validated against analytical solutions and real-world flood event simulations, such as the 2005 Carlisle flooding.

## Key Features:
- **Non-Uniform Structured Grids**: Enables local refinement to capture critical hydrodynamic features.
- **Machine-Learning-Based Implementation**: Uses neural network convolutional layers to represent numerical discretisations.
- **Platform Independence**: Compatible with CPUs, GPUs, and AI-optimized processors.
- **Validated on Real-World Data**: Successfully tested on the 2005 Carlisle flooding event, showing agreement with observational data.

## Applications:
- **Idealised Problems**: Validated against analytical solutions.
- **Real-World Test Case**: Applied to the 2005 Carlisle flood, with significant potential speed-ups with AI processors.

## Getting Started
To get started with the code, clone this repository using the code at the bottom of this page or click on each file to download them separately or click on the links below:
- [Carlisle Bathymetry](https://github.com/Amin-Nadimy/Shallow_Water_Equations_NN4PDEs/blob/main/carlisle-5m.dem.raw)
- [Inlet Points](https://github.com/Amin-Nadimy/Shallow_Water_Equations_NN4PDEs/blob/main/carlisle.bci)
- [Flowrates](https://github.com/Amin-Nadimy/Shallow_Water_Equations_NN4PDEs/blob/main/flowrates.csv)
---
### Carlisle 2005 Flood Event Modeled with NN4PDEs
```geojson
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "id": 1,
      "properties": {
        "ID": 0
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
              [-2.886863, 54.910878],
              [-2.959449, 54.910878],
              [-2.959449, 54.883988],
              [-2.886863, 54.883988],
              [-2.886863, 54.910878]
          ]
        ]
      }
    }
  ]
}
```

<!-- <img src="https://github.com/Amin-Nadimy/Shallow_Water_Equations_-SWE-/blob/main/SWE_2.gif" width="512" /> -->

---
## Repository Structure
```plaintext
. Solving the Discretised Shallow Water Equations Using Non-Uniform Grids and Machine-Learning Libraries
├── Demos/                                 # Folder for animation files
│   ├── 68-hours_simulation.mp4            # mp4 format of the 68 h simulation results
│   └── SWE_2_gif.gif                      # gif format of the 68 h simulation results
├── Source_code/                           # Python codes for Linea, Quadratic and parallel simulations
│   ├── Linear/                     
│           ├── AI4SWE_Linear.py           # Main code
│           └── SWE_2D_diff_Linear.py      # External library
```
---

https://github.com/Amin-Nadimy/Shallow_Water_Equations_NN4PDEs/assets/71018515/f44fed0b-9b45-499a-8383-9e65b6193202

## Installation

### Prerequisites

Before proceeding, ensure that you have the following:

- **Python 3.10**: It is recommended to use Python 3.10 for compatibility with the required packages and libraries.

- **Conda (Preferred)**: Although not essential, it is recommended to use [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) for managing your Python environment and dependencies. Conda simplifies package management and helps avoid conflicts.

- **GPU with CUDA support**: A GPU that supports CUDA and has at least 20GB of VRAM. Ensure that your CUDA drivers are correctly installed and configured.

### Environment Setup

To set up the environment, run:

```bash
conda env create -f environment.yml
```

Alternatively, you can install the required packages using `requirements.txt`:

```bash
pip install -r requirements.txt
```

Usage
Detailed instructions on how to use the code and run simulations can be found in the docs directory.

## License
This project is licensed under the MIT License. See the LICENSE.md file for details.

```ruby
git clone https://github.com/Amin-Nadimy/Flood-Modelling-on-Non-Uniform-Grids.git
```
```ruby
cd Shallow_Water_Equations_NN4PDEs
```

## Contact and references
For more information please contact me via:
- Email: amin.nadimy19@imperial.ac.uk
