# DASF is an Accelerated and Scalable Framework

[![Continuous Test](https://github.com/discovery-unicamp/dasf-core/actions/workflows/ci.yaml/badge.svg)](https://github.com/discovery-unicamp/dasf-core/actions/workflows/ci.yaml)
[![Commit Check Policy](https://github.com/discovery-unicamp/dasf-core/actions/workflows/commit-check.yaml/badge.svg)](https://github.com/discovery-unicamp/dasf-core/actions/workflows/commit-check.yaml)
![Interrogate](https://raw.githubusercontent.com/discovery-unicamp/dasf-core/badges/badges/interrogate_badge.svg)
![Coverage](https://img.shields.io/badge/Code%20Coverage-61%25-yellow?style=flat)

DASF is a generic framework specialized in acceleration and scaling common 
techniques for Machine Learning. DASF uses most methods and functions from 
the most common libraries to increase the speed up of most algorithms. Part 
of this is to use Dask data to scale computation and RAPIDS AI algorithms to 
extend the support to GPUs as well.

## Installation

For now, the installation can be done using docker or singularity (if available).

### Containers

To install DASF using docker or singularity, you must go to the `build/`
directory and execute the command below directory according to your build type:
`cpu` or `gpu`. Notice that DASF uses [HPC Container Maker](https://github.com/NVIDIA/hpc-container-maker)
(HPCCM) to generate recipes for all sorts of container types. You should install
HPCCM first, in order to generate them.

```bash
./build_container.sh --device <cpu|gpu>
```

You can also configure other parameters of the container if you want. Run `-h`
for further information. It includes the container backend: docker or
singularity.

The `dasf` image will be created and ready to use. Once it is ready, you 
can start a jupyter instance by executing the command:

```bash
./start_jupyter_server.sh --device <cpu|gpu>
```

You can also define a different port by using `--port PORT` argument.

### Install

To install this development version, all you need to do is run `pip` from the 
root project directory (the same where `pyproject.toml` lives).

```bash
pip3 install .
```

### Examples

If you want to see some examples of how to use DASF, you can visit the
[tutorials](https://discovery-unicamp.github.io/dasf-core/tutorials.html)
page to get more information of basic and advanced usage.

### Testing

If you have a working environment with DASF installed, you can execute all 
the test sets. Make sure you have all development packages installed such as 
**pytest**, **parameterized** and **mock**. To run, you need to execute 
`pytest` from the `tests/` directory.

```bash
pytest tests/
```

### Machine Learning Algorithms

The table below is a list of supported machine learning algorithms by DASF framework.

|     **ML Algorithm**     | **CPU** | **GPU** | **Multi-CPU** | **Multi-GPU** |       **Path**        |
|--------------------------|:-------:|:-------:|:-------------:|:-------------:|:---------------------:|
| K-Means                  |    X    |    X    |       X       |       X       |    dasf.ml.cluster    |
| Agglomerative Clustering |    X    |    X    |               |               |    dasf.ml.cluster    |
| DBSCAN                   |    X    |    X    |               |       X       |    dasf.ml.cluster    |
| HDBSCAN                  |    X    |    X    |               |               |    dasf.ml.cluster    |
| Spectral Clustering      |    X    |         |       X       |               |    dasf.ml.cluster    |
| Gaussian Mixture Models  |    X    |         |               |               |    dasf.ml.mixture    |
| PCA                      |    X    |    X    |       X       |       X       | dasf.ml.decomposition |
| SVM                      |    X    |    X    |               |               |      dasf.ml.svm      |
| Boosted Trees            |    X    |    X    |       X       |       X       |    dasf.ml.xgboost    |
| KNN                      |    X    |    X    |               |               |   dasf.ml.neighbors   |

### Cite

If you are using this project in your research, please cite our first paper where DASF was proposed.

```bibtex
@inproceedings{dasf,
  title        = {DASF: a high-performance and scalable framework for large seismic datasets},
  author       = {Julio C. Faracco and Otávio O. Napoli and João Seródio and Carlos A. Astudillo and Leandro Villas and Edson Borin and Alan A. Souza and Daniel C. Miranda and João Paulo Navarro},
  year         = {2024},
  month        = {August},
  booktitle    = {Proceedings of the International Meeting for Applied Geoscience and Energy},
  address      = {Houston, TX},
  organization = {AAPG/SEG}
}
```

### Authors

For further reference, below the authors list:

* Julio Faracco
* João Seródio
* Otavio Napoli
* Edson Borin


