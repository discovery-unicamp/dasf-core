# DASF is an Accelerated and Scalable Framework

DASF is a generic framework specialized in acceleration and scaling common 
techniques for Machine Learning. DASF uses most methods and function from 
the most common libraries to increase the speed up of most algorithms. Part 
of this is to use Dask data to scale computation and RAPIDS AI algorithms to 
extend the support to GPUs as well.

## Installation

For now, the installation can be done using docker or singularity (if available).

### Docker

To install DASF using docker or singularity, you must in the go to the `build/`
directory and execute the command below directory according to your build type:
`cpu` or `gpu`. Notice that DASF uses [HPC Container Maker](https://github.com/NVIDIA/hpc-container-maker)
(HPCCM) to generate recipes for all sort of container types. You should install
HPCCM first, in order to generate them.

```bash
./build_docker.sh --device <cpu|gpu>
```

You can also configure other parameters of the container if you want. Run `-h`
for further information. It includes the container backend: docker or
singularity.

The `dasf` image will be created and be ready to use. Once it is ready, you 
can start a jupyter instance by executing the command:

```bash
./start_jupyter_server.sh
```

### Install

To install this development version, all you need to do is run `pip` from the 
root project directory (the same where `pyproject.toml` lives).

```bash
pip3 install .
```

### Testing

If you have a working environment with DASF installed, you can execute the all 
the test set. Make sure you have all development packages installed such as 
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
| SOM                      |    X    |    X    |       X       |       X       |    dasf.ml.cluster    |
| Agglomerative Clustering |    X    |    X    |               |               |    dasf.ml.cluster    |
| DBSCAN                   |    X    |    X    |               |       X       |    dasf.ml.cluster    |
| HDBSCAN                  |    X    |    X    |               |               |    dasf.ml.cluster    |
| Spectral Clustering      |    X    |         |       X       |               |    dasf.ml.cluster    |
| Gaussian Mixture Models  |    X    |         |               |               |    dasf.ml.mixture    |
| PCA                      |    X    |    X    |       X       |       X       | dasf.ml.decomposition |
| SVM                      |    X    |    X    |               |               |      dasf.ml.svm      |
| Boosted Trees            |    X    |    X    |       X       |       X       |    dasf.ml.xgboost    |
| KNN                      |    X    |    X    |               |               |   dasf.ml.neighbors   |
