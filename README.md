# DASF is an Accelerated and Scalable Framework

DASF is a generic framework specialized in acceleration and scaling common 
techniques for Machine Learning. DASF uses most methods and function from 
the most common libraries to increase the speed up of most algorithms. Part 
of this is to use Dask data to scale computation and RAPIDS AI algorithms to 
extend the support to GPUs as well.

## Installation

The installation can be done using conda or docker

### Docker

To install DASF using docker, you must in the go to the `build/` directory and execute the command below directory according to your build type: `cpu` or `gpu`.

```bash
./build_docker.sh <cpu|gpu>
```

The `dasf` image will be created and be ready to use. Once it is ready, you can start a jupyter instance by executing the command:

```bash
./start_jupyter_server.sh
```

### Conda

### Machine Learning Algorithms

The table below is a list of supported machine learning algorithms by DASF framework.

|     **ML Algorithm**     | **CPU** | **GPU** | **Multi-CPU** | **Multi-GPU** |
|--------------------------|:-------:|:-------:|:-------------:|:-------------:|
| K-Means                  |    X    |    X    |       X       |       X       |
| SOM                      |    X    |    X    |       X       |       X       |
| Agglomerative Clustering |    X    |    X    |               |               |
| DBSCAN                   |    X    |    X    |               |       X       |
| HDBSCAN                  |    X    |    X    |               |               |
| Gaussian Mixture Models  |    X    |         |               |               |
| PCA                      |    X    |    X    |       X       |       X       |
