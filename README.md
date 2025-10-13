# DASF: An Accelerated and Scalable Framework for Machine Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Continuous Test](https://github.com/discovery-unicamp/dasf-core/actions/workflows/ci.yaml/badge.svg)](https://github.com/discovery-unicamp/dasf-core/actions/workflows/ci.yaml)
[![Commit Check Policy](https://github.com/discovery-unicamp/dasf-core/actions/workflows/commit-check.yaml/badge.svg)](https://github.com/discovery-unicamp/dasf-core/actions/workflows/commit-check.yaml)
![Interrogate](https://raw.githubusercontent.com/discovery-unicamp/dasf-core/badges/badges/interrogate_badge.svg)
![Coverage](https://raw.githubusercontent.com/discovery-unicamp/dasf-core/badges/badges/coverage.svg)
[![Docker](https://img.shields.io/badge/docker-required-blue.svg)](https://www.docker.com/)

DASF is a powerful, generic framework designed to accelerate and scale common machine learning techniques. By leveraging Dask for distributed computation and RAPIDS AI for GPU acceleration, DASF significantly speeds up algorithms, enabling you to tackle larger datasets and more complex problems.

## 🚀 Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

- [Docker](https://docs.docker.com/get-docker/) or [Singularity](https://sylabs.io/docs/)
- [HPC Container Maker (HPCCM)](https://github.com/NVIDIA/hpc-container-maker)

### Installation

#### 🐳 Container-Based Installation

The recommended way to get started with DASF is by using our pre-configured containers.

1.  **Navigate to the `build` directory:**

    ```bash
    cd build/
    ```

2.  **Build the container:**

    Choose the appropriate device type (`cpu` or `gpu`) for your environment.

    ```bash
    ./build_container.sh --device <cpu|gpu>
    ```

    For more build options, run `./build_container.sh -h`.

3.  **Start the Jupyter server:**

    Once the container is built, you can start a Jupyter server to begin working with DASF.

    ```bash
    ./start_jupyter_server.sh --device <cpu|gpu>
    ```

    You can specify a different port using the `--port` argument.

#### 🐍 Local Installation

For development purposes, you can install DASF locally using `pip`.

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/discovery-unicamp/dasf-core.git
    cd dasf-core
    ```

2.  **Install the package:**

    ```bash
    pip3 install .
    ```

## 📖 Usage

To learn how to use DASF, check out our comprehensive tutorials. They cover everything from basic usage to advanced features.

- [**Tutorials**](https://discovery-unicamp.github.io/dasf-core/tutorials.html)

## ✅ Testing

To ensure the stability and correctness of DASF, we have a comprehensive test suite. To run the tests, you'll need to have the development packages installed.

1.  **Install development dependencies:**

    ```bash
    pip3 install pytest parameterized mock
    ```

2.  **Run the tests:**

    ```bash
    pytest tests/
    ```

## 🤖 Supported Machine Learning Algorithms

DASF supports a wide range of machine learning algorithms, with varying levels of acceleration and scaling.

| **ML Algorithm**           | **CPU** | **GPU** | **Multi-CPU** | **Multi-GPU** | **Path**                |
| -------------------------- | :-----: | :-----: | :-----------: | :-----------: | :---------------------- |
| K-Means                    |    ✅    |    ✅    |       ✅       |       ✅       | `dasf.ml.cluster`       |
| Agglomerative Clustering   |    ✅    |    ✅    |               |               | `dasf.ml.cluster`       |
| DBSCAN                     |    ✅    |    ✅    |               |       ✅       | `dasf.ml.cluster`       |
| HDBSCAN                    |    ✅    |    ✅    |               |               | `dasf.ml.cluster`       |
| Spectral Clustering        |    ✅    |         |       ✅       |               | `dasf.ml.cluster`       |
| Gaussian Mixture Models    |    ✅    |         |               |               | `dasf.ml.mixture`       |
| PCA                        |    ✅    |    ✅    |       ✅       |       ✅       | `dasf.ml.decomposition` |
| SVM                        |    ✅    |    ✅    |               |               | `dasf.ml.svm`           |
| Boosted Trees              |    ✅    |    ✅    |       ✅       |       ✅       | `dasf.ml.xgboost`       |
| Nearest Neighbors          |    ✅    |    ✅    |               |               | `dasf.ml.neighbors`     |

## 🤝 Contributing

We welcome contributions from the community! If you'd like to contribute to DASF, please read our [**Contributing Guidelines**](CONTRIBUTING.md) for more information.

## 📄 License

This project is licensed under the permissive MIT License. This means you are free to:

- ✅ **Use:** Freely use the software in your own projects, whether personal, commercial, or open source.
- ✅ **Modify:** Adapt the code to your specific needs.
- ✅ **Distribute:** Share the original or your modified versions with others.

All we ask is that you include the original copyright and license notice in any copy of the software/source. For more details, see the [LICENSE](LICENSE) file.

## 📜 Citation

If you use DASF in your research, please cite our paper:

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

## 👥 Authors

- Julio Faracco
- João Seródio
- Otavio Napoli
- Edson Borin
