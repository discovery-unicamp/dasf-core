"""
HPC Container Maker for DASF core
"""


def str2bool(string):
    return string.lower() in ['true', '1', 't', 'y', 'yes']


device_target = USERARG.get('device-target', 'gpu')
is_devel = str2bool(USERARG.get('devel', 'False'))

if is_devel:
    # Devel packages always use the latest CUDA version
    cuda_version = "11.5"
else:
    cuda_version = USERARG.get('cuda-version', '11.2')

rapidsai_version = USERARG.get('rapids-version', '23.02')
ubuntu_version = USERARG.get('ubuntu-version', '20.04')
python_version = USERARG.get('python-version', '3.10')

if python_version:
    python_version = f"-py{python_version}"

gpu_image_devel = f"rapidsai/rapidsai-core-dev:{rapidsai_version}-cuda{cuda_version}-devel-ubuntu{ubuntu_version}{python_version}"

# GPU image needs to be fixed due to dependency matrix
gpu_image = "nvcr.io/nvidia/pytorch:23.06-py3"

cpu_image = f"ubuntu:{ubuntu_version}"

if device_target.lower() == "cpu":
    Stage0 += baseimage(image=cpu_image)
elif device_target.lower() == "gpu":
    # XXX: There is no way to use old GPUs with 11.5 CUDA.
    # if is_devel:
    #     Stage0 += baseimage(image=gpu_image_devel)
    # else:
    Stage0 += baseimage(image=gpu_image)
else:
    raise RuntimeError(f"Device target {device_target} is not known.")

ubuntu_unified_version = "".join(ubuntu_version.split("."))

apt_keys = [
    f"https://developer.download.nvidia.com/compute/cuda/repos/ubuntu{ubuntu_unified_version}/x86_64/3bf863cc.pub",
    f"https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu{ubuntu_unified_version}/x86_64/7fa2af80.pub"
]

packages_list = ["git", "graphviz", "gcc", "python3-dev", "g++", "openssh-client"]

if is_devel:
    # Install NVIDIA NSight packages
    package_list += ["nsight-compute"]

pip_package_install = "pip3 install --extra-index-url https://test.pypi.org/simple/ XPySom-dask git+https://github.com/discovery-unicamp/dasf-core.git"

if device_target.lower() == "cpu":
    packages_list.extend(["python3-pip"])

    Stage0 += apt_get(ospackages=packages_list)

    pip_package_install = ("%s jupyterlab" % pip_package_install)

elif device_target.lower() == "gpu":
    Stage0 += apt_get(keys=apt_keys, ospackages=packages_list)

    pip_package_install = ("%s cupy_xarray" % pip_package_install)

    if is_devel:
        pip_package_install = ("%s %s" % (pip_package_install, "git+https://github.com/cupy/cupy.git"))
    else:
        pip_package_install = ("%s %s" % (pip_package_install, "cupy==13.0.0b1"))
        Stage0 += shell(commands=["rm -r /usr/local/lib/python3.10/dist-packages/cupy_cuda12x-12.0.0b3.dist-info"]) # not the best solution but it works


Stage0 += shell(commands=["pip3 install pip --upgrade"])

Stage0 += shell(commands=[pip_package_install])

# TODO: fix numpy issue with version 1.24 and other fixed reqs
Stage0 += shell(commands=["pip install \"numpy<1.24\" bokeh==2.4.3 \"protobuf<=3.20.1\" \"charset-normalizer<3.0\" \"tornado<6.2\""])

Stage0 += workdir(directory='/dasf')

if is_devel:
    Stage0 += label(metadata={'dasf-devel': 'latest'})
else:
    Stage0 += label(metadata={'dasf': 'latest'})
