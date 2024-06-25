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

rapidsai_version = USERARG.get('rapids-version', '23.08')
ubuntu_version = USERARG.get('ubuntu-version', '20.04')
python_version = USERARG.get('python-version', '3.10')
repo_branch = USERARG.get('repo-branch', 'main')

if python_version:
    python_version = f"-py{python_version}"

gpu_image_devel = f"rapidsai/base:{rapidsai_version}-cuda{cuda_version}{python_version}"

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

packages_list = ["git", "graphviz", "gcc", "python3-dev", "g++", "openssh-client", "wget"]

if is_devel:
    # Install NVIDIA NSight packages
    package_list += ["nsight-compute"]

pip_package_install = f"pip3 install -U git+https://github.com/discovery-unicamp/dasf-core.git@{repo_branch}"

if device_target.lower() == "cpu":
    packages_list.extend(["python3-pip"])

    Stage0 += apt_get(ospackages=packages_list)

    pip_package_install = ("%s jupyterlab" % pip_package_install)

elif device_target.lower() == "gpu":
    if is_devel:
        Stage0 += shell(commands=["conda install -n base -c rapidsai git graphviz gcc cxx-compiler openssh wget kvikio -y"])
    else:
        Stage0 += apt_get(keys=apt_keys, ospackages=packages_list)
        Stage0 += apt_get(ospackages=packages_list)

    Stage0 += shell(commands=["pip install --no-dependencies cupy_xarray==0.1.3"]) # this avoids CuPy being installed twice and taking too long (installation process doesn't find CuPy because its named cupy_cuda12x)

    if is_devel:
        pip_package_install = ("%s %s" % (pip_package_install, "git+https://github.com/cupy/cupy.git"))
    else:
        pip_package_install = ("%s %s" % (pip_package_install, "cupy-cuda12x==13.2.0"))


Stage0 += shell(commands=["pip3 install pip --upgrade"])

Stage0 += shell(commands=[pip_package_install])

if is_devel:
    Stage0 += shell(commands=["wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh -b && cp /root/miniconda3/bin/conda /usr/bin/conda"])

    Stage0 += shell(commands=["conda create -n kvikio -c rapidsai"])

    Stage0 += shell(commands=["conda install -n kvikio -c rapidsai kvikio -y"])

Stage0 += workdir(directory='/dasf')

if is_devel:
    Stage0 += label(metadata={'dasf-devel': 'latest'})
else:
    Stage0 += label(metadata={'dasf': 'latest'})
