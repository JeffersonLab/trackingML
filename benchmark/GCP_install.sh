#!/bin/bash
#
# Install base software needed to run keras + tensorflow with
# GPU support on a VM created on the Google Cloud Platform.
#
# This assumes the VM was created with CentOS 7 as the OS.
# The user should be able to run sudo without a password.
#
# git clone https://github.com/JeffersonLab/trackingML
#

# Install CUDA (plus others)
curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo apt -y install ./cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo apt-key adv --fetch-keys  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda-runtime-10-0 python3-venv scons zlib1g-dev sysbench stress-ng

# Download and install cuDNN library
curl -O https://www.jlab.org/12gev_phys/ML/CUDA/libcudnn7_7.5.0.56-1+cuda10.0_amd64.deb
sudo apt -y install ./libcudnn7_7.5.0.56-1+cuda10.0_amd64.deb

# Create virtual python environment and install needed packages into it
python3 -m venv venv
source venv/bin/activate
pip install keras tensorflow-gpu pandas matplotlib imutils pillow scikit-learn opencv-python



# Install CUDA (also install python3 and c++ compiler)
#sudo yum install -y wget
#wget https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-repo-rhel7-10.1.105-1.x86_64.rpm
#sudo rpm -i cuda-repo-*.rpm
#sudo yum install -y cuda scons

# Create virtual python environment and install needed packages into it
#python3.6 -m venv venv
#source venv/bin/activate
#pip install keras tensorflow-gpu pandas matplotlib imutils pillow scikit-learn opencv-python

# Download and install cuDNN library
#wget https://www.jlab.org/12gev_phys/ML/CUDA/libcudnn7-7.4.2.24-1.cuda10.0.x86_64.rpm
#sudo rpm -i libcudnn7-7.4.2.24-1.cuda10.0.x86_64.rpm
