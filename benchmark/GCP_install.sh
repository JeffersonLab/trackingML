#!/bin/bash
#
# Install base software needed to run keras + tensorflow with
# GPU support on a VM created on the Google Cloud Platform.
#
# This assumes the VM was created with CentOS 7 as the OS.
# The user should be able to run sudo without a password.
#

# Install CUDA
sudo yum install -y wget
wget https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-repo-rhel7-10.1.105-1.x86_64.rpm
sudo rpm -i cuda-repo-*.rpm
sudo yum install -y cuda

# Install python3 and C++ compiler
sudo yum install python3 gcc-g++

# Create virtual python environment and install needed packages into it
pyvenv venv
source venv/bin/activate
sudo pip install keras tensorflow-gpu pandas matplotlib imutils pillow scikit-learn opencv-python

