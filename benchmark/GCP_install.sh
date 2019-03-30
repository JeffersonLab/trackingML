#!/bin/bash
#
# Install base software needed to run keras + tensorflow with
# GPU support on a VM created on the Google Cloud Platform.
#
# This assumes the VM was created with CentOS 7 as the OS.
# The user should be able to run sudo without a password.
#

sudo install python3 gcc-g++
sudo pip3 install keras tensorflow-gpu pandas matplotlib imutils pillow scikit-learn opencv-python

