# Getting Started with ML at JLab

Here are some instructions for setting up keras+tensorflow on
a JLab CUE computer. There are some toy examples in the examples
directory that can be used to test the sytem.

This system uses Python to build and train the network using the
keras package which itself uses tensorflow as a backend. Tensorflow
can optionally support GPUs (see instructions below).

These instructions also rely on python's easy_install and virtualenv
to set up a dedicate python environment where you can install the
needed packages without needing sysadmin privileges.


The basic steps are:
1. install pip in user directory using easy_install
2. use pip to install virtualenv
3. use virtualenv to create dedicated python environment (venv)
4. use pip from venv to install all required python packages
5. (optional) download cudNN and install precompiled Linux binaries
   in venv by hand


Here are the detailed commands:

```
easy_install --user pip
~/.local/bin/pip install virtualenv
~/.local/bin/virtualenv venv
source venv/bin/activate.csh
```

At this point the instructions are slightly different if you want
to setup for GPU support or not. Choose the correct section below
then continue to the ALL CONFIGURATIONS section.


INSTALL WITHOUT GPU SUPPORT
==============================
These are instructions for installing the necessary packages to 
run on an ifarm computer. It will not take advantage of any GPU
that may be present.

```
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org \
	tensorflow \
	keras \
	pandas \
	matplotlib \
	imutils \
	pillow \
	scikit-learn \
	opencv-python
```

You can now jump to the ALL CONFIGURATIONS section below.



INSTALL WITH GPU SUPPORT
==============================
This will install packages needed to utilze an Nvidia GPU device
on the system. The only interactive system I am aware of at the
moment is hpci12k01 which has an Nvidia K20.


Here are the detailed commands:

```
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org \
	tensorflow-gpu \
	keras \
	pandas \
	matplotlib \
	imutils \
	pillow \
	scikit-learn \
	opencv-python
```

At this point you should download the cudnn binary package from
Nvidia. That requires an active account (which is free) and a
little URL manipulation to get the correct version. To make it
easier, I've place a copy on the CUE as seen in the first
instruction below. The second just moves the libraries into a
directory where python will find them when running tensorflow.

```
tar xzf /group/12gev_phys/ml/downloads/cudnn-9.0-linux-x64-v7.3.1.20.tgz
mv cuda/lib64/libcudnn* venv/lib/python2.7/site-packages/tensorflow
rm -rf cuda
```

The last thing you need to do is pull cuda 9.0 into your
environment. This is only available on computers with GPUs
(e.g. hpci12k01).
 
```
module load cuda9.0
```

ALL CONFIGURATIONS
==================================
At this point you should have a python virtualenv setup and ready
use. If you installed it in a different shell you'll need to source
the venv/bin/activate.(c)sh script again to setup the environment
(but you won't have to re-run pip). If things are right, your prompt
will start with "[venv]".

Test that the installation has installed keras and tensorflow
correctly using an interactive python session:

 > python \
 Python 2.7.5 (default, Nov 20 2015, 02:00:19) \
 [GCC 4.8.5 20150623 (Red Hat 4.8.5-4)] on linux2 \
 Type "help", "copyright", "credits" or "license" for more information. \
 >>> import keras \
 Using TensorFlow backend. \
 >>> <ctl-D>

At this point you should be able to run a python script using keras.


