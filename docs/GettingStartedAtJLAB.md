# Getting Started with ML at JLab

Here are some instructions for setting up keras+tensorflow on
a JLab CUE computer. There are some toy examples in the docs/examples
directory that can be used to test the sytem.

This system uses Python to build and train the network using the
keras package which itself uses tensorflow as a backend. Tensorflow
can optionally support GPUs (see instructions below).

These instructions also rely on python's virtualenv
to set up a dedicated python environment where you can install the
needed packages without needing sysadmin privileges.


The basic steps are:
1. create a python3 virtual environment (venv)
2. update pip in the venv
3. use pip from venv to install all required python packages
4. (optional) download cudNN and install precompiled Linux binaries
   in venv by hand


Here are the detailed commands:

```
/apps/bin/python3 -m venv venv
source venv/bin/activate.csh
pip install --upgrade pip
```
(If using bash, use `source venv/bin/activate` in the 2nd line instead.)

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
	opencv-python \
	pypng
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
	opencv-python \
	pypng
```

At this point you should download the cudnn binary package from
Nvidia. That requires an active account (which is free) and a
little URL manipulation to get the correct version. To make it
easier, I've placed a copy on the CUE as seen in the first
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

```
>python
Python 3.4.3 (default, Apr 10 2015, 10:35:21) 
[GCC 4.8.3 20140911 (Red Hat 4.8.3-9)] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import keras
Using TensorFlow backend.
>>> <ctl-D>
```
	
At this point you should be able to run a python script using keras.


