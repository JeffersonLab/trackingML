#!/bin/bash
#
# Run the bm01 benchmark
#
# This script should be run from within the bm01 directory
# *AFTER* sourcing the python virtual environment script
# venv/bin/activate. One should also make sure the GPUS
# variable at the top of the script is set to the number
# of GPUs available on the VM
#
# This will run the benchmark and copy all relevant info
# into the archive bm01_results.tgz
#

# Make directory to hold results
results_dir=bm01_results
mkdir $results_dir

# Build mkimages program and run it
scons
./mkimages

# Capture system info
cat /proc/cpuinfo > $results_dir/cpuinfo.out
free              > $results_dir/memory.out
nvidia-smi        > $results_dir/nvidia-smi.out
uname             > $results_dir/uname.out

# Run training
./train.py &> $results_dir/train.out

# Run testing
./test.py  &> $results_dir/test.out

# Move tensorboard logs and results images to results dir
mv logs *.png $results_dir

# Tar and gzip results dir
tar czf ${results_dir}.tgz $results_dir

