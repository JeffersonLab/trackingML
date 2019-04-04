#!/bin/bash
#
# Run the bm02 benchmark
#
# This script should be run from within the bm01 directory
# *AFTER* sourcing the python virtual environment script
# venv/bin/activate. One should also make sure the GPUS
# variable at the top of the script is set to the number
# of GPUs available on the VM
#
# This will run the benchmark and copy all relevant info
# into the archive bm01_{hostname}.tgz
#

# Make directory to hold results
results_dir=bm02_`hostname`
mkdir $results_dir

# Build mkimages program and run it
scons
./mkimages

# Capture system info
cat /proc/cpuinfo > $results_dir/cpuinfo.out
free              > $results_dir/memory.out
nvidia-smi        > $results_dir/nvidia-smi.out
uname             > $results_dir/uname.out
sysbench cpu run  > $results_dir/sysbench.out
sudo stress-ng --cpu 1 --cpu-method all --perf -t 60 > $results_dir/stress-ng.out

#-------------------------------------------------------------------
# Capture GPU stats while training
nvidia-smi dmon -o DT -s puct -f $results_dir/nvidia-smi-train.out &

# Run training
./train.py &> $results_dir/train.out

# Kill nvidia-smi monitoring
pkill -9 nvidia-smi

#-------------------------------------------------------------------
# Capture GPU stats while testing
nvidia-smi dmon -o DT -s puct -f $results_dir/nvidia-smi-test.out &

# Run testing
./test.py  &> $results_dir/test.out

# Kill nvidia-smi monitoring
pkill -9 nvidia-smi

#-------------------------------------------------------------------
# Move tensorboard logs and results images to results dir
mv logs *.png *.dat $results_dir

# Tar and gzip results dir
tar czf ${results_dir}.tgz $results_dir

