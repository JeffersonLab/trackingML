#!/bin/bash
#
# Run the bm03 benchmark
#
# This script should be run from within the bm03 directory
# *AFTER* sourcing the python virtual environment script
# venv/bin/activate. One should also make sure the GPUS
# variable at the top of the script is set to the number
# of GPUs available on the VM
#
# This will run the benchmark and copy all relevant info
# into the archive bm03_{hostname}.tgz
#

GPUS=1

# Make directory to hold results
results_dir=bm03_`hostname`
mkdir $results_dir

# Download input data and unpack it
curl -O https://www.jlab.org/12gev_phys/ML/data_sets/bm03_dataset.tgz
tar xzf bm03_dataset.tgz
mkdir ./training_set
mv bm03_dataset/* ./training_set/

# Capture system info
cat /proc/cpuinfo > $results_dir/cpuinfo.out
free              > $results_dir/memory.out
nvidia-smi        > $results_dir/nvidia-smi.out
uname             > $results_dir/uname.out
sysbench cpu run  > $results_dir/sysbench.out
sudo stress-ng --cpu 1 --cpu-method all --perf -t 60 &> $results_dir/stress-ng.out

#-------------------------------------------------------------------
# Capture GPU stats while training
nvidia-smi dmon -o DT -s puct -f $results_dir/nvidia-smi-train.out &

# Run training
./gan_train.py -m test &> $results_dir/train.out

# Kill nvidia-smi monitoring
pkill -9 nvidia-smi

#-------------------------------------------------------------------
# Capture GPU stats while testing
nvidia-smi dmon -o DT -s puct -f $results_dir/nvidia-smi-test.out &

# Run testing
./predict_batch.py  &> $results_dir/test.out

# Kill nvidia-smi monitoring
pkill -9 nvidia-smi

#-------------------------------------------------------------------
# Move tensorboard logs and results images to results dir
mv logs *.png *.dat $results_dir

# Tar and gzip results dir
tar czf ${results_dir}.tgz $results_dir

