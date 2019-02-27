
ex_toy3

This example uses input images where each pixel represents a single
wire in a toy detector model. The detector is made up of 6 sections
with 6 wire layers each. Each layer is 100 wires. Thus, the images 
are 36 x 100 pixels. The geometry model has gaps between chambers and
between the vertex and the first chamber. This means the tracks
appear as discontinuous line segments. Details of the exact geometry
can be found in the comments at the top of mkimages.cc.


            <-6->  <-6->  <-6->  <-6->  <-6->  <-6->
            -----  -----  -----  -----  -----  -----
            |   |  |   |  |   |  |   |  |   |  |   | 
vertex      |   |  |   |  |   |  |   |  |   |  |   | 1
  o         |   |  |   |  |   |  |   |  |   |  |   | 0 
            |   |  |   |  |   |  |   |  |   |  |   | 0
            |   |  |   |  |   |  |   |  |   |  |   | 
            -----  -----  -----  -----  -----  -----


Each image will have a single track from the vertex and at an angle
between -10 degrees and +10 degrees from horizontal. The angle limits
were chosen such that last layer should always have a hit. This
means most wires in the first layer will never have a hit. The angle
is the only label to train to and is written to the track_parms.csv
file in degrees.

The network here uses a classification technique to fill 200 outputs
corresponding to -10 to +10 degrees in 0.1 degree increments. A custom
loss function is used to take a weighted average of the 200 outputs
as the angle. It does the same thing for the label (which is just an
array with 199 zeros and 1 one). This limits the resolution to the
bin size/sqrt(12) or 0.029. 

Images can be created with the mkphiimages program. You'll need
to create both a training and validation set. You may optionally want
to create a test set. Here are the steps:

scons

mkdir TRAIN VALIDATION TEST

cd TRAIN

../mkphiimages

cd ../VALIDATION

../mkphiimages

cd ../TEST

../mkphiimages

cd ..

The format of the images.raw.gz file is the same as in ex_toy2, but 
with a different image size. If you wish to see a few examples of the
images, run the raw2png.py script in one of the directories. It will
open the images.raw.gz file in the current directory and make PNG files
out of the first 25 images which you can then open with any viewer
(e.g. eog on Linux).

The train.py script can be used to train the network. It is setup to allow
using multiple GPUs. Modify the GPUS variable at the top of the file
to set how many it should use. Note that at this time, I have not 
tested this on a computer without any GPUs so it may not work correctly.

Once the network is trained, you can run the test.py script to run it
on the TEST data set. This will produce two ascii files used by the
plot_results.C and plot_outputs.C root macros. The first of these will
make a pair of plots showing the resolution achieved by the model. The
second will make a 2D plot of the first 100 tracks so you can see the
distribution of outputs.


