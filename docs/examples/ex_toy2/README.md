
ex_toy2

This example will attempt to train a network using 200x200 pixel
images as input and the angle of a line drawn in the image as
the label. In other words, it tries to train an AI to pick out
the phi angle of a straight line in a picture. 

Images can be created with the mkphiimages program. You'll need
to create both a training and validation set. Here are the steps:

> scons
> mkdir TRAIN VALIDATION
> cd TRAIN
> ../mkphiimages
> cd ../VALIDATION
> ../mkphiimages
> cd ..

The format of the images.raw.gz file is a simple one invented for
this exercise. A single file contains 50k "images" with each
40,000 bytes representing a single 8bit grayscale image. If you wish
to see a few examples of the images, run the raw2png.py script.
It will open the images.raw.gz file in the current directory and
make PNG files out of the first 25 images which you can then open
with any viewer (e.g. eog on Linux).


Run the train.py script to try training the network.

Be warned that the default, somewhat complex network will takes hours
train if using a GPU and days if using a CPU (and it doesn't converge).

The challenge here is to come up with a network configuration in
the train.py script that converges,

