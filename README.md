## Fast Feature-Less Quaternion-based Particle Swarm Optimization for Object Pose Estimation From RGB-D Images


The code in this repository is related to the paper:
http://www.bmva.org/bmvc/2016/papers/paper113/

This is an early GPU implementation, includes object segmentation from https://arxiv.org/abs/1605.03746

- - - -
Hardware:

Requires a CUDA enabled GPU (CUDA 7.0)

Sofware:

-OpenCV
-gcc with support for c++11

Compile:

`make`

Run (on the first 15 detected object clusters):

`for N in {0..15}
do
  ./main object_model rgb_image depth_image $N
done`

    
