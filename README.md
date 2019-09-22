# Squeeze-and-Excitiation Networks

## Environments
keras version : 2.2.5 <br />
python version : 3.6.4

## Info
<br />
This repo is the implementation of Squeeze and Excitation Networks, which can be used in Keras framework. <br />
In fact, it is more like a block. Here is the architecture of SEblock:<br />

![ScreenShot](imgs/SE_arch.png)

There are two part of this block, squeeze and excitiation. <br />
<br />
**Squeeze** <br />
In this part, first it turn 2D feature maps into 1D feature maps. (Batch_size, H, W, C) -> (Batch_size, 1, 1, C) <br />
Then it feed the tensor into a Dense(fully-connected) layer which might has less filters/units number than input filters/units. <br />
This reduction of filters/units is for saving the computation power. <br />
It can output the feature maps are so call channel descriptor in the original article, which aggregating feature maps across their spatial dimension. <br />
It is similar to concept of embedding which can produce the effect like putting the global receptive field information in each channels. <br />
<br />
**Excitiation** <br />
In this part, it will go through a Dense(fully-connected) layer which has the same filters/units number as input filters/units. <br />
Then use sigmoid activation to produce the weights for each channel of original tensor. <br />
It uses these weights to learn the importance of dependencies of each channel. <br />
In the end, the weights multiply to its corresponding channel in original tensor to enhance/decrease the importance. <br />
<br />
<br />
This block can be implemented in almost any kinds of famous neural network, like ResNet, Inception...<br />
The author also mention that how to add the SEBlock into these famouse architectures. <br />

![ScreenShot](imgs/other_archs.png)

<br />

## Try it on small CNN

I had simply implemented this block into small CNN to deal with Cifar10 classification problem. <br />
It worked very well, and only increase the computation effort a little. <br />
Here is the result compare of regular CNN and regular CNN + SENet <br />
<br />
<br />
Regular CNN <br />

![ScreenShot](imgs/regular_cnn_on_cifar10.png)

SE CNN <br />

![ScreenShot](imgs/SE_cnn_on_cifar10.png)

<br />
<br />
Validation on testing data <br />
Regular CNN :  loss 0.717884089794159, accuracy 0.7539200000190734<br />
SE CNN : loss 0.6834373204803467, accruacy 0.7627999999809265

## Reference:
1.https://arxiv.org/abs/1709.01507<br />
2.https://github.com/titu1994/keras-squeeze-excite-network


