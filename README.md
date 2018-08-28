<img src="imgs/DynamicObjectsInvariantSpace.png" width="900px"/>

Empty Cities: Image Inpainting for a Dynamic Objects Invariant Space  
[Berta Bescos](https://bertabescos.github.io), [Jose Neira](http://webdiis.unizar.es/~neira/), [Roland Siegwart], [Cesar Cadena](http://n.ethz.ch/~cesarc/)  
CoRL, 2018.

## Abstract
<div style="text-align: justify">
In this paper we present an end-to-end deep learning framework to turn images that show dynamic content, such as vehicles or pedestrians, into realistic static frames. This objective encounters two main challenges: detecting the dynamic objects, and inpainting the static occluded background. 
The second challenge is approached with a conditional generative adversarial model that, taking as input the original dynamic image and the computed dynamic/static binary mask, is capable of generating the final static image. The former challenge is addressed by the use of a convolutional network that learns a multi-class semantic segmentation of the image. The objective of this network is producing an accurate segmentation and helping the previous generative model to output a realistic static image. These generated images can be used for applications such as virtual reality or vision-based robot localization purposes. To validate our approach, we show both qualitative and quantitative comparisons with other methods by removing the dynamic objects and hallucinating (inpainting) the static structure behind them. </div>

## Try Our Code
- Torch implementation [here](https://github.com/BertaBescos/EmptyCities).  
- We are currently working on a PyTorch implementation.

## Paper


## Video


## Datasets
<div class="align-justify">
In our GitHub page we have some scripts available to generate the dataset with [CARLA](http://carla.org/). Also, you can download a small dataset from [here](https://drive.google.com/open?id=1XkgElMx4kgyhSNWgoarhBvKKDLuYkJ2o). Note that this dataset is valid for testing, but it contains very few images for training.
</div>

## Acknowledgments
