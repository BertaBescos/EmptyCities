
# EmptyCities

[[Project]](https://bertabescos.github.io/EmptyCities/)   [[Paper]](https://arxiv.org/pdf/1809.10239.pdf)

Torch implementation for learning a mapping from input images that contain dynamic objects in a city environment, such as vehicles and pedestrians, to output images which are static. In the example below, the top images are fed one by one into our model. The bottom row are the obtained results:

<img src="imgs/DynamicObjectsInvariantSpace.png" width="900px"/>

Empty Cities: Image Inpainting for a Dynamic-Object-Invariant Space  
[Berta Bescos](https://bertabescos.github.io/), [Jose Neira](http://webdiis.unizar.es/~neira/), [Roland Siegwart](http://www.asl.ethz.ch/the-lab/people/person-detail.html?persid=29981), [Cesar Cadena](http://n.ethz.ch/~cesarc/)  

## Setup

### Prerequisites
- Linux or OSX
- NVIDIA GPU + CUDA CuDNN (CPU mode and CUDA without CuDNN may work with minimal modification, but untested). We used CUDA 8.0 and CuDNN 5. Other versions might work too with minimal modification, but untested.

### Getting Started
- Install torch and dependencies from https://github.com/torch/distro
- Install torch packages `nngraph` and `display`  (optional)
```bash
luarocks install nngraph
luarocks install https://raw.githubusercontent.com/szym/display/master/display-scm-0.rockspec
```
- Clone this repo:
```bash
git clone git@github.com:BertaBescos/EmptyCities.git
cd EmptyCities
```
You might need the GitHub package https://git-lfs.github.com/ to clone it.

### Models
Pre-trained models are found within the folder `/checkpoints`. 
- `mGAN`: generative inpainting model.
- `SemSeg`: semantic segmentation model. The original model from [ERFNet](https://github.com/Eromera/erfnet) has been finetuned with our data.

## Inference

- You can fastly test our model with one image.
```bash
input=/path/to/input/image/ qlua test.lua
```
We provide some images in `/imgs/test` you can run our model on. For example:
```bash
input=imgs/test/0.png qlua test.lua
```
These images are from the [Cityscapes](https://www.cityscapes-dataset.com/) and the [SVS](https://svsdataset.github.io/) datasets.
- You can also store the inpainted result and the binary mask that has been used.
```bash
input=/path/to/input/image/ output=/path/to/output/image/ th test.lua
```
In the following example the binary mask is stored in `/imgs/test/0_output_mask.png`:
```bash
input=imgs/test/0.png output=imgs/test/0_output.png th test.lua
```
- If the stored mask is not accurate enough, you can provide yourself a better one:
```bash
input=imgs/test/0.png mask=imgs/test/0_mask.png output=imgs/test/0_output.png th test.lua
```

## Test

- If you want to work with more than one image, we encourage you to keep your data in a folder of your choice `/path/to/data/` with three subfolders `train`, `test` and `val`. The following command will run our model within all the images inside the folder `test` and keep the results in `./results/mGAN`. Images within the folder `test` should be RGB images of any size.
```bash
DATA_ROOT=/path/to/data/ th test.lua
```
For example:
```bash
DATA_ROOT=/path/to/EmptyCities/imgs/ th test.lua
```
- If you prefer to feed the dynamic/static binary masks, you should concatenate it to the RGB image. We provide a python script for this.
```bash
DATA_ROOT=/path/to/data/ mask=1 th test.lua
```
- Finally, if the groundtruth images are available you should concatenate them too (RGB | GT | Mask).
```bash
DATA_ROOT=/path/to/data/ mask=1 target=1 th test.lua
```
The test results will be saved to an html file here: `./results/mGAN/latest_net_G_val/index.html`.

## Datasets
Our synthetic dataset has been generated with [CARLA 0.8.2](https://drive.google.com/file/d/1ZtVt1AqdyGxgyTm69nzuwrOYoPUn_Dsm/view). Within our folder `/scripts/CARLA` we provide some python and bash scripts to generate the paired images. The files `/scripts/CARLA/client_example_read.py` and `/scripts/CARLA/client_example_write.py` shoule be run instead of the `/PythonClient/client_example.py` provided in CARLA_0.8.2. Images with different weather conditions should be generated. 
- The following bash scripts store the images with dynamic objects in `path/to/dataset/`, as well as the control inputs of the driving car and the trajectory that has been followed in `Control.txt` and `Trajectory.txt` respectively. CARLA provides two different towns setups: Town01 has been used for generating the training and validations sets, and Town02 for the testing set. 
```bash
bash scripts/CARLA/CreateDynamicDatasetTown01.sh path/to/my/folder/
```
```bash
bash scripts/CARLA/CreateDynamicDatasetTown02.sh path/to/my/folder/
```
- These scripts read the previous stored `Control.txt` files and try to replicate the same trajectories in the same scenarios with no dynamic objects. The followed trajectory and the one in `Trajectory.txt` are compared to check that the vehicle position is kept the same.
```bash
bash scripts/CARLA/CreateStaticDatasetTown01.sh path/to/my/folder/
```
```bash
bash scripts/CARLA/CreateStaticDatasetTown02.sh path/to/my/folder/
```
- Once all these images are generated, the dynamic images should be stored together in a folder with the subfolders `/train/`, `/test/` and `/val/`. The same for the static images and for the dynamic/static binary masks. We provide the following bash script:
```bash
bash scripts/CARLA/setup.sh path/to/my/folder/ path/to/output/
```
For better adaptation to real world images, we have used the [Cityscapes dataset](https://www.cityscapes-dataset.com/).

## Setup Training/Validation/Test data
### Generating Pairs
- We provide a python script to generate the CARLA training, validation and test data in the needed format. The following script concatenates the images {A,B,C} where A is the image with dynamic objects, B is the groundtruth static image, and C is the dynamic/static binary mask. 
```bash
python scripts/setup/combineCARLA.py --fold_A /path/to/output/A/ --fold_B /path/to/output/B/ --fold_C /path/to/output/C/ --fold_ABC /path/to/output/ABC/
```
- Also, to format the Cityscapes images we provide the following python script. You should run it for the `/val` folder too.
```bash
bash scripts/setup/combineCITYSCAPES.py --fold_A /path/to/CITYSCAPES/leftImg8bit_trainvaltest/leftImg8bit/train --fold_B /path/to/CITYSCAPES/gtFine_trainvaltest/gtFine/train --fold_AB /path/to/output/train
```
**Further notes**: Also, you can download a small dataset from [here](https://drive.google.com/open?id=1XkgElMx4kgyhSNWgoarhBvKKDLuYkJ2o). Note that this dataset is valid for testing, but it contains very few images for training.

## Citation
If you use this code for your research, please cite our paper Empty Cities: Image Inpainting for a Dynamic-Object-Invariant Space</a>:

```
@article{emptycities2018,
  title={Empty Cities: Image Inpainting for a Dynamic-Object-Invariant Space},
  author={Bescos, Berta and Neira, José and Siegwart, Roland and Cadena, Cesar},
  journal={arXiv},
  year={2018}
}
```

## Acknowledgments
Our code is heavily inspired by [pix2pix](https://github.com/phillipi/pix2pix).

# EmptyCities
