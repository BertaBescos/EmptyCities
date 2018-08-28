<img src="imgs/DynamicObjectsInvariantSpace.png" width="900px"/>

Empty Cities: Image Inpainting for a Dynamic Objects Invariant Space  
[Berta Bescos](https://bertabescos.github.io), [Cesar Cadena](http://n.ethz.ch/~cesarc/), [Jose Neira](http://webdiis.unizar.es/~neira/), [Roland Siegwart]   
CoRL, 2018.

<center>## Abstract</center>

In this paper we present an end-to-end deep learning framework to turn images that show dynamic content, such as vehicles or pedestrians, into realistic static frames. This objective encounters two main challenges: detecting the dynamic objects, and inpainting the static occluded background. 
The second challenge is approached with a conditional generative adversarial model that, taking as input the original dynamic image and the computed dynamic/static binary mask, is capable of generating the final static image. The former challenge is addressed by the use of a convolutional network that learns a multi-class semantic segmentation of the image. The objective of this network is producing an accurate segmentation and helping the previous generative model to output a realistic static image. These generated images can be used for applications such as virtual reality or vision-based robot localization purposes. To validate our approach, we show both qualitative and quantitative comparisons with other methods by removing the dynamic objects and hallucinating (inpainting) the static structure behind them.

## Models
Download the pre-trained models with the following script. You need to rename the model (*e.g.* `mGAN` to `/checkpoints/mGAN/latest_net_G.t7`) after the download has finished.
```bash
bash ./models/download_model.sh mGAN
```
- `mGAN`: trained only on synthetic data coming from [CARLA](http://carla.org/).
- `mGAN_DA`: trained on synthetic data coming from CARLA with data augmentation.
- `mGAN_RD`: trained on synthectic data coming from CARLA and real data from the Cityscapes dataset. Real data is added during training with a probability of 0.5 from epoch 50 on.
- `SemSeg`: semantic segmentation model. The original model from [ERFNet](https://github.com/Eromera/erfnet) has been finetuned with our data.
## Test
- Test one image
```bash
DATA_ROOT=/path/to/data/ name=expt_name phase=val th test.lua
```




- Download the dataset (e.g. [CMP Facades](http://cmp.felk.cvut.cz/~tylecr1/facade/)):
```bash
bash ./datasets/download_dataset.sh facades
```
- Train the model
```bash
DATA_ROOT=/home/bescosb/CARLA_0.8.2/dataset/data/RGBMask name=mGAN th train.lua
```
- If we want to add real data with a probability of 0.5 from epoch 50 on:
```bash
DATA_ROOT=/home/bescosb/CARLA_0.8.2/dataset/data/RGBMask NSYNTH_DATA_ROOT=/home/bescosb/CARLA_0.8.2/dataset/data/CITYSCAPES/Mask name=mGAN add_real_data=1 epoch_synth=50 pNonSynth=0.5 th train.lua
```
- If we do not have the masks:
```bash
DATA_ROOT=/home/bescosb/CARLA_0.8.2/dataset/data/RGBFullMask NSYNTH_DATA_ROOT=/home/bescosb/CARLA_0.8.2/dataset/data/CITYSCAPES/FullMask name=SS add_real_data=1 epoch_synth=50 pNonSynth=0.5 th train_ss.lua
```
- (CPU only) The same training command without using a GPU or CUDNN. Setting the environment variables ```gpu=0 cudnn=0``` forces CPU only
```bash
DATA_ROOT=./datasets/facades name=facades_generation which_direction=BtoA gpu=0 cudnn=0 batchSize=10 save_epoch_freq=5 th train.lua
```
- (Optionally) start the display server to view results as the model trains. ( See [Display UI](#display-ui) for more details):
```bash
th -ldisplay.start 8000 0.0.0.0
```

- Finally, test the model:
```bash
DATA_ROOT=/home/bescosb/CARLA_0.8.2/dataset/data/RGBMask name=mGAN which_epoch=50 phase=test th test.lua
```
- If we do not have the ground-truth:
```bash
DATA_ROOT=/home/bescosb/CARLA_0.8.2/dataset/data/CITYSCAPES/Mask/val name=mGAN which_epoch=50 phase=val th inference.lua
```
- If we do not have the masks:
```bash
DATA_ROOT=/home/bescosb/CARLA_0.8.2/dataset/data/CITYSCAPES/Mask/val name=mGAN which_epoch=50 phase=val th inference_ss.lua
```
The test results will be saved to an html file here: `./results/facades_generation/latest_net_G_val/index.html`.

## Train
```bash
DATA_ROOT=/path/to/data/ name=expt_name th train.lua
```

Models are saved to `./checkpoints/expt_name` (can be changed by passing `checkpoint_dir=your_dir` in train.lua).

See `opt` in train.lua for additional training options.

## Test
```bash
DATA_ROOT=/path/to/data/ name=expt_name phase=val th test.lua
```

This will run the model named `expt_name` on all images in `/path/to/data/val`.

Result images, and a webpage to view them, are saved to `./results/expt_name` (can be changed by passing `results_dir=your_dir` in test.lua).

See `opt` in test.lua for additional testing options.


## Datasets
Download the datasets using the following script. Some of the datasets are collected by other researchers. Please cite their papers if you use the data.
```bash
bash ./datasets/download_dataset.sh dataset_name
```
- `facades`: 400 images from [CMP Facades dataset](http://cmp.felk.cvut.cz/~tylecr1/facade/). [[Citation](datasets/bibtex/facades.tex)]
- `cityscapes`: 2975 images from the [Cityscapes training set](https://www.cityscapes-dataset.com/).  [[Citation](datasets/bibtex/cityscapes.tex)]
- `maps`: 1096 training images scraped from Google Maps
- `edges2shoes`: 50k training images from [UT Zappos50K dataset](http://vision.cs.utexas.edu/projects/finegrained/utzap50k/). Edges are computed by [HED](https://github.com/s9xie/hed) edge detector + post-processing.
[[Citation](datasets/bibtex/shoes.tex)]
- `edges2handbags`: 137K Amazon Handbag images from [iGAN project](https://github.com/junyanz/iGAN). Edges are computed by [HED](https://github.com/s9xie/hed) edge detector + post-processing. [[Citation](datasets/bibtex/handbags.tex)]

## Models
Download the pre-trained models with the following script. You need to rename the model (e.g. `facades_label2image` to `/checkpoints/facades/latest_net_G.t7`) after the download has finished.
```bash
bash ./models/download_model.sh model_name
```
- `facades_label2image` (label -> facade): trained on the CMP Facades dataset.
- `cityscapes_label2image` (label -> street scene): trained on the Cityscapes dataset.
- `cityscapes_image2label` (street scene -> label): trained on the Cityscapes dataset.
- `edges2shoes` (edge -> photo): trained on UT Zappos50K dataset.
- `edges2handbags` (edge -> photo): trained on Amazon handbags images.
- `day2night` (daytime scene -> nighttime scene): trained on around 100 [webcams](http://transattr.cs.brown.edu/).

## Setup Training and Test data
### Generating Pairs
We provide a python script to generate training data in the form of pairs of images {A,B}, where A and B are two different depicitions of the same underlying scene. For example, these might be pairs {label map, photo} or {bw image, color image}. Then we can learn to translate A to B or B to A:

Create folder `/path/to/data` with subfolders `A` and `B`. `A` and `B` should each have their own subfolders `train`, `val`, `test`, etc. In `/path/to/data/A/train`, put training images in style A. In `/path/to/data/B/train`, put the corresponding images in style B. Repeat same for other data splits (`val`, `test`, etc).

Corresponding images in a pair {A,B} must be the same size and have the same filename, e.g. `/path/to/data/A/train/1.jpg` is considered to correspond to `/path/to/data/B/train/1.jpg`.

Once the data is formatted this way, call:
```bash
python scripts/combine_A_and_B.py --fold_A /path/to/data/A --fold_B /path/to/data/B --fold_AB /path/to/data
```

This will combine each pair of images (A,B) into a single image file, ready for training.

### Evaluating Labels2Photos on Cityscapes
We provide scripts for running the evaluation of the Labels2Photos task on the Cityscapes validation set. We assume that you have installed `caffe` (and `pycaffe`) in your system. If not, see the [official website](http://caffe.berkeleyvision.org/installation.html) for installation instructions. Once `caffe` is successfully installed, download the pre-trained FCN-8s semantic segmentation model (512MB) by running
```bash
bash ./scripts/eval_cityscapes/download_fcn8s.sh
```
Then make sure `./scripts/eval_cityscapes/` is in your system's python path. If not, run the following command to add it
```bash
export PYTHONPATH=${PYTHONPATH}:./scripts/eval_cityscapes/
```
Now you can run the following command to evaluate your predictions:
```bash
python ./scripts/eval_cityscapes/evaluate.py --cityscapes_dir /path/to/original/cityscapes/dataset/ --result_dir /path/to/your/predictions/ --output_dir /path/to/output/directory/
```
By default, images in your prediction result directory have the same naming convention as the Cityscapes dataset (e.g. `frankfurt_000001_038418_leftImg8bit.png`). The script will output a txt file under `--output_dir` containing the metric.

**Further notes**: The pre-trained model does not work well on Cityscapes in the original resolution (1024x2048) as it was trained on 256x256 images that are resized to 1024x2048. The purpose of the resizing was to 1) keep the label maps in the original high resolution untouched and 2) avoid the need of changing the standard FCN training code for Cityscapes. To get the *ground-truth* numbers in the paper, you need to resize the original Cityscapes images to 256x256 before running the evaluation code.

## Display UI
Optionally, for displaying images during training and test, use the [display package](https://github.com/szym/display).

- Install it with: `luarocks install https://raw.githubusercontent.com/szym/display/master/display-scm-0.rockspec`
- Then start the server with: `th -ldisplay.start`
- Open this URL in your browser: [http://localhost:8000](http://localhost:8000)

By default, the server listens on localhost. Pass `0.0.0.0` to allow external connections on any interface:
```bash
th -ldisplay.start 8000 0.0.0.0
```
Then open `http://(hostname):(port)/` in your browser to load the remote desktop.

L1 error is plotted to the display by default. Set the environment variable `display_plot` to a comma-seperated list of values `errL1`, `errG` and `errD` to visualize the L1, generator, and descriminator error respectively. For example, to plot only the generator and descriminator errors to the display instead of the default L1 error, set `display_plot="errG,errD"`.

## Citation
If you use this code for your research, please cite our paper Empty Cities: Image Inpainting for a Dynamic Objects Invariant Space</a>:

```
@article{emptycities2018,
  title={Empty Cities: Image Inpainting for a Dynamic Objects Invariant Space},
  author={Bescos, Berta and Neira, José and Siegwart, Roland and Cadena, Cesar},
  journal={CoRL},
  year={2019}
}
```

## Acknowledgments
Our code is heavily inspired by [pix2pix](https://github.com/phillipi/pix2pix), [DCGAN](https://github.com/soumith/dcgan.torch) and [Context-Encoder](https://github.com/pathak22/context-encoder).

# EmptyCities
