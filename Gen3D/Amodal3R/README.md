<h1 align="center">Amodal3R: Amodal 3D Reconstruction from Occluded 2D Images</h1>
  <p align="center">
    <a href="https://sm0kywu.github.io/CV/CV.html">Tianhao Wu</a>
    ·
    <a href="https://chuanxiaz.com/">Chuanxia Zheng</a>
    ·
    <a href="https://www.singaporetech.edu.sg/directory/faculty/frank-guan">Frank Guan</a>
    .
    <a href="https://www.robots.ox.ac.uk/~vedaldi/">Andrea Vedaldi</a>
    .
    <a href="https://personal.ntu.edu.sg/astjcham/index.html">Tat-Jen Cham</a>

  </p>
  <h3 align="center">ICCV 2025</h3>
  <h3 align="center"><a href="https://arxiv.org/abs/2503.13439">Paper</a> | <a href="https://sm0kywu.github.io/Amodal3R/">Project Page</a> | <a href="https://huggingface.co/Sm0kyWu/Amodal3R">Pretrain Weight</a> | <a href="https://huggingface.co/spaces/Sm0kyWu/Amodal3R">Demo</a></h3>
  <div align="center"></div>
</p>

### Demo Video
<div align="center">

![Demo Video](asset/teaser.gif)

</div>

### Setup
This code has been tested on Ubuntu 22.02 with torch 2.4.0 & CUDA 11.8. We sincerely thank [TRELLIS](https://github.com/Microsoft/TRELLIS) for providing the environment setup and follow exactly as their instruction in this work.

Create a new conda environment named `amodal3r` and install the dependencies:
```sh
. ./setup.sh --new-env --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
```
The detailed usage of `setup.sh` can be found by running `. ./setup.sh --help`.
```sh
Usage: setup.sh [OPTIONS]
Options:
    -h, --help              Display this help message
    --new-env               Create a new conda environment
    --basic                 Install basic dependencies
    --train                 Install training dependencies
    --xformers              Install xformers
    --flash-attn            Install flash-attn
    --diffoctreerast        Install diffoctreerast
    --vox2seq               Install vox2seq
    --spconv                Install spconv
    --mipgaussian           Install mip-splatting
    --kaolin                Install kaolin
    --nvdiffrast            Install nvdiffrast
    --demo                  Install all dependencies for demo
```

### Pretrained models
We have provided our pretrained weights of both sparse structure module and SLAT module on [HuggingFace](https://huggingface.co/Sm0kyWu/Amodal3R).

### Data Preprocessing

#### Training Data
We use three datasets for training: [ABO](https://amazon-berkeley-objects.s3.amazonaws.com/index.html), [3D-FUTURE](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future), and [HSSD](https://huggingface.co/datasets/hssd/hssd-models). To obtain the training data, please also refer to [TRELLIS](https://github.com/microsoft/TRELLIS/blob/main/DATASET.md). **Thanks to them for the amazing work!!!**. 

When the data is ready, combine them and put under `./dataset/abo_3dfuture_hssd`. If you want to train on a single dataset, feel free to modify the dataloader. For training, rendering images, Sparse Structure and SLAT are required.

### Training

To train you own model, you can start either on [our weights](https://huggingface.co/Sm0kyWu/Amodal3R) or [TRELLIS original weights](https://huggingface.co/microsoft/TRELLIS-image-large/tree/main). Please download the weights and put them under `./ckpts`.

To train the sparse structure module with our designed mask-weighted cross-attention and occlusion-aware attention, please run:
```sh
. ./train_ss.sh
```
To train the sparse structure module with our designed mask-weighted cross-attention and occlusion-aware attention, please run:
```sh
. ./train_slat.sh
```
The output folder where the model will be saved can be changed by modifying `--vis` parameter in the script.


### Inference
We have prepared examples under ./example folder. It supports both single and multiple image as input. For inference, please run:
```sh
python ./inference.py
```

If you want to try on you own data. You should prepare: 1) original image and 2) mask image (background is white (255,255,255), visible area is gray (188,188,188), occluded area is black (0,0,0)).

You can use [Segment Anything](https://github.com/facebookresearch/segment-anything) to obtain the corresponding mask, which is used for our in-the-wild examples in the paper and also in our demo.


### Evalutation
We render Toys4K and GSO exactly the same as training data. To obtain the evaluation dataset, please modify the directory in `3d_mask_render.py` and run:
```sh
python ./3d_mask_render.py
```
It will create a `renders_mask` folder with the 3D consistent mask in it.



