# Description

Train VAE and GAN to generate face images with [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

Here is the images my model generate:

**VAE**

![image](https://github.com/weifanhaha/semantic-segmentation/blob/master/images/vae.png)

**GAN**

![image](https://github.com/weifanhaha/semantic-segmentation/blob/master/images/gan.png)

# Usage

## Download Dataset

```
./get_dataset.sh
```

## Install packages

```
pip install -r requirements.txt
```

## Train

To train VAE / GAN model, you need to enter the corresponding folder and run training script. please take care of the path of training data and output model path.

Train VAE

```
cd VAE
python train.py
```

Train GAN

```
cd GAN
python train.py
```

## Generate images

You can generate random images with trained VAE and GAN.

generate image with VAE

```
cd VAE
python generate.py
```

generate image with GAN

```
cd GAN
python generate.py
```
