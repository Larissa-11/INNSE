# INNSE: Invertible Neural Network-Based DNA Image Storage with Self-Correction Encoding
Pytorch Implementation of the paper "INNSE: Invertible Neural Network-Based DNA Image Storage with Self-Correction Encoding".

## Dependencies and Installation
- Python 3 via Anaconda (recommended)
- PyTorch >= 1.0
- NVIDIA GPU + CUDA
- Python Package: pip install numpy opencv-python lmdb pyyaml

## Dataset Preparation
The training and test data sets for the image can be downloaded [here](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md).
The training and test data sets for the video can be downloaded [here](http://toflow.csail.mit.edu/).
## Usage
All the implementation is in `/codes`. To run the code, 
select the corresponding configuration file in `/codes/options/` and run as following command:
#### Training
```
python train.py -opt options/train/train_I_IRN_x4.yml
python train.py -opt options/train/train_V_IRN_x4.yml
```
#### Testing
```
python test.py -opt options/test/test_I_IRN_x4.yml
python test.py -opt options/test/test_V_IRN_x4.yml

```
#### Codec
```
python encode.py -opt options/en_decode/encode_I_IRN_x4.yml
python decode.py -opt options/en_decode/decode_I_IRN_x4.yml
python encode.py -opt options/en_decode/encode_V_IRN_x4.yml
python decode.py -opt options/en_decode/decode_V_IRN_x4.yml
```
## Acknowledgement
Our project is heavily based on [Invertible-Image-Rescaling](https://github.com/pkuxmq/Invertible-Image-Rescaling) and [Video Rescaling Networks with Joint Optimization Strategies for Downscaling and Upscaling]([https://github.com/pkuxmq/Invertible-Image-Rescaling](https://github.com/ding3820/MIMO-VRN) as basic framework.
