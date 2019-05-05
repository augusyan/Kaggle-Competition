# Kaggle-Competition

This repo is code of [Public project of my Kaggle Competition](https://github.com/augusyan/Kaggle-Competition/) based on sklearn&Keras&Tensorflow. **This repo only for learning.**

## Environment
- Operating system: Ubuntu 16.04 or CentOS 7
- Data would take up to 10GB disk memory
- Memory cost would be around 16GB
- Dependencies: 
  - [CUDA](https://developer.nvidia.com/cuda-toolkit) and [cuDNN](https://developer.nvidia.com/cudnn) with GPU
  - [Tensorflow](https://github.com/tensorflow/tensorflow) with packages ([Keras](https://github.com/keras-team/keras)) installed

## Prerequisites
- Download this repo
  ```bash
  git clone https://git@github.com:augusyan/Kaggle-Competition.git
  cd forward_mahjong_processiong
  ```

- Install requirements
  ```bash
  pip3 install -r requirements.txt
  ```

- (Unnecessary) 

## Usage
The training and testing scripts come with several options, which can be listed with the `--help` flag.
```bash
python3 main.py --help
```

To run the training and testing, simply run main.py. By default, the script runs resnet34 on attribute 'coat_length_labels' with 50 epochs.

To training and testing resnet34 on attribute 'collar_design_labels' with 100 epochs and some learning parameters:
```bash
python3 main.py --model 'resnet34' --attribute 'collar_design_labels' --epochs 100 --batch-size 128 --lr 0.01 --momentum 0.5
```

Every epoch trained model will be saved in the folder `save/[attribute]/[model]`.

## License
The code is licensed with the [MIT](LICENSE) license.
