# EPDiff

The evaluation code for the paper "EPDiff: Erasure Perception Diffusion Model for Unsupervised Anomaly Detection in Preoperative Multimodal Images", which is submitted to IEEE Transactions on Medical Imaging.

Our implementation is based on [DDPM](https://github.com/dome272/Diffusion-Models-pytorch) and [ANDi](https://github.com/alexanderfrotscher/andi). The entire code will be updated after acceptance.

## How to run
### 1. Environment
Please prepare an virtual environment with Python 3.9, and then use the command "pip install -r requirements.txt" for the dependencies.

### 2. Dataset
BraTS21 dataset can be found in [official website](http://braintumorsegmentation.org/) or [kaggle](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1).

Download the dataset, and change the dataset_path in eval.yml

### 3. Evaluation

The evaluation model can be downloaded from [Baidu Netdisk](https://pan.baidu.com/s/1aWtGUT-jm7zuWQNuCLFHZQ?pwd=m5yv) or [Google Drive](https://drive.google.com/file/d/1_EePlNxMPZHlPCI5ptAfncQErHEy-abr/view?usp=sharing).

python3 eval.py

