# Image Classifier with Deep Learning

## Project Overview
This project is part of Udacity's **AI Programming with Python Nanodegree**, sponsored by AWS. The goal is to develop an **image classifier** using deep learning techniques with PyTorch. The classifier is trained to recognize different species of flowers but can be adapted to classify any set of labeled images.

## Features
- Load and preprocess an image dataset (flowers or any labeled dataset).
- Train a **deep learning model** using a pre-trained neural network (VGG16 by default).
- Save and load a trained model checkpoint.
- Predict the class of an image using the trained model.
- Run the model using a **command-line interface (CLI)**.

## Files Included
- `train.py` - Trains a new deep learning model on a dataset and saves the model as a checkpoint.
- `predict.py` - Loads a trained model checkpoint and predicts the class of an input image.
- `image_classifier.py` - Contains helper functions used for training and inference.
- `README.md` - Documentation for the project.

## Installation & Setup
### Prerequisites
Make sure you have Python installed along with the required libraries:

```sh
pip install torch torchvision numpy matplotlib PIL argparse json
```

Alternatively, you can install dependencies using:

```sh
pip install -r requirements.txt
```

### Dataset
This project was trained using the **102 Category Flower Dataset**, but you can use any labeled dataset for training. The dataset should be organized as follows:

```
flowers/
    train/
        class1/
        class2/
        ...
    valid/
        class1/
        class2/
        ...
    test/
        class1/
        class2/
        ...
```

If using a different dataset, ensure it follows a similar directory structure.

## Usage

### 1. Training the Model
Run `train.py` to train the model:

```sh
python train.py flowers --save_dir checkpoint/ --arch vgg16 --learning_rate 0.003 --hidden_units 512 --epochs 10 --gpu
```

#### Training Options
| Argument | Description | Default |
|----------|-------------|---------|
| `data_dir` | Path to dataset (Required) | - |
| `--save_dir` | Directory to save the checkpoint | Current directory |
| `--arch` | Model architecture (e.g., vgg16, resnet18) | vgg16 |
| `--learning_rate` | Learning rate for training | 0.003 |
| `--hidden_units` | Number of hidden units in classifier | 512 |
| `--epochs` | Number of training epochs | 10 |
| `--gpu` | Use GPU for training | False |

### 2. Making Predictions
Run `predict.py` to classify an image:

```sh
python predict.py path/to/image checkpoint.pth --top_k 3 --category_names cat_to_name.json --gpu
```

#### Prediction Options
| Argument | Description | Default |
|----------|-------------|---------|
| `input` | Path to input image (Required) | - |
| `checkpoint` | Path to trained model checkpoint (Required) | - |
| `--top_k` | Number of top most likely classes to return | 5 |
| `--category_names` | Path to JSON file mapping classes to names | cat_to_name.json |
| `--gpu` | Use GPU for inference | False |

### 3. Example Output
```
Daisy: 87.45%
Sunflower: 7.12%
Rose: 3.33%
```

## Model Architecture
The classifier is built using **transfer learning** with a pre-trained **VGG16** model. The classifier replaces the fully connected layers with a new feedforward network:
- **Input Layer:** 25088 neurons (VGG16 output size)
- **Hidden Layers:**
  - 4096 neurons (ReLU activation + Dropout)
  - 1024 neurons (ReLU activation + Dropout)
- **Output Layer:** 102 classes (LogSoftmax activation)

## Contributions
This project was implemented as part of Udacity's **AI Programming with Python Nanodegree**, utilizing AWS-sponsored GPU-powered workspaces.

## License
This project is open-source and available for educational use.

