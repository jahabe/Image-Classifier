# 🌸 Image Classifier with Deep Learning

## Overview
This was one of the first projects I worked on for the Udacity AI Programming with Python Nanodegree (sponsored by AWS). The goal was to build an image classifier that can recognize flower species using a deep learning model (VGG16) from PyTorch. Although the dataset was focused on flowers, the model and structure can be reused for any labeled image dataset.

It was my first hands-on experience training a model from scratch, saving/loading model checkpoints, and writing a prediction script with a command-line interface. I also learned how to structure a real ML project more clearly and write flexible code.

---

## What I Learned
- How to use a **pretrained model** like VGG16 and update the classifier part only
- How to handle **image transformations and normalization**
- How to write a **training loop** with validation steps to monitor performance
- How to build a **predict.py** file that runs from the command line and returns the top predictions
- The process of **saving and loading models** using `torch.save()` and `torch.load()`
- GPU vs CPU issues and how to write device-agnostic code

---

## 🎯 Why I Built This

I wanted to understand how image classification models work behind the scenes—not just using APIs, but actually training one from scratch. Working through this helped me connect theory to implementation and gave me confidence to explore more complex ML projects in the future.

---

## A Few Challenges I Faced

### 🧩 Device Mismatch (GPU vs CPU)
I ran into a problem where the model was on GPU but the inputs were still on CPU (or vice versa), which caused annoying runtime errors. After researching, I fixed it by making sure both model and inputs were moved to the same device using:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
inputs, labels = inputs.to(device), labels.to(device)
```
### 📉Struggling to Meet Test Accuracy

When I first submitted the notebook, my test accuracy was **way too low (~0.7%)**, which obviously wasn't acceptable. I realized my model was likely underfitting or not generalizing well. I went back to:

- Tune hyperparameters (like learning rate, hidden units, and epochs)
- Double check my image transformations and normalization
- Simplify the classifier architecture a bit

It took me a few tries, but I eventually pushed the test accuracy above 60%. That process really helped me understand how important validation feedback and iteration are when working with deep learning models.

### Mapping Predictions Back to Class Names
PyTorch’s topk() function gives indices, not labels. I had to reverse the class_to_idx dictionary to map predictions back to the correct class name. This took a bit of trial and error but really helped me understand how PyTorch handles datasets and outputs.

## 🛠️ Features

- Train an image classifier using a pretrained model like VGG16 or ResNet18  
- Run predictions on new images and get top K results  
- Save and load model checkpoints  
- Use any labeled dataset (not just flowers) by following the same folder structure  
- Everything runs from the command line  

---

## 🗂️ Folder Structure

| File                  | Description                                               |
|-----------------------|-----------------------------------------------------------|
| `train.py`            | Trains the model and saves the checkpoint                 |
| `predict.py`          | Predicts image classes using a saved model                |
| `image_classifier.py` | (Optional) Helper functions (used during notebook testing)|
| `name.json`           | JSON file mapping label indices to flower names           |
| `checkpoint.pth`      | Saved model state dictionary                              |
| `README.md`           | This file you're reading                                  |

---

## 🗂️ Dataset Info

I used the [102 Category Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html).  
The dataset should be organized like this:
```
flowers/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
├── valid/
└── test/
```

---

# How to Use

## To Train the Model

```bash
python train.py flowers --save_dir checkpoint/ --arch vgg16 --learning_rate 0.003 --hidden_units 512 --epochs 10 --gpu
```

## To predict prediction: 
python predict.py path/to/image.jpg checkpoint.pth --top_k 3 --category_names cat_to_name.json --gpu

## Model Architecture
Pretrained model: VGG16
Custom classifier:
Linear(25088 → 4096) + ReLU + Dropout
Linear(4096 → 1024) + ReLU + Dropout
Linear(1024 → 102) + LogSoftmax

## Example output
Daisy: 87.45%
Sunflower: 7.12%
Rose: 3.33%

---

# Additional

### 🚧 Future Improvements

- Integrate a web interface (Flask or Streamlit) to upload and classify images  
- Add model benchmarking across architectures (e.g., ResNet18 vs. VGG16)  
- Try training on a different dataset, such as dogs vs. cats  
- Visualize model activations and filters  

### About me
Hi! I’m Jane Choi, currently studying Computer Science at university. 
I’m always learning, always curious, and this project is one of the many small steps in my journey into machine learning and software development.

### 🙏 Acknowledgments

- Udacity AI Programming with Python Nanodegree  
- AWS for GPU workspace  
- PyTorch docs and community tutorials  

### License
This project is open-source and available for educational use.
