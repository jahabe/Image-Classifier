# In[1]:
# Check torch version and CUDA status if GPU is enabled.
import torch
print(torch.__version__)
print(torch.cuda.is_available()) # Should return True when GPU is enabled. 

"""
Developing an AI application
    The project is broken down into multiple steps:
        * Load and preprocess the image dataset
        * Train the image classifier on your dataset
        * Use the trained classifier to predict image content
"""
# In[2]:
import torch # used to build and train the neural network
from torch import nn, optim #nn for neural network modules and optim for optimization algorithms
from torchvision import datasets, transforms, models # for loading datasets, applying transformations, and pre-trained models

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json # for reading label mappings

"""
Load the data
    used `torchvision` to load the data.

Data Description
    The dataset is split into three parts, training, validation, and testing. 
    For the training, apply transformations such as random scaling, cropping, and flipping. 
    This will help the network generalize leading to better performance. 
    make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.

    The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. 
    For this, no scaling or rotation transformations. 
    Resize then crop the images to the appropriate size.

    The pre-trained networks were trained on the ImageNet dataset where each color channel was normalized separately. 
    For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects. 
    For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  
    These values will shift each color channel to be centered at 0 and range from -1 to 1. 
 
"""
# In[3]:
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# In[5]:
# Define data transformations with improvements
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]), 
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]), 
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Load datasets
image_datasets = {
    'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
    'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
    'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
}

# Use DataLoaders for batch processing
dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=128, shuffle=True),
    'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64),
    'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=64)
}

"""
Label mapping
    load in a mapping from category label to category name. 
    You can find this in the file `cat_to_name.json`. 
    This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.
"""

# In[6]:
import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# In[9]: Building and training the classifier
model = models.vgg16(pretrained=True)

for param in model.features.parameters():
    param.requires_grad = False

classifier = nn.Sequential(
    nn.Linear(25088, 4096),
    nn.ReLU(),
    nn.Dropout(0.6),
    nn.Linear(4096, 1024),
    nn.ReLU(),
    nn.Dropout(0.5),  
    nn.Linear(1024, 102),
    nn.LogSoftmax(dim=1)
)

model.classifier = classifier

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.003, weight_decay=1e-4)  
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs = 10  
best_accuracy = 0

for epoch in range(epochs):
    running_loss = 0
    model.train()
    
    for inputs, labels in dataloaders['train']:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logps = model(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Validation phase
    valid_loss = 0
    accuracy = 0
    model.eval()
    
    with torch.no_grad():
        for inputs, labels in dataloaders['valid']:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model(inputs)
            loss = criterion(logps, labels)
            valid_loss += loss.item()
            
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    
    scheduler.step()
    
    print(f"Epoch {epoch+1}/{epochs}.. "
          f"Train Loss: {running_loss/len(dataloaders['train']):.3f}.. "
          f"Validation Loss: {valid_loss/len(dataloaders['valid']):.3f}.. "
          f"Validation Accuracy: {accuracy/len(dataloaders['valid']):.3f}")

# In[ ]:
"""
Testing your network
    It's good practice to test your trained network on test data, 
    images the network has never seen either in training or validation. 
    This will give you a good estimate for the model's performance on completely new images. 
    Run the test images through the network and measure the accuracy, 
    the same way you did validation. 
    You should be able to reach around 70% accuracy 
    on the test set if the model has been trained well.
"""
def test_model(model, dataloader):
    model.eval()
    accuracy = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model(inputs)
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.sum(equals).item()
            total_samples += labels.size(0)

    print(f'Test Accuracy: {accuracy / total_samples:.3f}')

test_model(model, dataloaders['test'])

# In[ ]:

"""
Save the checkpoint
    Now that your network is trained, save the model so you can load it later for 
    making predictions. You probably want to save other things such as the mapping 
    of classes to indices which you get from one of the image datasets: 
    `image_datasets['train'].class_to_idx`. 
    You can attach this to the model as an attribute which makes inference easier later on.

    Remember that you'll want to completely rebuild the model later so you can use it 
    for inference. Make sure to include any information you need in the checkpoint. 
    If you want to load the model and keep training, you'll want to save the number of 
    epochs as well as the optimizer state, `optimizer.state_dict`. You'll likely want to use 
    this trained model in the next part of the project, so best to save it now.
"""

# TODO: Save the checkpoint 
model.class_to_idx = image_datasets['train'].class_to_idx

checkpoint = {
    'architecture': 'vgg16',
    'class_to_idx': model.class_to_idx, 
    'state_dict': model.state_dict(), 
    'classifier': model.classifier
}

torch.save(checkpoint, 'checkpoint.pth')

# In[ ]:
"""
Loading the checkpoint
    At this point it's good to write a function that can load a checkpoint and rebuild 
    the model. That way you can come back to this project and keep working on it without 
    having to retrain the network.
"""
# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg16(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return model

# Load and move the model to the correct device
model = load_checkpoint('checkpoint.pth')

# In[37]:

"""
Inference for classification
    Now you'll write a function to use a trained network for inference. 
    That is, you'll pass an image into the network and predict the class of 
    the flower in the image. Write a function called `predict` that takes an 
    image and a model, then returns the top $K$ most likely classes along 
    with the probabilities. It should look like 
        ```python
        probs, classes = predict(image_path, model)
        print(probs)
        print(classes)
        > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
        > ['70', '3', '45', '62', '55']
        ```

    First you'll need to handle processing the input image such that it can be used in your network. 

Image Preprocessing
    You'll want to use `PIL` to load the image. 
    It's best to write a function that preprocesses the image so it can be used as input for 
    the model. This function should process the images in the same manner used for training. 

    First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. 
    Color channels of images are typically encoded as integers 0-255, but the model expected 
    floats 0-1. You'll need to convert the values. It's easiest with a Numpy array, which you 
    can get from a PIL image like so `np_image = np.array(pil_image)`.

    As before, the network expects the images to be normalized in a specific way. 
    For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations 
    `[0.229, 0.224, 0.225]`. You'll want to subtract the means from each color channel, 
    then divide by the standard deviation. 

    And finally, PyTorch expects the color channel to be the first dimension but it's the 
    third dimension in the PIL image and Numpy array. You can reorder dimensions using 
    [`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html). 
    The color channel needs to be first and retain the order of the other two dimensions.
"""
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    image = Image.open(image)
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return preprocess(image)
    
# In[38]:
"""
To check your work, the function below converts a PyTorch tensor and displays it in the notebook. 
If your `process_image` function works, running the output through this function should return 
the original image (except for the cropped out portions).
"""
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

# In[39]:
"""
Class Prediction

    Once you can get images in the correct format, it's time to write a function for making 
    predictions with your model. A common practice is to predict the top 5 or so 
    (usually called top-$K$) most probable classes. You'll want to calculate the class 
    probabilities then find the $K$ largest values.

    To get the top $K$ largest values in a tensor use [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk). 
    This method returns both the highest `k` probabilities and the indices of those probabilities 
    corresponding to the classes. You need to convert from these indices to the actual class 
    labels using `class_to_idx` which hopefully you added to the model or from an 
    `ImageFolder` you used to load the data ([see here](#Save-the-checkpoint)). 
    Make sure to invert the dictionary so you get a mapping from index to class as well.

    Again, this method should take a path to an image and a model checkpoint, 
    then return the probabilities and classes.

    ```python
    probs, classes = predict(image_path, model)
    print(probs)
    print(classes)
    > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
    > ['70', '3', '45', '62', '55']
    ```
"""

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model. '''
    model.eval()
    
    # Move model to the same device as input
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Process the image and move it to the correct device
    image = process_image(image_path)
    image = image.unsqueeze(0)  # Add batch dimension
    image = image.to(device)  # Move image to the same device as the model
    
    with torch.no_grad(): 
        output = model(image)
    
    ps = torch.exp(output)
    top_p, top_class = ps.topk(topk, dim=1)
    
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[c.item()] for c in top_class[0]]
    
    return top_p[0].tolist(), top_classes

# In[40]:
"""
Sanity Checking

    Can use a trained model for predictions, check to make sure it makes sense. 
    Even if the testing accuracy is high, it's always good to check that there aren't 
    obvious bugs. Use `matplotlib` to plot the probabilities for the top 5 classes as 
    a bar graph, along with the input image. It should look like this:
        <img src='assets/inference_example.png' width=300px>
    You can convert from the class integer encoding to actual flower names with the 
    `cat_to_name.json` file (should have been loaded earlier in the notebook). 
    To show a PyTorch tensor as an image, use the `imshow` function defined above.
"""

# TODO: Display an image along with the top 5 classes
image_path = 'flowers/test/1/image_06743.jpg'
probs, classes = predict(image_path, model)

# Convert class indices to class names
class_names = [cat_to_name[c] for c in classes]

# Display the image and predicted classes
def plot_prediction(image_path, probs, class_names):
    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), nrows=2)
    
    # Display the image
    image = process_image(image_path)
    imshow(image, ax=ax1, title=class_names[0])
    
    # Plot the probabilities
    y_pos = np.arange(len(class_names))
    ax2.barh(y_pos, probs)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(class_names)
    ax2.invert_yaxis()
    
    plt.show()

# Run the plot function
plot_prediction(image_path, probs, class_names)


# In[ ]:

"""
Reminder for Workspace users
    If your network becomes very large when saved as a checkpoint, there might be issues 
    with saving backups in your workspace. You should reduce the size of your hidden layers 
    and train again. 
    
    We strongly encourage you to delete these large interim files and directories 
    before navigating to another page or closing the browser tab.
"""
# In[41]:


# TODO remove .pth files or move it to a temporary `~/opt` directory in this Workspace
get_ipython().system('rm checkpoint.pth')


# In[ ]:




