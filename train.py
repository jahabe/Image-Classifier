import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import argparse
import os

# Argument parser
parser = argparse.ArgumentParser(description='Train a neural network on a dataset.')
parser.add_argument('data_dir', help='Directory of the dataset')
parser.add_argument('--save_dir', default='.', help='Directory to save checkpoints')
parser.add_argument('--arch', default='vgg16', help='Model architecture (default: vgg16)')
parser.add_argument('--learning_rate', type=float, default=0.003, help='Learning rate')
parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
parser.add_argument('--gpu', action='store_true', help='Use GPU for training')

args = parser.parse_args()

# Data directories
train_dir = os.path.join(args.data_dir, 'train')
valid_dir = os.path.join(args.data_dir, 'valid')

# Data transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Load datasets
image_datasets = {
    'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
    'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])
}

dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
    'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64)
}

# Load pre-trained model
model = getattr(models, args.arch)(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

# Define classifier
classifier = nn.Sequential(
    nn.Linear(model.classifier[0].in_features, args.hidden_units),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(args.hidden_units, 102),
    nn.LogSoftmax(dim=1)
)

model.classifier = classifier

# Criterion and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

# Device configuration
device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
for epoch in range(args.epochs):
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
            batch_loss = criterion(logps, labels)
            valid_loss += batch_loss.item()

            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Epoch {epoch+1}/{args.epochs}.. ",
          f"Train loss: {running_loss/len(dataloaders['train']):.3f}.. ",
          f"Valid loss: {valid_loss/len(dataloaders['valid']):.3f}.. ",
          f"Valid accuracy: {accuracy/len(dataloaders['valid']):.3f}")

# Save checkpoint
model.class_to_idx = image_datasets['train'].class_to_idx
checkpoint = {
    'architecture': args.arch,
    'class_to_idx': model.class_to_idx,
    'state_dict': model.state_dict(),
    'classifier': model.classifier
}
torch.save(checkpoint, os.path.join(args.save_dir, 'checkpoint.pth'))
