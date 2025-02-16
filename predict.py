import torch
from torchvision import models, transforms
from PIL import Image
import argparse
import json

# Argument parser
parser = argparse.ArgumentParser(description='Predict flower name from an image.')
parser.add_argument('input', help='Path to input image')
parser.add_argument('checkpoint', help='Path to model checkpoint')
parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
parser.add_argument('--category_names', default='cat_to_name.json', help='Path to category names mapping')
parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')

args = parser.parse_args()

# Load checkpoint
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = getattr(models, checkpoint['architecture'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model

model = load_checkpoint(args.checkpoint)

# Move model to GPU if available
device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
model.to(device)

# Load category names
with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)

# Process image
def process_image(image_path):
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return preprocess(image)

# Predict function
def predict(image_path, model, topk=5):
    model.eval()
    image = process_image(image_path)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
    
    ps = torch.exp(output)
    top_p, top_class = ps.topk(topk, dim=1)

    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[c.item()] for c in top_class[0]]

    return top_p[0].tolist(), top_classes

# Run prediction
probs, classes = predict(args.input, model, args.top_k)
class_names = [cat_to_name[c] for c in classes]

# Display results
for i in range(len(probs)):
    print(f"{class_names[i]}: {probs[i]*100:.2f}%")
