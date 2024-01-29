import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image

# Charger le modèle pré-entraîné ResNet50
weights = ResNet50_Weights.DEFAULT
model = resnet50(pretrained=True)
model.eval()

# Fonction pour extraire les caractéristiques


def extract_features(image_path, bbox):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])

    # Charger l'image et extraire la région de la boîte englobante
    image = Image.open(image_path).convert('RGB')
    region = image.crop((bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]))
    region_tensor = transform(region).unsqueeze(0)

    # Passer l'image dans le modèle pour obtenir les caractéristiques
    with torch.no_grad():
        features = model(region_tensor)
    return features.squeeze(0)


# Exemple d'utilisation
# Exemple de boîte englobante [x, y, largeur, hauteur]
bbox_example = [100, 100, 50, 50]
image_path = 'ADL-Rundle-6/img1/000001.jpg'
features = extract_features(image_path, bbox_example)
print(features)