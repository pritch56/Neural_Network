import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(img_path, max_size=512):
    """Load and preprocess image."""
    image = Image.open(img_path).convert('RGB')
    size = min(max(image.size), max_size)
    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[:3, :, :]),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = in_transform(image).unsqueeze(0)
    return image.to(device)

def im_convert(tensor):
    """Convert tensor to PIL image."""
    image = tensor.clone().detach().cpu().squeeze(0)
    image = image * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    image = image + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    image = image.clamp(0, 1)
    image = transforms.ToPILImage()(image)
    return image

def get_features(image, model, layers=None):
    """Extract features from VGG19."""
    if layers is None:
        layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',  # content
            '28': 'conv5_1'
        }
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def gram_matrix(tensor):
    """Compute Gram matrix for style representation."""
    b, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

def run_style_transfer(content_img, style_img, num_steps=300, style_weight=1e6, content_weight=1):
    """Perform neural style transfer."""
    vgg = models.vgg19(pretrained=True).features.to(device).eval()
    for param in vgg.parameters():
        param.requires_grad = False

    content_features = get_features(content_img, vgg)
    style_features = get_features(style_img, vgg)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    target = content_img.clone().requires_grad_(True).to(device)
    optimizer = optim.LBFGS([target])

    mse_loss = nn.MSELoss()

    run = [0]
    while run[0] <= num_steps:
        def closure():
            optimizer.zero_grad()
            target_features = get_features(target, vgg)
            content_loss = mse_loss(target_features['conv4_2'], content_features['conv4_2'])
            style_loss = 0
            for layer in style_grams:
                target_feature = target_features[layer]
                target_gram = gram_matrix(target_feature)
                style_gram = style_grams[layer]
                layer_style_loss = mse_loss(target_gram, style_gram)
                style_loss += layer_style_loss / (target_feature.shape[1] ** 2)
            total_loss = content_weight * content_loss + style_weight * style_loss
            total_loss.backward()
            run[0] += 1
            return total_loss
        optimizer.step(closure)
    return target
