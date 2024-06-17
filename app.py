import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.models import vgg19

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image loader and transformer
imsize = 256
loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor()])

def load_image(image):
    image = loader(image).unsqueeze(0)  # Add batch dimension
    return image.to(device, torch.float)

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    batch_size, num_features, height, width = input.size()
    features = input.view(batch_size * num_features, height * width)
    gram = torch.mm(features, features.t())
    return gram.div(batch_size * num_features * height * width)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

cnn = vgg19(pretrained=True).features.to(device).eval()

mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
std = torch.tensor([0.229, 0.224, 0.225]).to(device)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

content_layers = ['conv_3']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, mean, std, style_img, content_img, content_layers=content_layers, style_layers=style_layers):
    normalization = Normalization(mean, std).to(device)
    content_losses = []
    style_losses = []
    model = nn.Sequential(normalization)
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
        
        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]
    return model, style_losses, content_losses

def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_(True)])
    return optimizer

def run_style_transfer(cnn, mean, std, content_img, style_img, input_img, num_steps=300, style_weight=10000, content_weight=0.001):
    model, style_losses, content_losses = get_style_model_and_losses(cnn, mean, std, style_img, content_img)
    optimizer = get_input_optimizer(input_img)

    run = [0]
    while run[0] <= num_steps:
        def closure():
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            return style_score + content_score

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)
    return input_img

# Streamlit app
st.title("Neural Style Transfer")

num_steps = 100
style_weight = 10000
content_weight = 0.001

style_image = st.file_uploader("Upload Style Image", type=["png", "jpg", "jpeg"])
content_image = st.file_uploader("Upload Content Image", type=["png", "jpg", "jpeg"])

if style_image and content_image:
    style_img = load_image(Image.open(style_image))
    content_img = load_image(Image.open(content_image))
    input_img = content_img.clone()

    if st.button("Generate Image"):
        with st.spinner('Processing...Please wait, it will take a minute to generate'):
            output = run_style_transfer(cnn, mean, std, content_img, style_img, input_img, num_steps, style_weight, content_weight)
            output_img = output.squeeze(0).cpu().detach().permute(1, 2, 0).numpy()

            st.image(output_img, width=500, caption="Stylized Image")

# To run the app, save the script as `app.py` and use the command `streamlit run app.py`
