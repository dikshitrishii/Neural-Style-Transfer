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
size = 256
image_loader = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor()])

def image_load(image):
    image = image_loader(image).unsqueeze(0)  # Add batch dimension
    return image.to(device, torch.float)

class CustomContentLoss(nn.Module):
    def __init__(self, target):
        super(CustomContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def compute_gram_matrix(input):
    batch_size, num_features, height, width = input.size()
    features = input.view(batch_size * num_features, height * width)
    gram = torch.mm(features, features.t())
    return gram.div(batch_size * num_features * height * width)

class CustomStyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(CustomStyleLoss, self).__init__()
        self.target = compute_gram_matrix(target_feature).detach()

    def forward(self, input):
        G = compute_gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

cnn = vgg19(pretrained=True).features.to(device).eval()

mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
std = torch.tensor([0.229, 0.224, 0.225]).to(device)

class ImageNormalization(nn.Module):
    def __init__(self, mean, std):
        super(ImageNormalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

# We will be taking the conv layers only as per original implementation.
ct_layers = ['conv_3']
st_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def style_model_and_losses(cnn, mean, std, style_img, content_img, ct_layers=ct_layers, st_layers=st_layers):
    normalization = ImageNormalization(mean, std).to(device)
    content_losses = []
    style_losses = []
    model = nn.Sequential(normalization)
    i = 0
    
    # Iterate through each layer in the CNN model and extracting it
    for layer in cnn.children():
        # Check if the layer is a Conv2d layer (Convolutional layer)
        if isinstance(layer, nn.Conv2d):
            i =i+ 1  # Increment the counter for convolutional layers
            name = 'conv_{}'.format(i)  # Create a name for the layer, e.g., 'conv_1', 'conv_2', etc.
        
        # Check if the layer is a ReLU layer (Activation function)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)  # Create a name for the layer, e.g., 'relu_1', 'relu_2', etc.
            layer = nn.ReLU(inplace=False)  # Ensure that ReLU is not i n-place, which means it does not overwrite input values
        
        # Check if the layer is a MaxPool2d layer (Pooling layer)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)  # Create a name for the layer, e.g., 'pool_1', 'pool_2', etc.
        
        # Check if the layer is a BatchNorm2d layer (Batch Normalization layer)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)  # Create a name for the layer, e.g., 'bn_1', 'bn_2', etc.
        
        # If the layer type is not recognized, raise an error
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
        
        # Add the layer to the model with the created name
        model.add_module(name, layer)

        # Add content loss:
        if name in ct_layers:
            target = model(content_img).detach()
            content_loss = CustomContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        # Add style loss:
        if name in st_layers:
            target_feature = model(style_img).detach()
            style_loss = CustomStyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # Now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], CustomContentLoss) or isinstance(model[i], CustomStyleLoss):
            break

    model = model[:(i + 1)]
    return model, style_losses, content_losses

def ModelOptimizer(input):
    optimizer = optim.LBFGS([input.requires_grad_(True)])  # Tried adam but it was very very stylelossow.
    return optimizer

def execution(cnn, mean, std, content_img, style_img, input, total_steps_taken=300, st_Wt=10000, ct_Wt=0.001):
    model, style_losses, content_losses = style_model_and_losses(cnn, mean, std, style_img, content_img)
    optimizer = ModelOptimizer(input)

    run = [0]
    while run[0] <= total_steps_taken:
        def endsteps():
            input.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input)
            style_score = 0
            content_score = 0

            # Accumulate style losses
            for styleloss in style_losses:
                style_score += styleloss.loss
            # Accumulate content losses
            for cl in content_losses:
                content_score += cl.loss

            style_score *= st_Wt
            content_score *= ct_Wt

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            return style_score + content_score

        optimizer.step(endsteps)

    # Clamp the values of the output image to [0, 1]
    input.data.clamp_(0, 1)
    return input

# Streamlit app
st.title("Neural Style Transfer")

total_steps_taken = 100
st_Wt = 10000
ct_Wt = 0.001

st_img = st.file_uploader("Upload Style Image", type=["png", "jpg", "jpeg"])
ct_img = st.file_uploader("Upload Content Image", type=["png", "jpg", "jpeg"])

if st_img and ct_img:
    style_img = image_load(Image.open(st_img))
    content_img = image_load(Image.open(ct_img))
    input = content_img.clone()

    if st.button("Generate Image"):
        with st.spinner('Processing...Please wait, it will take a minute to generate'):
            output = execution(cnn, mean, std, content_img, style_img, input, total_steps_taken, st_Wt, ct_Wt)
            output_img = output.squeeze(0).cpu().detach().permute(1, 2, 0).numpy()

            st.image(output_img, width=500, caption="Stylized Image")

# To run the app, save the script as `app.py` and use the command `streamlit run app.py`
