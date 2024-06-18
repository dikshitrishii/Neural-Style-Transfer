# Neural Style Transfer with Streamlit

## Project Overview

This project demonstrates a Neural Style Transfer application using a VGG-19 model pretrained on ImageNet. The application allows users to upload a style image and a content image, then generates a new image that applies the artistic style of the first image to the content of the second image. The project leverages Streamlit to provide an interactive web interface for users to easily perform style transfer.

## Installation Instructions

Follow these steps to set up the environment and run the project:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/dikshitrishii/Neural-Style-Transfer.git
    cd Neural-Style-Transfer
    ```

2. **Set Up a Virtual Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Application**:
    ```bash
    streamlit run app.py
    ```

## Usage

1. **Start the Streamlit App**:
    Run the command:
    ```bash
    streamlit run app.py
    ```

2. **Upload Images**:
    - Upload a style image (e.g., a painting).
    - Upload a content image (e.g., a photograph).

3. **Generate Stylized Image**:
    - Click the "Generate Image" button.
    - Wait for the processing to complete (it might take a minute).
    - View and download the generated stylized image.

## Dependencies

The project requires the following libraries:

- `streamlit`: For creating the web interface.
- `torch`: For building and running the neural network.
- `torchvision`: For loading the pretrained VGG-19 model.
- `PIL`: For image processing.
- `matplotlib`: For image visualization (optional, mainly for development).

## Referances
 "A Neural Algorithm of Artistic Style" by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge.
