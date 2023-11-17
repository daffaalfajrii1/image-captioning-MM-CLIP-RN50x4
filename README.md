# Image Captioning with CLIP - RN50x4 for visually impaired people

![CLIP](https://github.com/daffaalfajrii1/image-captioning-MM-CLIP-RN50x4/blob/main/architecture.png)

This repository contains the implementation of an image captioning model using CLIP (Contrastive Language-Image Pretraining) with a ResNet-50x4 backbone. The model was developed as part of a thesis project and achieved the following performance metrics:

| Model  | BLEU@1 | BLEU@2 | BLEU@3 | BLEU@4 | METEOR |
|--------|--------|--------|--------|--------|--------|
| RN50x4 | 0.82   | 0.79   | 0.75   | 0.73   | 0.50   |

## Overview

### CLIP (Contrastive Language-Image Pretraining)

CLIP is a powerful vision-language pretraining model that learns joint representations of images and text. It has demonstrated state-of-the-art performance on various vision and language tasks.

### Model Architecture

The image captioning model utilizes a ResNet-50x4 backbone for feature extraction and is fine-tuned using CLIP. The training process involves learning to generate descriptive captions for images based on the joint understanding of textual and visual data.

### Dataset

The dataset used for training consists of 3800 images captured in public spaces, and each image is associated with four captions. This diverse dataset aims to enhance the model's ability to provide detailed and informative captions for various scenarios encountered in public environments.
* [data_caption_here](https://drive.google.com/drive/u/0/folders/11tcspegZxbrwQnx9SnlCDcMXYPAbzR6V)
* [data_image_here](https://drive.google.com/drive/u/0/folders/1ZfstVhqay7GzZDOWTIEhcZCTRAJFE6LB)

## Installation

To run the image captioning model, follow these steps:

1. Clone this repository:

   ```bash
   git clone image-captioning-MM-CLIP-RN50x4-for-visually-impaired-people

2. Install transformers
   ```bash
   !pip install transformers

3. Install CLIP
   ```bash
   ! pip install git+https://github.com/openai/CLIP.git

4. Open the Colab inference 
   Navigate to the Colab notebook (.ipynb) and follow the steps outlined, including:
   a. Image Embedding
   b. Train

5. Output train is on .pt format. 
   The trained model will produce output in .pt format. You can find the model weights in the output directory. The file may be named something like image_captioning_model.pt.

6.Create a new folder named deploy in the project directory. 
   Move the trained model file (image_captioning_model.pt) to the deploy folder. Include your Flask deployment script (main.py) in the deploy folder.
