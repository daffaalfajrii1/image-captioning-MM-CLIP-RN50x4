# Image Captioning with CLIP - RN50x4

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

## Installation

To run the image captioning model, follow these steps:

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/image-captioning-MM-CLIP-RN50x4.git
   cd image-captioning-MM-CLIP-RN50x4
