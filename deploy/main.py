from flask import Flask, request, jsonify, render_template
from PIL import Image
import os
import torch
import clip
from transformers import GPT2Tokenizer
from clip_caption_model import ClipCaptionPrefix
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from time import time
from numpy import array
import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse
import IPython.display
from IPython.display import Markdown, display
import ipywidgets as widgets
import clip
import os
from torch import nn
import numpy as np
import torch
import torch.nn.functional as nnf
import sys
from typing import Tuple, List, Union, Optional
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
# from google.colab import files
import skimage.io as io
import PIL.Image
# from IPython.display import Image
from enum import Enum
import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union
import matplotlib.pyplot as plt
import base64
from gtts import gTTS
from googletrans import Translator
app = Flask(__name__)

# Muat model CLIP dan tokenizer
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
clip_model, preprocess = clip.load("RN50x4", device=device, jit=False)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load the captioning model
prefix_length = 40
model = ClipCaptionPrefix(prefix_length, clip_length=40, prefix_size=640, num_layers=8, mapping_type='transformer')
model.load_state_dict(torch.load('model_skrpsi_RN50x4-029.pt', map_location=device))
model = model.eval()
model = model.to(device)

# Fungsi untuk memastikan direktori "foto" ada atau dibuat jika belum ada
def ensure_foto_directory():
    foto_directory = './foto/'
    if not os.path.exists(foto_directory):
        os.makedirs(foto_directory)
    return foto_directory

# Fungsi untuk menyimpan gambar ke dalam folder "foto"
def save_image_to_foto(image, image_name):
    foto_directory = ensure_foto_directory()
    image_path = os.path.join(foto_directory, image_name)
    image.save(image_path)

def generate_beam(model, tokenizer, beam_size: int = 5, prompt=None, embed=None,
                  entry_length=40, temperature=1., stop_token: str = '.'):

    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [tokenizer.decode(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]
    order = scores.argsort(descending=True)
    #output_texts = [output_texts]
    output_texts = [output_texts[i] for i in order]
    return output_texts
# Function to generate captions
def generate_caption(image_path):
    images = Image.open(image_path).convert("RGB")
    image = preprocess(images).unsqueeze(0).to(device)

    with torch.no_grad():
        prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
        prefix = prefix / prefix.norm(2, -1).item()
        prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)

    # Use beam search for caption generation
    generated_text_prefix = generate_beam(model, tokenizer, embed=prefix_embed, beam_size=7, stop_token='.', entry_length=40)[0]
    return generated_text_prefix
def generate_translation(image_path):
    # Inisialisasi objek translator
    translator = Translator()

    # Lakukan terjemahan dari deskripsi gambar dalam bahasa Inggris ke bahasa Indonesia
    description = generate_caption(image_path)  # Gantilah dengan cara Anda mendapatkan deskripsi gambar
    translation = translator.translate(description, src='en', dest='id')

    return translation.text
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided."})
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({"error": "No selected file."})
        
        # Save the image to a temporary location
        save_image_to_foto(image_file, image_file.filename)

        # Generate caption for the image
        image_path = os.path.join('./foto/', image_file.filename)
        caption = generate_caption(image_path)
        tts = gTTS(text=caption, lang='en', slow=False)
        audio_path = os.path.join('./audio/', 'caption.mp3')
        tts.save(audio_path)

        # Encode the audio file to base64
        with open(audio_path, 'rb') as audio_file:
            audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')

        # Return the caption as JSON response
        return jsonify({"caption": caption, "audio_data": audio_base64})
    
    return render_template('index.html')
@app.route('/translation', methods=['GET', 'POST'])
def translation():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided."})
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({"error": "No selected file."})

        # Simpan gambar ke lokasi sementara
        save_image_to_foto(image_file, image_file.filename)

        # Hasilkan teks terjemahan untuk gambar
        image_path = os.path.join('./foto/', image_file.filename)
        translation = generate_translation(image_path)  # Fungsi ini harus Anda implementasikan

        # Buat audio terjemahan dalam bahasa Indonesia
        tts_id = gTTS(text=translation, lang='id', slow=False)
        audio_id_path = os.path.join('./audio_id/', 'translation.mp3')
        tts_id.save(audio_id_path)

        # Encode file audio ke base64
        with open(audio_id_path, 'rb') as audio_id_file:
            audio_id_base64 = base64.b64encode(audio_id_file.read()).decode('utf-8')

        return jsonify({"translation": translation, "audio_id_data": audio_id_base64})

    return render_template('translation.html')
if __name__ == '__main__':
    app.run(debug=True)