import os
import telebot
from flask import Flask, render_template, request, send_file
from diffusers import StableDiffusionPipeline
from io import BytesIO
import torch

# Initialize Flask app
app = Flask(__name__)

# Initialize the Telegram bot with your token
API_TOKEN = 'YOUR_TELEGRAM_BOT_TOKEN'
bot = telebot.TeleBot(API_TOKEN)

# Initialize Stable Diffusion Pipeline
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4-original")
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Function to generate image using Stable Diffusion
def generate_image(prompt: str):
    image = pipe(prompt).images[0]
    byte_io = BytesIO()
    image.save(byte_io, 'PNG')
    byte_io.seek(0)
    return byte_io

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form['prompt']
    try:
        image_io = generate_image(prompt)
        return send_file(image_io, mimetype='image/png', as_attachment=True, download_name='generated_image.png')
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
