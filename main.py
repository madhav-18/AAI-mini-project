import gradio as gr
from PIL import Image
import numpy as np
from inference import test

def load_image(image, x32=False):
    if x32:
        def to_32s(x):
            return 256 if x < 256 else x - x % 32
        w, h = image.size
        image = image.resize((to_32s(w), to_32s(h)))

    return image

def process_image(input_image):

    output = test(input_image)

    return output

iface = gr.Interface(fn=process_image, 
                     inputs=gr.Image(), 
                     outputs=gr.Image(type="pil"),
                     title="Image Processing Example",
                     description="Upload an image and convert it to Anime")

# Launch the interface share=True
iface.launch(debug= True, share=True)