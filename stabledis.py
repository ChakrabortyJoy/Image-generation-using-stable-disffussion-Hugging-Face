import os
import streamlit as st
import torch
from diffusers import StableDiffusionPipeline 
from PIL import Image

@st.experimental_singleton
def load_model(model_id, auth_token):
    pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=auth_token)
    pipe.to(device)
    return pipe

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" # Use CUDA if available
auth_token = os.getenv("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")

pipe = load_model(model_id, auth_token)

st.title("Stable Bud")
prompt = st.text_input("Enter a prompt:", "")

if st.button("Generate"):
    with torch.no_grad():
        output = pipe(prompt, guidance_scale=8.5)
        st.write("Output structure:", output)  # Debug: print the output structure

        # Assuming the output is an image tensor, convert it to a format that Streamlit can display
        generated_image = output['images'][0] if 'images' in output else output['sample'][0]
        generated_image = generated_image.cpu().detach().numpy()  # Move tensor to CPU for conversion
        generated_image = Image.fromarray(generated_image)
        st.image(generated_image, caption="Generated Image", use_column_width=True)
