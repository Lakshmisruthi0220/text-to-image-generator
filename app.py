import torch
from diffusers import StableDiffusionPipeline
import gradio as gr

# Determine the device and correct dtype
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32  

# Load model with appropriate dtype
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=dtype,
)

pipe = pipe.to(device)

def generate_image(prompt):
    image = pipe(prompt, num_inference_steps=25).images[0]
    return image

# Gradio app
demo = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(label="Enter your prompt"),
    outputs=gr.Image(type="pil", label="Generated Image"),
    title="Text-to-Image Generator (Stable Diffusion)",
    description="Type a text prompt to generate an image."
).launch()


if __name__ == "__main__":
    demo.launch()
