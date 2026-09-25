import requests, base64
from huggingface_hub import login
from huggingface_hub import model_info

with open("test.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

payload = {
    "anon_type": "single_align",
    "anon_degree": 1.25,
    "images": {0: {"src": img_b64, "height": 512, "width": 512, "begin": 0, "end": 0}},
    "redact_type": "blur",
    "blur": 51,
    "pixel": 16,
    "diffusion_model": "sd2-community/stable-diffusion-2-1",
    "clip_model": "openai/clip-vit-large-patch14",
    "seed": 42,
    "guidance": 4.0,
    "inference_steps": 50,
    "vis_input": True,
    "hf_token": "..."
}



r = requests.post("http://localhost:9711/v1/process", json=payload)
print(r.json())