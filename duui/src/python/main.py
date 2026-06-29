import argparse
import base64
from io import BytesIO

from pydantic import BaseModel
import face_alignment
import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModel
from fastapi import FastAPI, Response
from starlette.responses import JSONResponse, PlainTextResponse
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.utils import load_image
from diffusers.models.referencenet.referencenet_unet_2d_condition import (
    ReferenceNetModel,
)
from diffusers.models.referencenet.unet_2d_condition import UNet2DConditionModel
from diffusers.pipelines.referencenet.pipeline_referencenet import (
    StableDiffusionReferenceNetPipeline,
)
from utils.anonymize_faces_in_image import anonymize_faces_in_image
from utils.redact_faces import redact_faces_in_image

# todo this from lua
# creating and loading models


# --- duui

class ImageType(BaseModel):
    src: str
    height: int
    width: int
    begin: int
    end: int
class DUUIRequest(BaseModel):
    anon_type: str
    anon_degree: float
    images: dict[int, InputImage]
    redact_type: str
    blur: int
    pixel: int
    diffusion_model: str
    clip_model: str
    seed: int
    guidance: float
    inference_steps: int
    vis_input: bool
    height: int
    width: int
class DUUIResponse(BaseModel):
    output_images: dict[int, InputImage]



# ===== All the different options =====
def single_aligned_face(source_image,
                        inference_steps,
                        guidance_sclae,
                        anonymization_degree,
                        height,
                        width, vis_input)-> Image:
    """

    :param source_image: image to be anonymized
    :param inference_steps: number of inference steps
    :param guidance_sclae: the guidance scale
    :param anonymization_degree: degree of anonymization
    :param height: output image height
    :param width: input image height
    :param vis_input: weather to visualize input-output next to another


    :return: anonymized image
    """
    # generate an image that anonymizes faces
    anon_image = pipe(
        source_image=source_image,
        conditioning_image=source_image,
        num_inference_steps=inference_steps,
        guidance_scale=guidance_sclae,
        generator=generator,
        anonymization_degree=anonymization_degree,
        width=width,
        height=height,
    ).images[0]
    if vis_input:
        return combine_images([anon_image, source_image])

    return anon_image

def multiple_aligned_face(
        source_image,
        image_size,
        inference_steps,
        guidance_scale,
        anonymization_degree,
)->Image:
    """

    :param source_image: image to be anonymized
    :param image_size: image resize
    :param inference_steps: number of inference steps
    :param guidance_scale:
    :param anonymization_degree:
    :return:
    """
    # SFD (likely best results, but slower)
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, face_detector="sfd"
    )

    # generate an image that anonymizes faces
    anon_image = anonymize_faces_in_image(
        image=source_image,
        face_alignment=fa,
        pipe=pipe,
        generator=generator,
        face_image_size=image_size,
        num_inference_steps=inference_steps,
        guidance_scale=guidance_scale,
        anonymization_degree=anonymization_degree,
    )

    return anon_image

def combine_images(images):
    # Get the total width and maximum height of all images
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)

    # Create a new image with the combined width and maximum height
    new_image = Image.new("RGB", (total_width, max_height))

    # Paste each image onto the new image horizontally
    x_offset = 0
    for img in images:
        new_image.paste(img, (x_offset, 0))
        x_offset += img.width

    return new_image

def swap_faces(
        source_image,
        conditioning_image,
        inference_steps,
        guidance_scale,
        anonymization_degree,
        width,
        height,
        vis_input,
    ):
    """
    
    :param source_image: image to be anonymized
    :param conditioning_image: face to swap with
    :param inference_steps: number of infrence steps
    :param guidance_scale: guidance scale 
    :param anonymization_degree: degree of anonymization
    :param width: output image width (if not vis True)
    :param height: output image height (if not vis True)
    :param vis_input: weather to visualize input-output next to another
    :return: 
    """""
    # generate an image that swaps faces
    swap_image = pipe(
        source_image=source_image,
        conditioning_image=conditioning_image,
        num_inference_steps=inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        anonymization_degree=anonymization_degree,
        width=width,
        height=height,
    ).images[0]
    if vis_input:
        return combine_images([swap_image, source_image])
    return swap_image


def redact_faces(
        source_image,
        image_size,
        redaction_method,
        blur_strength,
        pixel_size,
        vis_input,
    )->Image:
    """

    :param source_image: image to be redacted
    :param image_size: image size for resizing
    :param redaction_method: which method to choose
    :param blur_strength: blur strength for blurring
    :param pixel_size: pixel size for pixelation
    :param vis_input: weather to visualize input-output next to another

    :return:
    """
    redact_image = redact_faces_in_image(
        source_image=source_image,
        face_image_size=image_size,
        redaction_method=redaction_method,
        blur_strength=blur_strength,
        pixel_size=pixel_size,
    )
    if vis_input:
        return combine_images([redact_image, source_image])
    return redact_image

def pil_to_b64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def b64_to_pil(b64_str):
    img_bytes = base64.b64decode(b64_str)
    return Image.open(BytesIO(img_bytes)).convert("RGB")

# === the container
app = FastAPI(
    openapi_url="/openapi.json",
    docs_url="/api",
    redoc_url=None,
    terms_of_service="https://www.texttechnologylab.org/legal_notice/",
    title="duui-face_anon",
    description="Implementation of [WACV 2025] 'Face Anonymization Made Simple' for DUUI.",
    version="0.1",
        contact={
            "name": "Coco Sittardt",
            "url": "https://texttechnologylab.org",
            "email": "sittardt@em.uni-frankfurt.de",
        },
        license_info={
            "name": "AGPL",
            "url": "http://www.gnu.org/licenses/agpl-3.0.en.html",
        },
)



@app.on_event("startup")
def startup():
    global pipe, generator
    # todo adjust so this takes other pretrained from duui
    face_model_id = "hkung/face-anon-simple"
    clip_model_id = "openai/clip-vit-large-patch14"
    sd_model_id = "stabilityai/stable-diffusion-2-1"

    unet = UNet2DConditionModel.from_pretrained(
        face_model_id, subfolder="unet", use_safetensors=True
    )
    referencenet = ReferenceNetModel.from_pretrained(
        face_model_id, subfolder="referencenet", use_safetensors=True
    )
    conditioning_referencenet = ReferenceNetModel.from_pretrained(
        face_model_id, subfolder="conditioning_referencenet", use_safetensors=True
    )
    vae = AutoencoderKL.from_pretrained(sd_model_id, subfolder="vae", use_safetensors=True)
    scheduler = DDPMScheduler.from_pretrained(
        sd_model_id, subfolder="scheduler", use_safetensors=True
    )
    feature_extractor = CLIPImageProcessor.from_pretrained(
        clip_model_id, use_safetensors=True
    )
    image_encoder = CLIPVisionModel.from_pretrained(clip_model_id, use_safetensors=True)

    pipe = StableDiffusionReferenceNetPipeline(
        unet=unet,
        referencenet=referencenet,
        conditioning_referencenet=conditioning_referencenet,
        vae=vae,
        feature_extractor=feature_extractor,
        image_encoder=image_encoder,
        scheduler=scheduler,
    )
    pipe = pipe.to("cuda")

    # todo manual seed from input
    generator = torch.manual_seed(1)
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")



@app.get("/v1/details/input_output")
def get_input_output()-> JSONResponse:
    json_item = {
       "inputs" : ["org.texttechnologylab.annotation.type.Image"],
        "outputs" : ["org.texttechnologylab.annotation.type.Image"]
    }
    json_compatible_item_data = jsonable_encoder(json_item)
    return JSONResponse(content=json_compatible_item_data)


#Load the predefined typesystem that is needed for this annotator to work
typesystem_filename = 'resources/typesystem_face_anon.xml'
with open(typesystem_filename, 'rb') as f:
    typesystem = f.read()
# Get typesystem of this annotator
@app.get("/v1/typesystem")
def get_typesystem() -> Response:
    return Response(
        content=typesystem,
        media_type="application/xml"
    )


# Load the Lua communication script
communication = "resources/communication.lua"
with open(communication, 'rb') as f:
    communication = f.read().decode("utf-8")

# Return Lua communication script
@app.get("/v1/communication_layer", response_class=PlainTextResponse)
def get_communication_layer() -> str:
    return communication

@app.post("/v1/process")
def post_process(request:DUUIRequest)-> DUUIResponse:
    """


    """
    # the base selection between which anonymization is run
    anon_type = request.anon_type
    # the amount of anonymization
    anon_degree = request.anon_degree
    # input images
    images = request.images
    # set if the anon_type is redaction, then can choose again between blur, black or pixel
    redact_type = request.redact_type
    blur = request.blur
    pixel = request.pixel
    diffusion_model = request.diffusion_model
    clip_model = request.clip_model
    seed = request.seed
    guidance = request.guidance
    inference_steps = request.inference_steps
    vis_input = request.vis_input
    height = request.height
    width = request.width

    output_images = {}

    # selection between the different anon types:
    # options: single_align, multiple_align, combine, swap, redact
    match anon_type:
        # only one image
        case "single_align":
            input_img = images[1]
            output = single_aligned_face(
                source_image=b64_to_pil(input_img.src),
                inference_steps=inference_steps,
                guidance_sclae=guidance,
                anonymization_degree=anon_degree,
                height=height,
                width=width,
                vis_input=vis_input,
            )
            output_images[1] = ImageType(
                src=pil_to_b64(output),
                height=height,
                width=width,
                begin = input_img.begin,
                end = input_img.end,
            )
        # for multiple iterate