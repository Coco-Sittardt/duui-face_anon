import argparse

import face_alignment
import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModel

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

# todo adjust this to take the Lua arguments and parse them
def parse_args():
    parser = argparse.ArgumentParser(description="Inference")
    # todo arg model diffusion
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2-1",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    # todo arg model clip
    parser.add_argument(
        "--pretrained_clip_model_name_or_path",
        type=str,
        default="openai/clip-vit-large-patch14",
        required=False,
        help="Path to pretrained CLIP model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        required=True,
        help="Path to the model trained by yourself",
    )
    parser.add_argument(
        "--dataset_loading_script_path",
        type=str,
        default=None,
        required=True,
        help="Path to the dataset loading script file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test-infer/",
        help="The output directory where predictions are saved",
    )
    # todo arg resulution
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="The resolution for input images, all the images in the test dataset will be resized to this resolution",
    )
    # todo some other args (scale, seed etc)
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible inference.")
    parser.add_argument(
        "--anonymization_degree_start",
        type=float,
        default=0.0,
        help="Increasing the anonymization scale value encourages the model to produce images that diverge significantly from the conditioning image.",
    )
    parser.add_argument("--anonymization_degree_end", type=float, default=0.0)
    parser.add_argument("--num_anonymization_degrees", type=int, default=1)
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=None,
        help="Truncate the number of test examples to this value if set.",
    )
    # todo arg vis input
    parser.add_argument(
        "--vis_input",
        action="store_true",
        help="If set, save the input and generated images together as a single output image for easy visualization",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=1,
        help=(
            "The batch size for the test dataloader per device should be set to 1."
            "This setting does not affect performance, no matter how large the batch size is."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )

    args = parser.parse_args()
    return args


def single_aligned_face(source_image,
                        inference_steps=25,
                        guidance_sclae=4.0,
                        anonymization_degree=1.25,
                        height=512,
                        width=512, vis_input=False)-> Image:
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
        image_size=512,
        inference_steps=25,
        guidance_scale=4.0,
        anonymization_degree=1.25,
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
        inference_steps=200,
        guidance_scale=4.0,
        anonymization_degree=0.00,
        width=512,
        height=512,
        vis_input=False,
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
        image_size=512,
        redaction_method="blur",
        blur_strength=51,
        pixel_size=16,
        vis_input=False,
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
