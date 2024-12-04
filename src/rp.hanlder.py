'''
Contains the handler function that will be called by the serverless.
'''

import os
import base64
import concurrent.futures

import torch
from diffusers import StableDiffusion3Pipeline
from transformers import T5EncoderModel, BitsAndBytesConfig


import runpod
from runpod.serverless.utils import rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate

from huggingface_hub import login
login(token="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")
from rp_schemas import INPUT_SCHEMA

torch.cuda.empty_cache()

# ------------------------------- Model Handler ------------------------------ #


class ModelHandler:
    def __init__(self):
        self.quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model_id = "stabilityai/stable-diffusion-3-medium-diffusers"

    def load_models(self):
        text_encoder = T5EncoderModel.from_pretrained(
            self.model_id,
            subfolder="text_encoder_3",
            quantization_config=self.quantization_config,
        )
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            self.model_id,
            text_encoder_3=text_encoder,
            device_map="balanced",
            torch_dtype=torch.float16
        )

MODELS = ModelHandler()
MODELS.load_models()
# ---------------------------------- Helper ---------------------------------- #


def _save_and_upload_images(images, job_id):
    os.makedirs(f"/{job_id}", exist_ok=True)
    image_urls = []
    for index, image in enumerate(images):
        image_path = os.path.join(f"/{job_id}", f"{index}.png")
        image.save(image_path)

        if os.environ.get('BUCKET_ENDPOINT_URL', False):
            image_url = rp_upload.upload_image(job_id, image_path)
            image_urls.append(image_url)
        else:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(
                    image_file.read()).decode("utf-8")
                image_urls.append(f"data:image/png;base64,{image_data}")

    rp_cleanup.clean([f"/{job_id}"])
    return image_urls





@torch.inference_mode()
def generate_image(job):
    '''
    Generate an image from text using your Model
    '''
    job_input = job["input"]

    # Input validation
    validated_input = validate(job_input, INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    job_input = validated_input['validated_input']

    starting_image = job_input['image_url']

    if job_input['seed'] is None:
        job_input['seed'] = int.from_bytes(os.urandom(2), "big")

    generator = torch.Generator("cuda").manual_seed(job_input['seed'])

    
    output = MODELS.pipe(
        prompt=job_input['prompt'],
        negative_prompt=job_input['negative_prompt'],
        num_inference_steps=job_input['refiner_inference_steps'],
        height=job_input['height'],
        width=job_input['width'],
        guidance_scale=job_input['guidance_scale'],
        num_images_per_prompt=job_input['num_images'],
        generator=generator
    ).images

    image_urls = _save_and_upload_images(output, job['id'])

    results = {
        "images": image_urls,
        "image_url": image_urls[0],
        "seed": job_input['seed']
    }

    if starting_image:
        results['refresh_worker'] = True

    return results


runpod.serverless.start({"handler": generate_image})
