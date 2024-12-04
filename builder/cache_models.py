# builder/model_fetcher.py

from huggingface_hub import login
login(token="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")
import torch
# from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, AutoencoderKL
from transformers import T5EncoderModel, BitsAndBytesConfig
from diffusers import StableDiffusion3Pipeline

def fetch_pretrained_model(model_class, model_name, **kwargs):
    '''
    Fetches a pretrained model from the HuggingFace model hub.
    '''
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return model_class.from_pretrained(model_name, **kwargs)
        except OSError as err:
            if attempt < max_retries - 1:
                print(
                    f"Error encountered: {err}. Retrying attempt {attempt + 1} of {max_retries}...")
            else:
                raise


def get_diffusion_pipelines():
    '''
    Fetches the Stable Diffusion XL pipelines from the HuggingFace model hub.
    '''
    

    # pipe = fetch_pretrained_model(StableDiffusionXLPipeline,
    #                               "stabilityai/stable-diffusion-xl-base-1.0", **common_args)
    # vae = fetch_pretrained_model(
    #     AutoencoderKL, "madebyollin/sdxl-vae-fp16-fix", **{"torch_dtype": torch.float16}
    # )
    # print("Loaded VAE")
    # refiner = fetch_pretrained_model(StableDiffusionXLImg2ImgPipeline,
    #                                  "stabilityai/stable-diffusion-xl-refiner-1.0", **common_args)
    # quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    # config = {
    #         "quantization_config":quantization_config,
    #         "subfolder": "text_encoder_3"
    #     }
    # t5 = fetch_pretrained_model(T5EncoderModel, "stabilityai/stable-diffusion-3-medium-diffusers", **config)

    # common_args = {
    #     "text_encoder_3": t5,
    #     "device_map": "balanced",
    #     "torch_dtype": torch.float16,
    # }
    # pipe = fetch_pretrained_model(StableDiffusion3Pipeline,"stabilityai/stable-diffusion-3-medium-diffusers",**common_args)
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
    text_encoder = T5EncoderModel.from_pretrained(
    model_id,
    subfolder="text_encoder_3",
    quantization_config=quantization_config,
    )
    pipe = StableDiffusion3Pipeline.from_pretrained(
    model_id,
    text_encoder_3=text_encoder,
    device_map="balanced",
    torch_dtype=torch.float16
    )

    return pipe


if __name__ == "__main__":
    get_diffusion_pipelines()
