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
    torch.set_float32_matmul_precision("high")

    torch._inductor.config.conv_1x1_as_mm = True
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.epilogue_fusion = False
    torch._inductor.config.coordinate_descent_check_all_directions = True

    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        # text_encoder_3=text_encoder,
        # device_map="balanced",
        torch_dtype=torch.float16
    )
    # .to("cuda")
    pipe.set_progress_bar_config(disable=True)
    pipe.transformer.to(memory_format=torch.channels_last)
    pipe.vae.to(memory_format=torch.channels_last)
    pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)
    pipe.vae.decode = torch.compile(pipe.vae.decode, mode="max-autotune", fullgraph=True)

    return pipe


if __name__ == "__main__":
    get_diffusion_pipelines()
