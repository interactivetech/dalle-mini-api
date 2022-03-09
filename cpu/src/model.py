import jax
import jax.numpy as jnp

# type used for computation - use bfloat16 on TPU's
dtype = jnp.bfloat16 if jax.local_device_count() == 8 else jnp.float32

# TODO: fix issue with bfloat16
dtype = jnp.float32

# Load models & tokenizer
from dalle_mini.model import DalleBart, DalleBartTokenizer
from vqgan_jax.modeling_flax_vqgan import VQModel
from transformers import CLIPProcessor, FlaxCLIPModel
# import wandb
from time import time
import random
from dalle_mini.text import TextNormalizer
from flax.training.common_utils import shard_prng_key
import numpy as np
from PIL import Image
from tqdm import trange
from flax.training.common_utils import shard
from flax.jax_utils import replicate
from base64 import encodebytes
import io
import wandb
import os
wandb.login(key='7ac71ae3a1191ddb6de757ad0c093bb311f44a04')
# DALLE_WEIGHTS='/Users/mendeza/Documents/github_projects/dalle-mini/tools/inference/artifacts/model-1reghx5l:v9'
# VQGAN_WEIGHTS='/Users/mendeza/Documents/github_projects/dalle-mini/tools/inference/vqgan_weights'
# CLIP_WEIGHTS='/Users/mendeza/Documents/github_projects/dalle-mini/tools/inference/clip_weights'
# PROCESSOR_WEIGHTS='/Users/mendeza/Documents/github_projects/dalle-mini/tools/inference/clip_processor_weights'

DALLE_WEIGHTS='model-1reghx5l:v9/'
VQGAN_WEIGHTS='vqgan_weights/'
CLIP_WEIGHTS='clip_weights/'
PROCESSOR_WEIGHTS='clip_processor_weights/'
from functools import partial

def load_model2():
    # dalle-mini
    DALLE_MODEL = "dalle-mini/dalle-mini/model-1reghx5l:latest"  # can be wandb artifact or 🤗 Hub or local folder
    DALLE_COMMIT_ID = None

    # VQGAN model
    VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
    VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"

    # CLIP model
    CLIP_REPO = "openai/clip-vit-base-patch16"
    CLIP_COMMIT_ID = None
    # # Load dalle-mini
    model = DalleBart.from_pretrained(
        DALLE_MODEL, revision=DALLE_COMMIT_ID, dtype=dtype, abstract_init=True
    )
    tokenizer = DalleBartTokenizer.from_pretrained(DALLE_MODEL, revision=DALLE_COMMIT_ID)

    # Load VQGAN
    vqgan = VQModel.from_pretrained(VQGAN_REPO, revision=VQGAN_COMMIT_ID)

    # Load CLIP
    clip = FlaxCLIPModel.from_pretrained(CLIP_REPO, revision=CLIP_COMMIT_ID)
    processor = CLIPProcessor.from_pretrained(CLIP_REPO, revision=CLIP_COMMIT_ID)
    model.save_pretrained('model-1reghx5l:v9')
    vqgan.save_pretrained('vqgan_weights')
    clip.save_pretrained('clip_weights')
    processor.save_pretrained('clip_processor_weights')# Needs to be same dir
    
    # convert model parameters for inference if requested
    if dtype == jnp.bfloat16:
        model.params = model.to_bf16(model.params)

    model_params = replicate(model.params)
    vqgan_params = replicate(vqgan.params)
    clip_params = replicate(clip.params)

    return model,tokenizer,vqgan,clip,processor,model_params, vqgan_params, clip_params

def load_model():
    '''
    Load Model
    '''
    t0=time()
    print("Loading Dalle Bart...")
    model = DalleBart.from_pretrained(
        DALLE_WEIGHTS
    )
    print("Done: {:.3f} seconds".format(time()-t0))

    print("Loading Dalle Tokenizer...")
    t0=time()
    tokenizer = DalleBartTokenizer.from_pretrained(
        DALLE_WEIGHTS
    )
    print("Done: {:.3f} seconds".format(time()-t0))
    # # # Load VQGAN
    print("Load VQGAN...")
    t0=time()

    vqgan = VQModel.from_pretrained(VQGAN_WEIGHTS)
    print("Done: {:.3f} seconds".format(time()-t0))
    # # Load CLIP
    print("Loading Clip...")
    t0=time()

    clip = FlaxCLIPModel.from_pretrained(CLIP_WEIGHTS)
    processor = CLIPProcessor.from_pretrained(PROCESSOR_WEIGHTS)
    print("Done: {:.3f} seconds".format(time()-t0))

    from flax.jax_utils import replicate

    # convert model parameters for inference if requested
    if dtype == jnp.bfloat16:
        model.params = model.to_bf16(model.params)

    model_params = replicate(model.params)
    vqgan_params = replicate(vqgan.params)
    clip_params = replicate(clip.params)

    return model,tokenizer,vqgan,clip,processor,model_params, vqgan_params, clip_params

def load_model():
    '''
    Load Model
    '''
    t0=time()
    print("Loading Dalle Bart...")
    model = DalleBart.from_pretrained(
        DALLE_WEIGHTS
    )
    print("Done: {:.3f} seconds".format(time()-t0))

    print("Loading Dalle Tokenizer...")
    t0=time()
    tokenizer = DalleBartTokenizer.from_pretrained(
        DALLE_WEIGHTS
    )
    print("Done: {:.3f} seconds".format(time()-t0))
    # # # Load VQGAN
    print("Load VQGAN...")
    t0=time()

    vqgan = VQModel.from_pretrained(VQGAN_WEIGHTS)
    print("Done: {:.3f} seconds".format(time()-t0))
    # # Load CLIP
    print("Loading Clip...")
    t0=time()

    clip = FlaxCLIPModel.from_pretrained(CLIP_WEIGHTS)
    processor = CLIPProcessor.from_pretrained(PROCESSOR_WEIGHTS)
    print("Done: {:.3f} seconds".format(time()-t0))

    from flax.jax_utils import replicate

    # convert model parameters for inference if requested
    if dtype == jnp.bfloat16:
        model.params = model.to_bf16(model.params)

    model_params = replicate(model.params)
    vqgan_params = replicate(vqgan.params)
    clip_params = replicate(clip.params)

    return model,tokenizer,vqgan,clip,processor,model_params, vqgan_params, clip_params

def generate_images(
                prompt,
                n_images,
                gen_top_k,
                model,
                tokenizer,
                vqgan,
                clip,
                processor,
                model_params,
                vqgan_params,
                clip_params,
                gen_top_k=4
                ):
    '''
    '''
    # print("Prompt: {}".format(prompt))
    # model inference
    @partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4))
    def p_generate(tokenized_prompt, key, params, top_k, top_p):
        return model.generate(
            **tokenized_prompt,
            do_sample=True,
            num_beams=1,
            prng_key=key,
            params=params,
            top_k=top_k,
            top_p=top_p,
            max_length=257
        )


    # decode images
    @partial(jax.pmap, axis_name="batch")
    def p_decode(indices, params):
        return vqgan.decode_code(indices, params=params)


    # score images
    @partial(jax.pmap, axis_name="batch")
    def p_clip(inputs, params):
        logits = clip(params=params, **inputs).logits_per_image
        return logits

    # create a random key
    seed = random.randint(0, 2**32 - 1)
    key = jax.random.PRNGKey(seed)
    text_normalizer = TextNormalizer() if model.config.normalize_text else None
    # prompt = "a blue table"

    processed_prompt = text_normalizer(prompt) if model.config.normalize_text else prompt

    tokenized_prompt = tokenizer(
    processed_prompt,
    return_tensors="jax",
    padding="max_length",
    truncation=True,
    max_length=128,
    ).data

    tokenized_prompt = replicate(tokenized_prompt)

    # number of predictions
    n_predictions = n_images

    # We can customize top_k/top_p used for generating samples
    gen_top_k = gen_top_k
    gen_top_p = None

    # generate images
    images = []
    for i in trange(n_predictions // jax.device_count()):
        # get a new key
        key, subkey = jax.random.split(key)
        # generate images
        encoded_images = p_generate(
            tokenized_prompt, shard_prng_key(subkey), model_params, gen_top_k, gen_top_p
        )
        # remove BOS
        encoded_images = encoded_images.sequences[..., 1:]
        # decode images
        decoded_images = p_decode(encoded_images, vqgan_params)
        decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
        for img in decoded_images:
            images.append(Image.fromarray(np.asarray(img * 255, dtype=np.uint8)))

    # get clip scores
    clip_inputs = processor(
        text=[prompt] * jax.device_count(),
        images=images,
        return_tensors="np",
        padding="max_length",
        max_length=77,
        truncation=True).data
    logits = p_clip(shard(clip_inputs), clip_params)
    logits = logits.squeeze().flatten()
    print(f"Prompt: {prompt}\n")
    imgs = []
    for idx in logits.argsort()[::-1]:
        # display(images[idx])
        # print(f"Score: {logits[idx]:.2f}\n")
        print(f"{idx}-Score: {logits[idx]:.2f}\n")
        imgs.append(images[idx])
    return imgs

def get_response_image(im):
    # pil_img = Image.fromarra(image_path, mode='r') # reads the PIL image
    byte_arr = io.BytesIO()
    im.save(byte_arr, format='jpeg') # convert the PIL image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
    return encoded_img

if __name__ == '__main__':
    model,tokenizer,vqgan,clip,processor,model_params, vqgan_params, clip_params = load_model()
    n_images=2
    gen_top_k=4
    images =generate_images(
                    "A red cat",
                    n_images,
                    gen_top_k,
                    model,
                    tokenizer,
                    vqgan,
                    clip,
                    processor,
                    model_params, 
                    vqgan_params, 
                    clip_params)
    print(get_response_image(images[0]) )
