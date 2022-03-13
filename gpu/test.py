# -*- coding: utf-8 -*-
"""3_13_2021_DALL·E mini - Inference_TPU_issue.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1uLhRwBrUR0n4ETRFo01UNl57kES0PHWn

# DALL·E mini - Inference pipeline

*Generate images from a text prompt*

<img src="https://github.com/borisdayma/dalle-mini/blob/main/img/logo.png?raw=true" width="200">

This notebook illustrates [DALL·E mini](https://github.com/borisdayma/dalle-mini) inference pipeline.

Just want to play? Use [the demo](https://huggingface.co/spaces/flax-community/dalle-mini).

For more understanding of the model, refer to [the report](https://wandb.ai/dalle-mini/dalle-mini/reports/DALL-E-mini--Vmlldzo4NjIxODA).

## 🛠️ Installation and set-up
"""

# Install required libraries
# !pip install -q transformers
# !pip install -q git+https://github.com/patil-suraj/vqgan-jax.git
# !pip install -q git+https://github.com/borisdayma/dalle-mini.git

# !pip freeze | grep jax
# !pip freeze | grep tpu



"""We load required models:
* dalle·mini for text to encoded images
* VQGAN for decoding images
* CLIP for scoring predictions
"""

# pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# !pip freeze | grep jax
# !pip freeze | grep tpu

# !pip freeze | grep tpu

# Model references

# dalle-mini
DALLE_MODEL = "dalle-mini/dalle-mini/model-1reghx5l:latest"  # can be wandb artifact or 🤗 Hub or local folder
DALLE_COMMIT_ID = None

# VQGAN model
VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"

# CLIP model
CLIP_REPO = "openai/clip-vit-base-patch16"
CLIP_COMMIT_ID = None

import jax.tools.colab_tpu
# # jax.tools.colab_tpu.setup_tpu('tpu_driver-0.1dev20211031')
jax.tools.colab_tpu.setup_tpu()

import jax
import jax.numpy as jnp

# check how many devices are available
jax.local_device_count()

# type used for computation - use bfloat16 on TPU's
dtype = jnp.bfloat16 if jax.local_device_count() == 8 else jnp.float32

# TODO: fix issue with bfloat16
# dtype = jnp.float32

dtype

# model

# Load models & tokenizer
from dalle_mini.model import DalleBart, DalleBartTokenizer
from vqgan_jax.modeling_flax_vqgan import VQModel
from transformers import CLIPProcessor, FlaxCLIPModel
import wandb
from time import time
t0 = time()
# Load dalle-mini
model = DalleBart.from_pretrained(
    DALLE_MODEL, revision=DALLE_COMMIT_ID, dtype=dtype, abstract_init=True
)
tokenizer = DalleBartTokenizer.from_pretrained(DALLE_MODEL, revision=DALLE_COMMIT_ID)

# Load VQGAN
vqgan = VQModel.from_pretrained(VQGAN_REPO, revision=VQGAN_COMMIT_ID)

# Load CLIP
clip = FlaxCLIPModel.from_pretrained(CLIP_REPO, revision=CLIP_COMMIT_ID)
processor = CLIPProcessor.from_pretrained(CLIP_REPO, revision=CLIP_COMMIT_ID)
print("Loading Done! Takes: {} sec".format(time()-t0))

"""Model parameters are replicated on each device for faster inference."""

from flax.jax_utils import replicate
import numpy as np
# # convert model parameters for inference if requested
# if dtype == jnp.bfloat16:
#     model.params = model.to_bf16(model.params)
def to_jnp(param):
    if isinstance(param, np.ndarray):
        param = jnp.array(param).astype(dtype)
    return param

if dtype == jnp.bfloat16:
  model.params = jax.tree_map(to_jnp, model.params)

model_params = replicate(model.params)
vqgan_params = replicate(vqgan.params)
clip_params = replicate(clip.params)

dtype

"""Model functions are compiled and parallelized to take advantage of multiple devices."""

from functools import partial

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

"""Keys are passed to the model on each device to generate unique inference per device."""

import random

# create a random key
seed = random.randint(0, 2**32 - 1)
key = jax.random.PRNGKey(seed)

"""## 🖍 Text Prompt

Our model may require to normalize the prompt.
"""

from dalle_mini.text import TextNormalizer

text_normalizer = TextNormalizer() if model.config.normalize_text else None

"""Let's define a text prompt."""

prompt = "a blue table"

processed_prompt = text_normalizer(prompt) if model.config.normalize_text else prompt
processed_prompt

"""We tokenize the prompt."""

tokenized_prompt = tokenizer(
    processed_prompt,
    return_tensors="jax",
    padding="max_length",
    truncation=True,
    max_length=128,
).data
tokenized_prompt

"""Notes:

* `0`: BOS, special token representing the beginning of a sequence
* `2`: EOS, special token representing the end of a sequence
* `1`: special token representing the padding of a sequence when requesting a specific length

Finally we replicate it onto each device.
"""

tokenized_prompt = replicate(tokenized_prompt)

"""## 🎨 Generate images

We generate images using dalle-mini model and decode them with the VQGAN.
"""

# number of predictions
n_predictions = 32

# We can customize top_k/top_p used for generating samples
gen_top_k = None
gen_top_p = None

# from jax import vmap
# from jax import pmap
# x = jnp.array([1., 2., 3.])
# y = jnp.array([2., 4., 6.])

# print(vmap(jnp.add)(x, y))
# print(pmap(jnp.add)(x, y))

from flax.training.common_utils import shard_prng_key
import numpy as np
from PIL import Image
from tqdm.notebook import trange
def process(*args):
    key=args[0]
    model_params = args[1]
    gen_top_k = args[2]
    gen_top_p = args[3]
    vqgan_params = args[4]
    # print(args)
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
    decoded_images = [Image.fromarray(np.asarray(img * 255, dtype=np.uint8)) for img in decoded_images]
    return decoded_images
# generate images
# images = []
# for i in trange(n_predictions // jax.device_count()):
#     images+=process(key,
#             model_params,
#             gen_top_k,
#             gen_top_p,
#             vqgan_params)
    # get a new key
    # key, subkey = jax.random.split(key)
    # # generate images
    # encoded_images = p_generate(
    #     tokenized_prompt, shard_prng_key(subkey), model_params, gen_top_k, gen_top_p
    # )
    # # remove BOS
    # encoded_images = encoded_images.sequences[..., 1:]
    # # decode images
    # decoded_images = p_decode(encoded_images, vqgan_params)
    # decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
    # for img in decoded_images:
    #     images.append(Image.fromarray(np.asarray(img * 255, dtype=np.uint8)))

process(key,
            model_params,
            gen_top_k,
            gen_top_p,
            vqgan_params)

# import torch.multiprocessing as mp
# for rank in range(8):
#         p = mp.Process(target=process, args=[(key,
#             model_params,
#             gen_top_k,
#             gen_top_p,
#             vqgan_params) for i in  range(0,n_predictions)])
#         p.start()
#         processes.append(p)
#     for p in processes:
#         p.join()

from tqdm import tqdm
# from multiprocessing import Pool
# from functools import partial
# import time
# import random
# def run_apply_async_multiprocessing(func, argument_list, num_processes):

#     pool = Pool(processes=num_processes)

#     jobs = [pool.apply_async(func=func, args=(*argument,)) if isinstance(argument, tuple) else pool.apply_async(func=func, args=(argument,)) for argument in argument_list]
#     pool.close()
#     result_list_tqdm = []
#     for job in tqdm(jobs):
#         result_list_tqdm.append(job.get())

#     return result_list_tqdm

# def func(a,b,c,d,e,f):

#   return a+b+c+d+e+f
                                        
      # for img in decoded_images:
      #     images.append())

import torch.multiprocessing as mp
ctx = mp.get_context('spawn')
pool = ctx.Pool(processes=1)
jobs = [pool.apply_async(func=process, args=(key,
            model_params,
            gen_top_k,
            gen_top_p,
            vqgan_params)) for _ in range(2) ]
pool.close()
res = []
for job in tqdm(jobs):
  res.append(job.get())
print(len(res))

# from multiprocessing import get_context

import os
os.cpu_count()

# arg_list = [(key,
#             model_params,
#             gen_top_k,
#             gen_top_p,
#             vqgan_params) for i in  range(0,n_predictions)]
# print(len(arg_list))

# if __name__=='__main__':
result_list = run_apply_async_multiprocessing(func=process, argument_list= arg_list, num_processes=1)

result_list = run_apply_async_multiprocessing(func=process, argument_list=[(5,5,5,5,5,5) for i in  range(0,10000)], num_processes=8)

"""Let's calculate their score with CLIP."""

from flax.training.common_utils import shard

# get clip scores
clip_inputs = processor(
    text=[prompt] * jax.device_count(),
    images=images,
    return_tensors="np",
    padding="max_length",
    max_length=77,
    truncation=True,
).data
logits = p_clip(shard(clip_inputs), clip_params)
logits = logits.squeeze().flatten()

"""Let's display images ranked by CLIP score."""

print(f"Prompt: {prompt}\n")
for idx in logits.argsort()[::-1]:
    display(images[idx])
    print(f"Score: {logits[idx]:.2f}\n")
