import requests, json, argparse
import numpy as np
import torch
from pipes.sana_img2img import SanaPipelineImg2Img


parser = argparse.ArgumentParser()

parser.add_argument("--prompt","-p")
parser.add_argument("--model","-m")

args = parser.parse_args()

model_name = args.model

pipe = SanaPipelineImg2Img.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    transformer=None,
    vae=None

)
pipe.to('cpu')
prompt = args.prompt
embeds = pipe(
    prompt=prompt,
    height=512,
    width=512,
    guidance_scale=5.0,
    num_inference_steps=2,
    textonly=True
)


data = {}
for i in embeds:
    embs = json.dumps(embeds[i].type(torch.float32).numpy().astype(np.float32).tolist())
    data[i] = embs
req = requests.post("http://127.0.0.1:5000/getEmbeds",json=data,headers={'accept': 'content_type_value'})
