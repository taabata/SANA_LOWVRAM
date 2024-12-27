

from flask import Flask, request
from PIL import Image
import requests, subprocess, json, os, datetime
import torch

from pipes.sana_img2img import SanaPipelineImg2Img
from pipes.sana_pag import SanaPAGPipeline
from pipes.sana_pag_img2img import SanaPipelineImg2ImgPAG
import numpy as np
import argparse
from signal import SIGKILL

parser = argparse.ArgumentParser()

parser.add_argument("--prompt","-p")
parser.add_argument("--steps","-s")
parser.add_argument("--cfg","-c")

args = parser.parse_args()

steps = args.steps
cfg = args.cfg if args.cfg else 5.0
prompt = args.prompt

params = {
    "steps":steps,
    "cfg":cfg,
    "prompt":prompt
}



app = Flask(__name__)

embeds = {}
processid = ""
pid = os.getpid()
print("pid:   ",pid)
flag = True
output_image = ""

@app.route("/endApp",methods=["POST","GET"])
def endApp():
    global pid
    print("ending....: ",pid)
    os.kill(pid,SIGKILL)
    return {}

@app.route("/getSharedData",methods=["POST","GET"])
def getSharedData():
    global flag, output_image
    return {"flag":flag,"image":output_image,"embeds":embeds}

@app.route("/encode",methods=["POST","GET"])
def encode():
    global params
    print("encoding......")
    subprocess.Popen(["python3","getembeds.py","--prompt",request.json["prompt"],"--model",request.json["model"]])
    return {}

@app.route("/getEmbeds",methods=["POST","GET"])
def getEmbeds():
    global embeds,flag
    embeds = request.json
    flag = False
    return {}

@app.route("/diffuse",methods=["POST","GET"])
def diffuse():
    global embeds,params, output_image, flag
    print("diffusing......")
    if request.json["img2img"] == "enable":
        pipe = SanaPipelineImg2ImgPAG.from_pretrained(
            request.json["model"],
            torch_dtype = torch.float16,
            text_encoder=None
        )
        pipe.enable_model_cpu_offload()
        image = pipe(
            prompt_embeds = torch.Tensor(np.array(json.loads(request.json["embeds"]["prompt_embeds"]))).half().to('cuda'),
            prompt_attention_mask= torch.Tensor(np.array(json.loads(request.json["embeds"]["prompt_attention_mask"]))).half().to('cuda'),
            negative_prompt_embeds=torch.Tensor(np.array(json.loads(request.json["embeds"]["negative_prompt_embeds"]))).half().to('cuda'),
            negative_prompt_attention_mask = torch.Tensor(np.array(json.loads(request.json["embeds"]["negative_prompt_attention_mask"]))).half().to('cuda'),
            height=int(request.json["height"]),
            width=int(request.json["width"]),
            guidance_scale=float(request.json["cfg"]),
            pag_scale=float(request.json["pag_scale"]),
            num_inference_steps=int(request.json["steps"]),
            image=json.loads(request.json["image"]),
            strength=float(request.json["strength"])
        )[0]
    else:
        pipe = SanaPAGPipeline.from_pretrained(
            request.json["model"],
            torch_dtype = torch.float16,
            text_encoder=None
        )
        pipe.enable_model_cpu_offload()
        image = pipe(
            prompt_embeds = torch.Tensor(np.array(json.loads(request.json["embeds"]["prompt_embeds"]))).half().to('cuda'),
            prompt_attention_mask= torch.Tensor(np.array(json.loads(request.json["embeds"]["prompt_attention_mask"]))).half().to('cuda'),
            negative_prompt_embeds=torch.Tensor(np.array(json.loads(request.json["embeds"]["negative_prompt_embeds"]))).half().to('cuda'),
            negative_prompt_attention_mask = torch.Tensor(np.array(json.loads(request.json["embeds"]["negative_prompt_attention_mask"]))).half().to('cuda'),
            height=int(request.json["height"]),
            width=int(request.json["width"]),
            guidance_scale=float(request.json["cfg"]),
            pag_scale=float(request.json["pag_scale"]),
            num_inference_steps=int(request.json["steps"]),
        )[0]
    output_image = json.dumps(np.array(image[0]).tolist())
    flag = False
    return{}




if __name__ == "__main__":
    app.run()
